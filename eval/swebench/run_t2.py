#!/usr/bin/env python3
"""T2 memory track: SWE-Bench-CL, 273 Verified tasks in 8 repo sequences.

Arms: M (mini-swe-agent, no memory), M-notes (mini plus /memory/NOTES.md,
carried within a repo, 4k-token cap), G-fresh (Graphirm, new graph per
task), G-persist (Graphirm, one graph per repo). A job is one (repo, arm)
pair and walks that repo's tasks strictly in the published order. Memory
is keyed by repo, so it starts empty at every repo switch. Jobs run in a
pool; arms of one repo are kept within DRIFT tasks of each other so only a
few Verified images are on disk at once.

Grading: site-packages swebench 4.1.0 on the swebench/ Verified images
(grade_verified.py). The three tasks whose gold patch fails in that
harness run and are graded but carry scored=false. Outputs stay under
/home/krs/graphirm-runs/t2. Stdout progress lines omit resolve flags.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import docker

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters.memory_notes import carry, token_count
from config_guard import graphirm_problem, mini_problem, provider_error, read_run_metadata
from durable import (
    append_ledger,
    backup_record,
    pending_backup_count,
    push_backup,
    remember_failed_backup,
    retry_failed_backups,
    runs_dir,
)
from failure_class import CONFIG, INFRASTRUCTURE, REQUEST, classify_arm
from grade_verified import SETTLED, grade_patch, image_name, remove_task_image
from reconstruct_check import sampled as reconstruction_sampled
from run_arm import (
    crlf_safe_submission,
    load_dotenv,
    load_prices,
    mini_costs,
    read_patch,
    usage_cost,
    render_notes_task,
    render_task,
    strip_test_hunks,
    write_patch,
)
from run_t1 import BINARY, LOCK, MINI_CONFIG, REPO, RUNNER, SWEBENCH_YAML, merge_run_checks
from run_t1 import preflight as _preflight

ARMS = ["M", "M-notes", "G-fresh", "G-persist"]
TRACK = "t2"
SPEND_CAP = 40.0  # primary (frozen-price) cost, as T1's limit was measured
BILLED_STOP = 100.0  # account safety stop on OpenRouter's billed cost
DISK_LIMIT = 78.0
TOTAL = 273
WORKERS = int(os.environ.get("T2_WORKERS", "8"))
DRIFT = int(os.environ.get("T2_DRIFT", "2"))
PROGRESS_EVERY = 25
GRADE_TIMEOUT = 1800
# Gold patch does not resolve under swebench 4.1.0 (t2-validity). Run, not scored.
UNSCORED = {"django__django-10097", "sphinx-doc__sphinx-9711", "astropy__astropy-7606"}

load_dotenv(REPO / ".env")
ledger_lock = threading.Lock()
streak_lock = threading.Lock()
preflight_lock = threading.Lock()
progress_lock = threading.Lock()
drift_cv = threading.Condition()
stop_event = threading.Event()
streaks = {"request": 0}
progress: dict[str, dict[str, int]] = {}
dead_arms: dict[str, set[str]] = {}
tasks_finished = {"count": 0, "reported": 0}


def preflight(root: Path, image: str) -> None:
    with preflight_lock:
        _preflight(root, image)


def used_pct() -> float:
    usage = shutil.disk_usage("/")
    return 100.0 * usage.used / usage.total


def image_present(client, image: str) -> bool:
    try:
        client.images.get(image)
        return True
    except docker.errors.ImageNotFound:
        return False
    except Exception:
        return False


def light_reclaim() -> None:
    subprocess.run(["docker", "image", "prune", "-f"], check=False, capture_output=True)
    subprocess.run(["docker", "container", "prune", "-f"], check=False, capture_output=True)


def wait_disk(client, image: str) -> None:
    """Block before pulling a new image while the disk is above the line.
    An image already on disk never waits, so the slowest arm keeps moving."""
    reclaimed = False
    while used_pct() >= DISK_LIMIT and not image_present(client, image):
        if not reclaimed:
            light_reclaim()
            reclaimed = True
            continue
        print(f"DISK {used_pct():.1f} waiting", flush=True)
        time.sleep(30)
        if stop_event.is_set():
            return


def repo_key(repo: str) -> str:
    return repo.replace("/", "__")


def spent(path: Path, field: str = "cost_usd") -> float:
    """Summed over every ledger row (retries included). The $40 track limit is
    measured on cost_usd, the primary formula at frozen prices, as T1's was.
    field="cost_usd_billed" gives OpenRouter's bill for the $100 safety stop."""
    if not path.exists():
        return 0.0
    total = 0.0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        value = row.get(field)
        if value is None and field == "cost_usd_billed":
            value = row.get("cost_usd")
        try:
            total += float(value or 0.0)
        except (TypeError, ValueError):
            continue
    return total


def latest_rows(path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if not path.exists():
        return latest
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[record["instance_id"]] = record
    return latest


def arm_done(record: dict | None) -> bool:
    if not record:
        return False
    return bool(record.get("request_caused")) or (
        record.get("complete") is True and not record.get("error")
    )


# ---------------------------------------------------------------- instances


def load_sequences() -> list[dict]:
    """Lock-file sequences joined with SWE-bench Verified rows, in order."""
    from datasets import load_dataset

    lock = json.loads(LOCK.read_text())["memory_tracks"]["swe_bench_cl"]["sequences"]
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    found = {row["instance_id"]: dict(row) for row in dataset}
    sequences = []
    for seq in lock:
        tasks = []
        for item in sorted(seq["tasks"], key=lambda t: t["position"]):
            row = found.get(item["instance_id"])
            if row is None:
                raise SystemExit(f"missing verified instance {item['instance_id']}")
            if row["base_commit"] != item["base_commit"]:
                raise SystemExit(f"base_commit differs from lock for {item['instance_id']}")
            row["position"] = item["position"]
            row["n_tasks"] = seq["n_tasks"]
            row["lock_difficulty"] = item.get("difficulty")
            row["image_name"] = image_name(row)
            row["scored"] = row["instance_id"] not in UNSCORED
            tasks.append(row)
        if len(tasks) != seq["n_tasks"]:
            raise SystemExit(f"{seq['repo']} has {len(tasks)} tasks, lock says {seq['n_tasks']}")
        sequences.append({"repo": seq["repo"], "tasks": tasks})
    total = sum(len(s["tasks"]) for s in sequences)
    if total != TOTAL:
        raise SystemExit(f"expected {TOTAL} tasks, found {total}")
    return sequences


# ---------------------------------------------------------------- memory


def notes_dir(root: Path, repo: str) -> Path:
    return root / "memory" / "M-notes" / repo


def persist_graph(root: Path, repo: str) -> Path:
    return root / "memory" / "G-persist" / repo / "graph.db"


def prepare_notes(root: Path, instance: dict, out: Path) -> Path:
    repo = repo_key(instance["repo"])
    carried = notes_dir(root, repo) / "NOTES.md"
    mount = out / "M-notes.memory"
    if mount.exists():
        shutil.rmtree(mount)
    log = carry(carried, mount / "NOTES.md")
    (out / "M-notes.carry.json").write_text(json.dumps(log, indent=2) + "\n")
    os.chmod(mount, 0o777)
    os.chmod(mount / "NOTES.md", 0o666)
    return mount


def promote_notes(root: Path, instance: dict, out: Path) -> None:
    repo = repo_key(instance["repo"])
    src = out / "M-notes.memory" / "NOTES.md"
    dest = notes_dir(root, repo) / "NOTES.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dest)
    else:
        dest.write_text("")


def promote_graph(root: Path, instance: dict, out: Path) -> None:
    repo = repo_key(instance["repo"])
    src = out / "G-persist.graph.db"
    dest = persist_graph(root, repo)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file() and src.stat().st_size > 0:
        shutil.copy2(src, dest)


def notes_compliance(out: Path, traj: Path) -> dict:
    """Did the M-notes agent read and write /memory/NOTES.md, and how big is it now?"""
    import re

    comp = {"notes_read": False, "notes_wrote": False, "notes_bytes": None, "notes_tokens": None}
    if traj.is_file():
        try:
            data = json.loads(traj.read_text())
        except json.JSONDecodeError:
            data = {}
        for msg in data.get("messages", []):
            if msg.get("role") != "assistant":
                continue
            for call in msg.get("tool_calls") or []:
                args = (call.get("function") or {}).get("arguments")
                try:
                    cmd = json.loads(args).get("command", "") if isinstance(args, str) else ""
                except json.JSONDecodeError:
                    cmd = args or ""
                if "/memory" not in cmd and "NOTES.md" not in cmd:
                    continue
                if re.search(r"\b(cat|head|tail|less|sed|grep)\b[^|;&]*NOTES\.md", cmd):
                    comp["notes_read"] = True
                if re.search(r"(>>?|tee(\s+-a)?)\s*\S*NOTES\.md", cmd) or re.search(
                    r"open\(['\"][^'\"]*NOTES\.md['\"],\s*['\"][wa]", cmd
                ):
                    comp["notes_wrote"] = True
    notes = out / "M-notes.memory" / "NOTES.md"
    if notes.is_file():
        comp["notes_bytes"] = notes.stat().st_size
        comp["notes_tokens"] = token_count(notes.read_text(errors="replace"))
    return comp


# ---------------------------------------------------------------- arms


def mini_env(instance: dict, run_args: list[str]):
    from minisweagent.config import get_config_from_spec
    from minisweagent.run.benchmarks.swebench import get_sb_environment
    from minisweagent.utils.serialize import recursive_merge

    base = get_config_from_spec(SWEBENCH_YAML)
    mini = get_config_from_spec(str(MINI_CONFIG))
    config = recursive_merge(base, mini)
    config.setdefault("environment", {})
    config["environment"]["pull_timeout"] = 600
    config["environment"]["container_timeout"] = "35m"
    config["environment"]["cwd"] = "/testbed"
    config["environment"]["run_args"] = ["--rm", *run_args]
    return config, get_sb_environment(config, instance)


def primary_cost_model(model):
    """Make mini-swe-agent sum primary (frozen-price) cost, not OpenRouter's bill.

    mini's OpenRouter model returns usage.cost from _calculate_cost and the agent
    applies cost_limit to that sum. Graphirm applies its $0.50 cap to the primary
    formula, so the same formula is used here; the bill stays in each message's
    extra.response.usage.cost and is recorded per row as cost_usd_billed.
    """
    prices = load_prices()
    original = model._calculate_cost

    def calculate(response):
        usage = response.get("usage", {}) or {}
        billed = original(response)  # keeps mini's "no cost info" error for broken responses
        return {"cost": usage_cost(usage, prices), "cost_billed": billed.get("cost", 0.0)}

    model._calculate_cost = calculate
    return model


def run_mini(instance: dict, arm: str, out: Path, root: Path) -> dict:
    from minisweagent.agents import get_agent
    from minisweagent.models import get_model

    run_args: list[str] = []
    if arm == "M-notes":
        mount = prepare_notes(root, instance, out)
        run_args = ["-v", f"{mount}:/memory"]
    record = {
        "arm": arm,
        "image": instance["image_name"],
        "start_commit": None,
        "cost_usd": 0.0,
        "cost_usd_billed": 0.0,
        "error": None,
    }
    env = None
    try:
        config, env = mini_env(instance, run_args)
        traj = out / f"{arm}.traj.json"
        agent_cfg = {
            "system_template": config["agent"]["system_template"],
            "instance_template": config["agent"]["instance_template"],
            "step_limit": 100,
            "cost_limit": 0.5,
            "wall_time_limit_seconds": 1800,
            "output_path": traj,
        }
        probed = env.execute({"command": "git -C /testbed rev-parse HEAD"}, timeout=60)
        record["start_commit"] = (probed.get("output") or "").strip() or None
        model = primary_cost_model(get_model(config=config.get("model", {})))
        agent = get_agent(model, env, agent_cfg, default_type="default")
        text = (
            render_notes_task(instance["problem_statement"])
            if arm == "M-notes"
            else instance["problem_statement"]
        )
        info = agent.run(text)
        submission = info.get("submission") or ""
        submission, capture = crlf_safe_submission(
            submission, getattr(env, "container_id", None) or "", "/testbed"
        )
        record["patch_capture"] = capture
        patch = strip_test_hunks(submission, instance.get("test_patch") or "")
        write_patch(out / f"{arm}.patch", patch)
        costs = mini_costs(traj)
        # agent.cost is now the primary formula (primary_cost_model); the bill comes
        # from each call's usage.cost in the trajectory.
        record["cost_usd"] = round(float(getattr(agent, "cost", 0.0) or costs["cost_usd"]), 6)
        record["cost_usd_billed"] = round(costs["cost_usd_billed"], 6)
        record["cap_hit"] = info.get("exit_status") == "LimitsExceeded"
        stats = info.get("model_stats") or {}
        summary = {
            "arm": arm,
            "start_commit": record["start_commit"],
            "image": instance["image_name"],
            "cost_usd": record["cost_usd"],
            "cost_usd_billed": record["cost_usd_billed"],
            "tokens_in": costs["prompt"],
            "tokens_cached": costs["cached"],
            "tokens_out": costs["output"],
            "exit_status": info.get("exit_status"),
            "api_calls": stats.get("api_calls"),
            "patch_capture": capture,
        }
        if arm == "M-notes":
            comp = notes_compliance(out, traj)
            record.update(comp)
            summary.update(comp)
        (out / f"{arm}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    except Exception as exc:
        record["diag"] = f"{type(exc).__name__}: {exc}"[:4000]
        record["error"] = record["diag"][:400]
        if not (out / f"{arm}.patch").exists():
            (out / f"{arm}.patch").write_text("")
    finally:
        cleanup = getattr(env, "cleanup", None)
        if cleanup:
            cleanup()
    return record


def run_graphirm(instance: dict, arm: str, out: Path, root: Path) -> dict:
    iid = instance["instance_id"]
    name = f"t2-{arm}-{iid}".lower().replace("_", "-")[:60]
    client = docker.from_env(timeout=600)
    record = {
        "arm": arm,
        "image": instance["image_name"],
        "start_commit": None,
        "cost_usd": 0.0,
        "cost_usd_billed": 0.0,
        "error": None,
        "http_429": 0,
    }
    container = None
    try:
        try:
            client.containers.get(name).remove(force=True)
        except docker.errors.NotFound:
            pass
        container = client.containers.run(
            instance["image_name"],
            command=["sleep", "2100"],
            detach=True,
            name=name,
            working_dir="/testbed",
            user="root",
        )
        task = out / f"{arm}.issue.md"
        task.write_text(render_task(instance["problem_statement"]))
        test_patch = out / "test.patch"
        write_patch(test_patch, instance.get("test_patch") or "")
        graph_out = out / f"{arm}.graph.db"
        summary_out = out / f"{arm}.summary.json"
        retry_log = out / f"{arm}.retries.log"
        env = os.environ.copy()
        env.update(
            {
                "CONTAINER": container.id,
                "TASK_FILE": str(task),
                "WORKDIR": "/testbed",
                "GRAPH_OUT": str(graph_out),
                "SUMMARY_OUT": str(summary_out),
                "TEST_PATCH_FILE": str(test_patch),
                "GRAPHIRM_BIN": str(BINARY),
                "GRAPHIRM_IMAGE": instance["image_name"],
                "RETRY_LOG_OUT": str(retry_log),
                "GRAPHIRM_MAX_TURNS": "100",
                "GRAPHIRM_MAX_COST": "0.50",
            }
        )
        if arm == "G-persist":
            prior = persist_graph(root, repo_key(instance["repo"]))
            record["graph_in_bytes"] = prior.stat().st_size if prior.is_file() else 0
            if prior.is_file() and prior.stat().st_size > 0:
                env["GRAPH_IN"] = str(prior)
        if reconstruction_sampled(iid, arm):
            env["PACK_TRACE_OUT"] = str(out / f"{arm}.pack-trace.jsonl")
        for stale in (graph_out, summary_out, retry_log, Path(str(graph_out) + ".diff")):
            stale.unlink(missing_ok=True)
        proc = subprocess.run(
            ["sh", str(RUNNER)],
            cwd=REPO,
            env=env,
            timeout=1900,
            capture_output=True,
            text=True,
        )
        diag = proc.stderr or proc.stdout or ""
        record["diag"] = diag[-4000:]
        if proc.returncode != 0:
            record["error"] = (diag or f"exit {proc.returncode}")[-400:]
        if summary_out.is_file():
            summary = json.loads(summary_out.read_text())
            record["start_commit"] = summary.get("start_commit")
            record["cost_usd"] = float(summary.get("cost_usd") or 0.0)
            record["cost_usd_billed"] = float(summary.get("cost_usd_reported") or 0.0)
            record["cap_hit"] = summary.get("exit_reason") in ("max_cost", "cost_cap", "budget")
            record["exit_reason"] = summary.get("exit_reason")
        elif Path(str(summary_out) + ".start_commit").is_file():
            record["start_commit"] = Path(str(summary_out) + ".start_commit").read_text().strip()
        if retry_log.is_file():
            record["http_429"] = retry_log.read_text().count("429")
        patch_path = Path(str(graph_out) + ".diff")
        if not patch_path.is_file():
            patch_path.write_text("")
        write_patch(out / f"{arm}.patch", read_patch(patch_path))
    except Exception as exc:
        record["diag"] = f"{type(exc).__name__}: {exc}"[:4000]
        record["error"] = record["diag"][:400]
        if not (out / f"{arm}.patch").exists():
            (out / f"{arm}.patch").write_text("")
    finally:
        if container is not None:
            try:
                container.remove(force=True)
            except Exception:
                pass
    return record


# ---------------------------------------------------------------- guards


def t2_arm_problem(arm: str, out: Path, error_text: str) -> str | None:
    patch = read_patch(out / f"{arm}.patch")
    if arm in ("M", "M-notes"):
        traj_path = out / f"{arm}.traj.json"
        model_cfg = None
        if traj_path.is_file():
            try:
                model_cfg = json.loads(traj_path.read_text())["info"]["config"]["model"]
            except (json.JSONDecodeError, KeyError, TypeError):
                model_cfg = None
        if model_cfg is None and provider_error(error_text):
            return "provider error"
        if model_cfg is not None:
            problem = mini_problem(model_cfg)
            if problem:
                return problem
    else:
        summary = None
        summary_path = out / f"{arm}.summary.json"
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text())
            except json.JSONDecodeError:
                summary = None
        metadata = read_run_metadata(out / f"{arm}.graph.db")
        if summary is None and metadata is None:
            return "provider error" if provider_error(error_text) else None
        problem = graphirm_problem(summary, metadata)
        if problem:
            return problem
    if not patch.strip() and provider_error(error_text):
        return "empty patch from provider error"
    return None


def apply_config_guard(arm: str, out: Path, result: dict) -> None:
    reason = t2_arm_problem(arm, out, result.get("diag") or result.get("error") or "")
    result.pop("diag", None)
    if not reason:
        return
    result["invalid_config"] = True
    result["status"] = "invalid_config"
    result["error"] = f"invalid_config: {reason}"[:400]


# ---------------------------------------------------------------- drift and images


heads_lock = threading.Lock()
image_heads: dict[str, tuple[str | None, str | None]] = {}


def expected_starts(client, image: str, base_commit: str) -> tuple[set[str], str | None]:
    """Verified images carry one extra "SWE-bench" commit on top of base_commit
    (environment setup). A start at that HEAD whose parent is base_commit is
    the expected state, not a mismatch."""
    with heads_lock:
        cached = image_heads.get(image)
    if cached is None:
        head = parent = None
        try:
            raw = client.containers.run(
                image,
                entrypoint=["git"],
                command=["-C", "/testbed", "rev-parse", "HEAD", "HEAD^"],
                remove=True,
                user="root",
            )
            text = raw.decode() if isinstance(raw, bytes) else str(raw)
            parts = text.split()
            head = parts[0] if parts else None
            parent = parts[1] if len(parts) > 1 else None
        except Exception:
            pass
        cached = (head, parent)
        with heads_lock:
            image_heads[image] = cached
    head, parent = cached
    allowed = {base_commit}
    if head and parent == base_commit:
        allowed.add(head)
    return allowed, head


def wait_turn(repo: str, arm: str, index: int) -> None:
    """Arms of one repo stay within DRIFT tasks of the slowest live arm."""
    with drift_cv:
        while not stop_event.is_set():
            live = [n for a, n in progress[repo].items() if a not in dead_arms[repo]]
            if not live or min(live) >= index - DRIFT:
                return
            drift_cv.wait(timeout=60)


def mark_progress(repo: str, arm: str, count: int) -> None:
    with drift_cv:
        progress[repo][arm] = count
        drift_cv.notify_all()


def mark_dead(repo: str, arm: str) -> None:
    with drift_cv:
        dead_arms[repo].add(arm)
        drift_cv.notify_all()


def notes_rate(ledger: Path) -> str:
    """notes=read/wrote/n plus median NOTES.md tokens over finished M-notes rows."""
    latest = latest_rows(ledger)
    rows = [r for r in latest.values() if r.get("arm") == "M-notes" and arm_done(r)]
    if not rows:
        return "notes=0/0/0"
    read = sum(1 for r in rows if r.get("notes_read"))
    wrote = sum(1 for r in rows if r.get("notes_wrote"))
    sizes = sorted(int(r.get("notes_tokens") or 0) for r in rows)
    median = sizes[len(sizes) // 2]
    return f"notes=read {read}/wrote {wrote}/n {len(rows)} notes_tokens_median={median}"


def task_complete(root: Path, instance: dict, client) -> None:
    """Called after each arm; when all four arms are done the image goes."""
    out = root / instance["instance_id"]
    grades_path = out / "grades.json"
    with ledger_lock:
        grades = json.loads(grades_path.read_text()) if grades_path.is_file() else {}
    if not all(arm in grades for arm in ARMS):
        return
    remove_task_image(instance, client)
    with progress_lock:
        tasks_finished["count"] += 1
        count = tasks_finished["count"]
        due = count // PROGRESS_EVERY > tasks_finished["reported"] // PROGRESS_EVERY
        if due:
            tasks_finished["reported"] = count
    if due:
        print(
            f"PROGRESS tasks={count} primary={spent(root / 'arms.jsonl'):.2f} "
            f"billed={spent(root / 'arms.jsonl', 'cost_usd_billed'):.2f} disk={used_pct():.1f} "
            f"backup_failed={pending_backup_count(TRACK)} {notes_rate(root / 'arms.jsonl')}",
            flush=True,
        )


# ---------------------------------------------------------------- job walk


def walk_job(repo: str, arm: str, tasks: list[dict], root: Path, arm_ledger: Path, done_arms: set[str]) -> dict:
    counters = {"finished": 0, "errors": 0, "http_429": 0, "mismatch": 0, "invalid": 0, "unresolved": 0}
    client = docker.from_env(timeout=600)
    for index, instance in enumerate(tasks):
        iid = instance["instance_id"]
        key = f"{iid}::{arm}"
        if key in done_arms:
            counters["finished"] += 1
            mark_progress(repo, arm, index + 1)
            continue
        if stop_event.is_set():
            break
        if spent(arm_ledger) >= SPEND_CAP:
            print(f"STOP {repo} {arm} primary={spent(arm_ledger):.2f} cap={SPEND_CAP}", flush=True)
            stop_event.set()
            break
        if spent(arm_ledger, "cost_usd_billed") >= BILLED_STOP:
            print(f"STOP {repo} {arm} billed={spent(arm_ledger, 'cost_usd_billed'):.2f} safety={BILLED_STOP}", flush=True)
            (root / "stop-reason").write_text("billed-safety\n")
            stop_event.set()
            break
        wait_turn(repo, arm, index)
        if stop_event.is_set():
            break
        wait_disk(client, instance["image_name"])
        if not image_present(client, instance["image_name"]):
            client.images.pull(instance["image_name"])
        out = root / iid
        out.mkdir(parents=True, exist_ok=True)
        recover_wait = 60
        while True:
            if arm in ("M", "M-notes"):
                result = run_mini(instance, arm, out, root)
            else:
                result = run_graphirm(instance, arm, out, root)
                merge_run_checks(iid, arm, out)
            result["instance_id"] = key
            result["task"] = iid
            result["repo"] = instance["repo"]
            result["position"] = instance["position"]
            result["n_tasks"] = instance["n_tasks"]
            result["scored"] = instance["scored"]
            result["base_commit"] = instance["base_commit"]
            allowed, head = expected_starts(client, instance["image_name"], instance["base_commit"])
            result["image_head"] = head
            result["commit_mismatch"] = bool(
                result.get("start_commit") and result["start_commit"] not in allowed
            )
            apply_config_guard(arm, out, result)
            result["complete"] = not bool(result.get("error"))
            kind = classify_arm(out, arm, result.get("error") or "")
            if result.get("error"):
                result["failure_class"] = kind
                counters["errors"] += 1
            counters["http_429"] += int(result.get("http_429") or 0)
            if result.get("commit_mismatch"):
                counters["mismatch"] += 1
            if kind == CONFIG and result.get("error"):
                with ledger_lock:
                    append_ledger(arm_ledger, result)
                (root / "stop-reason").write_text("config\n")
                stop_event.set()
                raise SystemExit(f"STOP config {iid} {arm} {result.get('error')}")
            if kind == REQUEST and result.get("error"):
                result["request_caused"] = True
                result["complete"] = True
                result["unresolved"] = True
                counters["unresolved"] += 1
                with streak_lock:
                    streaks["request"] += 1
                    streak = streaks["request"]
                print(f"UNRESOLVED {iid} {arm} {result.get('error')}", flush=True)
                if streak > 3:
                    with ledger_lock:
                        append_ledger(arm_ledger, result)
                    (root / "stop-reason").write_text("request\n")
                    stop_event.set()
                    raise SystemExit(f"STOP request streak={streak} spend={spent(arm_ledger):.2f}")
                break
            if kind == INFRASTRUCTURE and result.get("error"):
                with ledger_lock:
                    append_ledger(arm_ledger, result)
                with streak_lock:
                    streaks["request"] = 0
                print(f"RECOVER {iid} arms={arm} wait={recover_wait}", flush=True)
                time.sleep(recover_wait)
                recover_wait = min(recover_wait * 2, 1800)
                if stop_event.is_set():
                    raise SystemExit(f"STOP during recovery {iid} {arm}")
                preflight(root, instance["image_name"])
                continue
            if result.get("invalid_config") or (
                result.get("error") and provider_error(result.get("error") or "")
            ):
                counters["invalid"] += 1
                with ledger_lock:
                    append_ledger(arm_ledger, result)
                print(f"INVALID_CONFIG {iid} {arm} {result.get('error')}", flush=True)
                (root / "stop-reason").write_text("config\n")
                stop_event.set()
                raise SystemExit(f"STOP invalid_config {iid} {arm}")
            if result.get("error"):
                with ledger_lock:
                    append_ledger(arm_ledger, result)
                if recover_wait > 60:
                    result["request_caused"] = True
                    result["complete"] = True
                    result["unresolved"] = True
                    counters["unresolved"] += 1
                    print(f"UNRESOLVED {iid} {arm} other: {result.get('error')}", flush=True)
                    break
                print(f"RECOVER {iid} arms={arm} wait={recover_wait} other", flush=True)
                time.sleep(recover_wait)
                recover_wait *= 2
                continue
            with streak_lock:
                streaks["request"] = 0
            break

        patch = read_patch(out / f"{arm}.patch")
        if result.get("unresolved"):
            resolved, status, grade_error = False, "unresolved", ""
        elif not patch.strip():
            resolved, status, grade_error = False, "empty_patch", ""
        else:
            try:
                grade = grade_patch(
                    instance, patch, client, GRADE_TIMEOUT, out / "grading" / arm, f"t2-{arm}"
                )
                resolved, status, grade_error = grade.resolved, grade.status, grade.error
            except Exception as exc:
                resolved, status, grade_error = None, "grade_exception", f"{type(exc).__name__}: {exc}"[:400]
        if resolved is None:
            print(f"HARNESS_ERR {iid} {arm} {status} {grade_error}", flush=True)
            resolved = False
            result["harness_error"] = f"{status}: {grade_error}"[:400]
        result["grade_status"] = status
        if status == "apply_failed":
            result["apply_error"] = grade_error
        grades_path = out / "grades.json"
        with ledger_lock:
            grades = json.loads(grades_path.read_text()) if grades_path.is_file() else {}
            grades[arm] = resolved
            grades_path.write_text(json.dumps(grades, indent=2) + "\n")
            append_ledger(arm_ledger, result)
        if arm == "M-notes":
            promote_notes(root, instance, out)
        if arm == "G-persist":
            promote_graph(root, instance, out)
        files = [
            p
            for p in (
                out / f"{arm}.patch",
                out / f"{arm}.summary.json",
                out / f"{arm}.traj.json",
                out / "grades.json",
                out / "M-notes.carry.json",
            )
            if p.is_file()
        ]
        if not backup_record(TRACK, result, files):
            remember_failed_backup(TRACK, result, files)
        done_arms.add(key)
        counters["finished"] += 1
        mark_progress(repo, arm, index + 1)
        task_complete(root, instance, client)
        print(
            f"DONE {repo_key(repo)} {arm} {counters['finished']}/{len(tasks)} primary={spent(arm_ledger):.2f}",
            flush=True,
        )
    return counters


def run_job(repo: str, arm: str, tasks, root, arm_ledger, done_arms) -> dict:
    try:
        return walk_job(repo, arm, tasks, root, arm_ledger, done_arms)
    finally:
        mark_dead(repo, arm)


# ---------------------------------------------------------------- main


def main() -> None:
    if not BINARY.is_file():
        raise SystemExit(f"missing binary {BINARY}")
    root = runs_dir(TRACK)
    os.chdir(root)
    if (root / "stop-reason").is_file():
        raise SystemExit(f"STOP held {(root / 'stop-reason').read_text().strip()}")
    (root / "running").write_text("t2\n")
    arm_ledger = root / "arms.jsonl"
    latest = latest_rows(arm_ledger)
    done_arms = {key for key, record in latest.items() if arm_done(record)}
    sequences = load_sequences()
    instances = [task for seq in sequences for task in seq["tasks"]]
    for seq in sequences:
        progress[seq["repo"]] = {arm: 0 for arm in ARMS}
        dead_arms[seq["repo"]] = set()
    done_tasks = {
        row["instance_id"]
        for row in instances
        if all(f"{row['instance_id']}::{arm}" in done_arms for arm in ARMS)
    }
    tasks_finished["count"] = len(done_tasks)
    tasks_finished["reported"] = len(done_tasks)
    pending_keys = sum(
        1 for row in instances for arm in ARMS if f"{row['instance_id']}::{arm}" not in done_arms
    )
    print(
        f"START total={TOTAL} done={len(done_tasks)} pending_arms={pending_keys} "
        f"primary={spent(arm_ledger):.2f} billed={spent(arm_ledger, 'cost_usd_billed'):.2f} "
        f"disk={used_pct():.1f} workers={WORKERS} drift={DRIFT}",
        flush=True,
    )
    if pending_keys == 0:
        (root / "running").unlink(missing_ok=True)
        print(f"HOLD tasks={len(done_tasks)} primary={spent(arm_ledger):.2f} billed={spent(arm_ledger, 'cost_usd_billed'):.2f}", flush=True)
        return
    if spent(arm_ledger) >= SPEND_CAP:
        (root / "running").unlink(missing_ok=True)
        print(f"STOP primary={spent(arm_ledger):.2f} cap={SPEND_CAP}", flush=True)
        return
    if spent(arm_ledger, "cost_usd_billed") >= BILLED_STOP:
        (root / "running").unlink(missing_ok=True)
        (root / "stop-reason").write_text("billed-safety\n")
        print(f"STOP billed={spent(arm_ledger, 'cost_usd_billed'):.2f} safety={BILLED_STOP}", flush=True)
        return
    first_pending = next(
        row for row in instances if any(f"{row['instance_id']}::{arm}" not in done_arms for arm in ARMS)
    )
    preflight(root, first_pending["image_name"])

    jobs = [(seq["repo"], arm, seq["tasks"]) for seq in sequences for arm in ARMS]
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(run_job, repo, arm, tasks, root, arm_ledger, done_arms): f"{repo_key(repo)}/{arm}"
            for repo, arm, tasks in jobs
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                results[label] = future.result()
            except SystemExit as exc:
                print(str(exc), flush=True)
                results[label] = {"stopped": str(exc)}
            except Exception:
                traceback.print_exc()
                results[label] = {"crashed": True}
    still = retry_failed_backups(TRACK)
    if push_backup():
        print(f"PUSH backup_failed={still}", flush=True)
    else:
        print(f"PUSH_WARN backup_failed={still}", flush=True)
    latest = latest_rows(arm_ledger)
    finished = sum(
        1
        for row in instances
        if all(arm_done(latest.get(f"{row['instance_id']}::{arm}")) for arm in ARMS)
    )
    errors = sum(int(c.get("errors", 0)) for c in results.values())
    http_429 = sum(int(c.get("http_429", 0)) for c in results.values())
    mismatch = sum(int(c.get("mismatch", 0)) for c in results.values())
    unresolved = sum(int(c.get("unresolved", 0)) for c in results.values())
    print(
        f"PROGRESS tasks={finished} primary={spent(arm_ledger):.2f} billed={spent(arm_ledger, 'cost_usd_billed'):.2f} disk={used_pct():.1f} "
        f"errors={errors} http_429={http_429} commit_mismatch={mismatch} "
        f"unresolved={unresolved} backup_failed={pending_backup_count(TRACK)} {notes_rate(arm_ledger)}",
        flush=True,
    )
    if finished >= TOTAL or (root / "stop-reason").is_file() or spent(arm_ledger) >= SPEND_CAP:
        (root / "running").unlink(missing_ok=True)
    print(f"HOLD tasks={finished} primary={spent(arm_ledger):.2f} billed={spent(arm_ledger, 'cost_usd_billed'):.2f}", flush=True)


if __name__ == "__main__":
    main()
