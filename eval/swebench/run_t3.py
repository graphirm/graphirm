#!/usr/bin/env python3
"""T3 memory track: the 19 CL-Bench codebase tasks, four arms.

Arms: M (mini-swe-agent, no memory), M-notes (mini plus /memory/NOTES.md,
carried within a repo, 4k-token cap), G-fresh (Graphirm, new graph per
task), G-persist (Graphirm, one graph per repo). Each arm walks the 19
tasks strictly in the published order; the four arms run in parallel.
Memory resets at the tablib -> tenacity switch.

Workspace setup and grading use CL-Bench's own generic PR runtime:
checkout base_commit, apply test_patch, run the suite. Outputs stay under
/home/krs/graphirm-runs/t3. Stdout progress lines omit resolve flags.
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
sys.path.insert(0, "/home/krs/graphirm-runs/continual-learning-bench/src")

from tasks.codebase_adaptation.generic_runtime import (
    evaluate_generic_pr_submission,
    initialize_generic_pr_container,
)

from adapters.memory_notes import carry
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
from run_arm import load_dotenv, render_notes_task, render_task, strip_test_hunks
from reconstruct_check import sampled as reconstruction_sampled
from run_t1 import BINARY, MINI_CONFIG, REPO, RUNNER, SWEBENCH_YAML, merge_run_checks
from run_t1 import preflight as _preflight
from validate_clbench import ORDER, SETTLED, load_instances

ARMS = ["M", "M-notes", "G-fresh", "G-persist"]
TRACK = "t3"
SPEND_CAP = 5.0
DISK_LIMIT = 78.0
TOTAL = 19

load_dotenv(REPO / ".env")
ledger_lock = threading.Lock()
streak_lock = threading.Lock()
preflight_lock = threading.Lock()
streaks = {"request": 0}


def preflight(root: Path, image: str) -> None:
    """run_t1's preflight uses one fixed container name, so arms take turns."""
    with preflight_lock:
        _preflight(root, image)


def used_pct() -> float:
    usage = shutil.disk_usage("/")
    return 100.0 * usage.used / usage.total


def wait_disk() -> None:
    while used_pct() >= DISK_LIMIT:
        print(f"DISK {used_pct():.1f} waiting", flush=True)
        time.sleep(30)


def repo_key(instance: dict) -> str:
    return instance["repo"].replace("/", "__")


def spent(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0.0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            total += float(json.loads(line).get("cost_usd") or 0.0)
        except (json.JSONDecodeError, TypeError, ValueError):
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


# ---------------------------------------------------------------- memory


def notes_dir(root: Path, repo: str) -> Path:
    return root / "memory" / "M-notes" / repo


def persist_graph(root: Path, repo: str) -> Path:
    return root / "memory" / "G-persist" / repo / "graph.db"


def prepare_notes(root: Path, instance: dict, out: Path) -> Path:
    """Carry the repo's NOTES.md into this task's mount directory."""
    repo = repo_key(instance)
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
    """After a clean task, the task's NOTES.md becomes the repo's carried copy."""
    repo = repo_key(instance)
    src = out / "M-notes.memory" / "NOTES.md"
    dest = notes_dir(root, repo) / "NOTES.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dest)
    else:
        dest.write_text("")


def promote_graph(root: Path, instance: dict, out: Path) -> None:
    repo = repo_key(instance)
    src = out / "G-persist.graph.db"
    dest = persist_graph(root, repo)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file() and src.stat().st_size > 0:
        shutil.copy2(src, dest)


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
    config["environment"]["cwd"] = instance.get("workdir", "/testbed")
    config["environment"]["run_args"] = ["--rm", *run_args]
    return config, get_sb_environment(config, instance)


def run_mini(instance: dict, arm: str, out: Path, root: Path) -> dict:
    from minisweagent.agents import get_agent
    from minisweagent.models import get_model

    run_args: list[str] = []
    if arm == "M-notes":
        mount = prepare_notes(root, instance, out)
        run_args = ["-v", f"{mount}:/memory"]
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
    record = {
        "arm": arm,
        "image": instance["image_name"],
        "start_commit": None,
        "cost_usd": 0.0,
        "error": None,
    }
    try:
        initialize_generic_pr_container(env.container_id, instance)
        probed = env.execute({"command": "git rev-parse HEAD"}, timeout=60)
        record["start_commit"] = (probed.get("output") or "").strip() or None
        agent = get_agent(get_model(config=config.get("model", {})), env, agent_cfg, default_type="default")
        text = (
            render_notes_task(instance["problem_statement"])
            if arm == "M-notes"
            else instance["problem_statement"]
        )
        info = agent.run(text)
        submission = info.get("submission") or ""
        patch = strip_test_hunks(submission, instance.get("test_patch") or "")
        (out / f"{arm}.patch").write_text(patch)
        record["cost_usd"] = float(getattr(agent, "cost", 0.0) or 0.0)
        stats = info.get("model_stats") or {}
        summary = {
            "arm": arm,
            "start_commit": record["start_commit"],
            "image": instance["image_name"],
            "cost_usd": record["cost_usd"],
            "exit_status": info.get("exit_status"),
            "api_calls": stats.get("api_calls"),
        }
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
    name = f"t3-{arm}-{iid}".lower().replace("_", "-")[:60]
    client = docker.from_env(timeout=600)
    record = {
        "arm": arm,
        "image": instance["image_name"],
        "start_commit": None,
        "cost_usd": 0.0,
        "error": None,
        "http_429": 0,
    }
    container = None
    workdir = instance.get("workdir", "/testbed")
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
            working_dir=workdir,
            user="root",
        )
        initialize_generic_pr_container(container.id, instance)
        task = out / f"{arm}.issue.md"
        task.write_text(render_task(instance["problem_statement"]))
        test_patch = out / "test.patch"
        test_patch.write_text(instance.get("test_patch") or "")
        graph_out = out / f"{arm}.graph.db"
        summary_out = out / f"{arm}.summary.json"
        retry_log = out / f"{arm}.retries.log"
        env = os.environ.copy()
        env.update(
            {
                "CONTAINER": container.id,
                "TASK_FILE": str(task),
                "WORKDIR": workdir,
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
            prior = persist_graph(root, repo_key(instance))
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
        elif Path(str(summary_out) + ".start_commit").is_file():
            record["start_commit"] = Path(str(summary_out) + ".start_commit").read_text().strip()
        if retry_log.is_file():
            record["http_429"] = retry_log.read_text().count("429")
        patch_path = Path(str(graph_out) + ".diff")
        if not patch_path.is_file():
            patch_path.write_text("")
        (out / f"{arm}.patch").write_text(patch_path.read_text())
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


# ---------------------------------------------------------------- grading


def grade(instance: dict, patch: str) -> tuple[bool | None, str, str]:
    """CL-Bench grade. Harness error: rerun once, then unresolved and logged."""
    last = None
    for _attempt in (1, 2):
        last = evaluate_generic_pr_submission(patch, instance)
        if last.status in SETTLED:
            return bool(last.success), last.status, ""
    error = str(getattr(last, "error", "") or "")[:400]
    return None, getattr(last, "status", "unknown"), error


def t3_arm_problem(arm: str, out: Path, error_text: str) -> str | None:
    """config_guard.arm_problem keys the mini check off the name "M" and the
    file M.traj.json. Here two arms are mini and two are Graphirm, each with
    its own file names, so the same checks run against the right files."""
    patch_path = out / f"{arm}.patch"
    patch = patch_path.read_text() if patch_path.is_file() else ""
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
    reason = t3_arm_problem(arm, out, result.get("diag") or result.get("error") or "")
    result.pop("diag", None)
    if not reason:
        return
    result["invalid_config"] = True
    result["status"] = "invalid_config"
    result["error"] = f"invalid_config: {reason}"[:400]


# ---------------------------------------------------------------- arm walk


def walk_arm(arm: str, instances: list[dict], root: Path, arm_ledger: Path, done_arms: set[str]) -> dict:
    """One arm over the 19 tasks in order. Returns counters for progress lines."""
    counters = {"finished": 0, "errors": 0, "http_429": 0, "mismatch": 0, "invalid": 0, "unresolved": 0}
    image_pulled: set[str] = set()
    client = docker.from_env(timeout=600)
    for instance in instances:
        iid = instance["instance_id"]
        key = f"{iid}::{arm}"
        if key in done_arms:
            counters["finished"] += 1
            continue
        if spent(arm_ledger) >= SPEND_CAP:
            print(f"STOP {arm} spend={spent(arm_ledger):.2f} cap={SPEND_CAP}", flush=True)
            break
        wait_disk()
        if instance["image_name"] not in image_pulled:
            client.images.pull(instance["image_name"])
            image_pulled.add(instance["image_name"])
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
            result["position"] = instance.get("sequence_rank")
            result["base_commit"] = instance["base_commit"]
            result["commit_mismatch"] = bool(
                result.get("start_commit") and result["start_commit"] != instance["base_commit"]
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
                raise SystemExit(f"STOP invalid_config {iid} {arm}")
            if result.get("error"):
                # Unknown failure: record it and try once more, then leave it unresolved.
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

        patch_path = out / f"{arm}.patch"
        patch = patch_path.read_text() if patch_path.is_file() else ""
        if result.get("unresolved"):
            resolved, status, grade_error = False, "unresolved", ""
        elif not patch.strip():
            resolved, status, grade_error = False, "empty_patch", ""
        else:
            try:
                resolved, status, grade_error = grade(instance, patch)
            except Exception as exc:
                resolved, status, grade_error = None, "grade_exception", f"{type(exc).__name__}: {exc}"[:400]
        if resolved is None:
            print(f"HARNESS_ERR {iid} {arm} {status} {grade_error}", flush=True)
            resolved = False
            result["harness_error"] = f"{status}: {grade_error}"[:400]
        result["grade_status"] = status
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
        print(f"DONE {arm} {counters['finished']}/{TOTAL} spend={spent(arm_ledger):.2f}", flush=True)
    return counters


# ---------------------------------------------------------------- main


def main() -> None:
    if not BINARY.is_file():
        raise SystemExit(f"missing binary {BINARY}")
    root = runs_dir(TRACK)
    os.chdir(root)
    if (root / "stop-reason").is_file():
        raise SystemExit(f"STOP held {(root / 'stop-reason').read_text().strip()}")
    (root / "running").write_text("t3\n")
    arm_ledger = root / "arms.jsonl"
    latest = latest_rows(arm_ledger)
    done_arms = {key for key, record in latest.items() if arm_done(record)}
    instances = load_instances()
    if [row["instance_id"] for row in instances] != ORDER:
        raise SystemExit("instance order does not match the published schedule")
    done_tasks = {
        row["instance_id"]
        for row in instances
        if all(f"{row['instance_id']}::{arm}" in done_arms for arm in ARMS)
    }
    pending_keys = sum(
        1 for row in instances for arm in ARMS if f"{row['instance_id']}::{arm}" not in done_arms
    )
    print(
        f"START total={TOTAL} done={len(done_tasks)} pending_arms={pending_keys} "
        f"spend={spent(arm_ledger):.2f} disk={used_pct():.1f}",
        flush=True,
    )
    if pending_keys == 0:
        (root / "running").unlink(missing_ok=True)
        print(f"HOLD tasks={len(done_tasks)} spend={spent(arm_ledger):.2f}", flush=True)
        return
    if spent(arm_ledger) >= SPEND_CAP:
        (root / "running").unlink(missing_ok=True)
        print(f"STOP spend={spent(arm_ledger):.2f} cap={SPEND_CAP}", flush=True)
        return
    for image in sorted({row["image_name"] for row in instances}):
        print(f"PULL {image}", flush=True)
        docker.from_env(timeout=600).images.pull(image)
    preflight(root, instances[0]["image_name"])

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=len(ARMS)) as pool:
        futures = {
            pool.submit(walk_arm, arm, instances, root, arm_ledger, done_arms): arm for arm in ARMS
        }
        for future in as_completed(futures):
            arm = futures[future]
            try:
                results[arm] = future.result()
            except SystemExit as exc:
                print(str(exc), flush=True)
                results[arm] = {"stopped": str(exc)}
            except Exception:
                traceback.print_exc()
                results[arm] = {"crashed": True}
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
        f"PROGRESS tasks={finished} spend={spent(arm_ledger):.2f} disk={used_pct():.1f} "
        f"errors={errors} http_429={http_429} commit_mismatch={mismatch} "
        f"unresolved={unresolved} backup_failed={pending_backup_count(TRACK)}",
        flush=True,
    )
    if finished >= TOTAL or (root / "stop-reason").is_file() or spent(arm_ledger) >= SPEND_CAP:
        (root / "running").unlink(missing_ok=True)
    print(f"HOLD tasks={finished} spend={spent(arm_ledger):.2f}", flush=True)


if __name__ == "__main__":
    main()
