#!/usr/bin/env python3
"""Exploratory G-persist follow-ups. Not pre-registered.

One binary, three arms, matplotlib then astropy (56 tasks, the published
sequences, including the unscored gold-fail). Each arm keeps one graph per
repo and starts the next repo from an empty graph. The control arm uses the
pre-registered eval config unchanged. The other two set one switch off.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import docker

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_guard import CONFIG_HASH, CONFIG_PATH, provider_error, read_run_metadata, toml_problem
from durable import append_ledger, refuse_tmp, runs_dir
from failure_class import CONFIG, INFRASTRUCTURE, REQUEST, classify_arm
from grade_verified import grade_patch, remove_task_image
from reconstruct_check import sampled as reconstruction_sampled
from run_arm import load_dotenv, read_patch, render_task, write_patch
from run_t1 import LOCK, RUNNER, merge_run_checks
from run_t2 import load_sequences

LABEL = "exploratory, not pre-registered"
TRACK = "explore"
HOST_REPO = Path("/home/krs/graphirm-repo")
BINARY = Path("/home/krs/graphirm-frozen-target/x86_64-unknown-linux-musl/release/graphirm")
BINARY_SHA = "58e6cbfeb54d0fa590177c65ebd5d3a47f4899a12730b8ce04c3b4b954f46dbb"
SOURCE_COMMIT = "53a8b55e58260697b18ef069447fef45ac530f9a"
REPOS = ("matplotlib/matplotlib", "astropy/astropy")
ARMS = ("G-persist-control", "G-persist-nonote", "G-persist-nooldtools")
SWITCHES = {
    "G-persist-control": None,
    "G-persist-nonote": "budget_note = false",
    "G-persist-nooldtools": "pack_previous_tool_results = false",
}
UNSCORED = {"astropy__astropy-7606"}
PRIMARY_CAP = 20.0
BILLED_STOP = 30.0
MIN_BALANCE = 20.0
DISK_LIMIT = 78.0
DRIFT = 2
GRADE_TIMEOUT = 1800
WORKERS = 3

load_dotenv(HOST_REPO / ".env")
ledger_lock = threading.Lock()
streak_lock = threading.Lock()
drift_cv = threading.Condition()
stop_event = threading.Event()
streaks = {"request": 0}
progress: dict[str, dict[str, int]] = {}
dead_arms: dict[str, set[str]] = {}
expected_hash: dict[str, str] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def balance() -> float:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is missing")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/credits",
        headers={"Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode())
    data = payload.get("data") or payload
    total = float(data["total_credits"])
    used = float(data["total_usage"])
    remaining = total - used
    print(
        f"BALANCE remaining={remaining:.2f} credits={total:.2f} usage={used:.2f}",
        flush=True,
    )
    return remaining


def used_pct() -> float:
    usage = shutil.disk_usage("/")
    return 100.0 * usage.used / usage.total


def spent(path: Path, field: str = "cost_usd") -> float:
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
        try:
            total += float(row.get(field) or 0.0)
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


def write_configs(root: Path) -> None:
    problem = toml_problem()
    if problem:
        raise SystemExit(f"CONFIG {problem}")
    base = CONFIG_PATH.read_bytes()
    if hashlib.sha256(base).hexdigest() != CONFIG_HASH:
        raise SystemExit("base eval config hash drifted")
    dest = root / "configs"
    dest.mkdir(parents=True, exist_ok=True)
    text = base.decode()
    if not text.endswith("\n"):
        text += "\n"
    for arm, line in SWITCHES.items():
        body = text if line is None else text + line + "\n"
        path = dest / f"{arm}.toml"
        path.write_text(body)
        expected_hash[arm] = hashlib.sha256(body.encode()).hexdigest()
    if expected_hash["G-persist-control"] != CONFIG_HASH:
        raise SystemExit("control config is not the pre-registered file")
    (dest / "hashes.json").write_text(json.dumps(expected_hash, indent=2) + "\n")


def persist_graph(root: Path, arm: str, repo: str) -> Path:
    return root / "memory" / arm / repo.replace("/", "__") / "graph.db"


def promote_graph(root: Path, arm: str, instance: dict, out: Path) -> None:
    src = out / f"{arm}.graph.db"
    dest = persist_graph(root, arm, instance["repo"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file() and src.stat().st_size > 0:
        shutil.copy2(src, dest)


def explore_problem(arm: str, summary: dict | None, metadata: dict | None) -> str | None:
    src = metadata or summary
    if not src:
        return "missing run_metadata"
    model = src.get("model")
    if model not in {"deepseek/deepseek-v4.1-flash", "openrouter/deepseek/deepseek-v4.1-flash"}:
        return f"model {model}"
    if src.get("reasoning_effort") != "high":
        return f"reasoning {src.get('reasoning_effort')}"
    if src.get("config_hash") != expected_hash.get(arm):
        return "config_hash does not match this arm"
    return None


def apply_guard(arm: str, out: Path, result: dict) -> None:
    summary = None
    summary_path = out / f"{arm}.summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            summary = None
    metadata = read_run_metadata(out / f"{arm}.graph.db")
    error_text = result.get("diag") or result.get("error") or ""
    if summary is None and metadata is None:
        reason = "provider error" if provider_error(error_text) else None
    else:
        reason = explore_problem(arm, summary, metadata)
    patch = read_patch(out / f"{arm}.patch") if (out / f"{arm}.patch").is_file() else ""
    if not reason and not patch.strip() and provider_error(error_text):
        reason = "empty patch from provider error"
    result.pop("diag", None)
    if not reason:
        return
    result["invalid_config"] = True
    result["status"] = "invalid_config"
    result["error"] = f"invalid_config: {reason}"[:400]


def run_graphirm(instance: dict, arm: str, out: Path, root: Path) -> dict:
    iid = instance["instance_id"]
    name = f"ex-{arm}-{iid}".lower().replace("_", "-")[:60]
    client = docker.from_env(timeout=600)
    record = {
        "arm": arm,
        "image": instance["image_name"],
        "start_commit": None,
        "cost_usd": 0.0,
        "cost_usd_billed": 0.0,
        "error": None,
        "http_429": 0,
        "label": LABEL,
        "binary_sha256": BINARY_SHA,
        "source_commit": SOURCE_COMMIT,
        "config_sha256": expected_hash[arm],
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
                "CONFIG": str(root / "configs" / f"{arm}.toml"),
            }
        )
        prior = persist_graph(root, arm, instance["repo"])
        record["graph_in_bytes"] = prior.stat().st_size if prior.is_file() else 0
        if prior.is_file() and prior.stat().st_size > 0:
            env["GRAPH_IN"] = str(prior)
        if reconstruction_sampled(iid, arm):
            env["PACK_TRACE_OUT"] = str(out / f"{arm}.pack-trace.jsonl")
        for stale in (graph_out, summary_out, retry_log, Path(str(graph_out) + ".diff")):
            stale.unlink(missing_ok=True)
        proc = subprocess.run(
            ["sh", str(RUNNER)],
            cwd=HOST_REPO,
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


def wait_disk(client, image: str) -> None:
    reclaimed = False
    while used_pct() >= DISK_LIMIT and not image_present(client, image):
        if not reclaimed:
            subprocess.run(["docker", "image", "prune", "-f"], check=False, capture_output=True)
            subprocess.run(["docker", "container", "prune", "-f"], check=False, capture_output=True)
            reclaimed = True
            continue
        print(f"DISK {used_pct():.1f} waiting", flush=True)
        time.sleep(30)
        if stop_event.is_set():
            return


def image_present(client, image: str) -> bool:
    try:
        client.images.get(image)
        return True
    except Exception:
        return False


def wait_turn(repo: str, arm: str, index: int) -> None:
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


def load_grades(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def save_grades(path: Path, grades: dict) -> None:
    tmp = path.parent / f"{path.name}.tmp"
    tmp.write_text(json.dumps(grades, indent=2) + "\n")
    tmp.replace(path)


def walk_job(repo: str, arm: str, tasks: list[dict], root: Path, ledger: Path, done: set[str]) -> dict:
    counters = {"finished": 0, "errors": 0, "http_429": 0, "unresolved": 0}
    client = docker.from_env(timeout=600)
    for index, instance in enumerate(tasks):
        iid = instance["instance_id"]
        key = f"{iid}::{arm}"
        if key in done:
            counters["finished"] += 1
            mark_progress(repo, arm, index + 1)
            continue
        if stop_event.is_set():
            break
        if spent(ledger) >= PRIMARY_CAP or spent(ledger, "cost_usd_billed") >= BILLED_STOP:
            print(
                f"STOP {arm} primary={spent(ledger):.2f} billed={spent(ledger, 'cost_usd_billed'):.2f}",
                flush=True,
            )
            (root / "stop-reason").write_text("spend\n")
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
            result = run_graphirm(instance, arm, out, root)
            merge_run_checks(iid, arm, out)
            result["instance_id"] = key
            result["task"] = iid
            result["repo"] = instance["repo"]
            result["position"] = instance["position"]
            result["n_tasks"] = instance["n_tasks"]
            result["scored"] = iid not in UNSCORED
            result["base_commit"] = instance["base_commit"]
            result["label"] = LABEL
            apply_guard(arm, out, result)
            result["complete"] = not bool(result.get("error"))
            kind = classify_arm(out, arm, result.get("error") or "")
            if result.get("error"):
                result["failure_class"] = kind
                counters["errors"] += 1
            counters["http_429"] += int(result.get("http_429") or 0)
            if kind == CONFIG and result.get("error"):
                with ledger_lock:
                    append_ledger(ledger, result)
                (root / "stop-reason").write_text("config\n")
                stop_event.set()
                raise SystemExit(f"STOP config {iid} {arm} {result.get('error')}")
            if result.get("invalid_config"):
                with ledger_lock:
                    append_ledger(ledger, result)
                (root / "stop-reason").write_text("config\n")
                stop_event.set()
                raise SystemExit(f"STOP invalid_config {iid} {arm} {result.get('error')}")
            if kind == REQUEST and result.get("error"):
                result["request_caused"] = True
                result["complete"] = True
                result["unresolved"] = True
                counters["unresolved"] += 1
                with streak_lock:
                    streaks["request"] += 1
                    streak = streaks["request"]
                print(f"UNRESOLVED {iid} {arm}", flush=True)
                if streak > 3:
                    with ledger_lock:
                        append_ledger(ledger, result)
                    (root / "stop-reason").write_text("request\n")
                    stop_event.set()
                    raise SystemExit(f"STOP request streak={streak}")
                break
            if kind == INFRASTRUCTURE and result.get("error"):
                with ledger_lock:
                    append_ledger(ledger, result)
                with streak_lock:
                    streaks["request"] = 0
                print(f"RECOVER {iid} {arm} wait={recover_wait}", flush=True)
                time.sleep(recover_wait)
                recover_wait = min(recover_wait * 2, 1800)
                if stop_event.is_set():
                    raise SystemExit(f"STOP during recovery {iid} {arm}")
                continue
            if result.get("error"):
                with ledger_lock:
                    append_ledger(ledger, result)
                print(f"RECOVER {iid} {arm} wait={recover_wait} other", flush=True)
                time.sleep(recover_wait)
                recover_wait = min(recover_wait * 2, 1800)
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
                grade = grade_patch(instance, patch, client, GRADE_TIMEOUT, out / "grading" / arm, f"ex-{arm}")
                resolved, status, grade_error = grade.resolved, grade.status, grade.error
            except Exception as exc:
                resolved, status, grade_error = None, "grade_exception", f"{type(exc).__name__}: {exc}"[:400]
        if resolved is None:
            print(f"HARNESS_ERR {iid} {arm} {status}", flush=True)
            resolved = False
            result["harness_error"] = f"{status}: {grade_error}"[:400]
        result["grade_status"] = status
        grades_path = out / "grades.json"
        with ledger_lock:
            grades = load_grades(grades_path)
            grades[arm] = resolved
            save_grades(grades_path, grades)
            append_ledger(ledger, result)
        promote_graph(root, arm, instance, out)
        done.add(key)
        counters["finished"] += 1
        mark_progress(repo, arm, index + 1)
        grades = load_grades(out / "grades.json")
        if all(name in grades for name in ARMS):
            remove_task_image(instance, client)
        print(
            f"DONE {repo.split('/')[-1]} {arm} {counters['finished']}/{len(tasks)} "
            f"primary={spent(ledger):.2f} billed={spent(ledger, 'cost_usd_billed'):.2f}",
            flush=True,
        )
    return counters


def run_job(repo, arm, tasks, root, ledger, done) -> dict:
    completed = False
    try:
        result = walk_job(repo, arm, tasks, root, ledger, done)
        completed = True
        return result
    finally:
        if completed or stop_event.is_set():
            mark_dead(repo, arm)


def selected_sequences() -> list[dict]:
    wanted = []
    for seq in load_sequences():
        if seq["repo"] in REPOS:
            wanted.append(seq)
    found = [seq["repo"] for seq in wanted]
    if found != list(REPOS):
        raise SystemExit(f"sequence order {found}")
    total = sum(len(seq["tasks"]) for seq in wanted)
    if total != 56:
        raise SystemExit(f"expected 56 tasks, found {total}")
    return wanted


def main() -> int:
    if sha256_file(BINARY) != BINARY_SHA:
        raise SystemExit(f"binary sha256 is not {BINARY_SHA}")
    if not LOCK.is_file():
        raise SystemExit(f"missing lock {LOCK}")
    root = refuse_tmp(runs_dir(TRACK))
    if (root / "stop-reason").is_file():
        raise SystemExit(f"STOP held {(root / 'stop-reason').read_text().strip()}")
    write_configs(root)
    (root / "LABEL.txt").write_text(LABEL + "\n")
    remaining = balance()
    if remaining < MIN_BALANCE:
        raise SystemExit(f"STOP balance {remaining:.2f} below {MIN_BALANCE:.0f}")
    (root / "running").write_text("explore\n")
    ledger = root / "arms.jsonl"
    done = {key for key, record in latest_rows(ledger).items() if arm_done(record)}
    sequences = selected_sequences()
    print(
        f"START {LABEL} tasks=56 arms=3 binary={BINARY_SHA[:12]} "
        f"primary={spent(ledger):.2f} billed={spent(ledger, 'cost_usd_billed'):.2f} "
        f"disk={used_pct():.1f} balance={remaining:.2f}",
        flush=True,
    )
    for seq in sequences:
        if stop_event.is_set():
            break
        progress[seq["repo"]] = {arm: 0 for arm in ARMS}
        dead_arms[seq["repo"]] = set()
        print(f"REPO {seq['repo']} tasks={len(seq['tasks'])} memory=empty", flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {
                pool.submit(run_job, seq["repo"], arm, seq["tasks"], root, ledger, done): arm
                for arm in ARMS
            }
            for future in as_completed(futures):
                arm = futures[future]
                try:
                    future.result()
                except SystemExit as exc:
                    print(str(exc), flush=True)
                    stop_event.set()
                except Exception:
                    traceback.print_exc()
                    stop_event.set()
    finished = sum(
        1
        for seq in sequences
        for task in seq["tasks"]
        if all(arm_done(latest_rows(ledger).get(f"{task['instance_id']}::{arm}")) for arm in ARMS)
    )
    print(
        f"HOLD tasks={finished}/56 primary={spent(ledger):.2f} "
        f"billed={spent(ledger, 'cost_usd_billed'):.2f}",
        flush=True,
    )
    if finished >= 56:
        from integrity import finish_track

        problems = finish_track(TRACK, root)
        if problems:
            print(f"INTEGRITY_FAIL {len(problems)}", flush=True)
            for line in problems[:30]:
                print(f"INTEGRITY {line}", flush=True)
            return 1
        (root / "running").unlink(missing_ok=True)
        print("INTEGRITY_OK explore", flush=True)
    return 0 if not (root / "stop-reason").is_file() else 1


if __name__ == "__main__":
    raise SystemExit(main())
