#!/usr/bin/env python3
"""Rerun the 13 T4 arms whose agent never started.

Attempt 1 died in bash on an empty statement in the test-file filter.
This is the one pre-registered harness rerun. Original rows, grades.json,
and attempt-1 reports stay. The new grade is a correction row
(rerun: launch, complete: false) so the integrity check still reads the
original empty_patch row.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_t4

PAIRS = [
    ("G", "ivov__lisette-491"),
    ("M", "ivov__lisette-491"),
    ("G", "ivov__lisette-627"),
    ("M", "ivov__lisette-627"),
    ("G", "ivov__lisette-657"),
    ("M", "ivov__lisette-657"),
    ("G", "kenn-io__agentsview-602"),
    ("M", "kenn-io__agentsview-602"),
    ("G", "kestra-io__kestra-16067"),
    ("M", "kestra-io__kestra-16067"),
    ("G", "kestra-io__kestra-16119"),
    ("M", "kestra-io__kestra-16119"),
    ("M", "kestra-io__kestra-16450"),
]


def done_pairs() -> set[tuple[str, str]]:
    found = set()
    if not run_t4.LEDGER.is_file():
        return found
    for line in run_t4.LEDGER.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("rerun") == "launch":
            found.add((row["arm"], row["task"]))
    return found


def pinned(task: str, arm: str) -> dict[str, bytes]:
    root = run_t4.RUNS / task
    files = {
        "grades": root / "grades.json",
        "manifest": root / f"{arm}.manifest.json",
        "report": root / "grading" / arm / "attempt1" / "report.json",
    }
    return {name: path.read_bytes() for name, path in files.items()}


def run_pair(arm: str, task_name: str) -> str:
    task = run_t4.DATA / task_name
    before = pinned(task_name, arm)
    if run_t4.disk_pct() >= run_t4.DISK_STOP:
        print(f"STOP disk {run_t4.disk_pct():.1f}", flush=True)
        return "disk"
    out, returncode, seen = run_t4.run_once(arm, task, 2)
    trial = run_t4.latest_trial(out)
    message = run_t4.exception_message(trial)
    if "syntax error near unexpected token" in message:
        print(f"SYNTAX {arm} {task_name}", flush=True)
        return "syntax"
    reward = run_t4.reward_of(trial)
    kind = run_t4.classify_trial(trial, returncode, reward)
    task_dir = run_t4.RUNS / task_name
    run_t4.copy_arm_files(arm, trial, task_dir)
    image = run_t4.image_of(task)
    cleaned = run_t4.cleanup_after_trial(image, task=task_name, known_size=seen.get("image_bytes"))
    after = pinned(task_name, arm)
    if after != before:
        print(f"PIN_BROKEN {arm} {task_name}", flush=True)
        return "pin"
    ok = run_t4.resolved(reward)
    patch = task_dir / f"{arm}.patch"
    patch_text = patch.read_text() if patch.is_file() else ""
    if kind == "wall":
        status = "unresolved"
        grade = False
    elif not patch_text.strip() and not ok:
        status = "empty_patch"
        grade = False
    elif reward is None:
        status = "error"
        grade = False
    else:
        status = "resolved" if ok else "unresolved"
        grade = ok
    run_t4.write_report(task_dir, arm, task_name, grade, 2)
    primary, billed = run_t4.costs_for(arm, task_dir)
    run_t4.record(
        {
            "task": task_name,
            "arm": arm,
            "complete": False,
            "counts_toward_cap": True,
            "corrects": "harness",
            "rerun": "launch",
            "grade": grade,
            "grade_status": status,
            "cost_usd": round(primary, 6),
            "cost_usd_billed": round(billed, 6),
            "returncode": returncode,
            "error": None if reward is not None or status == "empty_patch" else kind,
            "http_429": run_t4.count_429(message),
            "image": image,
            "image_bytes": cleaned.get("image_bytes"),
            "disk_pct_after": round(run_t4.disk_pct(), 1),
            "attempt": 2,
            "note": "harness rerun; attempt 1 never started the agent",
        }
    )
    print(
        f"DONE {arm} {task_name} status={status} rc={returncode} "
        f"primary={primary:.4f} billed={billed:.4f} disk={run_t4.disk_pct():.1f}",
        flush=True,
    )
    return "ok"


def main() -> None:
    pending = [pair for pair in PAIRS if pair not in done_pairs()]
    if not pending:
        print("NOTHING_TO_RUN", flush=True)
        return
    remaining = run_t4.balance()
    if remaining < run_t4.MIN_BALANCE:
        raise SystemExit(f"balance {remaining:.2f} is under {run_t4.MIN_BALANCE}")
    print(f"RERUN pairs={len(pending)} disk={run_t4.disk_pct():.1f}", flush=True)
    first, rest = pending[0], pending[1:]
    outcome = run_pair(*first)
    if outcome != "ok":
        raise SystemExit(f"canary {first[0]} {first[1]} {outcome}")
    slots = 1 if run_t4.disk_pct() >= run_t4.DISK_PARALLEL else 2
    if not rest:
        print("RERUN_DONE", flush=True)
        return
    with ThreadPoolExecutor(max_workers=slots) as pool:
        futures = {pool.submit(run_pair, arm, task): (arm, task) for arm, task in rest}
        for future in as_completed(futures):
            arm, task = futures[future]
            outcome = future.result()
            if outcome != "ok":
                raise SystemExit(f"stop {arm} {task} {outcome}")
    print("RERUN_DONE", flush=True)


if __name__ == "__main__":
    main()
