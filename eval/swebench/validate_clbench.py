#!/usr/bin/env python3
"""Gold and empty checks for the 19 CL-Bench codebase tasks.

Uses continual-learning-bench's generic PR runtime. The public
evaluate_submission helper treats an empty patch as a failure without
running tests, so this script calls evaluate_generic_pr_submission,
which applies no model patch and still runs the suite.

Outputs go under /home/krs/graphirm-runs/t3-validity/.
"""

from __future__ import annotations

import json
import shutil
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/krs/graphirm-runs/continual-learning-bench/src")

from durable import append_ledger, backup_record, load_done, runs_dir
from tasks.codebase_adaptation.generic_runtime import evaluate_generic_pr_submission

SOURCE = Path("/home/krs/graphirm-runs/sources/clbench-codebase.jsonl")
ORDER = [
    "jazzband__tablib-534",
    "jazzband__tablib-540",
    "jazzband__tablib-547",
    "jazzband__tablib-579",
    "jazzband__tablib-584",
    "jazzband__tablib-594",
    "jazzband__tablib-595",
    "jazzband__tablib-596",
    "jazzband__tablib-613",
    "jd__tenacity-597",
    "jd__tenacity-603",
    "jd__tenacity-604",
    "jd__tenacity-606",
    "jd__tenacity-609",
    "jd__tenacity-610",
    "jd__tenacity-611",
    "jd__tenacity-614",
    "jd__tenacity-615",
    "jd__tenacity-628",
]
TRACK = "t3-validity"
WORKERS = 1
DISK_LIMIT = 82.0
SETTLED = {"completed", "tests_failed", "apply_failed"}


def used_pct() -> float:
    usage = shutil.disk_usage("/")
    return 100.0 * usage.used / usage.total


def wait_disk() -> None:
    import time

    while used_pct() >= DISK_LIMIT:
        print(f"DISK {used_pct():.1f} waiting", flush=True)
        time.sleep(30)


def load_instances() -> list[dict]:
    rows = {
        json.loads(line)["instance_id"]: json.loads(line)
        for line in SOURCE.read_text().splitlines()
        if line.strip()
    }
    missing = [iid for iid in ORDER if iid not in rows]
    if missing:
        raise SystemExit(f"missing {missing}")
    return [rows[iid] for iid in ORDER]


def _grade(patch: str, instance: dict):
    last = None
    for _attempt in (1, 2):
        last = evaluate_generic_pr_submission(patch, instance)
        if last.status in SETTLED:
            return last
    return last


def one(instance: dict) -> dict:
    iid = instance["instance_id"]
    wait_disk()
    record = {
        "instance_id": iid,
        "gold_resolved": None,
        "empty_resolved": None,
        "gold_status": None,
        "empty_status": None,
        "gold_error": None,
        "empty_error": None,
        "complete": False,
    }
    gold = _grade(instance.get("patch") or "", instance)
    empty = _grade("", instance)
    record["gold_status"] = gold.status
    record["empty_status"] = empty.status
    record["gold_error"] = (gold.error or None) and str(gold.error)[:400]
    record["empty_error"] = (empty.error or None) and str(empty.error)[:400]
    if gold.status in SETTLED:
        record["gold_resolved"] = bool(gold.success)
    if empty.status in SETTLED:
        record["empty_resolved"] = bool(empty.success)
    record["complete"] = gold.status in SETTLED and empty.status in SETTLED
    return record


def main() -> None:
    root = runs_dir(TRACK)
    ledger = root / "ledger.jsonl"
    done = load_done(ledger)
    pending = [row for row in load_instances() if row["instance_id"] not in done]
    print(
        f"START total=19 done={len(done)} pending={len(pending)} disk={used_pct():.1f}",
        flush=True,
    )
    if not pending:
        print("HOLD nothing pending", flush=True)
        return

    def run_one(instance: dict) -> None:
        iid = instance["instance_id"]
        try:
            record = one(instance)
        except Exception as exc:
            record = {
                "instance_id": iid,
                "gold_resolved": None,
                "empty_resolved": None,
                "gold_status": None,
                "empty_status": None,
                "gold_error": f"{type(exc).__name__}: {exc}"[:400],
                "empty_error": None,
                "complete": False,
                "trace": traceback.format_exc()[-600:],
            }
        append_ledger(ledger, record)
        backup_record(TRACK, record)
        print(
            f"VALID {iid} gold={record.get('gold_resolved')} empty={record.get('empty_resolved')} "
            f"complete={record.get('complete')} gold_status={record.get('gold_status')} "
            f"empty_status={record.get('empty_status')}",
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(run_one, row) for row in pending]
        for future in as_completed(futures):
            future.result()
    print("HOLD validation finished", flush=True)


if __name__ == "__main__":
    main()
