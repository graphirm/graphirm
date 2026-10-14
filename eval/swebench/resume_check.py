#!/usr/bin/env python3
"""Five fake tasks that only touch the ledger.

Kill this process and start it again. Finished task ids are skipped.
A second start must not duplicate them.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from durable import append_ledger, backup_record, load_done, runs_dir

TRACK = "resume-test"
TASKS = [f"task-{index}" for index in range(5)]


def main() -> None:
    root = runs_dir(TRACK)
    ledger = root / "ledger.jsonl"
    done = load_done(ledger)
    print(f"START done={len(done)}", flush=True)
    for task in TASKS:
        if task in done:
            print(f"SKIP {task}", flush=True)
            continue
        time.sleep(1)
        record = {"instance_id": task, "complete": True}
        append_ledger(ledger, record)
        backup_record(TRACK, record)
        print(f"WROTE {task}", flush=True)
    print(f"HOLD {len(load_done(ledger))}", flush=True)


if __name__ == "__main__":
    main()
