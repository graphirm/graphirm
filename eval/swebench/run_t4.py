#!/usr/bin/env python3
"""T4 robustness driver.

The paid loop is not in this file yet. When it is added, it must call
finish() after the last task and must not clear `running` itself.
finish() is the integrity gate from eval/DATA_RULES.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

from integrity import finish_track

RUNS = Path("/home/krs/graphirm-runs/t4")


def finish(root: Path = RUNS) -> list[str]:
    problems = finish_track("t4", root)
    if problems:
        print(f"INTEGRITY_FAIL {len(problems)}", flush=True)
        for line in problems[:30]:
            print(f"INTEGRITY {line}", flush=True)
        raise SystemExit(1)
    (root / "running").unlink(missing_ok=True)
    print("INTEGRITY_OK t4", flush=True)
    return problems


def main() -> int:
    print(
        "T4 is not started. The oracle and empty-patch pass have to finish first, "
        "and finish() is the only way the track can count as complete.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
