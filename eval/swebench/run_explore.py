#!/usr/bin/env python3
"""Exploratory G-persist follow-ups. Not pre-registered.

Arms, when a run is started: G-persist-nonote and G-persist-nooldtools,
on the astropy and matplotlib sequences. finish() is required before either
arm counts as complete. This file does not start a run.
"""

from __future__ import annotations

import sys
from pathlib import Path

from integrity import finish_track

RUNS = Path("/home/krs/graphirm-runs/explore")


def finish(root: Path = RUNS) -> list[str]:
    problems = finish_track("explore", root)
    if problems:
        print(f"INTEGRITY_FAIL {len(problems)}", flush=True)
        for line in problems[:30]:
            print(f"INTEGRITY {line}", flush=True)
        raise SystemExit(1)
    (root / "running").unlink(missing_ok=True)
    print("INTEGRITY_OK explore", flush=True)
    return problems


def main() -> int:
    print(
        "Exploratory runs are not started. They wait until T4 has finished, "
        "and finish() is the only way a run can count as complete.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
