#!/usr/bin/env python3
"""Restart a track driver after an infrastructure exit. No phone or email.

Each driver keeps /home/krs/graphirm-runs/<track>/running while it should
be running, and removes that file when the track finishes. A request-caused
or config stop writes stop-reason and must not be restarted.

This timer does not contact the user. It only starts a driver again.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

RUNS_ROOT = Path("/home/krs/graphirm-runs")
PYTHON = "/home/krs/anaconda3/envs/py312/bin/python3"
REPO = "/home/krs/graphirm-repo"
EVAL = Path(REPO) / "eval" / "swebench"


@dataclass(frozen=True)
class Track:
    name: str
    unit: str
    driver: Path
    total: int

    @property
    def runs(self) -> Path:
        return RUNS_ROOT / self.name

    @property
    def log(self) -> Path:
        return self.runs / "run.log"

    @property
    def state(self) -> Path:
        return self.runs / "recover-state.json"


TRACKS = [
    Track("t1", "t1-rerun", EVAL / "run_t1.py", 210),
    Track("t3", "t3-memory", EVAL / "run_t3.py", 19),
]


def unit_active(unit: str) -> bool:
    proc = subprocess.run(
        ["systemctl", "--user", "is-active", unit],
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() == "active"


def hold_finished(track: Track) -> bool:
    if not track.log.is_file():
        return False
    tail = track.log.read_bytes()[-12000:].decode(errors="replace")
    return any(line.startswith(f"HOLD tasks={track.total} ") for line in tail.splitlines())


def load_state(track: Track) -> dict:
    if not track.state.is_file():
        return {"wait": 60, "next_at": 0}
    try:
        return json.loads(track.state.read_text())
    except json.JSONDecodeError:
        return {"wait": 60, "next_at": 0}


def save_state(track: Track, state: dict) -> None:
    track.state.write_text(json.dumps(state) + "\n")


def start_driver(track: Track) -> None:
    subprocess.run(["systemctl", "--user", "reset-failed", track.unit], check=False)
    subprocess.run(
        [
            "systemd-run",
            "--user",
            f"--unit={track.unit}",
            "--collect",
            f"--working-directory={REPO}",
            f"--property=StandardOutput=append:{track.log}",
            f"--property=StandardError=append:{track.log}",
            "--setenv=PYTHONUNBUFFERED=1",
            PYTHON,
            str(track.driver),
        ],
        check=True,
    )


def preflight_image(track: Track) -> str:
    if track.name == "t1":
        from run_t1 import headline_ids, load_instances

        return load_instances(headline_ids())[0]["image_name"]
    from validate_clbench import load_instances as load_clbench

    return load_clbench()[0]["image_name"]


def recover(track: Track) -> None:
    state = load_state(track)
    now = time.time()
    if now < float(state.get("next_at") or 0):
        print(f"WATCH {track.name} backoff wait={int(float(state['next_at']) - now)}", flush=True)
        return
    sys.path.insert(0, str(EVAL))
    from run_t1 import preflight, runs_dir

    root = runs_dir(track.name)
    try:
        preflight(root, preflight_image(track))
    except SystemExit as exc:
        wait = min(int(state.get("wait") or 60) * 2, 1800)
        state["wait"] = wait
        state["next_at"] = time.time() + wait
        save_state(track, state)
        print(f"WATCH {track.name} preflight failed {exc.code} next={wait}", flush=True)
        return
    if unit_active(track.unit):
        print(f"WATCH {track.name} ok", flush=True)
        return
    start_driver(track)
    wait = min(int(state.get("wait") or 60) * 2, 1800)
    state["wait"] = wait
    state["next_at"] = time.time() + wait
    save_state(track, state)
    print(f"WATCH resumed {track.name}", flush=True)


def check_track(track: Track) -> None:
    if track.name == "t1" and hold_finished(track):
        sys.path.insert(0, str(EVAL))
        from closing import write_report

        write_report(track.runs)
    stop = track.runs / "stop-reason"
    if stop.is_file():
        print(f"WATCH {track.name} stop {stop.read_text().strip()}", flush=True)
        return
    if not (track.runs / "running").is_file():
        print(f"WATCH {track.name} idle", flush=True)
        return
    if unit_active(track.unit):
        print(f"WATCH {track.name} ok", flush=True)
        return
    recover(track)


def main() -> None:
    import shutil

    sys.path.insert(0, str(EVAL))
    from disk_guard import clear_stamp_if_below, reclaim

    used = shutil.disk_usage("/")
    pct = 100.0 * used.used / used.total
    if pct >= 78.0:
        print(reclaim(), flush=True)
        used = shutil.disk_usage("/")
        pct = 100.0 * used.used / used.total
        if pct >= 78.0:
            print(f"WATCH disk {pct:.1f}", flush=True)
    else:
        clear_stamp_if_below(78.0, pct)
    for track in TRACKS:
        try:
            check_track(track)
        except Exception as exc:
            print(f"WATCH {track.name} error {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
