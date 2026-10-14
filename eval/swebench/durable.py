"""Durable eval outputs.

Run outputs live under /home/krs/graphirm-runs/<track>/. A path under /tmp
is refused. Ledger lines are flushed and fsynced. After each completed task
the record is copied into /home/krs/graphirm-runs-backup and committed there.
That backup is a separate git repo on the same disk. It survives a reboot
and a /tmp wipe. It does not survive loss of the disk.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

RUNS_ROOT = Path("/home/krs/graphirm-runs")
BACKUP_ROOT = Path("/home/krs/graphirm-runs-backup")
TMP_ROOT = Path("/tmp")


def refuse_tmp(path: Path) -> Path:
    """Return the resolved path, or exit if it is under /tmp."""
    resolved = path.expanduser().resolve()
    tmp = TMP_ROOT.resolve()
    if resolved == tmp or tmp in resolved.parents:
        raise SystemExit(f"refusing output under /tmp: {resolved}")
    return resolved


def runs_dir(track: str) -> Path:
    """Create and return the durable directory for one track."""
    if not track or "/" in track or track.startswith("."):
        raise SystemExit(f"refusing track name: {track}")
    path = refuse_tmp(RUNS_ROOT / track)
    root = RUNS_ROOT.resolve()
    if path != root and root not in path.parents:
        raise SystemExit(f"refusing output outside {root}: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_ledger(path: Path, record: dict) -> None:
    """Append one JSON line and fsync it."""
    path = refuse_tmp(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(record, sort_keys=True) + "\n").encode()
    with path.open("ab") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def load_done(path: Path) -> set[str]:
    """Instance ids already in the ledger.

    A partial final line, from a kill during the write, is ignored so that
    task can run again. A broken line anywhere earlier is an error.
    """
    path = refuse_tmp(path)
    if not path.exists():
        return set()
    done: set[str] = set()
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue
            raise
        done.add(record["instance_id"])
    return done


def ensure_backup_repo() -> Path:
    """Create the backup git repo if it does not exist yet."""
    root = refuse_tmp(BACKUP_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    if not (root / ".git").is_dir():
        subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    return root


def backup_record(track: str, record: dict, files: list[Path] | None = None) -> None:
    """Copy one task's record into the backup repo and commit it."""
    root = ensure_backup_repo()
    dest = refuse_tmp(root / track / record["instance_id"])
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    for src in files or []:
        if src.is_file():
            shutil.copy2(src, dest / src.name)
    append_ledger(root / track / "ledger.jsonl", record)
    subprocess.run(["git", "add", "-A", track], cwd=root, check=True)
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if not status.stdout.strip():
        return
    message = f"{track} {record['instance_id']}"
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
