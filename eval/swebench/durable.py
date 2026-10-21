"""Durable eval outputs.

Run outputs live under /home/krs/graphirm-runs/<track>/. A path under /tmp
is refused. Ledger lines are flushed and fsynced. After each completed task
the record is copied into /home/krs/graphirm-runs-backup and committed there.
That backup is a separate git repo on the same disk. It survives a reboot
and a /tmp wipe. It does not survive loss of the disk.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
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
        if record.get("complete") is True:
            done.add(record["instance_id"])
    return done


def ensure_backup_repo() -> Path:
    """Create the backup git repo if it does not exist yet."""
    root = refuse_tmp(BACKUP_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    if not (root / ".git").is_dir():
        subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    return root


@contextmanager
def _backup_lock(root: Path):
    """One git writer at a time. T1's arms and the T2 validator share this repo."""
    handle = (root / ".backup.lock").open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _backup_once(track: str, record: dict, files: list[Path] | None) -> None:
    root = ensure_backup_repo()
    dest = refuse_tmp(root / track / record["instance_id"])
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    for src in files or []:
        if src.is_file():
            shutil.copy2(src, dest / src.name)
    append_ledger(root / track / "ledger.jsonl", record)
    with _backup_lock(root):
        subprocess.run(["git", "add", "-A", track], cwd=root, check=True)
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", track],
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


def backup_record(track: str, record: dict, files: list[Path] | None = None) -> bool:
    """Copy one record into the backup repo and commit it.

    Three tries, then a warning. A failure does not raise.
    """
    last = "backup failed"
    for attempt in range(3):
        try:
            _backup_once(track, record, files)
            return True
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            if attempt < 2:
                time.sleep(2)
    print(f"BACKUP_WARN {track} {record.get('instance_id')} {last}", flush=True)
    return False


def _pending_path(track: str) -> Path:
    return runs_dir(track) / "backup-pending.jsonl"


def remember_failed_backup(track: str, record: dict, files: list[Path] | None = None) -> None:
    """Keep a failed backup so the next 10-task push can try it again."""
    append_ledger(
        _pending_path(track),
        {
            "instance_id": record["instance_id"],
            "record": record,
            "files": [str(path) for path in (files or [])],
        },
    )


def pending_backup_count(track: str) -> int:
    path = _pending_path(track)
    if not path.exists():
        return 0
    ids = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            ids.add(json.loads(line)["instance_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return len(ids)


def retry_failed_backups(track: str) -> int:
    """Retry pending backups. Return how many are still failing."""
    path = _pending_path(track)
    if not path.exists():
        return 0
    latest: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[record["instance_id"]] = record
    still = []
    for item in latest.values():
        payload = item.get("record") or item
        files = [Path(name) for name in item.get("files") or []]
        if not backup_record(track, payload, files):
            still.append(item)
    if still:
        path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in still))
    elif path.exists():
        path.unlink()
    return len(still)


def push_backup() -> bool:
    """Push the backup repo to origin. A failure does not raise."""
    root = ensure_backup_repo()
    last = "push failed"
    for attempt in range(3):
        try:
            with _backup_lock(root):
                subprocess.run(
                    ["git", "push", "origin", "HEAD"],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            return True
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            if attempt < 2:
                time.sleep(2)
    print(f"BACKUP_WARN push {last}", flush=True)
    return False
