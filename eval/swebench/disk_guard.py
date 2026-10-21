"""Free Docker space the remaining T1 tasks do not need.

One reclaim per time the disk is above the wait line. Runs, graphs,
ledgers, and the backup repo are not touched.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

LOCK = Path("/home/krs/graphirm-repo/eval/swebench/instances.lock.json")
RUNS = Path("/home/krs/graphirm-runs/t1")
STAMP = RUNS / "disk-reclaim.json"
EXCLUDE = {
    "sigma67__ytmusicapi-909_interface",
    "huggingface__trl-5135",
    "vyperlang__vyper-4801",
    "keras-team__keras-22316",
    "tox-dev__tox-3931",
}
ARMS = ("M", "G", "G-md", "G-b16", "G-b16-recency")


def image_name(instance_id: str) -> str:
    slug = instance_id.replace("__", "_1776_")
    return f"swerebench/sweb.eval.x86_64.{slug}:latest"


def needed_images() -> set[str]:
    lock = json.loads(LOCK.read_text())
    ids = [row["instance_id"] for row in lock["headline"] if row["instance_id"] not in EXCLUDE]
    latest: dict[str, dict] = {}
    ledger = RUNS / "arms.jsonl"
    if ledger.is_file():
        for line in ledger.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                latest[row["instance_id"]] = row
    by: dict[str, dict] = {}
    for key, row in latest.items():
        iid = row.get("task") or key.split("::", 1)[0]
        by.setdefault(iid, {})[row.get("arm")] = row
    done = set()
    for iid, arms in by.items():
        if all(
            arm in arms
            and (
                arms[arm].get("request_caused")
                or (arms[arm].get("complete") is True and not arms[arm].get("error"))
            )
            for arm in ARMS
        ):
            done.add(iid)
    needed = {image_name(iid) for iid in ids if iid not in done}
    # Memory tracks share two CL-Bench images for every task. Keep them while T3 runs.
    if (RUNS.parent / "t3" / "running").is_file():
        needed |= {
            "pgasawa2/continual-learning-bench:tablib",
            "pgasawa2/continual-learning-bench:tenacity",
        }
    return needed


def _run(args: list[str]) -> str:
    proc = subprocess.run(args, check=False, capture_output=True, text=True)
    return (proc.stdout or proc.stderr or "").strip()


def running_images() -> set[str]:
    text = _run(["docker", "ps", "--format", "{{.Image}}"])
    return {line.strip() for line in text.splitlines() if line.strip()}


def reclaim(force: bool = False) -> str:
    """Prune unused Docker data once. `force` ignores the once-per-episode stamp."""
    if STAMP.is_file() and not force:
        return "already reclaimed this episode"
    kept = needed_images() | running_images()
    notes = [
        "containers " + _run(["docker", "container", "prune", "-f"]),
        "dangling " + _run(["docker", "image", "prune", "-f"]),
        "builder " + _run(["docker", "builder", "prune", "-f"]),
    ]
    listed = _run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"])
    removed = []
    for name in listed.splitlines():
        name = name.strip()
        if not name or name in kept or name.endswith(":<none>"):
            continue
        if name.startswith("swerebench/") or name.startswith("pgasawa2/") or name.startswith("swebench/"):
            result = _run(["docker", "rmi", name])
            removed.append(result.splitlines()[-1] if result else name)
        elif name in {"ubuntu:22.04", "debian:bookworm-slim"}:
            result = _run(["docker", "rmi", name])
            removed.append(result.splitlines()[-1] if result else name)
    STAMP.parent.mkdir(parents=True, exist_ok=True)
    STAMP.write_text(json.dumps({"removed": removed}) + "\n")
    if removed:
        notes.append("images " + "; ".join(removed))
    return " | ".join(note for note in notes if note)


def clear_stamp_if_below(limit: float, used: float) -> None:
    if used < limit and STAMP.is_file():
        STAMP.unlink()
