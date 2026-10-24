#!/usr/bin/env python3
"""Post-track integrity check.

A track is complete only when check_track returns no problems. Ledgers are
not rewritten. A manifest is a new file; an existing manifest is left in place.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[1] / "registry.json"
REPO_LOGS = Path(__file__).resolve().parents[2] / "logs" / "run_evaluation"

T1_ARM_DIR = {
    "M": "t1m/M",
    "G": "t1g/G",
    "G-md": "t1gmd/G-md",
    "G-b16": "t1gb16/G-b16",
    "G-b16-recency": "t1gb16recency/G-b16-recency",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(run_dir: Path, arm: str, paths: list[Path]) -> Path:
    """Write arm.manifest.json, or arm.manifest.<n>.json if one already exists."""
    base = run_dir / f"{arm}.manifest.json"
    path = base
    n = 2
    while path.exists():
        path = run_dir / f"{arm}.manifest.{n}.json"
        n += 1
    files = []
    for item in paths:
        if item.is_file():
            files.append({"path": str(item.relative_to(run_dir)), "sha256": sha256(item)})
        else:
            files.append({"path": str(item.relative_to(run_dir)), "sha256": None, "missing": True})
    path.write_text(json.dumps({"arm": arm, "files": files}, indent=2) + "\n")
    return path


def _ledger(runs: Path) -> list[dict]:
    path = runs / "arms.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _grades(task_dir: Path) -> tuple[dict | None, str | None]:
    path = task_dir / "grades.json"
    if not path.is_file():
        return None, "grades.json missing"
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return None, f"grades.json {exc}"
    if not isinstance(data, dict):
        return None, "grades.json is not an object"
    return data, None


def _report_resolved(path: Path, iid: str) -> tuple[bool | None, str | None]:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return None, f"report {exc}"
    inner = data.get(iid) or next(iter(data.values()), None)
    if not isinstance(inner, dict) or "resolved" not in inner:
        return None, "report has no resolved field"
    return bool(inner["resolved"]), None


def _check_manifests(task_dir: Path, arm: str, problems: list[str], iid: str) -> None:
    manifests = sorted(task_dir.glob(f"{arm}.manifest*.json"))
    for manifest in manifests:
        try:
            data = json.loads(manifest.read_text())
        except json.JSONDecodeError as exc:
            problems.append(f"{iid} {arm} manifest {exc}")
            continue
        for item in data.get("files") or []:
            if item.get("missing"):
                problems.append(f"{iid} {arm} manifest lists missing {item.get('path')}")
                continue
            file_path = task_dir / item["path"]
            if not file_path.is_file():
                problems.append(f"{iid} {arm} manifest file gone {item.get('path')}")
                continue
            digest = sha256(file_path)
            if digest != item.get("sha256"):
                problems.append(f"{iid} {arm} hash mismatch {item.get('path')}")


def _t1(runs: Path, problems: list[str]) -> None:
    seen = set()
    for row in _ledger(runs):
        if row.get("complete") is not True:
            continue
        iid, arm = row.get("task"), row.get("arm")
        if not iid or not arm or (iid, arm) in seen:
            continue
        seen.add((iid, arm))
        task = runs / iid
        grades, err = _grades(task)
        if err:
            problems.append(f"{iid} {arm} {err}")
            continue
        if arm not in grades or grades[arm] is None:
            problems.append(f"{iid} {arm} missing from grades.json")
            continue
        grade = bool(grades[arm])
        patch = task / f"{arm}.patch"
        report = runs / "logs" / "run_evaluation" / T1_ARM_DIR.get(arm, arm) / iid / "report.json"
        if not report.is_file():
            report = REPO_LOGS / T1_ARM_DIR.get(arm, arm) / iid / "report.json"
        if patch.is_file() and patch.stat().st_size == 0:
            if grade:
                problems.append(f"{iid} {arm} empty patch graded true")
            continue
        if report.is_file():
            harness, rerr = _report_resolved(report, iid)
            if rerr:
                problems.append(f"{iid} {arm} {rerr}")
            elif harness != grade:
                problems.append(f"{iid} {arm} grades={grade} harness={harness}")
            continue
        log = runs / "logs" / "run_evaluation" / T1_ARM_DIR.get(arm, arm) / iid / "run_instance.log"
        text = log.read_text(errors="replace") if log.is_file() else ""
        if "Patch Apply Failed" in text and not grade:
            continue
        problems.append(f"{iid} {arm} no harness report")
        _check_manifests(task, arm, problems, iid)


def _t2(runs: Path, problems: list[str]) -> None:
    seen = set()
    for row in _ledger(runs):
        if row.get("complete") is not True:
            continue
        iid, arm = row.get("task"), row.get("arm")
        if not iid or not arm or (iid, arm) in seen:
            continue
        seen.add((iid, arm))
        task = runs / iid
        grades, err = _grades(task)
        if err:
            problems.append(f"{iid} {arm} {err}")
            continue
        reports = sorted((task / "grading" / arm).glob("attempt*/report.json"))
        status = row.get("grade_status")
        if status == "empty_patch":
            if arm in grades and grades[arm] is not False:
                problems.append(f"{iid} {arm} empty_patch grades={grades.get(arm)}")
            _check_manifests(task, arm, problems, iid)
            continue
        if arm not in grades or grades[arm] is None:
            if reports:
                harness, rerr = _report_resolved(reports[-1], iid)
                problems.append(
                    f"{iid} {arm} missing from grades.json harness={harness if not rerr else rerr}"
                )
            else:
                problems.append(f"{iid} {arm} missing from grades.json")
            continue
        if not reports:
            problems.append(f"{iid} {arm} no harness report")
            continue
        harness, rerr = _report_resolved(reports[-1], iid)
        if rerr:
            problems.append(f"{iid} {arm} {rerr}")
        elif harness != bool(grades[arm]):
            problems.append(f"{iid} {arm} grades={grades[arm]} harness={harness}")
        _check_manifests(task, arm, problems, iid)


def _t3(runs: Path, problems: list[str]) -> None:
    seen = set()
    settled_false = {"tests_failed", "apply_failed", "empty_patch", "unresolved", "error"}
    for row in _ledger(runs):
        if row.get("complete") is not True:
            continue
        iid, arm = row.get("task"), row.get("arm")
        if not iid or not arm or (iid, arm) in seen:
            continue
        seen.add((iid, arm))
        grades, err = _grades(runs / iid)
        if err:
            problems.append(f"{iid} {arm} {err}")
            continue
        if arm not in grades or grades[arm] is None:
            problems.append(f"{iid} {arm} missing from grades.json")
            continue
        status = row.get("grade_status")
        grade = bool(grades[arm])
        if status == "completed" and not grade:
            problems.append(f"{iid} {arm} status completed grades false")
        elif status in settled_false and grade:
            problems.append(f"{iid} {arm} status {status} grades true")
        elif status not in {"completed", *settled_false}:
            problems.append(f"{iid} {arm} status {status} has no harness rule")
        _check_manifests(runs / iid, arm, problems, iid)


def check_track(track: str, runs: Path) -> list[str]:
    problems: list[str] = []
    if not (runs / "arms.jsonl").is_file():
        return [f"{track} has no arms.jsonl"]
    if track == "t1":
        _t1(runs, problems)
    elif track == "t2":
        _t2(runs, problems)
    elif track == "t3":
        _t3(runs, problems)
    elif track in {"t4", "explore"}:
        _t2(runs, problems)
    else:
        return [f"unknown track {track}"]
    return problems


def finish_track(track: str, runs: Path) -> list[str]:
    """Write the problem list. The caller clears `running` only when this is empty."""
    problems = check_track(track, runs)
    text = "\n".join(problems) + ("\n" if problems else "")
    (runs / "integrity.txt").write_text(text)
    return problems
