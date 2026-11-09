"""Pick the calibration, headline, robustness, and sequential instance lists.

Writes eval/swebench/instances.lock.json. Does not commit it.

Headline window: SWE-rebench tasks whose created_at falls in January through
May 2026. xAI documents Grok 4.7's knowledge cutoff as May 2026
(https://docs.x.ai/developers/models). Tasks from 2026-06-01 onward are
post-cutoff. The robustness set is Harbor's July 2026 swe-rebench tasks,
which sit after that line.

`--n 150` draws a month-stratified subset of the headline window. The
default writes every task in the window.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

SEED = 20261011
WINDOW_START = datetime(2026, 1, 1)
WINDOW_END = datetime(2026, 6, 1)
CALIBRATION_N = 30
SEQUENTIAL_CAP = 20
HERE = Path(__file__).resolve().parent
LOCK = HERE / "instances.lock.json"

VERIFIED = "princeton-nlp/SWE-bench_Verified"
REBENCH = "nebius/SWE-rebench-leaderboard"
HARBOR_DATASET = "ibragim-badertdinov/swe-rebench-07-2026"


def _parse_created(value: str) -> datetime:
    text = value.strip().replace("Z", "")
    if len(text) >= 19 and text[10] in " T":
        return datetime.strptime(text[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S")
    return datetime.strptime(text[:10], "%Y-%m-%d")


def _parquet_urls(name: str, split: str) -> list[str]:
    import urllib.request

    api = f"https://huggingface.co/api/datasets/{name}/parquet"
    with urllib.request.urlopen(api, timeout=60) as response:
        listing = json.load(response)
    return listing["default"][split]


def _load_rows(name: str, split: str, columns: list[str]) -> list[dict]:
    import urllib.request

    import pyarrow.parquet as pq

    cache = Path("/tmp/graphirm-swebench-cache")
    cache.mkdir(exist_ok=True)
    rows: list[dict] = []
    for index, url in enumerate(_parquet_urls(name, split)):
        dest = cache / f"{name.replace('/', '__')}__{split}__{index}.parquet"
        if not dest.exists():
            urllib.request.urlretrieve(url, dest)
        table = pq.read_table(dest, columns=columns)
        rows.extend(table.to_pylist())
    return rows


def _in_window(value: str) -> bool:
    created = _parse_created(str(value))
    return WINDOW_START <= created < WINDOW_END


def stratify_by_month(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Proportional sample. The largest fractional parts fill the leftover seats."""
    if n >= len(rows):
        return list(rows)
    by_month: dict[str, list[dict]] = {}
    for row in rows:
        by_month.setdefault(str(row["created_at"])[:7], []).append(row)
    months = sorted(by_month)
    raw = {month: len(by_month[month]) * n / len(rows) for month in months}
    alloc = {month: int(raw[month]) for month in months}
    leftover = n - sum(alloc.values())
    for month in sorted(months, key=lambda item: (raw[item] - alloc[item], item), reverse=True):
        if leftover <= 0:
            break
        alloc[month] += 1
        leftover -= 1
    rng = random.Random(seed)
    picked: list[dict] = []
    for month in months:
        pool = list(by_month[month])
        rng.shuffle(pool)
        picked.extend(pool[: alloc[month]])
    picked.sort(key=lambda row: (row["created_at"], row["instance_id"]))
    return picked


def load_harbor_ids() -> list[dict]:
    """Task ids for Harbor's July 2026 swe-rebench set.

    The dataset is a Harbor task directory, not a Hugging Face parquet.
    A local `harbor dataset download` under /tmp/harbor-ds is preferred.
    A cached listing at /tmp/graphirm-swebench-cache/harbor-07-2026.json
    is used when that download is absent.
    """
    cache = Path("/tmp/graphirm-swebench-cache/harbor-07-2026.json")
    download_roots = [
        Path("/tmp/harbor-ds/swe-rebench-07-2026"),
        Path("/tmp/graphirm-swebench-cache/swe-rebench-07-2026"),
    ]
    for root in download_roots:
        if not root.is_dir():
            continue
        rows = []
        for config in sorted(root.glob("*/tests/config.json")):
            payload = json.loads(config.read_text())
            rows.append(
                {
                    "instance_id": payload.get("instance_id") or config.parents[1].name,
                    "language": payload.get("language"),
                    "dataset": HARBOR_DATASET,
                }
            )
        if rows:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(rows, indent=2) + "\n")
            return rows
    if cache.exists():
        return json.loads(cache.read_text())

    import urllib.request

    # The published set is consumed as `ibragim-badertdinov/swe-rebench-07-2026`.
    # Task ids are filled by harbor_task_ids() when that command is available.
    try:
        import shutil
        import subprocess

        if shutil.which("harbor"):
            listed = subprocess.run(
                ["harbor", "dataset", "list", HARBOR_DATASET],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if listed.returncode == 0 and listed.stdout.strip():
                rows = []
                for line in listed.stdout.splitlines():
                    text = line.strip()
                    if not text or text.startswith("#") or " " in text[:1]:
                        continue
                    rows.append({"instance_id": text, "dataset": HARBOR_DATASET})
                if rows:
                    cache.write_text(json.dumps(rows, indent=2) + "\n")
                    return rows
    except (OSError, subprocess.SubprocessError):
        pass

    # Last resort: the dataset card's file listing, if the repo is public.
    api = f"https://huggingface.co/api/datasets/{HARBOR_DATASET}"
    try:
        with urllib.request.urlopen(api, timeout=60) as response:
            payload = json.load(response)
    except Exception:
        return []
    siblings = payload.get("siblings") or []
    rows = []
    for item in siblings:
        name = item.get("rfilename") or ""
        if name.endswith("/task.toml") or name.endswith("/instruction.md"):
            instance = name.split("/")[0]
            rows.append({"instance_id": instance, "dataset": HARBOR_DATASET})
    dedup = {row["instance_id"]: row for row in rows}
    found = [dedup[key] for key in sorted(dedup)]
    if found:
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(found, indent=2) + "\n")
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Month-stratified headline size. Omit to keep every Jan–May 2026 task.",
    )
    args = parser.parse_args()

    verified = _load_rows(VERIFIED, "test", ["instance_id", "repo"])
    rng = random.Random(SEED)
    order = list(range(len(verified)))
    rng.shuffle(order)
    calibration = []
    for index in order[:CALIBRATION_N]:
        row = verified[index]
        calibration.append({"instance_id": row["instance_id"], "repo": row["repo"]})

    rebench = _load_rows(REBENCH, "test", ["instance_id", "repo", "created_at"])
    window = []
    for row in rebench:
        if not _in_window(str(row["created_at"])):
            continue
        window.append(
            {
                "instance_id": row["instance_id"],
                "repo": row["repo"],
                "created_at": str(row["created_at"]),
            }
        )
    window.sort(key=lambda row: (row["created_at"], row["instance_id"]))
    headline = stratify_by_month(window, args.n, SEED) if args.n else window

    counts = Counter(row["repo"] for row in window)
    top = counts.most_common(5)
    sequential_repo = top[0][0] if top else None
    sequential = [row for row in window if row["repo"] == sequential_repo][:SEQUENTIAL_CAP]

    robustness = load_harbor_ids()
    lock = {
        "seed": SEED,
        "headline_window": ["2026-01-01", "2026-06-01"],
        "headline_note": "SWE-rebench tasks with created_at in January through May 2026.",
        "calibration_n": CALIBRATION_N,
        "headline_n_requested": args.n,
        "headline_available": len(window),
        "calibration": calibration,
        "headline": headline,
        "robustness_dataset": HARBOR_DATASET,
        "robustness": robustness,
        "repo_counts_top5": [{"repo": repo, "count": count} for repo, count in top],
        "sequential_repo": sequential_repo,
        "sequential": sequential,
    }
    LOCK.write_text(json.dumps(lock, indent=2) + "\n")
    print(
        json.dumps(
            {
                "verified_rows": len(verified),
                "rebench_test_rows": len(rebench),
                "headline_available": len(window),
                "headline": len(headline),
                "calibration": len(calibration),
                "robustness": len(robustness),
                "sequential_repo": sequential_repo,
                "sequential": len(sequential),
                "repo_counts_top5": lock["repo_counts_top5"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
