"""End-of-T1 notes that do not change the run.

Budgets in the driver are Graphirm's word estimate (words / 0.75). This report
prints the provider's prompt-token counts beside that nominal budget, and the
median cache-hit ratio before and after the four-day gap.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from pathlib import Path

RUNS = Path("/home/krs/graphirm-runs/t1")
ARMS = ("M", "G", "G-md", "G-b16", "G-b16-recency")
GRAPH_ARMS = ("G", "G-md", "G-b16", "G-b16-recency")
NOMINAL = {"G": 100_000, "G-md": 100_000, "G-b16": 16_000, "G-b16-recency": 16_000}


def _assistant_usage(graph: Path) -> list[tuple[int, int]]:
    if not graph.is_file() or graph.stat().st_size == 0:
        return []
    try:
        conn = sqlite3.connect(f"file:{graph}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    rows: list[tuple[int, int]] = []
    try:
        for data, meta in conn.execute("SELECT data, metadata FROM nodes WHERE node_type='interaction'"):
            try:
                role = json.loads(data).get("role")
                fields = json.loads(meta)
            except json.JSONDecodeError:
                continue
            if role != "assistant":
                continue
            prompt = int(fields.get("usage_input") or 0)
            cached = int(fields.get("usage_cache_read") or 0)
            if prompt > 0:
                rows.append((prompt, cached))
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    return rows


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[index]


def _median_ratio(ratios: list[float]) -> float | None:
    if not ratios:
        return None
    return float(statistics.median(ratios))


def build_report(root: Path = RUNS) -> dict:
    before_path = root / "before-gap.json"
    before = set(json.loads(before_path.read_text())) if before_path.is_file() else set()
    prompts: dict[str, list[int]] = {arm: [] for arm in GRAPH_ARMS}
    ratios: dict[str, dict[str, list[float]]] = {
        arm: {"before": [], "after": []} for arm in ARMS
    }
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        iid = task_dir.name
        side = "before" if iid in before else "after"
        for arm in ARMS:
            usage = _assistant_usage(task_dir / f"{arm}.graph.db")
            if not usage:
                continue
            if arm in prompts:
                prompts[arm].extend(prompt for prompt, _cached in usage)
            prompt_sum = sum(prompt for prompt, _cached in usage)
            cache_sum = sum(cached for _prompt, cached in usage)
            if prompt_sum:
                ratios[arm][side].append(cache_sum / prompt_sum)
    tokens = {}
    for arm, values in prompts.items():
        tokens[arm] = {
            "nominal": NOMINAL[arm],
            "calls": len(values),
            "median": int(statistics.median(values)) if values else 0,
            "p90": _percentile(values, 0.9),
        }
    cache = {}
    for arm, sides in ratios.items():
        cache[arm] = {
            "before_median": _median_ratio(sides["before"]),
            "before_n": len(sides["before"]),
            "after_median": _median_ratio(sides["after"]),
            "after_n": len(sides["after"]),
        }
    return {"before_tasks": len(before), "tokens": tokens, "cache": cache}


def write_report(root: Path = RUNS) -> Path | None:
    dest = root / "closing.json"
    if dest.is_file():
        return dest
    if not (root / "before-gap.json").is_file():
        return None
    report = build_report(root)
    dest.write_text(json.dumps(report, indent=2) + "\n")
    for arm, row in report["tokens"].items():
        print(
            f"TOKENS arm={arm} nominal={row['nominal']} calls={row['calls']} "
            f"median={row['median']} p90={row['p90']}",
            flush=True,
        )
    for arm, row in report["cache"].items():
        print(
            f"CACHE arm={arm} before={row['before_median']} n={row['before_n']} "
            f"after={row['after_median']} n={row['after_n']}",
            flush=True,
        )
    return dest
