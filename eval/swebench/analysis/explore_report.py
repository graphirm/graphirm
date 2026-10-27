#!/usr/bin/env python3
"""Exploratory memory arms on matplotlib and astropy. Not pre-registered.

Compares G-persist-control, G-persist-nonote and G-persist-nooldtools with
each other and with the recorded T2 G-persist and G-fresh runs on the same
tasks. Resolve counts use the 55 scored tasks. astropy__astropy-7606 ran so
the carried graph matches T2, and it is left out of the scored counts the
same way T2 leaves it out. Nothing here rewrites a run file.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from metrics import primary_cost_from_calls, recorded_billed  # noqa: E402
from t1_report import (  # noqa: E402
    N_BOOT,
    SEED,
    bootstrap_ratio_of_medians,
    paired_bootstrap_diff,
    win_tie_loss,
)

EXPLORE = Path("/home/krs/graphirm-runs/explore")
T2 = Path("/home/krs/graphirm-runs/t2")
OUT = EXPLORE / "analysis"
LOCK = HERE.parent / "instances.lock.json"
REPOS = ("matplotlib/matplotlib", "astropy/astropy")
EXPLORE_ARMS = ("G-persist-control", "G-persist-nonote", "G-persist-nooldtools")
RECORDED_ARMS = ("G-persist", "G-fresh")
ARMS = EXPLORE_ARMS + RECORDED_ARMS
UNSCORED = {"astropy__astropy-7606"}
THIRDS = ("first", "middle", "last")
CAP_EXITS = {"max_cost", "cost_cap", "budget", "LimitsExceeded"}
LABEL = "exploratory, not pre-registered"


def load_tasks() -> list[dict]:
    lock = json.loads(LOCK.read_text())
    tasks = []
    found = []
    for seq in lock["memory_tracks"]["swe_bench_cl"]["sequences"]:
        if seq["repo"] not in REPOS:
            continue
        found.append(seq["repo"])
        n = int(seq["n_tasks"])
        for row in seq["tasks"]:
            pos = int(row["position"])
            iid = row["instance_id"]
            tasks.append(
                {
                    "iid": iid,
                    "repo": seq["repo"],
                    "position": pos,
                    "n_tasks": n,
                    "third": THIRDS[min(2, (pos - 1) * 3 // n)],
                    "scored": iid not in UNSCORED,
                }
            )
    if found != list(REPOS):
        raise SystemExit(f"sequence order {found}")
    if len(tasks) != 56:
        raise SystemExit(f"expected 56 tasks, found {len(tasks)}")
    if sum(1 for t in tasks if t["scored"]) != 55:
        raise SystemExit("expected 55 scored tasks")
    return tasks


def full_order() -> list[dict]:
    """Every T2 task in published order, so G-persist can drop earlier nodes."""
    lock = json.loads(LOCK.read_text())
    tasks = []
    for seq in lock["memory_tracks"]["swe_bench_cl"]["sequences"]:
        n = int(seq["n_tasks"])
        for row in seq["tasks"]:
            pos = int(row["position"])
            tasks.append(
                {
                    "iid": row["instance_id"],
                    "repo": seq["repo"],
                    "position": pos,
                    "n_tasks": n,
                }
            )
    return tasks


def graph_calls(path: Path) -> dict[str, dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT id, metadata, data FROM nodes WHERE node_type = 'interaction' "
            "AND json_extract(metadata, '$.usage_input') IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    out = {}
    for nid, meta, data in rows:
        parsed = json.loads(meta)
        parsed["_budget_note"] = "[Budget]" in (meta or "") or "[Budget]" in (data or "")
        out[nid] = parsed
    return out


def new_calls(path: Path, known: set[str]) -> tuple[list[dict], set[str]]:
    current = graph_calls(path)
    calls = [meta for nid, meta in current.items() if nid not in known]
    return calls, set(current)


def read_summary(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def grade_of(root: Path, iid: str, arm: str) -> bool:
    grades = json.loads((root / iid / "grades.json").read_text())
    if arm not in grades or grades[arm] is None:
        raise SystemExit(f"{iid} {arm} has no grade")
    return bool(grades[arm])


def task_row(calls: list[dict], summary: dict, resolved: bool) -> dict:
    prompts = [int(c.get("usage_input") or 0) for c in calls]
    dropped = [int(c.get("context_dropped_nodes") or 0) for c in calls]
    notes = sum(1 for c in calls if c.get("_budget_note"))
    exit_reason = summary.get("exit_reason") or summary.get("exit_status") or "missing"
    wall = summary.get("wall_seconds")
    return {
        "resolved": resolved,
        "calls": len(calls),
        "prompt_per_call": float(np.median(prompts)) if prompts else None,
        "cost_usd": primary_cost_from_calls(calls),
        "cost_usd_billed": recorded_billed(summary),
        "wall_seconds": None if wall is None else float(wall),
        "exit_reason": exit_reason,
        "budget_stop": exit_reason in CAP_EXITS,
        "dropped_nodes": int(sum(dropped)),
        "calls_with_drops": int(sum(1 for d in dropped if d)),
        "budget_notes_stored": notes,
    }


def collect_explore(tasks: list[dict]) -> dict[str, dict[str, dict]]:
    rows = {arm: {} for arm in EXPLORE_ARMS}
    for repo in REPOS:
        known = {arm: set() for arm in EXPLORE_ARMS}
        for task in tasks:
            if task["repo"] != repo:
                continue
            iid = task["iid"]
            for arm in EXPLORE_ARMS:
                calls, known[arm] = new_calls(EXPLORE / iid / f"{arm}.graph.db", known[arm])
                summary = read_summary(EXPLORE / iid / f"{arm}.summary.json")
                rows[arm][iid] = task_row(calls, summary, grade_of(EXPLORE, iid, arm))
    return rows


def collect_recorded(wanted: set[str]) -> dict[str, dict[str, dict]]:
    rows = {arm: {} for arm in RECORDED_ARMS}
    known: dict[str, set[str]] = {}
    for task in full_order():
        iid = task["iid"]
        repo = task["repo"]
        for arm in RECORDED_ARMS:
            if arm == "G-fresh":
                calls, _ = new_calls(T2 / iid / f"{arm}.graph.db", set())
            else:
                calls, known[repo] = new_calls(T2 / iid / f"{arm}.graph.db", known.get(repo, set()))
            if iid in wanted:
                summary = read_summary(T2 / iid / f"{arm}.summary.json")
                rows[arm][iid] = task_row(calls, summary, grade_of(T2, iid, arm))
    return rows


def arr(rows: dict[str, dict], tasks: list[dict], field: str) -> np.ndarray:
    vals = []
    for task in tasks:
        value = rows[task["iid"]][field]
        vals.append(np.nan if value is None else float(value))
    return np.array(vals, dtype=float)


def med(values: np.ndarray) -> float | None:
    clean = values[~np.isnan(values)]
    if len(clean) == 0:
        return None
    return float(np.median(clean))


def share(flags: np.ndarray) -> float | None:
    clean = flags[~np.isnan(flags)]
    if len(clean) == 0:
        return None
    return float(clean.mean())


def slice_summary(rows: dict[str, dict], tasks: list[dict]) -> dict:
    resolved = arr(rows, tasks, "resolved")
    calls = arr(rows, tasks, "calls")
    prompt = arr(rows, tasks, "prompt_per_call")
    cost = arr(rows, tasks, "cost_usd")
    billed = arr(rows, tasks, "cost_usd_billed")
    wall = arr(rows, tasks, "wall_seconds")
    stops = arr(rows, tasks, "budget_stop")
    dropped = arr(rows, tasks, "dropped_nodes")
    drop_calls = arr(rows, tasks, "calls_with_drops")
    call_counts = arr(rows, tasks, "calls")
    notes = arr(rows, tasks, "budget_notes_stored")
    exits = Counter(rows[t["iid"]]["exit_reason"] for t in tasks)
    any_drop = (~np.isnan(dropped)) & (dropped > 0)
    return {
        "n": len(tasks),
        "resolved": int(np.nansum(resolved)),
        "calls_median": None if med(calls) is None else round(med(calls), 1),
        "prompt_per_call_median": None if med(prompt) is None else int(round(med(prompt))),
        "cost_usd_median": None if med(cost) is None else round(med(cost), 6),
        "cost_usd_sum": round(float(np.nansum(cost)), 4),
        "cost_usd_billed_median": None if med(billed) is None else round(med(billed), 6),
        "cost_usd_billed_sum": round(float(np.nansum(billed)), 4),
        "wall_seconds_median": None if med(wall) is None else round(med(wall), 1),
        "budget_stop_share": None if share(stops) is None else round(share(stops), 4),
        "budget_stop_tasks": int(np.nansum(stops)),
        "tasks_with_drops": int(any_drop.sum()),
        "dropped_nodes_median": None if med(dropped) is None else int(round(med(dropped))),
        "dropped_nodes_median_when_any": None
        if not any_drop.any()
        else int(round(float(np.median(dropped[any_drop])))),
        "calls_with_drops": int(np.nansum(drop_calls)),
        "calls_sum": int(np.nansum(call_counts)),
        "budget_notes_stored": int(np.nansum(notes)),
        "exit_reasons": dict(exits),
    }


def compare(arm_rows: dict[str, dict], base_rows: dict[str, dict], tasks: list[dict]) -> dict:
    a = arr(arm_rows, tasks, "resolved")
    b = arr(base_rows, tasks, "resolved")
    diff, lo, hi, _ = paired_bootstrap_diff(a, b, np.random.default_rng(SEED))
    out = {
        "n": len(tasks),
        "resolved_arm": int(a.sum()),
        "resolved_base": int(b.sum()),
        "diff_pp": round(diff * 100, 2),
        "ci95_pp": [round(lo * 100, 2), round(hi * 100, 2)],
        **win_tie_loss(a.astype(int), b.astype(int)),
    }
    ratios = {}
    for field in ("calls", "prompt_per_call", "cost_usd", "cost_usd_billed", "wall_seconds"):
        left = arr(arm_rows, tasks, field)
        right = arr(base_rows, tasks, field)
        mask = ~(np.isnan(left) | np.isnan(right))
        if int(mask.sum()) == 0 or float(np.median(right[mask])) == 0:
            ratios[field] = None
            continue
        point, rlo, rhi = bootstrap_ratio_of_medians(left[mask], right[mask], np.random.default_rng(SEED))
        ratios[field] = {
            "n": int(mask.sum()),
            "ratio": round(point, 3),
            "ci95": [round(rlo, 3), round(rhi, 3)],
        }
    out["ratio_of_medians"] = ratios
    return out


def render(report: dict) -> str:
    lines = [
        "# Exploratory memory arms",
        "",
        "Exploratory, not pre-registered. Matplotlib then astropy. "
        "Resolve counts are the 55 scored tasks. `astropy__astropy-7606` ran so the "
        "carried graph matches T2, and it is left out of these counts. "
        "The explore integrity check reported no problems.",
        "",
        "Calls on a persistent arm are interaction nodes that were not in the previous "
        "task's graph. Recorded G-persist walks the whole T2 sequence before counting "
        "the matplotlib tasks. Each exploratory arm starts empty at matplotlib and again "
        "at astropy. Primary cost is the per-call micro formula. Billed cost is "
        f"OpenRouter usage.cost. Bootstrap seed {SEED}, {N_BOOT} draws. "
        "A ratio resamples tasks and divides the two medians. Each ratio uses its own "
        "generator with that seed. Compare cost with the primary formula. "
        "The stored OpenRouter bill on the October runs sits above that formula, "
        "and on these new runs it sits below.",
        "",
        "## Resolved",
        "",
        "| arm | resolved | vs control | vs G-persist | vs G-fresh |",
        "|---|---:|---|---|---|",
    ]
    for arm in EXPLORE_ARMS:
        summary = report["overall"][arm]
        cell = [f"{summary['resolved']}/{summary['n']}"]
        for base in ("G-persist-control", "G-persist", "G-fresh"):
            if arm == "G-persist-control" and base == "G-persist-control":
                cell.append("—")
                continue
            pair = report["comparisons"][arm][base]
            cell.append(
                f"{pair['diff_pp']:+.2f} pp [{pair['ci95_pp'][0]:+.2f}, {pair['ci95_pp'][1]:+.2f}] "
                f"{pair['win']}/{pair['tie']}/{pair['loss']}"
            )
        lines.append("| " + arm + " | " + " | ".join(cell) + " |")
    lines.append("")
    lines.append(
        "Win/tie/loss counts the row arm against the column arm. "
        "A win is a task the row arm solved and the column arm did not."
    )
    lines.append("")
    lines.append("## Cost, time, calls")
    lines.append("")
    lines.append(
        "| arm | calls | prompt / call | primary median | primary sum | billed median | billed sum | wall s |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        s = report["overall"][arm]
        lines.append(
            f"| {arm} | {s['calls_median']} | {s['prompt_per_call_median']:,} | "
            f"{s['cost_usd_median']:.4f} | {s['cost_usd_sum']:.2f} | "
            f"{s['cost_usd_billed_median']:.4f} | {s['cost_usd_billed_sum']:.2f} | "
            f"{s['wall_seconds_median']} |"
        )
    lines.append("")
    lines.append("## Budget stops and dropped nodes")
    lines.append("")
    lines.append(
        "A budget stop is an exit reason of `max_cost`, `cost_cap`, `budget`, or "
        "`LimitsExceeded`. The `[Budget]` note is appended to the request and is not "
        "stored on the graph. `budget_notes_stored` is the count of this task's calls "
        "whose saved text contains that note."
    )
    lines.append("")
    lines.append(
        "| arm | budget stops | exit reasons | tasks with drops | median dropped nodes when any | calls with drops | notes stored |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    for arm in ARMS:
        s = report["overall"][arm]
        exits = ", ".join(f"{k} {v}" for k, v in sorted(s["exit_reasons"].items(), key=lambda kv: -kv[1]))
        lines.append(
            f"| {arm} | {s['budget_stop_tasks']}/{s['n']} | {exits} | "
            f"{s['tasks_with_drops']} | {s['dropped_nodes_median_when_any']} | "
            f"{s['calls_with_drops']}/{s['calls_sum']} | {s['budget_notes_stored']} |"
        )
    lines.append("")
    lines.append("## By sequence third")
    lines.append("")
    lines.append(
        "A third is `min(2, (position-1)*3/n_tasks)` on the full repo sequence. "
        "The unscored task is then left out."
    )
    lines.append("")
    for third in THIRDS:
        lines.append(f"### {third}")
        lines.append("")
        lines.append(
            "| arm | resolved | calls | prompt / call | primary median | billed median | wall s | budget stops | tasks with drops |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for arm in ARMS:
            s = report["thirds"][third][arm]
            lines.append(
                f"| {arm} | {s['resolved']}/{s['n']} | {s['calls_median']} | "
                f"{s['prompt_per_call_median']:,} | {s['cost_usd_median']:.4f} | "
                f"{s['cost_usd_billed_median']:.4f} | {s['wall_seconds_median']} | "
                f"{s['budget_stop_tasks']}/{s['n']} | {s['tasks_with_drops']} |"
            )
        lines.append("")
    lines.append("## Recorded pair on these tasks")
    lines.append("")
    pair = report["recorded_pair"]
    lines.append(
        f"G-persist minus G-fresh on these scored tasks: {pair['diff_pp']:+.2f} pp "
        f"[{pair['ci95_pp'][0]:+.2f}, {pair['ci95_pp'][1]:+.2f}], "
        f"win/tie/loss {pair['win']}/{pair['tie']}/{pair['loss']}."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    tasks = load_tasks()
    scored = [t for t in tasks if t["scored"]]
    explore = collect_explore(tasks)
    recorded = collect_recorded({t["iid"] for t in tasks})
    rows = {**explore, **recorded}
    comparisons: dict[str, dict] = {}
    for arm in EXPLORE_ARMS:
        comparisons[arm] = {}
        for base in ("G-persist-control", "G-persist", "G-fresh"):
            if arm == base:
                continue
            comparisons[arm][base] = compare(rows[arm], rows[base], scored)
    report = {
        "label": LABEL,
        "ran": 56,
        "scored_n": 55,
        "unscored": sorted(UNSCORED),
        "unscored_grades": {
            arm: rows[arm]["astropy__astropy-7606"]["resolved"] for arm in ARMS
        },
        "seed": SEED,
        "bootstrap": N_BOOT,
        "overall": {arm: slice_summary(rows[arm], scored) for arm in ARMS},
        "thirds": {
            third: {
                arm: slice_summary(rows[arm], [t for t in scored if t["third"] == third])
                for arm in ARMS
            }
            for third in THIRDS
        },
        "comparisons": comparisons,
        "recorded_pair": compare(rows["G-persist"], rows["G-fresh"], scored),
        "gold_fail": {arm: rows[arm]["astropy__astropy-7606"] for arm in ARMS},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "explore_report.json").write_text(json.dumps(report, indent=2) + "\n")
    text = render(report)
    (OUT / "explore_report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
