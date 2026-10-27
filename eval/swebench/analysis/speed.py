#!/usr/bin/env python3
"""Secondary speed metrics (prereg §4) and the pooled T1+T4 summary.

Wall time for a Graphirm arm on T1-T3 is summary wall_seconds. Wall time
for a mini arm is the span of message timestamps in the trajectory. T4 uses
Harbor's agent_execution window, which covers the arms that never wrote a
summary. Provider wait is
the backoff sleep reconstructed from retries.log with the same delay the
agent uses. A missing log on a track that wrote logs means no retry.
Mini arms never wrote that log, and T4's Harbor agent did not set it.

Nothing here changes a run file. Outputs land in each track's analysis
directory, spliced into the existing report after a marker so a
hand-written section stays.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from metrics import primary_cost_from_calls  # noqa: E402
from run_arm import mini_costs  # noqa: E402
from t1_report import (  # noqa: E402
    N_BOOT,
    SEED,
    bootstrap_ratio_of_medians,
    grades as t1_grades,
    task_ids as t1_ids,
    win_tie_loss,
)
from t2_report import load_tasks as t2_tasks  # noqa: E402
from t2_report import resolved_of as t2_resolved  # noqa: E402
from t3_report import load_tasks as t3_tasks  # noqa: E402
from t4_report import filtered_grades, ledger as t4_ledger  # noqa: E402
from t4_report import original_grades, scored_ids  # noqa: E402

RUNS = {
    "t1": Path("/home/krs/graphirm-runs/t1"),
    "t2": Path("/home/krs/graphirm-runs/t2"),
    "t3": Path("/home/krs/graphirm-runs/t3"),
    "t4": Path("/home/krs/graphirm-runs/t4"),
}
GRAPH = {"G", "G-md", "G-b16", "G-b16-recency", "G-fresh", "G-persist"}
# Harbor's T4 agent does not set GRAPHIRM_RETRY_LOG, so a missing file
# there is not a recorded zero.
WAIT_TRACKS = {"t1", "t2", "t3"}
PAIRS = {
    "t1": [("G", "M"), ("G-md", "M"), ("G-b16", "M"), ("G-b16-recency", "M")],
    "t2": [
        ("G-persist", "G-fresh"),
        ("G-persist", "M-notes"),
        ("G-persist", "M"),
        ("G-fresh", "M"),
        ("M-notes", "M"),
    ],
    "t3": [
        ("G-persist", "G-fresh"),
        ("G-persist", "M-notes"),
        ("G-persist", "M"),
        ("G-fresh", "M"),
        ("M-notes", "M"),
    ],
    "t4": [("G", "M")],
}
ARMS = {
    "t1": ["M", "G", "G-md", "G-b16", "G-b16-recency"],
    "t2": ["M", "M-notes", "G-fresh", "G-persist"],
    "t3": ["M", "M-notes", "G-fresh", "G-persist"],
    "t4": ["M", "G"],
}
BEGIN = "<!-- speed-secondary -->"
END = "<!-- /speed-secondary -->"
POOL_BEGIN = "<!-- pooled-exploratory -->"
POOL_END = "<!-- /pooled-exploratory -->"


def backoff_seconds(text: str) -> float:
    """Sleep the agent would insert after each logged provider error.

    Records are split on a line that starts with ``Provider error``. Within
    one file they are one retry counter, because the log does not mark the
    next model call. The first record uses try_n = 1, matching
    ``provider_retry_delay`` after ``same_model_try`` has been incremented.
    """
    records: list[str] = []
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("Provider error") and buf:
            records.append("\n".join(buf))
            buf = [line]
        elif line.startswith("Provider error") or buf:
            buf.append(line)
    if buf:
        records.append("\n".join(buf))
    total_ms = 0
    for index, record in enumerate(records, start=1):
        base = 2000 if "429" in record else 1000
        shift = min(index, 4)
        total_ms += min(base * (1 << shift), 30_000)
    return total_ms / 1000.0


def graph_calls(path: Path) -> dict[str, dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT id, metadata FROM nodes WHERE node_type = 'interaction' "
            "AND json_extract(metadata, '$.usage_input') IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    return {node_id: json.loads(meta) for node_id, meta in rows}


def persist_calls(track: str) -> dict[str, list[dict]]:
    """Calls added on this task. A G-persist graph keeps every earlier task."""
    if track == "t2":
        tasks = t2_tasks()
    elif track == "t3":
        tasks = t3_tasks()
    else:
        return {}
    prior: dict[str, set[str]] = {}
    added: dict[str, list[dict]] = {}
    root = RUNS[track]
    for task in tasks:
        current = graph_calls(root / task["iid"] / "G-persist.graph.db")
        known = prior.get(task["repo"], set())
        if task.get("scored", True):
            added[task["iid"]] = [meta for node_id, meta in current.items() if node_id not in known]
        if current:
            prior[task["repo"]] = set(current)
    return added


def harbor_wall(track_root: Path, arm: str, iid: str) -> float | None:
    """Seconds from Harbor's agent_execution window on the last attempt.

    That window is the agent command. On the 100 T4 graphs that also wrote
    wall_seconds it is about a second longer, which is the shell around the
    process. Image pull sits in environment_setup and is not included.
    """
    attempts = sorted((track_root / "trials" / arm / iid).glob("attempt*"))
    if not attempts:
        return None
    results = sorted(attempts[-1].rglob("result.json"))
    if not results:
        return None
    window = json.loads(results[-1].read_text()).get("agent_execution") or {}
    start, end = window.get("started_at"), window.get("finished_at")
    if not start or not end:
        return None
    a = datetime.fromisoformat(start.replace("Z", "+00:00"))
    b = datetime.fromisoformat(end.replace("Z", "+00:00"))
    return (b - a).total_seconds()


def mini_wall(path: Path) -> float | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    stamps = []
    for msg in data.get("messages") or []:
        extra = msg.get("extra") or {}
        if isinstance(extra, dict) and isinstance(extra.get("timestamp"), (int, float)):
            stamps.append(float(extra["timestamp"]))
    if len(stamps) < 2:
        return None
    return max(stamps) - min(stamps)


def graph_wall(path: Path) -> float | None:
    if not path.is_file():
        return None
    summary = json.loads(path.read_text())
    if summary.get("wall_seconds") is None:
        return None
    return float(summary["wall_seconds"])


def wait_seconds(track: str, arm: str, path: Path) -> float | None:
    if arm not in GRAPH or track not in WAIT_TRACKS:
        return None
    if not path.is_file() or path.stat().st_size == 0:
        return 0.0
    return backoff_seconds(path.read_text(errors="replace"))


def finite_median(values: np.ndarray) -> float | None:
    kept = values[np.isfinite(values)]
    if len(kept) == 0:
        return None
    return float(np.median(kept))


def ratio_block(a: np.ndarray, b: np.ndarray, rng) -> dict | None:
    mask = np.isfinite(a) & np.isfinite(b) & (a >= 0) & (b > 0)
    if int(mask.sum()) < 2:
        return None
    point, lo, hi = bootstrap_ratio_of_medians(a[mask], b[mask], rng)
    return {
        "n": int(mask.sum()),
        "point": round(point, 3),
        "ci95": [round(lo, 3), round(hi, 3)],
    }


def _round(value: float | None, digits: int) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def arm_row(wall: np.ndarray, calls: np.ndarray, per_call: np.ndarray, wait: np.ndarray | None) -> dict:
    row = {
        "n_wall": int(np.isfinite(wall).sum()),
        "median_wall_s": _round(finite_median(wall), 1),
        "median_calls": _round(finite_median(calls), 1),
        "median_seconds_per_call": _round(finite_median(per_call), 2),
        "provider_wait_recorded": wait is not None,
    }
    if wait is not None:
        net = np.where(np.isfinite(wall), np.maximum(0.0, wall - np.nan_to_num(wait)), np.nan)
        row["median_wall_excluding_provider_wait_s"] = _round(finite_median(net), 1)
        row["median_provider_wait_s"] = _round(finite_median(wait), 2)
    return row


def load_track(track: str) -> tuple[list[str], dict[str, dict[str, bool]]]:
    if track == "t1":
        ids = t1_ids()
        return ids, t1_grades(ids)
    if track == "t2":
        tasks = [task for task in t2_tasks() if task["scored"]]
        ids = [task["iid"] for task in tasks]
        grades = {}
        root = RUNS[track]
        for iid in ids:
            raw = json.loads((root / iid / "grades.json").read_text())
            grades[iid] = {}
            for arm in ARMS[track]:
                ok, _src = t2_resolved(root / iid, iid, arm, raw)
                grades[iid][arm] = ok
        return ids, grades
    if track == "t3":
        ids = [task["iid"] for task in t3_tasks()]
        grades = {}
        for iid in ids:
            raw = json.loads((RUNS[track] / iid / "grades.json").read_text())
            grades[iid] = {arm: bool(raw[arm]) for arm in ARMS[track]}
        return ids, grades
    ids = scored_ids()
    rows = t4_ledger()
    return ids, filtered_grades(ids, original_grades(ids), rows)


def collect(track: str, ids: list[str]) -> dict[str, dict[str, np.ndarray]]:
    root = RUNS[track]
    added = persist_calls(track)
    out = {
        arm: {
            "wall": np.full(len(ids), np.nan),
            "calls": np.full(len(ids), np.nan),
            "per_call": np.full(len(ids), np.nan),
            "wait": np.full(len(ids), np.nan),
            "cost": np.full(len(ids), np.nan),
            "resolved": np.zeros(len(ids)),
            "wait_recorded": arm in GRAPH and track in WAIT_TRACKS,
        }
        for arm in ARMS[track]
    }
    for i, iid in enumerate(ids):
        task = root / iid
        for arm in ARMS[track]:
            if arm in GRAPH:
                if arm == "G-persist":
                    calls = added[iid]
                else:
                    calls = list(graph_calls(task / f"{arm}.graph.db").values())
                wall = graph_wall(task / f"{arm}.summary.json")
                cost = primary_cost_from_calls(calls) if calls else 0.0
                n_calls = float(len(calls))
            else:
                traj = task / f"{arm}.traj.json"
                wall = mini_wall(traj)
                stats = mini_costs(traj)
                n_calls = float(stats["calls"])
                cost = float(stats["cost_usd"])
            if track == "t4":
                # One clock for both arms, including the runs that never
                # wrote a summary or a second timestamp.
                wall = harbor_wall(root, arm, iid)
            out[arm]["wall"][i] = np.nan if wall is None else wall
            out[arm]["calls"][i] = n_calls
            out[arm]["cost"][i] = cost
            if wall is not None and n_calls > 0:
                out[arm]["per_call"][i] = wall / n_calls
            waited = wait_seconds(track, arm, task / f"{arm}.retries.log")
            if waited is not None:
                out[arm]["wait"][i] = waited
    return out


def analyse(track: str) -> dict:
    ids, grades = load_track(track)
    series = collect(track, ids)
    for arm in ARMS[track]:
        series[arm]["resolved"] = np.array([1 if grades[iid][arm] else 0 for iid in ids], dtype=float)
    arms = {}
    for arm in ARMS[track]:
        wait = series[arm]["wait"] if series[arm]["wait_recorded"] else None
        arms[arm] = arm_row(series[arm]["wall"], series[arm]["calls"], series[arm]["per_call"], wait)
    ratios = []
    for left, right in PAIRS[track]:
        wall_ratio = ratio_block(
            series[left]["wall"], series[right]["wall"], np.random.default_rng(SEED)
        )
        item = {"pair": f"{left}/{right}", "wall_ratio": wall_ratio}
        if series[left]["wait_recorded"] and series[right]["wait_recorded"]:
            net_l = np.maximum(0.0, series[left]["wall"] - series[left]["wait"])
            net_r = np.maximum(0.0, series[right]["wall"] - series[right]["wait"])
            item["wall_ratio_excluding_provider_wait"] = ratio_block(
                net_l, net_r, np.random.default_rng(SEED)
            )
        else:
            item["wall_ratio_excluding_provider_wait"] = None
        both = (
            (series[left]["resolved"] == 1)
            & (series[right]["resolved"] == 1)
            & np.isfinite(series[left]["wall"])
            & np.isfinite(series[right]["wall"])
        )
        item["both_resolved"] = {
            "n": int(both.sum()),
            left: _round(finite_median(series[left]["wall"][both]), 1),
            right: _round(finite_median(series[right]["wall"][both]), 1),
        }
        ratios.append(item)
    lacking = [
        arm
        for arm in ARMS[track]
        if not series[arm]["wait_recorded"]
    ]
    return {
        "track": track,
        "label": "secondary, prereg §4",
        "n": len(ids),
        "seed": SEED,
        "bootstrap": N_BOOT,
        "wall_definition": (
            "T4 uses Harbor agent_execution on the last attempt. "
            "T1-T3 Graphirm uses summary wall_seconds. "
            "T1-T3 mini uses the span of trajectory timestamps."
            if track == "t4"
            else "Graphirm wall_seconds on the summary. Mini is the span of trajectory timestamps."
        ),
        "provider_wait": (
            "backoff reconstructed from retries.log; missing file on T1-T3 graph arms means no retry. "
            "The failed call's own duration is not in the log, so it stays inside wall time."
        ),
        "arms": arms,
        "ratios": ratios,
        "provider_wait_not_recorded": lacking,
        "missing_wall": {
            arm: [iid for iid, value in zip(ids, series[arm]["wall"]) if not np.isfinite(value)]
            for arm in ARMS[track]
            if not np.isfinite(series[arm]["wall"]).all()
        },
        "_series": series,
        "_ids": ids,
    }


def render_track(report: dict) -> str:
    lines = [
        "## Speed (secondary, prereg §4)",
        "",
        report["wall_definition"]
        + " The ratio is the median of the first arm over the median of the second, paired by task, 10,000 resamples, seed 20261022.",
        "",
        "| arm | n | median wall s | median calls | median s/call | provider wait recorded | median wall excluding wait s |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm, row in report["arms"].items():
        lines.append(
            f"| {arm} | {row['n_wall']} | {row['median_wall_s']} | {row['median_calls']} | "
            f"{row['median_seconds_per_call']} | {row['provider_wait_recorded']} | "
            f"{row.get('median_wall_excluding_provider_wait_s', '—')} |"
        )
    lines += ["", "| pair | n | wall ratio | 95% CI | ratio excluding provider wait | both resolved n | median wall on those |", "|---|---|---|---|---|---|---|"]
    for item in report["ratios"]:
        wall = item["wall_ratio"]
        net = item["wall_ratio_excluding_provider_wait"]
        both = item["both_resolved"]
        left, right = item["pair"].split("/")
        solved = f"{left} {both[left]}, {right} {both[right]}"
        lines.append(
            f"| {item['pair']} | {wall['n'] if wall else 0} | "
            f"{wall['point'] if wall else '—'} | {wall['ci95'] if wall else '—'} | "
            f"{(str(net['point']) + ' ' + str(net['ci95'])) if net else 'not recorded'} | "
            f"{both['n']} | {solved} |"
        )
    lacking = ", ".join(report["provider_wait_not_recorded"]) or "none"
    missing = report.get("missing_wall") or {}
    missing_line = (
        "Tasks with no wall clock: "
        + "; ".join(f"{arm} ({len(ids)})" for arm, ids in missing.items())
        + "."
        if missing
        else "Every scored task has a wall clock."
    )
    lines += [
        "",
        missing_line,
        f"Provider-wait backoff is not recorded for: {lacking}.",
        *(
            ["G-persist calls are the interaction nodes that were not in the previous task's graph."]
            if "G-persist" in report["arms"]
            else []
        ),
        "On T1, T2 and T3 the Graphirm arms wrote `retries.log` when a call was retried. A task with no file is a recorded zero, not a missing instrument. "
        "The surviving T1 graph arms all have that field. Mini never wrote it, and T4's Harbor launch did not set `GRAPHIRM_RETRY_LOG`.",
        "",
    ]
    return "\n".join(lines)


def splice(path: Path, begin: str, end: str, body: str) -> None:
    block = f"{begin}\n{body.rstrip()}\n{end}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        path.write_text(block)
        return
    text = path.read_text()
    if begin in text and end in text:
        pre, rest = text.split(begin, 1)
        _old, post = rest.split(end, 1)
        path.write_text(pre.rstrip() + "\n\n" + block + post.lstrip("\n"))
        return
    if text and not text.endswith("\n"):
        text += "\n"
    path.write_text(text + "\n" + block)


def public_report(report: dict) -> dict:
    clean = dict(report)
    clean.pop("_series", None)
    clean.pop("_ids", None)
    return clean


def pooled(t1: dict, t4: dict, rng) -> dict:
    g_res = np.concatenate([t1["_series"]["G"]["resolved"], t4["_series"]["G"]["resolved"]])
    m_res = np.concatenate([t1["_series"]["M"]["resolved"], t4["_series"]["M"]["resolved"]])
    g_cost = np.concatenate([t1["_series"]["G"]["cost"], t4["_series"]["G"]["cost"]])
    m_cost = np.concatenate([t1["_series"]["M"]["cost"], t4["_series"]["M"]["cost"]])
    g_wall = np.concatenate([t1["_series"]["G"]["wall"], t4["_series"]["G"]["wall"]])
    m_wall = np.concatenate([t1["_series"]["M"]["wall"], t4["_series"]["M"]["wall"]])
    strata = np.array(["t1"] * t1["n"] + ["t4"] * t4["n"])
    diff, lo, hi = stratified_diff(g_res, m_res, strata, rng)
    return {
        "label": "exploratory, not pre-registered",
        "tracks": {"t1": t1["n"], "t4": t4["n"]},
        "n": int(len(g_res)),
        "seed": SEED,
        "bootstrap": N_BOOT,
        "method": "stratified paired bootstrap, resampled inside each track",
        "resolved_G": int(g_res.sum()),
        "resolved_M": int(m_res.sum()),
        "diff_pp": round(diff * 100, 2),
        "ci95_pp": [round(lo * 100, 2), round(hi * 100, 2)],
        **win_tie_loss(g_res, m_res),
        "cost_ratio": stratified_ratio(g_cost, m_cost, strata, rng),
        "wall_ratio": stratified_ratio(g_wall, m_wall, strata, rng),
    }


def stratified_diff(a: np.ndarray, b: np.ndarray, strata: np.ndarray, rng) -> tuple[float, float, float]:
    groups = {label: np.where(strata == label)[0] for label in dict.fromkeys(strata.tolist())}
    point = float(a.mean() - b.mean())
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = np.concatenate([rng.choice(loc, size=len(loc), replace=True) for loc in groups.values()])
        boots[i] = a[idx].mean() - b[idx].mean()
    return point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def stratified_ratio(a: np.ndarray, b: np.ndarray, strata: np.ndarray, rng) -> dict | None:
    mask = np.isfinite(a) & np.isfinite(b) & (b > 0)
    if int(mask.sum()) < 2:
        return None
    a, b, strata = a[mask], b[mask], strata[mask]
    groups = {label: np.where(strata == label)[0] for label in dict.fromkeys(strata.tolist())}
    point = float(np.median(a) / np.median(b))
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = np.concatenate([rng.choice(loc, size=len(loc), replace=True) for loc in groups.values()])
        boots[i] = np.median(a[idx]) / np.median(b[idx])
    return {
        "n": int(len(a)),
        "point": round(point, 3),
        "ci95": [round(float(np.quantile(boots, 0.025)), 3), round(float(np.quantile(boots, 0.975)), 3)],
    }


def render_pooled(report: dict) -> str:
    cost = report["cost_ratio"]
    wall = report["wall_ratio"]
    return "\n".join(
        [
            "## Pooled T1 and T4 (exploratory, not pre-registered)",
            "",
            "T1 (n=210) and T4 after the launch rerun (n=106), G versus M. "
            "The interval is a stratified paired bootstrap: each draw resamples tasks inside a track, then pools the tracks. "
            "10,000 resamples, seed 20261022.",
            "",
            f"- Resolved: G {report['resolved_G']}, M {report['resolved_M']} of {report['n']}.",
            f"- Difference {report['diff_pp']:+.2f} pp, 95% CI [{report['ci95_pp'][0]:+.2f}, {report['ci95_pp'][1]:+.2f}] pp.",
            f"- Win/tie/loss {report['win']}/{report['tie']}/{report['loss']}, discordant {report['discordant']}.",
            f"- Median primary cost G/M {cost['point']}, 95% CI {cost['ci95']}.",
            f"- Median wall time G/M {wall['point']}, 95% CI {wall['ci95']}.",
            "",
        ]
    )


def main() -> None:
    # The delay formula is the one in workflow.rs. One 429 record sleeps 4s.
    if abs(backoff_seconds("Provider error: OpenRouter 429\n") - 4.0) > 1e-9:
        raise SystemExit("backoff formula drifted from provider_retry_delay")
    reports = {}
    for track in ("t1", "t2", "t3", "t4"):
        print(f"TRACK {track}", flush=True)
        reports[track] = analyse(track)
        out = RUNS[track] / "analysis"
        out.mkdir(parents=True, exist_ok=True)
        (out / "speed.json").write_text(json.dumps(public_report(reports[track]), indent=2) + "\n")
        splice(out / f"{track}_report.md", BEGIN, END, render_track(reports[track]))
        print(render_track(reports[track]), flush=True)
    # A fresh generator so the pooled interval does not depend on how many
    # speed ratios were drawn first. Same seed, documented in the report.
    pool = pooled(reports["t1"], reports["t4"], np.random.default_rng(SEED))
    for track in ("t1", "t4"):
        out = RUNS[track] / "analysis"
        (out / "pooled_t1_t4.json").write_text(json.dumps(pool, indent=2) + "\n")
        splice(out / f"{track}_report.md", POOL_BEGIN, POOL_END, render_pooled(pool))
    print(render_pooled(pool), flush=True)


if __name__ == "__main__":
    main()
