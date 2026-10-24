#!/usr/bin/env python3
"""T2 memory track (SWE-Bench-CL, 273 tasks) — pre-registered H5 analysis.

Scored n = 270. django__django-10097, sphinx-doc__sphinx-9711 and
astropy__astropy-7606 ran in sequence and are not scored (gold patch fails).
Position thirds use the full sequence length, then those three are dropped.

Reads /home/krs/graphirm-runs/t2, writes /home/krs/graphirm-runs/t2/analysis.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from metrics import cache_hit, primary_cost_from_calls, recorded_billed, tokens  # noqa: E402
from run_arm import mini_costs  # noqa: E402
from t1_report import classify_retry as _classify_retry  # noqa: E402
from t1_report import paired_bootstrap_diff, win_tie_loss  # noqa: E402

RUNS = Path("/home/krs/graphirm-runs/t2")
OUT = RUNS / "analysis"
LOCK = HERE.parent / "instances.lock.json"
ARROW = Path(
    "/home/krs/.cache/huggingface/datasets/princeton-nlp___swe-bench_verified/"
    "default/0.0.0/c104f840cc67f8b6eec6f759ebc8b2693d585d4a/swe-bench_verified-test.arrow"
)
ARMS = ["M", "M-notes", "G-fresh", "G-persist"]
MINI = {"M", "M-notes"}
GRAPH = {"G-fresh", "G-persist"}
PAIRS = [("G-persist", "G-fresh"), ("G-persist", "M-notes"), ("M-notes", "M")]
# Prereg H5 names the first two. M-notes − M is the same extra pair the T3 report used.
H5_PAIRS = {("G-persist", "G-fresh"), ("G-persist", "M-notes")}
METRICS = ["resolved", "calls", "tokens", "cost"]
UNSCORED = {"django__django-10097", "sphinx-doc__sphinx-9711", "astropy__astropy-7606"}
N_BOOT = 10_000
SEED = 20261022
THIRDS = ["first", "middle", "last"]
CAP_EXITS = {"max_cost", "cost_cap", "budget", "LimitsExceeded"}
BILLED_CAP_ROWS = 228  # journal 2026-10-22 11:15, before any per-arm result


def classify_retry(error: str) -> str:
    return _classify_retry(error)


def load_tasks() -> list[dict]:
    lock = json.loads(LOCK.read_text())
    sequences = lock["memory_tracks"]["swe_bench_cl"]["sequences"]
    cal = {row["instance_id"] for row in lock["memory_tracks"]["swe_bench_cl"]["calibration_overlap"]}
    tasks = []
    for seq in sequences:
        n = seq["n_tasks"]
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
                    "calibration": iid in cal,
                    "difficulty": row.get("difficulty"),
                }
            )
    if len(tasks) != 273:
        raise SystemExit(f"expected 273 tasks, found {len(tasks)}")
    if sum(1 for t in tasks if t["scored"]) != 270:
        raise SystemExit("expected 270 scored tasks")
    if len(cal) != 14:
        raise SystemExit(f"expected 14 calibration ids, found {len(cal)}")
    return tasks


def ledger_rows() -> list[dict]:
    return [json.loads(line) for line in (RUNS / "arms.jsonl").read_text().splitlines() if line.strip()]


def gold_sizes(ids: set[str]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    with pa.memory_map(str(ARROW), "r") as source:
        table = pa.ipc.open_stream(source).read_all().select(["instance_id", "patch"])
        for row in table.to_pylist():
            iid = row["instance_id"]
            if iid not in ids:
                continue
            count = 0
            for line in (row["patch"] or "").splitlines():
                if (line.startswith("+") and not line.startswith("+++")) or (
                    line.startswith("-") and not line.startswith("---")
                ):
                    count += 1
            sizes[iid] = count
    missing = ids - set(sizes)
    if missing:
        raise SystemExit(f"gold patch missing for {sorted(missing)[:5]}")
    return sizes


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
    return {i: json.loads(meta) for i, meta in rows}


def mini_calls(traj: Path) -> list[dict]:
    if not traj.is_file():
        return []
    data = json.loads(traj.read_text())
    calls = []
    for msg in data.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        usage = ((msg.get("extra") or {}).get("response") or {}).get("usage") or {}
        if not usage:
            continue
        calls.append(
            {
                "usage_input": int(usage.get("prompt_tokens") or 0),
                "usage_output": int(usage.get("completion_tokens") or 0),
                "usage_cache_read": int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
                "context_dropped_nodes": 0,
                "context_dropped_tokens": 0,
            }
        )
    return calls


def call_stats(calls: list[dict]) -> dict:
    if not calls:
        return {
            "calls": None,
            "tokens": None,
            "prompt_median": None,
            "cache_hit": None,
            "dropped_nodes": 0,
            "dropped_tokens": 0,
            "max_prompt": None,
        }
    prompts = [int(c.get("usage_input") or 0) for c in calls]
    outputs = [int(c.get("usage_output") or 0) for c in calls]
    cached = [int(c.get("usage_cache_read") or 0) for c in calls]
    inp = sum(prompts)
    return {
        "calls": len(calls),
        "tokens": tokens(inp, sum(outputs)),
        "prompt_median": float(np.median(prompts)),
        "cache_hit": cache_hit(inp, sum(cached)),
        "dropped_nodes": int(sum(int(c.get("context_dropped_nodes") or 0) for c in calls)),
        "dropped_tokens": int(sum(int(c.get("context_dropped_tokens") or 0) for c in calls)),
        "max_prompt": int(max(prompts)),
    }


def cluster_bootstrap_diff(a: np.ndarray, b: np.ndarray, clusters: np.ndarray, rng, n_boot=N_BOOT):
    labels = sorted(set(clusters.tolist()))
    members = [np.where(clusters == lab)[0] for lab in labels]
    k = len(labels)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        draw = rng.integers(0, k, size=k)
        idx = np.concatenate([members[j] for j in draw])
        diffs[i] = a[idx].mean() - b[idx].mean()
    return float(a.mean() - b.mean()), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def summarize_pair(a: np.ndarray, b: np.ndarray, clusters: np.ndarray, rng, binary: bool) -> dict:
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b, clusters = a[mask], b[mask], clusters[mask]
    out: dict = {"n": int(mask.sum())}
    if out["n"] == 0:
        return out
    d, lo, hi, _ = paired_bootstrap_diff(a, b, rng)
    cd, clo, chi = cluster_bootstrap_diff(a, b, clusters, rng)
    scale = 100 if binary else 1
    out["diff"] = round(d * scale, 4)
    out["task_ci95"] = [round(lo * scale, 4), round(hi * scale, 4)]
    out["cluster_ci95"] = [round(clo * scale, 4), round(chi * scale, 4)]
    out["clusters"] = len(set(clusters.tolist()))
    if binary:
        out.update(win_tie_loss(a.astype(int), b.astype(int)))
    else:
        mb = float(np.median(b))
        out["median_a"] = round(float(np.median(a)), 6)
        out["median_b"] = round(mb, 6)
        out["ratio_of_medians"] = round(float(np.median(a)) / mb, 4) if mb else None
    return out


def read_summary(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def resolved_of(out: Path, iid: str, arm: str, grades: dict) -> tuple[bool, str]:
    """grades.json is the record. One matplotlib file lost M-notes when a later
    write replaced it; that boolean is still in the harness report."""
    if arm in grades and grades[arm] is not None:
        return bool(grades[arm]), "grades.json"
    reports = sorted((out / "grading" / arm).glob("attempt*/report.json"))
    if not reports:
        raise SystemExit(f"{iid} {arm} has no grade")
    data = json.loads(reports[-1].read_text())
    inner = data.get(iid) or next(iter(data.values()))
    return bool(inner["resolved"]), str(reports[-1].relative_to(RUNS))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    all_tasks = load_tasks()
    tasks = [t for t in all_tasks if t["scored"]]
    ids = [t["iid"] for t in tasks]
    n = len(ids)
    index = {iid: i for i, iid in enumerate(ids)}
    rows = ledger_rows()
    latest: dict[str, dict] = {}
    by_key: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        latest[row["instance_id"]] = row
        by_key[row["instance_id"]].append(row)

    sizes = gold_sizes(set(ids))
    metric = {m: {arm: np.full(n, np.nan) for arm in ARMS} for m in METRICS}
    prompt_med = {arm: np.full(n, np.nan) for arm in ARMS}
    cache = {arm: np.full(n, np.nan) for arm in ARMS}
    exits = {arm: Counter() for arm in ARMS}
    per_task = []
    prior_ids: dict[str, set[str]] = {}
    notes_series = []
    persist_growth = []
    grade_recovered = []

    for t in all_tasks:
        iid = t["iid"]
        out = RUNS / iid
        if not t["scored"]:
            # Keep the persisted graph's node ids so the next scored task
            # does not count this task's calls as its own.
            gpath = out / "G-persist.graph.db"
            if gpath.is_file():
                prior_ids[t["repo"]] = set(graph_calls(gpath))
            continue
        i = index[iid]
        grades = json.loads((out / "grades.json").read_text())
        rec = {**t, "arms": {}}
        for arm in ARMS:
            resolved, grade_src = resolved_of(out, iid, arm, grades)
            if grade_src != "grades.json":
                grade_recovered.append({"task": iid, "arm": arm, "resolved": resolved, "source": grade_src})
            summary = read_summary(out / f"{arm}.summary.json")
            if arm in MINI:
                calls = mini_calls(out / f"{arm}.traj.json")
                costs = mini_costs(out / f"{arm}.traj.json")
                cost = float(costs["cost_usd"])
                billed = float(costs["cost_usd_billed"])
                exit_reason = summary.get("exit_status") or summary.get("exit_reason")
            else:
                current = graph_calls(out / f"{arm}.graph.db")
                if arm == "G-persist":
                    known = prior_ids.get(t["repo"], set())
                    calls = [md for nid, md in current.items() if nid not in known]
                    prior_ids[t["repo"]] = set(current)
                else:
                    calls = list(current.values())
                cost = primary_cost_from_calls(calls)
                billed = recorded_billed(summary)
                exit_reason = summary.get("exit_reason") or summary.get("exit_status")
            stats = call_stats(calls)
            metric["resolved"][arm][i] = 1.0 if resolved else 0.0
            metric["calls"][arm][i] = np.nan if stats["calls"] is None else stats["calls"]
            metric["tokens"][arm][i] = np.nan if stats["tokens"] is None else stats["tokens"]
            metric["cost"][arm][i] = cost
            if stats["prompt_median"] is not None:
                prompt_med[arm][i] = stats["prompt_median"]
            if stats["cache_hit"] is not None:
                cache[arm][i] = stats["cache_hit"]
            exits[arm][exit_reason or "missing"] += 1
            rec["arms"][arm] = {
                "resolved": resolved,
                "calls": stats["calls"],
                "tokens": stats["tokens"],
                "prompt_median": stats["prompt_median"],
                "cache_hit": None if stats["cache_hit"] is None else round(stats["cache_hit"], 4),
                "cost_usd": round(cost, 6),
                "cost_usd_billed": round(billed, 6),
                "exit_reason": exit_reason,
                "dropped_nodes": stats["dropped_nodes"],
                "dropped_tokens": stats["dropped_tokens"],
                "max_prompt": stats["max_prompt"],
                "graph_bytes": (out / f"{arm}.graph.db").stat().st_size if (out / f"{arm}.graph.db").is_file() else None,
            }
        carry_path = out / "M-notes.carry.json"
        carry = json.loads(carry_path.read_text()) if carry_path.is_file() else {}
        summary_notes = read_summary(out / "M-notes.summary.json")
        rec["notes"] = {
            "read": bool(summary_notes.get("notes_read")),
            "wrote": bool(summary_notes.get("notes_wrote")),
            "tokens_before": carry.get("tokens_before"),
            "tokens_after": carry.get("tokens_after"),
            "truncated": carry.get("truncated"),
            "notes_tokens": summary_notes.get("notes_tokens"),
        }
        notes_series.append({"repo": t["repo"], "position": t["position"], **rec["notes"]})
        gp = rec["arms"]["G-persist"]
        persist_growth.append(
            {
                "iid": iid,
                "repo": t["repo"],
                "position": t["position"],
                "third": t["third"],
                "graph_bytes": gp["graph_bytes"],
                "prompt_median": gp["prompt_median"],
                "max_prompt": gp["max_prompt"],
                "dropped_nodes": gp["dropped_nodes"],
                "dropped_tokens": gp["dropped_tokens"],
                "calls": gp["calls"],
            }
        )
        per_task.append(rec)

    clusters = np.array([t["repo"] for t in tasks])
    thirds = np.array([t["third"] for t in tasks])
    cal_mask = np.array([t["calibration"] for t in tasks])
    report: dict = {
        "n_ran": 273,
        "n": n,
        "unscored": sorted(UNSCORED),
        "arms": ARMS,
        "seed": SEED,
        "bootstrap": N_BOOT,
        "repos": {repo: int((clusters == repo).sum()) for repo in sorted(set(clusters.tolist()))},
        "thirds": {k: int((thirds == k).sum()) for k in THIRDS},
        "third_rule": "third = min(2, (position-1)*3/n_tasks) on the full sequence, then the 3 unscored tasks are dropped",
        "cost_note": (
            "Primary cost is the frozen formula (uncached × 4.2e-8 + cache read × 9e-9 + output × 6e-7). "
            "Mini is recomputed from the trajectory. Graphirm is summary.cost_usd. "
            "Billed cost is OpenRouter usage.cost. Tokens are prompt + completion on this task's calls only. "
            "G-persist calls are interaction nodes that were not in the previous task's graph."
        ),
    }

    def arm_table(sel: np.ndarray | None = None) -> dict:
        table = {}
        for arm in ARMS:
            r = metric["resolved"][arm] if sel is None else metric["resolved"][arm][sel]
            c = metric["calls"][arm] if sel is None else metric["calls"][arm][sel]
            tk = metric["tokens"][arm] if sel is None else metric["tokens"][arm][sel]
            cost = metric["cost"][arm] if sel is None else metric["cost"][arm][sel]
            billed = np.array([per_task[j]["arms"][arm]["cost_usd_billed"] for j in range(n)], dtype=float)
            if sel is not None:
                billed = billed[sel]
            pm = prompt_med[arm] if sel is None else prompt_med[arm][sel]
            ch = cache[arm] if sel is None else cache[arm][sel]
            table[arm] = {
                "resolved": int(np.nansum(r)),
                "n": int(len(r)),
                "resolve_rate": round(float(np.nanmean(r)), 4),
                "calls_median": float(np.nanmedian(c)),
                "tokens_median": int(np.nanmedian(tk)),
                "prompt_per_call_median": None if np.all(np.isnan(pm)) else int(np.nanmedian(pm)),
                "cache_hit_median": None if np.all(np.isnan(ch)) else round(float(np.nanmedian(ch)), 4),
                "cost_median_usd": round(float(np.nanmedian(cost)), 6),
                "cost_total_usd": round(float(np.nansum(cost)), 4),
                "cost_billed_median_usd": round(float(np.nanmedian(billed)), 6),
                "cost_billed_total_usd": round(float(np.nansum(billed)), 4),
                "exit_reasons": dict(exits[arm]) if sel is None else None,
            }
        return table

    report["table"] = arm_table()
    report["grade_recovered"] = grade_recovered
    report["ledger_primary_total"] = round(sum(float(r.get("cost_usd") or 0) for r in rows), 2)
    report["ledger_billed_total"] = round(sum(float(r.get("cost_usd_billed") or 0) for r in rows), 2)

    def h5_block(sel: np.ndarray | None, rng_local) -> dict:
        h5: dict = {}
        for a, b in PAIRS:
            key = f"{a} - {b}"
            h5[key] = {"registered": (a, b) in H5_PAIRS}
            for m in METRICS:
                binary = m == "resolved"
                aa = metric[m][a] if sel is None else metric[m][a][sel]
                bb = metric[m][b] if sel is None else metric[m][b][sel]
                cc = clusters if sel is None else clusters[sel]
                entry = {"overall": summarize_pair(aa, bb, cc, rng_local, binary)}
                th = thirds if sel is None else thirds[sel]
                for third in THIRDS:
                    mask = th == third
                    entry[third] = summarize_pair(aa[mask], bb[mask], cc[mask], rng_local, binary)
                h5[key][m] = entry
        return h5

    report["H5"] = h5_block(None, rng)
    report["cluster_ci_note"] = (
        "Repo-cluster bootstrap draws 8 sequences with replacement and keeps every scored task "
        "of each drawn sequence. Tasks inside a sequence are never split, so the interval is "
        "between-sequence uncertainty. With 8 sequences it is wide, and django or sympy can be "
        "drawn more than once. The task-level interval treats tasks as independent and is too "
        "narrow when tasks in one sequence fail together. Read them side by side."
    )

    # ---- retry sensitivity
    provider_kinds = {"config", "harness", "provider_429_or_overload", "provider_connection", "llm_call_timeout"}
    retry_types = {arm: Counter() for arm in ARMS}
    first_attempt = {arm: np.zeros(n, dtype=int) for arm in ARMS}
    wall_only = {arm: np.zeros(n, dtype=int) for arm in ARMS}
    flips_first, flips_wall = [], []
    for i, iid in enumerate(ids):
        final = {arm: bool(metric["resolved"][arm][i]) for arm in ARMS}
        for arm in ARMS:
            attempts = by_key.get(f"{iid}::{arm}", [])
            if not attempts:
                continue
            real = [r for r in attempts if classify_retry(r.get("error") or "") not in ("config", "harness")]
            for r in real[:-1]:
                if r.get("error"):
                    retry_types[arm][classify_retry(r["error"])] += 1
            first = real[0] if real else attempts[0]
            if first.get("error") and not first.get("request_caused"):
                first_attempt[arm][i] = 0
                if final[arm]:
                    flips_first.append({"task": iid, "arm": arm, "first_error": classify_retry(first["error"])})
            else:
                first_attempt[arm][i] = 1 if final[arm] else 0
            agent_attempts = [
                r for r in attempts
                if classify_retry(r.get("error") or "") not in provider_kinds or not r.get("error")
            ]
            firstw = agent_attempts[0] if agent_attempts else attempts[-1]
            if firstw.get("error") and not firstw.get("request_caused"):
                wall_only[arm][i] = 0
                if final[arm]:
                    flips_wall.append({"task": iid, "arm": arm, "first_error": classify_retry(firstw["error"])})
            else:
                wall_only[arm][i] = 1 if final[arm] else 0
    rng2 = np.random.default_rng(SEED + 1)
    sens: dict = {
        "retries_by_type": {arm: dict(retry_types[arm]) for arm in ARMS},
        "all_retries": {arm: int(np.nansum(metric["resolved"][arm])) for arm in ARMS},
        "first_attempt_table": {arm: int(first_attempt[arm].sum()) for arm in ARMS},
        "wall_clock_table": {arm: int(wall_only[arm].sum()) for arm in ARMS},
        "flips_first_attempt": flips_first,
        "flips_wall_clock": flips_wall,
    }
    for label, arrs in (("first_attempt", first_attempt), ("wall_clock", wall_only)):
        sens[f"{label}_H5"] = {}
        for a, b in PAIRS:
            d, lo, hi, _ = paired_bootstrap_diff(arrs[a].astype(float), arrs[b].astype(float), rng2)
            sens[f"{label}_H5"][f"{a} - {b}"] = {
                "diff_pp": round(d * 100, 2),
                "task_ci95_pp": [round(lo * 100, 2), round(hi * 100, 2)],
                **win_tie_loss(arrs[a], arrs[b]),
                "registered": (a, b) in H5_PAIRS,
            }
    base = report["H5"]
    changes = []
    for pair_name, metrics in base.items():
        overall = metrics["resolved"]["overall"]
        base_lo = overall["task_ci95"][0]
        for label in ("first_attempt", "wall_clock"):
            alt = sens[f"{label}_H5"][pair_name]
            alt_lo = alt["task_ci95_pp"][0]
            # A conclusion change: the task-level CI crosses or leaves zero, or the sign of the point estimate flips.
            base_cross = base_lo <= 0 <= overall["task_ci95"][1]
            alt_cross = alt_lo <= 0 <= alt["task_ci95_pp"][1]
            if (overall["diff"] > 0) != (alt["diff_pp"] > 0) or base_cross != alt_cross:
                changes.append({"pair": pair_name, "rule": label, "base": overall["diff"], "alt": alt["diff_pp"]})
    sens["conclusion_changes"] = changes
    report["sensitivity"] = sens

    # ---- cost caps
    cap_rows = []
    for row in rows:
        reason = row.get("exit_reason")
        if row.get("cap_hit") or reason in CAP_EXITS:
            cap_rows.append(
                {
                    "task": row.get("task"),
                    "arm": row.get("arm"),
                    "exit_reason": reason,
                    "cap_hit": row.get("cap_hit"),
                    "cost_usd": row.get("cost_usd"),
                    "cost_usd_billed": row.get("cost_usd_billed"),
                }
            )
    early = rows[:BILLED_CAP_ROWS]
    early_primary = [float(r.get("cost_usd") or 0) for r in early]
    early_billed = [float(r.get("cost_usd_billed") or 0) for r in early]
    report["cost_caps"] = {
        "exits": cap_rows,
        "early_rows": BILLED_CAP_ROWS,
        "early_note": (
            "Journal 2026-10-22 11:15: the first 228 arm-runs had mini's $0.50 cap applied to "
            "OpenRouter's bill. The decision was logged at 228 runs, before any per-arm result."
        ),
        "early_max_ledger_cost_usd": round(max(early_primary), 4) if early_primary else None,
        "early_max_billed": round(max(early_billed), 4) if early_billed else None,
        "early_near_half_cap": bool(early_primary and (max(early_primary) >= 0.25 or max(early_billed) >= 0.25)),
    }

    # ---- notes
    read_n = sum(1 for row in notes_series if row["read"])
    wrote_n = sum(1 for row in notes_series if row["wrote"])
    by_repo_notes = {}
    for repo in report["repos"]:
        series = [row for row in notes_series if row["repo"] == repo]
        series.sort(key=lambda row: row["position"])
        after = [row["tokens_after"] for row in series if row["tokens_after"] is not None]
        before = [row["tokens_before"] for row in series if row["tokens_before"] is not None]
        by_repo_notes[repo] = {
            "n": len(series),
            "read": sum(1 for row in series if row["read"]),
            "wrote": sum(1 for row in series if row["wrote"]),
            "truncated": sum(1 for row in series if row["truncated"]),
            "tokens_after_first": after[0] if after else None,
            "tokens_after_last": after[-1] if after else None,
            "tokens_before_max": max(before) if before else None,
            "tokens_after_max": max(after) if after else None,
        }
    report["notes"] = {
        "scored_tasks": n,
        "read": read_n,
        "wrote": wrote_n,
        "by_repo": by_repo_notes,
        "cap_tokens": 4000,
    }

    # ---- G-persist growth
    growth = {}
    for third in THIRDS:
        chunk = [row for row in persist_growth if row["third"] == third]
        prompts = [row["prompt_median"] for row in chunk if row["prompt_median"] is not None]
        graphs = [row["graph_bytes"] for row in chunk if row["graph_bytes"]]
        drops = [row for row in chunk if (row["dropped_nodes"] or 0) > 0]
        growth[third] = {
            "n": len(chunk),
            "graph_bytes_median": int(np.median(graphs)) if graphs else None,
            "prompt_per_call_median": int(np.median(prompts)) if prompts else None,
            "tasks_with_drops": len(drops),
            "dropped_nodes_median_when_any": int(np.median([row["dropped_nodes"] for row in drops])) if drops else 0,
            "max_prompt": max((row["max_prompt"] or 0) for row in chunk) if chunk else None,
        }
    bound = [row for row in persist_growth if (row["dropped_nodes"] or 0) > 0]
    report["g_persist"] = {
        "by_third": growth,
        "tasks_with_any_drop": len(bound),
        "max_prompt_any_call": max((row["max_prompt"] or 0) for row in persist_growth),
        "max_graph_bytes": max((row["graph_bytes"] or 0) for row in persist_growth),
    }

    # ---- calibration overlap, exploratory
    report["calibration"] = {
        "n": int(cal_mask.sum()),
        "with_all_270": {
            pair: report["H5"][pair]["resolved"]["overall"] for pair in report["H5"]
        },
        "without_14": h5_block(~cal_mask, np.random.default_rng(SEED + 3)),
    }
    # keep only resolve overall in the without block's stored comparison to limit size? keep full, it's small enough
    # The full h5_block is large but fine.

    # ---- exploratory
    by_repo = {}
    for repo in report["repos"]:
        sel = clusters == repo
        by_repo[repo] = {
            "n": int(sel.sum()),
            **{arm: int(np.nansum(metric["resolved"][arm][sel])) for arm in ARMS},
        }
    gold = np.array([sizes[iid] for iid in ids])
    t1, t2 = np.quantile(gold, [1 / 3, 2 / 3])
    bands = {"small": gold <= t1, "medium": (gold > t1) & (gold <= t2), "large": gold > t2}
    by_size = {
        "cutpoints_lines": [int(t1), int(t2)],
        **{
            band: {
                "n": int(mask.sum()),
                **{arm: int(np.nansum(metric["resolved"][arm][mask])) for arm in ARMS},
            }
            for band, mask in bands.items()
        },
    }
    report["exploratory"] = {"by_repo": by_repo, "by_gold_patch_lines": by_size}

    (OUT / "t2_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "t2_per_task.json").write_text(json.dumps(per_task, indent=2) + "\n")
    (OUT / "t2_report.md").write_text(render_md(report))
    print(f"wrote {OUT / 't2_report.md'}")


def fmt_ci(ci, nd=2):
    return f"[{ci[0]:,.{nd}f}, {ci[1]:,.{nd}f}]"


def render_md(r: dict) -> str:
    L = []
    L.append("# T2 memory track — H5 (pre-registered)\n")
    L.append(
        f"273 tasks ran. {', '.join(f'`{i}`' for i in r['unscored'])} are not scored, so n = {r['n']}. "
        f"Thirds {r['thirds']} ({r['third_rule']}). "
        f"Paired bootstrap {r['bootstrap']} resamples, seed {r['seed']}.\n"
    )
    L.append(r["cost_note"] + "\n")
    if r.get("grade_recovered"):
        L.append(
            "One grade was read from the harness report because `grades.json` no longer has that arm: "
            + ", ".join(f"{g['task']} {g['arm']} resolved={g['resolved']}" for g in r["grade_recovered"])
            + ".\n"
        )
    L.append("## 1. Per arm\n")
    L.append("| arm | resolved | rate | calls median | prompt tokens / call median | cache hit median | tokens median | primary median | primary total | billed median | billed total | exits |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in r["arms"]:
        t = r["table"][arm]
        exits = ", ".join(f"{k} {v}" for k, v in sorted(t["exit_reasons"].items(), key=lambda kv: -kv[1]))
        L.append(
            f"| {arm} | {t['resolved']}/{t['n']} | {t['resolve_rate']:.3f} | {t['calls_median']:.0f} | "
            f"{t['prompt_per_call_median']:,} | {t['cache_hit_median']:.3f} | {t['tokens_median']:,} | "
            f"${t['cost_median_usd']:.4f} | ${t['cost_total_usd']:.2f} | ${t['cost_billed_median_usd']:.4f} | "
            f"${t['cost_billed_total_usd']:.2f} | {exits} |"
        )
    L.append("")
    L.append(
        f"Ledger sum over every row, including retries: primary ${r['ledger_primary_total']:.2f}, "
        f"billed ${r['ledger_billed_total']:.2f}. The table totals are the final attempt of each scored task.\n"
    )
    L.append("### Reading\n")
    L.append(
        "G-persist resolves fewer tasks than G-fresh and fewer than M-notes. "
        "Both registered gaps are about −25 pp. The task-level and the 8-repo cluster intervals both exclude zero. "
        "The gap is small in the first third and about −48 pp in the last third. "
        "G-persist makes fewer calls and spends more tokens: the carried graph is packed into each call. "
        "M-notes resolves more tasks than M by 3.3 pp. That pair is not named in H5. "
        "Its task-level interval excludes zero when every retry is kept, and includes zero on the first attempt only.\n"
    )
    L.append("## 1. H5\n")
    L.append(
        "G-persist − G-fresh and G-persist − M-notes are the registered comparisons. "
        "M-notes − M is reported with the same method; it is not named in H5.\n"
    )
    L.append(r["cluster_ci_note"] + "\n")
    for pair, metrics in r["H5"].items():
        registered = "registered" if metrics["registered"] else "not named in H5"
        L.append(f"### {pair} ({registered})\n")
        L.append("| metric | slice | n | diff | task-level 95% CI | cluster 95% CI | W/T/L or medians |")
        L.append("|---|---|---|---|---|---|---|")
        for m, slices in metrics.items():
            if m == "registered":
                continue
            for sl, e in slices.items():
                if "diff" not in e:
                    L.append(f"| {m} | {sl} | {e.get('n', 0)} | – | – | – | – |")
                    continue
                nd = {"resolved": 2, "calls": 2, "tokens": 0, "cost": 4}[m]
                if m == "resolved":
                    extra = f"{e['win']}/{e['tie']}/{e['loss']} disc {e['discordant']}"
                    unit = " pp"
                else:
                    extra = f"{e['median_a']:,.{nd}f} vs {e['median_b']:,.{nd}f} (ratio {e['ratio_of_medians']})"
                    unit = ""
                L.append(
                    f"| {m} | {sl} | {e['n']} | {e['diff']:+,.{nd}f}{unit} | {fmt_ci(e['task_ci95'], nd)} | "
                    f"{fmt_ci(e['cluster_ci95'], nd)} | {extra} |"
                )
        L.append("")

    L.append("## 2. Retry sensitivity\n")
    s = r["sensitivity"]
    L.append(f"Resolved — all retries: {s['all_retries']}.")
    L.append(f"First attempt only: {s['first_attempt_table']}.")
    L.append(f"Wall-clock counts (provider errors retried): {s['wall_clock_table']}.")
    L.append(f"Retries by type: {s['retries_by_type']}.")
    L.append(f"Flips, first attempt: {s['flips_first_attempt']}.")
    L.append(f"Flips, wall clock: {s['flips_wall_clock']}.")
    L.append(f"Resolve-rate conclusions that change (sign, or whether the task CI includes zero): {s['conclusion_changes']}.\n")
    L.append("| rule | pair | diff pp | task CI | W/T/L |")
    L.append("|---|---|---|---|---|")
    for label in ("first_attempt", "wall_clock"):
        for pair, e in s[f"{label}_H5"].items():
            L.append(
                f"| {label} | {pair} | {e['diff_pp']:+.2f} | {fmt_ci(e['task_ci95_pp'])} | "
                f"{e['win']}/{e['tie']}/{e['loss']} |"
            )
    L.append("")

    L.append("## 2. Cost caps\n")
    c = r["cost_caps"]
    L.append(f"Runs that exited on a cost cap: {len(c['exits'])}.")
    if c["exits"]:
        for row in c["exits"]:
            L.append(f"- {row['task']} {row['arm']} exit={row['exit_reason']} primary={row['cost_usd']} billed={row['cost_usd_billed']}")
    L.append(c["early_note"])
    L.append(
        f"On those {c['early_rows']} ledger rows the max `cost_usd` is ${c['early_max_ledger_cost_usd']} "
        f"and the max billed is ${c['early_max_billed']}. "
        f"Either figure at least half of $0.50: {c['early_near_half_cap']}.\n"
    )

    L.append("## 2. Notes\n")
    notes = r["notes"]
    L.append(
        f"On the {notes['scored_tasks']} scored tasks, M-notes read NOTES.md in {notes['read']} "
        f"and wrote it in {notes['wrote']}. `M-notes.carry.json` is the file carried into the task: "
        f"`tokens_before` is its size before the {notes['cap_tokens']}-token trim, and `tokens_after` is what the task received. "
        f"The first task of a repo receives an empty file. The agent's own write is `notes_tokens` on the summary, "
        f"and the next task's carry trims that write.\n"
    )
    L.append("| repo | read | wrote | truncated | incoming after trim, first | incoming after trim, last | max incoming before trim | max incoming after trim |")
    L.append("|---|---|---|---|---|---|---|---|")
    for repo, row in notes["by_repo"].items():
        L.append(
            f"| {repo} | {row['read']}/{row['n']} | {row['wrote']}/{row['n']} | {row['truncated']} | "
            f"{row['tokens_after_first']} | {row['tokens_after_last']} | {row['tokens_before_max']} | {row['tokens_after_max']} |"
        )
    L.append("")

    L.append("## 2. G-persist growth\n")
    g = r["g_persist"]
    L.append(
        f"Tasks whose new calls dropped nodes: {g['tasks_with_any_drop']} of {r['n']}. "
        f"Largest prompt on any new call: {g['max_prompt_any_call']:,} tokens. "
        f"Largest graph file: {g['max_graph_bytes']:,} bytes.\n"
    )
    L.append("| third | n | graph bytes median | prompt tokens / call median | tasks with drops | median dropped nodes when any | max prompt |")
    L.append("|---|---|---|---|---|---|---|")
    for third, row in g["by_third"].items():
        L.append(
            f"| {third} | {row['n']} | {row['graph_bytes_median']:,} | {row['prompt_per_call_median']:,} | "
            f"{row['tasks_with_drops']} | {row['dropped_nodes_median_when_any']} | {row['max_prompt']:,} |"
        )
    L.append("")

    L.append("## 2. Calibration overlap (exploratory)\n")
    cal = r["calibration"]
    L.append(f"{cal['n']} calibration tasks sit inside the sequences. H5 resolve rate with them is the section above. Without them:\n")
    L.append("| pair | n | diff pp | task CI | cluster CI | W/T/L |")
    L.append("|---|---|---|---|---|---|")
    for pair, metrics in cal["without_14"].items():
        e = metrics["resolved"]["overall"]
        L.append(
            f"| {pair} | {e['n']} | {e['diff']:+.2f} | {fmt_ci(e['task_ci95'])} | {fmt_ci(e['cluster_ci95'])} | "
            f"{e['win']}/{e['tie']}/{e['loss']} |"
        )
    L.append("")

    L.append("## 3. Exploratory\n")
    L.append("### By repo\n")
    L.append("| repo | n | M | M-notes | G-fresh | G-persist |")
    L.append("|---|---|---|---|---|---|")
    for repo, row in r["exploratory"]["by_repo"].items():
        L.append(
            f"| {repo} | {row['n']} | {row['M']} | {row['M-notes']} | {row['G-fresh']} | {row['G-persist']} |"
        )
    L.append("")
    gold = r["exploratory"]["by_gold_patch_lines"]
    L.append(f"### By gold-patch size\n")
    L.append(f"Terciles of added plus removed lines. Cuts at {gold['cutpoints_lines']} lines.\n")
    L.append("| band | n | M | M-notes | G-fresh | G-persist |")
    L.append("|---|---|---|---|---|---|")
    for band in ("small", "medium", "large"):
        row = gold[band]
        L.append(f"| {band} | {row['n']} | {row['M']} | {row['M-notes']} | {row['G-fresh']} | {row['G-persist']} |")
    L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
