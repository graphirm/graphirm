#!/usr/bin/env python3
"""T1 analysis, in the order prereg.md fixes.

Section 1 is the pre-registered analysis: resolve rates, H1 (parity,
paired bootstrap), H2 (scoring, exact one-sided McNemar), Holm across H1
and H2, H3 (format), H4 (cost), the 90% ceiling statement, and H6
(provenance). Section 2 is the required robustness work: first-attempt-only
grades, retry counts by type, the mini-swe-agent cache metric from
`prompt_tokens_details.cached_tokens`, and the request-caused list with the
eodag-2176 root cause. Section 3 is exploratory and labelled as such.

Inputs: /home/krs/graphirm-runs/t1 (grades.json, arms.jsonl, summaries,
graphs, trajectories, before-gap.json). Outputs: <runs>/analysis/.
Nothing here changes a run file.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from metrics import cache_hit, primary_cost_from_calls  # noqa: E402

RUNS = Path("/home/krs/graphirm-runs/t1")
OUT = RUNS / "analysis"
LOCK = HERE.parent / "instances.lock.json"
ARMS = ["M", "G", "G-md", "G-b16", "G-b16-recency"]
GRAPH_ARMS = ["G", "G-md", "G-b16", "G-b16-recency"]
N_BOOT = 10_000
SEED = 20261022
EXCLUDE = {
    "sigma67__ytmusicapi-909_interface",
    "huggingface__trl-5135",
    "vyperlang__vyper-4801",
    "keras-team__keras-22316",
    "tox-dev__tox-3931",
}
NOMINAL = {"G": 100_000, "G-md": 100_000, "G-b16": 16_000, "G-b16-recency": 16_000}


# ------------------------------------------------------------------ loading


def task_ids() -> list[str]:
    lock = json.loads(LOCK.read_text())
    ids = [row["instance_id"] for row in lock["headline"] if row["instance_id"] not in EXCLUDE]
    if len(ids) != 210:
        raise SystemExit(f"expected 210 ids, found {len(ids)}")
    return ids


def ledger_rows() -> list[dict]:
    rows = []
    for line in (RUNS / "arms.jsonl").read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def grades(ids: list[str]) -> dict[str, dict[str, bool]]:
    out = {}
    for iid in ids:
        path = RUNS / iid / "grades.json"
        g = json.loads(path.read_text())
        for arm in ARMS:
            if g.get(arm) is None:
                raise SystemExit(f"{iid} {arm} has no grade")
        out[iid] = {arm: bool(g[arm]) for arm in ARMS}
    return out


def latest_by_key(rows: list[dict]) -> dict[str, dict]:
    latest = {}
    for row in rows:
        latest[row["instance_id"]] = row
    return latest


# ------------------------------------------------------------------ statistics


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, rng, n_boot=N_BOOT, alphas=(0.025, 0.975)):
    """Difference of means a - b over paired resamples of tasks."""
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.quantile(diffs, alphas)
    return float(a.mean() - b.mean()), float(lo), float(hi), diffs


def win_tie_loss(a: np.ndarray, b: np.ndarray) -> dict:
    return {
        "win": int(((a == 1) & (b == 0)).sum()),
        "tie": int((a == b).sum()),
        "loss": int(((a == 0) & (b == 1)).sum()),
        "discordant": int((a != b).sum()),
    }


def mcnemar_one_sided(wins: int, losses: int) -> float:
    """P(X >= wins | Binom(wins + losses, 0.5)). Exact."""
    from math import comb

    n = wins + losses
    if n == 0:
        return 1.0
    return float(sum(comb(n, k) for k in range(wins, n + 1)) / 2**n)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted = {}
    running = 0.0
    for rank, (name, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        adjusted[name] = running
    return adjusted


def bootstrap_ratio_of_medians(a: np.ndarray, b: np.ndarray, rng, n_boot=N_BOOT):
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    ratios = np.median(a[idx], axis=1) / np.median(b[idx], axis=1)
    return float(np.median(a) / np.median(b)), float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def bootstrap_stat(values: np.ndarray, fn, rng, n_boot=N_BOOT):
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = np.array([fn(values[row]) for row in idx])
    return float(fn(values)), float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


# ------------------------------------------------------------------ per-arm inputs


def cache_hit_graph(graph: Path) -> float | None:
    """Token-weighted cache hit over the assistant calls in one Graphirm graph."""
    if not graph.is_file() or graph.stat().st_size == 0:
        return None
    conn = sqlite3.connect(f"file:{graph}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT metadata FROM nodes WHERE node_type = 'interaction'").fetchall()
    finally:
        conn.close()
    inp = cached = 0
    for (meta,) in rows:
        md = json.loads(meta)
        if md.get("usage_input") is None:
            continue
        inp += int(md.get("usage_input") or 0)
        cached += int(md.get("usage_cache_read") or 0)
    return cache_hit(inp, cached)


def cache_hit_mini(traj: Path) -> float | None:
    """Token-weighted cache hit from `usage.prompt_tokens_details.cached_tokens`."""
    if not traj.is_file():
        return None
    data = json.loads(traj.read_text())
    inp = cached = 0
    for msg in data.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        usage = ((msg.get("extra") or {}).get("response") or {}).get("usage") or {}
        inp += int(usage.get("prompt_tokens") or 0)
        cached += int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    return cache_hit(inp, cached)


def dropped_tokens(graph: Path) -> int | None:
    if not graph.is_file() or graph.stat().st_size == 0:
        return None
    conn = sqlite3.connect(f"file:{graph}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT metadata FROM nodes WHERE node_type = 'interaction'").fetchall()
    finally:
        conn.close()
    total = 0
    for (meta,) in rows:
        total += int(json.loads(meta).get("context_dropped_tokens") or 0)
    return total


def classify_retry(error: str) -> str:
    text = error or ""
    low = text.lower()
    if text.startswith("invalid_config") and "provider error" not in low:
        return "config"
    if "timeoutexpired" in low:
        return "wall_clock_1900s"
    if "timed out" in low or "timeout" in low:
        return "llm_call_timeout"
    if "error sending request" in low or "connection" in low:
        return "provider_connection"
    if "429" in text or "service_overloaded" in low or "shared_pool" in low or "provider error" in low:
        return "provider_429_or_overload"
    if "400" in text or "context length" in low:
        return "request_400"
    if "graphirm_agent::" in text and "INFO" in text:
        # 2026-10-14 model-bug rows: the runner exited non-zero with only INFO
        # logs in the tail. Those 20 arms ran the wrong model and were rerun.
        return "config"
    return "other"


# ------------------------------------------------------------------ main


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    ids = task_ids()
    rows = ledger_rows()
    latest = latest_by_key(rows)
    final = grades(ids)
    n = len(ids)

    resolved = {arm: np.array([1 if final[iid][arm] else 0 for iid in ids]) for arm in ARMS}
    # Primary cost (prereg §4) = frozen prices x provider-reported tokens for every arm.
    # The ledger's cost_usd is primary for Graphirm arms (summary.cost_usd) but for M it
    # is mini-swe-agent's agent.cost, which is OpenRouter's bill. M is recomputed from
    # its trajectory; the billed figure is kept alongside for every arm.
    from run_arm import mini_costs

    cost = {}
    for arm in GRAPH_ARMS:
        vals = []
        for iid in ids:
            graph = RUNS / iid / f"{arm}.graph.db"
            calls = []
            if graph.is_file() and graph.stat().st_size > 0:
                conn = sqlite3.connect(f"file:{graph}?mode=ro", uri=True)
                try:
                    meta_rows = conn.execute(
                        "SELECT metadata FROM nodes WHERE node_type = 'interaction' "
                        "AND json_extract(metadata, '$.usage_input') IS NOT NULL"
                    ).fetchall()
                finally:
                    conn.close()
                calls = [json.loads(meta) for (meta,) in meta_rows]
            vals.append(primary_cost_from_calls(calls))
        cost[arm] = np.array(vals)
    m_costs = [mini_costs(RUNS / iid / "M.traj.json") for iid in ids]
    cost["M"] = np.array([c["cost_usd"] for c in m_costs])
    billed = {"M": np.array([c["cost_usd_billed"] for c in m_costs])}
    ledger_m = np.array([float(latest[f"{iid}::M"].get("cost_usd") or 0.0) for iid in ids])
    for arm in GRAPH_ARMS:
        vals = []
        for iid in ids:
            path = RUNS / iid / f"{arm}.summary.json"
            rep = json.loads(path.read_text()).get("cost_usd_reported") if path.is_file() else None
            vals.append(float(rep or 0.0))
        billed[arm] = np.array(vals)
    cache = {}
    for arm in ARMS:
        vals = []
        for iid in ids:
            if arm == "M":
                vals.append(cache_hit_mini(RUNS / iid / "M.traj.json"))
            else:
                vals.append(cache_hit_graph(RUNS / iid / f"{arm}.graph.db"))
        cache[arm] = vals

    report: dict = {"n": n, "arms": ARMS, "seed": SEED, "bootstrap": N_BOOT}

    # ---- table
    table = {}
    for arm in ARMS:
        c = cache[arm]
        known = [v for v in c if v is not None]
        table[arm] = {
            "resolved": int(resolved[arm].sum()),
            "resolve_rate": round(float(resolved[arm].mean()), 4),
            "median_cost_usd": round(float(np.median(cost[arm])), 6),
            "total_cost_usd": round(float(cost[arm].sum()), 4),
            "median_billed_usd": round(float(np.median(billed[arm])), 6),
            "total_billed_usd": round(float(billed[arm].sum()), 4),
            "median_cache_hit": round(float(np.median(known)), 4) if known else None,
            "cache_hit_n": len(known),
        }
    report["table"] = table

    # ---- H1 parity: G - M, non-inferior if lower bound > -0.05
    d, lo, hi, diffs = paired_bootstrap_diff(resolved["G"], resolved["M"], rng)
    p_h1 = float((diffs <= -0.05).mean())  # one-sided: P(diff <= margin)
    h1 = {
        "diff_pp": round(d * 100, 2),
        "ci95_pp": [round(lo * 100, 2), round(hi * 100, 2)],
        "non_inferior_at_minus_5pp": bool(lo > -0.05),
        "p_one_sided_vs_minus_5pp": p_h1,
        **win_tie_loss(resolved["G"], resolved["M"]),
    }
    # ---- H2 scoring: G-b16 >= G-b16-recency, exact one-sided McNemar
    wtl = win_tie_loss(resolved["G-b16"], resolved["G-b16-recency"])
    p_h2 = mcnemar_one_sided(wtl["win"], wtl["loss"])
    h2 = {
        **wtl,
        "p_mcnemar_one_sided": p_h2,
        "underpowered_under_10_discordant": wtl["discordant"] < 10,
        "rate_b16": round(float(resolved["G-b16"].mean()), 4),
        "rate_recency": round(float(resolved["G-b16-recency"].mean()), 4),
    }
    adj = holm({"H1": p_h1, "H2": p_h2})
    h1["p_holm"] = adj["H1"]
    h2["p_holm"] = adj["H2"]
    # Holm at alpha 0.05 over two tests means the smaller raw p is judged at 0.025.
    _, lo975, hi975, _ = paired_bootstrap_diff(resolved["G"], resolved["M"], rng, alphas=(0.0125, 0.9875))
    h1["ci975_pp_for_holm_step"] = [round(lo975 * 100, 2), round(hi975 * 100, 2)]
    report["H1"] = h1
    report["H2"] = h2

    # ---- H3 format: G vs G-md
    d3, lo3, hi3, _ = paired_bootstrap_diff(resolved["G"], resolved["G-md"], rng)
    report["H3"] = {
        "diff_pp": round(d3 * 100, 2),
        "ci95_pp": [round(lo3 * 100, 2), round(hi3 * 100, 2)],
        **win_tie_loss(resolved["G"], resolved["G-md"]),
    }

    # ---- H4 cost
    h4 = {"median_cost_usd": {arm: round(float(np.median(cost[arm])), 6) for arm in ARMS}}
    ratio, rlo, rhi = bootstrap_ratio_of_medians(cost["G"], cost["M"], rng)
    h4["G_over_M_median_ratio"] = {"point": round(ratio, 3), "ci95": [round(rlo, 3), round(rhi, 3)]}
    per_resolved = {}
    for arm in ARMS:
        pairs = np.stack([cost[arm], resolved[arm]], axis=1)

        def cpr(sample):
            r = sample[:, 1].sum()
            return sample[:, 0].sum() / r if r else float("nan")

        idx = rng.integers(0, n, size=(N_BOOT, n))
        boots = np.array([cpr(pairs[row]) for row in idx])
        boots = boots[~np.isnan(boots)]
        per_resolved[arm] = {
            "point": round(float(cpr(pairs)), 4),
            "ci95": [round(float(np.quantile(boots, 0.025)), 4), round(float(np.quantile(boots, 0.975)), 4)],
        }
    h4["cost_per_resolved_task_usd"] = per_resolved
    other_ratios = {}
    for arm in ("G-md", "G-b16", "G-b16-recency"):
        r, a, b = bootstrap_ratio_of_medians(cost[arm], cost["M"], rng)
        other_ratios[arm] = {"point": round(r, 3), "ci95": [round(a, 3), round(b, 3)]}
    h4["other_arms_over_M_median_ratio"] = other_ratios
    bratio, blo, bhi = bootstrap_ratio_of_medians(billed["G"], billed["M"], rng)
    h4["G_over_M_billed_median_ratio"] = {"point": round(bratio, 3), "ci95": [round(blo, 3), round(bhi, 3)]}
    h4["M_ledger_cost_was_billed"] = {
        "ledger_total_usd": round(float(ledger_m.sum()), 4),
        "billed_total_from_traj_usd": round(float(billed["M"].sum()), 4),
        "primary_total_usd": round(float(cost["M"].sum()), 4),
        "note": "First report (2026-10-22 morning) used the ledger figure for M, i.e. billed cost "
        "against primary cost for G; its G/M ratio 1.712 is superseded by G_over_M_median_ratio here.",
    }
    report["H4"] = h4

    # ---- ceiling
    report["ceiling"] = {
        "any_arm_over_90pct": any(float(resolved[arm].mean()) > 0.9 for arm in ARMS),
        "both_G_and_M_over_90pct": float(resolved["G"].mean()) > 0.9 and float(resolved["M"].mean()) > 0.9,
    }

    # ---- H6 provenance (pooled over tasks, per Graphirm arm)
    h6 = {}
    for arm in GRAPH_ARMS:
        agg = Counter()
        tasks_with = 0
        for iid in ids:
            path = RUNS / iid / f"{arm}.summary.json"
            if not path.is_file():
                continue
            prov = json.loads(path.read_text()).get("provenance") or {}
            if "items" not in prov:
                continue
            tasks_with += 1
            for k in ("items", "items_with_src", "citations", "strict_match", "lenient_match"):
                agg[k] += int(prov.get(k) or 0)
        h6[arm] = {
            "tasks_with_summary": tasks_with,
            "items": agg["items"],
            "items_with_src": agg["items_with_src"],
            "share_with_src": round(agg["items_with_src"] / agg["items"], 4) if agg["items"] else None,
            "citations": agg["citations"],
            "strict_exists_and_type_match": round(agg["strict_match"] / agg["citations"], 4) if agg["citations"] else None,
            "lenient_exists_and_type_match": round(agg["lenient_match"] / agg["citations"], 4) if agg["citations"] else None,
        }
    report["H6"] = h6

    # ================================================================ robustness
    # ---- retries by type and first-attempt-only grades
    by_key: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_key[row["instance_id"]].append(row)
    retry_types = {arm: Counter() for arm in ARMS}
    first_attempt = {arm: np.zeros(n, dtype=int) for arm in ARMS}
    first_changes = []
    for i, iid in enumerate(ids):
        for arm in ARMS:
            attempts = by_key[f"{iid}::{arm}"]
            # The first attempt that ran the registered model. Rows marked invalid_config
            # (model guard) are not attempts of the agent and are skipped.
            real = [r for r in attempts if classify_retry(r.get("error") or "") != "config"]
            for r in real[:-1]:
                if r.get("error"):
                    retry_types[arm][classify_retry(r["error"])] += 1
            first = real[0] if real else attempts[0]
            if first.get("error") and not first.get("request_caused"):
                first_attempt[arm][i] = 0
                if final[iid][arm]:
                    first_changes.append({"task": iid, "arm": arm, "first_error": classify_retry(first["error"])})
            else:
                first_attempt[arm][i] = 1 if final[iid][arm] else 0
    flips = {arm: Counter() for arm in ARMS}
    for change in first_changes:
        flips[change["arm"]][change["first_error"]] += 1
    sens = {
        "retries_by_type": {arm: dict(retry_types[arm]) for arm in ARMS},
        "resolved_flipped_to_unresolved": first_changes,
        "flips_by_arm_and_type": {arm: dict(flips[arm]) for arm in ARMS},
    }
    fa = {}
    for arm in ARMS:
        fa[arm] = {"resolved": int(first_attempt[arm].sum()), "rate": round(float(first_attempt[arm].mean()), 4)}
    sens["first_attempt_table"] = fa
    rng2 = np.random.default_rng(SEED + 1)
    d1, l1, u1, dd1 = paired_bootstrap_diff(first_attempt["G"], first_attempt["M"], rng2)
    w2 = win_tie_loss(first_attempt["G-b16"], first_attempt["G-b16-recency"])
    d3f, l3f, u3f, _ = paired_bootstrap_diff(first_attempt["G"], first_attempt["G-md"], rng2)
    sens["first_attempt_H1"] = {
        "diff_pp": round(d1 * 100, 2),
        "ci95_pp": [round(l1 * 100, 2), round(u1 * 100, 2)],
        "non_inferior_at_minus_5pp": bool(l1 > -0.05),
        **win_tie_loss(first_attempt["G"], first_attempt["M"]),
    }
    sens["first_attempt_H2"] = {**w2, "p_mcnemar_one_sided": mcnemar_one_sided(w2["win"], w2["loss"]), "underpowered_under_10_discordant": w2["discordant"] < 10}
    sens["first_attempt_H3"] = {"diff_pp": round(d3f * 100, 2), "ci95_pp": [round(l3f * 100, 2), round(u3f * 100, 2)], **win_tie_loss(first_attempt["G"], first_attempt["G-md"])}
    # (c) wall-clock kills count against the arm; provider-side errors (429, connection
    # drop, LLM call timeout) may still be retried. This isolates the retry asymmetry:
    # mini submits its partial work at 1800s, Graphirm was killed at 1900s with no patch.
    provider_kinds = {"config", "provider_429_or_overload", "provider_connection", "llm_call_timeout"}
    wall_only = {arm: np.zeros(n, dtype=int) for arm in ARMS}
    wall_flips = {arm: 0 for arm in ARMS}
    for i, iid in enumerate(ids):
        for arm in ARMS:
            attempts = by_key[f"{iid}::{arm}"]
            agent_attempts = [r for r in attempts if classify_retry(r.get("error") or "") not in provider_kinds or not r.get("error")]
            first = agent_attempts[0] if agent_attempts else attempts[-1]
            if first.get("error") and not first.get("request_caused"):
                wall_only[arm][i] = 0
                if final[iid][arm]:
                    wall_flips[arm] += 1
            else:
                wall_only[arm][i] = 1 if final[iid][arm] else 0
    rng3 = np.random.default_rng(SEED + 2)
    dw, lw, uw, _ = paired_bootstrap_diff(wall_only["G"], wall_only["M"], rng3)
    w2w = win_tie_loss(wall_only["G-b16"], wall_only["G-b16-recency"])
    d3w, l3w, u3w, _ = paired_bootstrap_diff(wall_only["G"], wall_only["G-md"], rng3)
    sens["wall_clock_only_table"] = {arm: {"resolved": int(wall_only[arm].sum()), "rate": round(float(wall_only[arm].mean()), 4), "resolves_lost": wall_flips[arm]} for arm in ARMS}
    sens["wall_clock_only_H1"] = {"diff_pp": round(dw * 100, 2), "ci95_pp": [round(lw * 100, 2), round(uw * 100, 2)], "non_inferior_at_minus_5pp": bool(lw > -0.05), **win_tie_loss(wall_only["G"], wall_only["M"])}
    sens["wall_clock_only_H2"] = {**w2w, "p_mcnemar_one_sided": mcnemar_one_sided(w2w["win"], w2w["loss"])}
    sens["wall_clock_only_H3"] = {"diff_pp": round(d3w * 100, 2), "ci95_pp": [round(l3w * 100, 2), round(u3w * 100, 2)]}
    sens["conclusions_change"] = {
        "H1_non_inferior": h1["non_inferior_at_minus_5pp"] != sens["first_attempt_H1"]["non_inferior_at_minus_5pp"],
        "H2_underpowered": h2["underpowered_under_10_discordant"] != sens["first_attempt_H2"]["underpowered_under_10_discordant"],
        "H2_significant_at_0.05": (h2["p_mcnemar_one_sided"] < 0.05) != (sens["first_attempt_H2"]["p_mcnemar_one_sided"] < 0.05),
        "H3_ci_excludes_zero": (lo3 > 0 or hi3 < 0) != (l3f > 0 or u3f < 0),
    }
    report["sensitivity"] = sens

    # ---- cache before/after gap, now including M
    before = set(json.loads((RUNS / "before-gap.json").read_text()))
    gap = {}
    for arm in ARMS:
        b = [v for iid, v in zip(ids, cache[arm]) if v is not None and iid in before]
        a = [v for iid, v in zip(ids, cache[arm]) if v is not None and iid not in before]
        gap[arm] = {
            "before_median": round(float(np.median(b)), 4) if b else None,
            "before_n": len(b),
            "after_median": round(float(np.median(a)), 4) if a else None,
            "after_n": len(a),
        }
    report["cache_before_after_gap"] = gap

    # ---- request-caused list
    req = []
    for key, row in latest.items():
        if not row.get("request_caused"):
            continue
        iid, arm = key.split("::", 1)
        cause = (row.get("error") or "")[:200]
        retries = RUNS / iid / f"{arm}.retries.log"
        detail = ""
        if retries.is_file():
            last = retries.read_text(errors="replace").strip().splitlines()
            if last:
                detail = last[-1][:400]
        req.append({"task": iid, "arm": arm, "ledger_error": cause, "last_provider_line": detail})
    report["request_caused"] = sorted(req, key=lambda r: (r["task"], r["arm"]))
    report["eodag_2176"] = eodag_root_cause()

    # ================================================================ exploratory
    explor = {}
    # by repo owner (first path segment)
    owners = defaultdict(list)
    for i, iid in enumerate(ids):
        owners[iid.split("__", 1)[0]].append(i)
    by_owner = {}
    for owner, idx in sorted(owners.items()):
        if len(idx) < 3:
            continue
        by_owner[owner] = {"n": len(idx), **{arm: int(resolved[arm][idx].sum()) for arm in ARMS}}
    explor["by_repo_owner_min3"] = by_owner
    # by gold patch size (difficulty proxy): terciles of added+removed lines in the gold patch
    sizes = np.array([gold_patch_lines(iid) for iid in ids])
    t1, t2 = np.quantile(sizes, [1 / 3, 2 / 3])
    bands = {"small": sizes <= t1, "medium": (sizes > t1) & (sizes <= t2), "large": sizes > t2}
    explor["by_gold_patch_lines"] = {
        "cutpoints": [int(t1), int(t2)],
        **{
            band: {"n": int(mask.sum()), **{arm: int(resolved[arm][mask].sum()) for arm in ARMS}}
            for band, mask in bands.items()
        },
    }
    interface = np.array([iid.endswith("_interface") for iid in ids])
    explor["by_interface_flag"] = {
        "interface": {"n": int(interface.sum()), **{arm: int(resolved[arm][interface].sum()) for arm in ARMS}},
        "other": {"n": int((~interface).sum()), **{arm: int(resolved[arm][~interface].sum()) for arm in ARMS}},
    }
    # by tokens dropped in the 16k arms
    dropped = {}
    for arm in ("G-b16", "G-b16-recency"):
        vals = np.array([dropped_tokens(RUNS / iid / f"{arm}.graph.db") or 0 for iid in ids])
        zero = vals == 0
        pos = vals[vals > 0]
        med = float(np.median(pos)) if len(pos) else 0.0
        low = (vals > 0) & (vals <= med)
        high = vals > med
        dropped[arm] = {
            "tasks_with_any_drop": int((vals > 0).sum()),
            "median_dropped_tokens_when_any": int(med),
            "none": {"n": int(zero.sum()), "resolved": int(resolved[arm][zero].sum())},
            "low": {"n": int(low.sum()), "resolved": int(resolved[arm][low].sum())},
            "high": {"n": int(high.sum()), "resolved": int(resolved[arm][high].sum())},
        }
    explor["by_dropped_tokens_16k_arms"] = dropped
    report["exploratory"] = explor

    (OUT / "t1_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "t1_report.md").write_text(render_md(report))
    per_task = []
    for i, iid in enumerate(ids):
        per_task.append(
            {
                "task": iid,
                **{f"resolved_{arm}": int(resolved[arm][i]) for arm in ARMS},
                **{f"cost_{arm}": round(float(cost[arm][i]), 6) for arm in ARMS},
                **{f"cache_{arm}": (round(cache[arm][i], 4) if cache[arm][i] is not None else None) for arm in ARMS},
                "gold_patch_lines": int(sizes[i]),
            }
        )
    (OUT / "t1_per_task.json").write_text(json.dumps(per_task, indent=2) + "\n")
    print((OUT / "t1_report.md").read_text())


def gold_patch_lines(iid: str) -> int:
    """Added plus removed lines in the gold patch, read from the dataset."""
    return _gold_sizes()[iid]


_GOLD: dict[str, int] | None = None


def _gold_sizes() -> dict[str, int]:
    global _GOLD
    if _GOLD is not None:
        return _GOLD
    import pyarrow as pa

    from run_t1 import ARROW_DIR

    sizes: dict[str, int] = {}
    for path in sorted(ARROW_DIR.glob("swe-rebench-leaderboard-2026_0[1-5].arrow")):
        with pa.memory_map(str(path), "r") as source:
            table = pa.ipc.open_stream(source).read_all().select(["instance_id", "patch"])
            for row in table.to_pylist():
                patch = row["patch"] or ""
                count = 0
                for line in patch.splitlines():
                    if (line.startswith("+") and not line.startswith("+++")) or (
                        line.startswith("-") and not line.startswith("---")
                    ):
                        count += 1
                sizes[row["instance_id"]] = count
    _GOLD = sizes
    return sizes


def eodag_root_cause() -> dict:
    """Which node made the eodag-2176 requests exceed the model's context."""
    out = {}
    task = RUNS / "cs-si__eodag-2176"
    for arm in GRAPH_ARMS:
        graph = task / f"{arm}.graph.db"
        if not graph.is_file():
            continue
        conn = sqlite3.connect(f"file:{graph}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT node_type, data, metadata, created_at FROM nodes ORDER BY created_at"
            ).fetchall()
        finally:
            conn.close()
        calls = 0
        biggest = (0, None)
        command = None
        last_assistant_tools = None
        for nt, data, meta, _ts in rows:
            md = json.loads(meta)
            if nt == "interaction" and md.get("usage_input") is not None:
                calls += 1
                if md.get("tool_calls"):
                    last_assistant_tools = md["tool_calls"]
            body = json.loads(data)
            size = len(json.dumps(body))
            if size > biggest[0]:
                biggest = (size, {"node_type": nt, "label": md.get("label"), "tool": md.get("tool_name"), "calls_before": calls, "path": body.get("path")})
                command = last_assistant_tools
        size, node = biggest
        body_words = None
        if node and node["node_type"] == "content":
            for nt, data, meta, _ts in rows:
                if json.loads(meta).get("label") == node["label"]:
                    text = json.loads(data).get("body") or ""
                    body_words = {"chars": len(text), "whitespace_words": len(text.split()), "words_over_0_75": int(len(text.split()) / 0.75), "newlines": text.count("\n")}
                    break
        summary = json.loads((task / f"{arm}.summary.json").read_text()) if (task / f"{arm}.summary.json").is_file() else {}
        out[arm] = {
            "assistant_calls": calls,
            "largest_node_bytes": size,
            "largest_node": node,
            "largest_node_text": body_words,
            "tool_calls_before_it": command,
            "exit_reason": summary.get("exit_reason"),
            "graded": json.loads((task / "grades.json").read_text()).get(arm),
        }
    traj = task / "M.traj.json"
    if traj.is_file():
        data = json.loads(traj.read_text())
        obs = [str(m.get("content") or "") for m in data["messages"] if m.get("role") == "user"]
        out["M"] = {
            "observations": len(obs),
            "longest_observation_chars": max((len(o) for o in obs), default=0),
            "truncation_warnings": sum(1 for o in obs if "output of your last command was too long" in o),
            "api_calls": (data["info"].get("model_stats") or {}).get("api_calls"),
            "graded": json.loads((task / "grades.json").read_text()).get("M"),
        }
    return out


def render_md(r: dict) -> str:
    lines = []
    lines.append(f"# T1 report (pre-registered analysis), n = {r['n']}\n")
    lines.append("## 1. Per-arm table\n")
    lines.append("| arm | resolved | rate | median cost USD | total cost USD | median cache hit | cache n |")
    lines.append("|---|---|---|---|---|---|---|")
    for arm in r["arms"]:
        t = r["table"][arm]
        lines.append(f"| {arm} | {t['resolved']}/{r['n']} | {t['resolve_rate']*100:.1f}% | {t['median_cost_usd']:.6f} | {t['total_cost_usd']:.2f} | {t['median_cache_hit']} | {t['cache_hit_n']} |")
    h1 = r["H1"]
    lines.append("\n## H1 parity (confirmatory): G - M")
    lines.append(f"- diff {h1['diff_pp']:+.2f} pp, paired bootstrap 95% CI [{h1['ci95_pp'][0]:+.2f}, {h1['ci95_pp'][1]:+.2f}] pp; 97.5% CI for the Holm step [{h1['ci975_pp_for_holm_step'][0]:+.2f}, {h1['ci975_pp_for_holm_step'][1]:+.2f}]")
    lines.append(f"- win/tie/loss {h1['win']}/{h1['tie']}/{h1['loss']}, discordant {h1['discordant']}")
    lines.append(f"- non-inferior at -5 pp: {h1['non_inferior_at_minus_5pp']}; one-sided bootstrap p vs -5 pp = {h1['p_one_sided_vs_minus_5pp']:.4f}; Holm-adjusted p = {h1['p_holm']:.4f}")
    h2 = r["H2"]
    lines.append("\n## H2 graph scoring (confirmatory): G-b16 vs G-b16-recency")
    lines.append(f"- rates {h2['rate_b16']*100:.1f}% vs {h2['rate_recency']*100:.1f}%; win/tie/loss {h2['win']}/{h2['tie']}/{h2['loss']}, discordant {h2['discordant']}")
    lines.append(f"- exact one-sided McNemar p = {h2['p_mcnemar_one_sided']:.4f}; Holm-adjusted p = {h2['p_holm']:.4f}; underpowered (<10 discordant): {h2['underpowered_under_10_discordant']}")
    h3 = r["H3"]
    lines.append("\n## H3 format (secondary): G vs G-md")
    lines.append(f"- diff {h3['diff_pp']:+.2f} pp, 95% CI [{h3['ci95_pp'][0]:+.2f}, {h3['ci95_pp'][1]:+.2f}] pp; win/tie/loss {h3['win']}/{h3['tie']}/{h3['loss']}")
    h4 = r["H4"]
    lines.append("\n## H4 cost (secondary)")
    lines.append("Primary cost = frozen prices x provider-reported tokens for every arm (M recomputed from its trajectory; "
                 "the ledger's M cost_usd was OpenRouter's bill). Billed figures alongside.")
    lines.append(f"- M ledger (billed) total {h4['M_ledger_cost_was_billed']['ledger_total_usd']}, M primary total {h4['M_ledger_cost_was_billed']['primary_total_usd']}; "
                 f"G/M median ratio on billed cost {h4['G_over_M_billed_median_ratio']['point']} CI {h4['G_over_M_billed_median_ratio']['ci95']}")
    lines.append(f"- median primary cost per task: " + ", ".join(f"{a} {v:.6f}" for a, v in h4["median_cost_usd"].items()))
    g = h4["G_over_M_median_ratio"]
    lines.append(f"- G/M ratio of medians {g['point']:.3f}, 95% CI [{g['ci95'][0]:.3f}, {g['ci95'][1]:.3f}]")
    for arm, v in h4["other_arms_over_M_median_ratio"].items():
        lines.append(f"- {arm}/M ratio of medians {v['point']:.3f}, 95% CI [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
    lines.append("- cost per resolved task: " + ", ".join(f"{a} {v['point']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]" for a, v in h4["cost_per_resolved_task_usd"].items()))
    c = r["ceiling"]
    lines.append(f"\n## Ceiling\n- any arm over 90%: {c['any_arm_over_90pct']}; both G and M over 90%: {c['both_G_and_M_over_90pct']}")
    lines.append("\n## H6 provenance (descriptive), pooled over tasks")
    lines.append("| arm | items | with src | share | citations | strict exists+type | lenient exists+type |")
    lines.append("|---|---|---|---|---|---|---|")
    for arm, v in r["H6"].items():
        lines.append(f"| {arm} | {v['items']} | {v['items_with_src']} | {v['share_with_src']} | {v['citations']} | {v['strict_exists_and_type_match']} | {v['lenient_exists_and_type_match']} |")
    s = r["sensitivity"]
    lines.append("\n## 2. Robustness\n### Retry sensitivity")
    lines.append("- retries by type (attempts before the final one, model-guard rows excluded): " + json.dumps(s["retries_by_type"]))
    lines.append("- first-attempt-only resolved: " + ", ".join(f"{a} {v['resolved']} ({v['rate']*100:.1f}%)" for a, v in s["first_attempt_table"].items()))
    lines.append(f"- arm results whose final resolve came from a retry: {len(s['resolved_flipped_to_unresolved'])}; by arm and first error: " + json.dumps(s["flips_by_arm_and_type"]))
    f1 = s["first_attempt_H1"]
    lines.append(f"- H1 first-attempt: diff {f1['diff_pp']:+.2f} pp, CI [{f1['ci95_pp'][0]:+.2f}, {f1['ci95_pp'][1]:+.2f}], non-inferior {f1['non_inferior_at_minus_5pp']}, W/T/L {f1['win']}/{f1['tie']}/{f1['loss']}")
    f2 = s["first_attempt_H2"]
    lines.append(f"- H2 first-attempt: W/T/L {f2['win']}/{f2['tie']}/{f2['loss']}, p = {f2['p_mcnemar_one_sided']:.4f}, underpowered {f2['underpowered_under_10_discordant']}")
    f3 = s["first_attempt_H3"]
    lines.append(f"- H3 first-attempt: diff {f3['diff_pp']:+.2f} pp, CI [{f3['ci95_pp'][0]:+.2f}, {f3['ci95_pp'][1]:+.2f}]")
    lines.append("- conclusions change (all retries vs first attempt): " + json.dumps(s["conclusions_change"]))
    lines.append("- (c) wall-clock kills count, provider errors still retried: resolved " + ", ".join(f"{a} {v['resolved']} ({v['rate']*100:.1f}%, lost {v['resolves_lost']})" for a, v in s["wall_clock_only_table"].items()))
    fw = s["wall_clock_only_H1"]
    lines.append(f"- H1 under (c): diff {fw['diff_pp']:+.2f} pp, CI [{fw['ci95_pp'][0]:+.2f}, {fw['ci95_pp'][1]:+.2f}], non-inferior {fw['non_inferior_at_minus_5pp']}, W/T/L {fw['win']}/{fw['tie']}/{fw['loss']}")
    fw2 = s["wall_clock_only_H2"]
    lines.append(f"- H2 under (c): W/T/L {fw2['win']}/{fw2['tie']}/{fw2['loss']}, p = {fw2['p_mcnemar_one_sided']:.4f}")
    fw3 = s["wall_clock_only_H3"]
    lines.append(f"- H3 under (c): diff {fw3['diff_pp']:+.2f} pp, CI [{fw3['ci95_pp'][0]:+.2f}, {fw3['ci95_pp'][1]:+.2f}]")
    lines.append("\n### Cache hit before/after the 2026-10-15 gap (median per-task token-weighted ratio)")
    for arm, v in r["cache_before_after_gap"].items():
        lines.append(f"- {arm}: before {v['before_median']} (n={v['before_n']}), after {v['after_median']} (n={v['after_n']})")
    lines.append("\n### Request-caused failures")
    for row in r["request_caused"]:
        lines.append(f"- {row['task']} {row['arm']}: {row['ledger_error']} | {row['last_provider_line'][:200]}")
    lines.append("\n### eodag-2176 root cause")
    lines.append("```json\n" + json.dumps(r["eodag_2176"], indent=2) + "\n```")
    e = r["exploratory"]
    lines.append("\n## 3. Exploratory (not pre-registered)")
    lines.append("### By repo owner (owners with at least 3 tasks): resolved counts")
    lines.append("| owner | n | " + " | ".join(r["arms"]) + " |")
    lines.append("|---|---|" + "---|" * len(r["arms"]))
    for owner, v in e["by_repo_owner_min3"].items():
        lines.append(f"| {owner} | {v['n']} | " + " | ".join(str(v[a]) for a in r["arms"]) + " |")
    lines.append(f"\n### By gold patch size (added+removed lines; terciles at {e['by_gold_patch_lines']['cutpoints']})")
    lines.append("| band | n | " + " | ".join(r["arms"]) + " |")
    lines.append("|---|---|" + "---|" * len(r["arms"]))
    for band in ("small", "medium", "large"):
        v = e["by_gold_patch_lines"][band]
        lines.append(f"| {band} | {v['n']} | " + " | ".join(str(v[a]) for a in r["arms"]) + " |")
    lines.append("\n### Interface tasks vs other")
    for band, v in e["by_interface_flag"].items():
        lines.append(f"- {band}: n={v['n']}, " + ", ".join(f"{a} {v[a]}" for a in r["arms"]))
    lines.append("\n### 16k arms by tokens dropped")
    lines.append("```json\n" + json.dumps(e["by_dropped_tokens_16k_arms"], indent=2) + "\n```")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
