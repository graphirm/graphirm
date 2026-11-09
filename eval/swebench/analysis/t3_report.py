#!/usr/bin/env python3
"""T3 memory track (CL-Bench codebase, 19 tasks) — pre-registered H5 analysis.

Arms: M, M-notes, G-fresh, G-persist. Per arm: resolved, LLM calls, tokens
(prompt + completion), primary cost. Forward transfer: G-persist − G-fresh,
M-notes − M, and G-persist − M-notes, overall and by sequence-position third,
paired by task. CIs: repo-cluster bootstrap as registered (two clusters, so
not informative, stated as such) and task-level paired bootstrap alongside.
Retry sensitivity uses the same three rules as the T1 report.

Reads /home/krs/graphirm-runs/t3, writes to /home/krs/graphirm-runs/t3/analysis.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from t1_report import classify_retry as _classify_retry  # noqa: E402
from t1_report import paired_bootstrap_diff, win_tie_loss  # noqa: E402
from run_arm import mini_costs  # noqa: E402


def classify_retry(error: str) -> str:
    """T1 rules plus one T3 harness row: the first G-persist carry-in failed with
    'attempt to write a readonly database' (docker cp kept the host uid). That
    is a harness fault fixed before the rerun, not an agent attempt; it is
    excluded from attempts the same way config rows are."""
    if "readonly database" in (error or "").lower():
        return "harness"
    return _classify_retry(error)

RUNS = Path("/home/krs/graphirm-runs/t3")
OUT = RUNS / "analysis"
SOURCES = Path("/home/krs/graphirm-runs/sources/clbench-codebase.jsonl")
ARMS = ["M", "M-notes", "G-fresh", "G-persist"]
MINI_ARMS = {"M", "M-notes"}
PAIRS = [("G-persist", "G-fresh"), ("M-notes", "M"), ("G-persist", "M-notes")]
METRICS = ["resolved", "calls", "tokens", "cost", "cost_billed"]
N_BOOT = 10_000
SEED = 20261022
THIRDS = ["first", "middle", "last"]


# ------------------------------------------------------------------ loading


def load_tasks() -> list[dict]:
    rows = [json.loads(line) for line in SOURCES.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: (0 if r["repo"].startswith("jazzband") else 1, int(r["sequence_rank"])))
    by_repo: dict[str, int] = Counter(r["repo"] for r in rows)
    tasks = []
    for r in rows:
        n = by_repo[r["repo"]]
        pos = int(r["sequence_rank"])
        tasks.append(
            {
                "iid": r["instance_id"],
                "repo": r["repo"],
                "position": pos,
                "n_tasks": n,
                "third": THIRDS[min(2, (pos - 1) * 3 // n)],
            }
        )
    if len(tasks) != 19:
        raise SystemExit(f"expected 19 tasks, found {len(tasks)}")
    return tasks


def ledger_rows() -> list[dict]:
    return [json.loads(l) for l in (RUNS / "arms.jsonl").read_text().splitlines() if l.strip()]


def assistant_calls(graph: Path) -> int | None:
    if not graph.is_file() or graph.stat().st_size == 0:
        return None
    conn = sqlite3.connect(f"file:{graph}?mode=ro", uri=True)
    try:
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE node_type = 'interaction' "
            "AND json_extract(metadata, '$.usage_input') IS NOT NULL"
        ).fetchone()
    finally:
        conn.close()
    return int(count)


def mini_usage(traj: Path) -> tuple[int | None, int | None]:
    """(api calls, prompt + completion tokens) from the trajectory."""
    if not traj.is_file():
        return None, None
    data = json.loads(traj.read_text())
    calls = (data.get("info") or {}).get("model_stats", {}).get("api_calls")
    tokens = 0
    seen = 0
    for msg in data.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        usage = ((msg.get("extra") or {}).get("response") or {}).get("usage") or {}
        tokens += int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
        seen += 1
    return (int(calls) if calls is not None else seen), tokens


def graph_usage(out: Path, arm: str, prior_graph: Path | None) -> tuple[int | None, int | None]:
    """(LLM calls this task, prompt + completion tokens this task).

    summary.json tokens are per run. Call counts live in the graph, which for
    G-persist is cumulative, so the previous task's persisted graph is subtracted."""
    summary_path = out / f"{arm}.summary.json"
    tokens = None
    if summary_path.is_file():
        s = json.loads(summary_path.read_text())
        tokens = int(s.get("tokens_in") or 0) + int(s.get("tokens_out") or 0)
    calls = assistant_calls(out / f"{arm}.graph.db")
    if calls is not None and prior_graph is not None:
        prior = assistant_calls(prior_graph) or 0
        calls -= prior
    return calls, tokens


def notes_usage(traj: Path) -> dict:
    """Did the M-notes agent read or write /memory/NOTES.md in this task?"""
    import re

    out = {"read": False, "wrote": False, "commands": 0}
    if not traj.is_file():
        return out
    data = json.loads(traj.read_text())
    for msg in data.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            args = (call.get("function") or {}).get("arguments")
            try:
                cmd = json.loads(args).get("command", "") if isinstance(args, str) else ""
            except json.JSONDecodeError:
                cmd = args or ""
            if "/memory" not in cmd:
                continue
            out["commands"] += 1
            if re.search(r"cat\s+(/memory/NOTES\.md|\S*NOTES\.md)", cmd):
                out["read"] = True
            if re.search(r"(>>?|tee( -a)?)\s*/memory/NOTES\.md", cmd) or re.search(
                r"open\(['\"]/memory/NOTES\.md['\"],\s*['\"][wa]", cmd
            ):
                out["wrote"] = True
    return out


# ------------------------------------------------------------------ statistics


def cluster_bootstrap_diff(a: np.ndarray, b: np.ndarray, clusters: np.ndarray, rng, n_boot=N_BOOT):
    """Resample repos with replacement, keep every task of each drawn repo."""
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


# ------------------------------------------------------------------ main


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    tasks = load_tasks()
    ids = [t["iid"] for t in tasks]
    n = len(ids)
    rows = ledger_rows()
    latest: dict[str, dict] = {}
    by_key: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        latest[r["instance_id"]] = r
        by_key[r["instance_id"]].append(r)

    # ---- per task per arm inputs
    final: dict[str, dict[str, bool]] = {}
    metric = {m: {arm: np.full(n, np.nan) for arm in ARMS} for m in METRICS}
    prior_graph: dict[str, Path | None] = {}
    per_task = []
    for i, t in enumerate(tasks):
        iid = t["iid"]
        out = RUNS / iid
        g = json.loads((out / "grades.json").read_text())
        final[iid] = {}
        rec = {**t, "arms": {}}
        for arm in ARMS:
            if g.get(arm) is None:
                raise SystemExit(f"{iid} {arm} has no grade")
            final[iid][arm] = bool(g[arm])
            row = latest[f"{iid}::{arm}"]
            if arm in MINI_ARMS:
                calls, tokens = mini_usage(out / f"{arm}.traj.json")
            else:
                prior = prior_graph.get(t["repo"]) if arm == "G-persist" else None
                calls, tokens = graph_usage(out, arm, prior)
            # Primary cost = frozen prices x provider-reported tokens (prereg §4) for
            # every arm. The ledger's cost_usd is primary for Graphirm arms but
            # OpenRouter's bill for mini arms, so mini is recomputed from its trajectory.
            if arm in MINI_ARMS:
                mc = mini_costs(out / f"{arm}.traj.json")
                cost, billed = mc["cost_usd"], mc["cost_usd_billed"]
            else:
                summary = json.loads((out / f"{arm}.summary.json").read_text())
                cost = float(summary.get("cost_usd") or 0.0)
                billed = float(summary.get("cost_usd_reported") or 0.0)
            metric["resolved"][arm][i] = 1.0 if final[iid][arm] else 0.0
            metric["calls"][arm][i] = np.nan if calls is None else calls
            metric["tokens"][arm][i] = np.nan if tokens is None else tokens
            metric["cost"][arm][i] = cost
            metric["cost_billed"][arm][i] = billed
            rec["arms"][arm] = {
                "resolved": final[iid][arm],
                "calls": calls,
                "tokens": tokens,
                "cost_usd": round(cost, 6),
                "cost_usd_billed": round(billed, 6),
                "ledger_cost_usd": round(float(row.get("cost_usd") or 0.0), 6),
                "grade_status": row.get("grade_status"),
                "graph_in_bytes": row.get("graph_in_bytes"),
                "patch_bytes": (out / f"{arm}.patch").stat().st_size if (out / f"{arm}.patch").is_file() else None,
            }
        if (out / "G-persist.graph.db").is_file():
            prior_graph[t["repo"]] = out / "G-persist.graph.db"
        carry = out / "M-notes.carry.json"
        if carry.is_file():
            rec["notes_carry"] = json.loads(carry.read_text())
        rec["notes_usage"] = notes_usage(out / "M-notes.traj.json")
        notes_file = out / "M-notes.memory" / "NOTES.md"
        rec["notes_bytes_after_task"] = notes_file.stat().st_size if notes_file.is_file() else None
        per_task.append(rec)

    clusters = np.array([t["repo"] for t in tasks])
    thirds = np.array([t["third"] for t in tasks])
    report: dict = {
        "n": n,
        "arms": ARMS,
        "seed": SEED,
        "bootstrap": N_BOOT,
        "repos": {repo: int((clusters == repo).sum()) for repo in sorted(set(clusters.tolist()))},
        "thirds": {k: int((thirds == k).sum()) for k in THIRDS},
        "third_rule": "third = floor((position-1)*3/n_tasks); tablib 3/3/3, tenacity 4/3/3",
    }

    # ---- per-arm table
    table = {}
    for arm in ARMS:
        r = metric["resolved"][arm]
        c = metric["calls"][arm]
        tk = metric["tokens"][arm]
        cost = metric["cost"][arm]
        billed = metric["cost_billed"][arm]
        table[arm] = {
            "cost_billed_median_usd": round(float(np.nanmedian(billed)), 6),
            "cost_billed_total_usd": round(float(np.nansum(billed)), 4),
            "billed_over_primary": round(float(np.nansum(billed) / np.nansum(cost)), 3) if np.nansum(cost) else None,
            "resolved": int(np.nansum(r)),
            "resolve_rate": round(float(np.nanmean(r)), 4),
            "calls_median": float(np.nanmedian(c)),
            "calls_mean": round(float(np.nanmean(c)), 2),
            "calls_total": int(np.nansum(c)),
            "tokens_median": int(np.nanmedian(tk)),
            "tokens_mean": int(np.nanmean(tk)),
            "tokens_total": int(np.nansum(tk)),
            "cost_median_usd": round(float(np.nanmedian(cost)), 6),
            "cost_total_usd": round(float(np.nansum(cost)), 4),
            "by_repo": {
                repo: {
                    "resolved": int(np.nansum(r[clusters == repo])),
                    "n": int((clusters == repo).sum()),
                }
                for repo in report["repos"]
            },
        }
    report["table"] = table
    report["cost_note"] = (
        "cost = primary formula at frozen prices (input 4.2e-8, cache read 9e-9, output 6e-7 USD/token) "
        "for every arm; cost_billed = OpenRouter's billed cost. The ledger's cost_usd for M and "
        "M-notes was the billed figure, which is why the first draft of this report showed "
        "Graphirm cheaper than mini. On the primary formula it is not."
    )

    # ---- H5 forward transfer: pairs × metrics × (overall, thirds)
    h5: dict = {}
    for a, b in PAIRS:
        key = f"{a} - {b}"
        h5[key] = {}
        for m in METRICS:
            binary = m == "resolved"
            entry = {"overall": summarize_pair(metric[m][a], metric[m][b], clusters, rng, binary)}
            for third in THIRDS:
                sel = thirds == third
                entry[third] = summarize_pair(
                    metric[m][a][sel], metric[m][b][sel], clusters[sel], rng, binary
                )
            h5[key][m] = entry
    report["H5"] = h5
    report["cluster_ci_note"] = (
        "Repo-cluster bootstrap over 2 clusters has three distinct resamples "
        "(tablib×2, tenacity×2, one of each); its interval is not informative and is "
        "shown only because it was registered. Read the task-level paired CI."
    )

    # ---- position curves (resolve rate per third per arm)
    report["by_third"] = {
        arm: {
            third: {
                "resolved": int(np.nansum(metric["resolved"][arm][thirds == third])),
                "n": int((thirds == third).sum()),
                "cost_median_usd": round(float(np.nanmedian(metric["cost"][arm][thirds == third])), 6),
                "tokens_median": int(np.nanmedian(metric["tokens"][arm][thirds == third])),
            }
            for third in THIRDS
        }
        for arm in ARMS
    }

    # ---- memory state: graph_in sizes and notes carry
    report["memory"] = {
        "G-persist_graph_in_bytes": {
            t["iid"]: latest[f"{t['iid']}::G-persist"].get("graph_in_bytes") for t in tasks
        },
        "M-notes_carry": {
            t["iid"]: (per_task[i].get("notes_carry") or {}) for i, t in enumerate(tasks)
        },
    }

    report["memory"]["M-notes_usage"] = {
        "tasks": n,
        "tasks_that_read_notes": sum(1 for t in per_task if t["notes_usage"]["read"]),
        "tasks_that_wrote_notes": sum(1 for t in per_task if t["notes_usage"]["wrote"]),
        "max_notes_bytes_after_any_task": max((t["notes_bytes_after_task"] or 0) for t in per_task),
        "note": "NOTES.md stayed empty for all 19 tasks: the agent read it when told it may, "
        "and never wrote to it. M-notes therefore carried no information in T3 and differs "
        "from M only by one prompt line.",
    }

    # ---- retry sensitivity, three rules as in T1
    retry_types = {arm: Counter() for arm in ARMS}
    first_attempt = {arm: np.zeros(n, dtype=int) for arm in ARMS}
    wall_only = {arm: np.zeros(n, dtype=int) for arm in ARMS}
    flips_first, flips_wall = [], []
    provider_kinds = {"config", "harness", "provider_429_or_overload", "provider_connection", "llm_call_timeout"}
    for i, iid in enumerate(ids):
        for arm in ARMS:
            attempts = by_key[f"{iid}::{arm}"]
            real = [r for r in attempts if classify_retry(r.get("error") or "") not in ("config", "harness")]
            for r in real[:-1]:
                if r.get("error"):
                    retry_types[arm][classify_retry(r["error"])] += 1
            first = real[0] if real else attempts[0]
            if first.get("error") and not first.get("request_caused"):
                first_attempt[arm][i] = 0
                if final[iid][arm]:
                    flips_first.append({"task": iid, "arm": arm, "first_error": classify_retry(first["error"])})
            else:
                first_attempt[arm][i] = 1 if final[iid][arm] else 0
            agent_attempts = [
                r for r in attempts
                if classify_retry(r.get("error") or "") not in provider_kinds or not r.get("error")
            ]
            firstw = agent_attempts[0] if agent_attempts else attempts[-1]
            if firstw.get("error") and not firstw.get("request_caused"):
                wall_only[arm][i] = 0
                if final[iid][arm]:
                    flips_wall.append({"task": iid, "arm": arm, "first_error": classify_retry(firstw["error"])})
            else:
                wall_only[arm][i] = 1 if final[iid][arm] else 0
    non_agent_rows = [
        {"key": r["instance_id"], "class": classify_retry(r.get("error") or ""), "error": (r.get("error") or "")[:160]}
        for r in rows
        if r.get("error")
    ]
    rng2 = np.random.default_rng(SEED + 1)
    sens: dict = {
        "rows_with_error": non_agent_rows,
        "retries_by_type": {arm: dict(retry_types[arm]) for arm in ARMS},
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
            }
    sens["conclusions_change"] = any(
        sens["first_attempt_table"][arm] != table[arm]["resolved"]
        or sens["wall_clock_table"][arm] != table[arm]["resolved"]
        for arm in ARMS
    )
    report["sensitivity"] = sens

    (OUT / "t3_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "t3_per_task.json").write_text(json.dumps(per_task, indent=2) + "\n")
    (OUT / "t3_report.md").write_text(render_md(report))
    print(f"wrote {OUT / 't3_report.md'}")


# ------------------------------------------------------------------ rendering


def fmt_ci(ci, nd=2):
    return f"[{ci[0]:,.{nd}f}, {ci[1]:,.{nd}f}]"


def render_md(r: dict) -> str:
    L = []
    L.append("# T3 memory track — H5 (pre-registered)\n")
    L.append(f"n = {r['n']} tasks, repos {r['repos']}, thirds {r['thirds']} ({r['third_rule']}). ")
    L.append(f"Paired bootstrap {r['bootstrap']} resamples, seed {r['seed']}.\n")
    L.append("## Per-arm table\n")
    L.append(r["cost_note"] + "\n")
    L.append("| arm | resolved | rate | calls median (mean) | tokens median | tokens total | cost median | cost total | billed total | tablib | tenacity |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in r["arms"]:
        t = r["table"][arm]
        br = t["by_repo"]
        tab = br.get("jazzband/tablib", {})
        ten = br.get("jd/tenacity", {})
        L.append(
            f"| {arm} | {t['resolved']}/{r['n']} | {t['resolve_rate']:.3f} | {t['calls_median']:.0f} ({t['calls_mean']}) | "
            f"{t['tokens_median']:,} | {t['tokens_total']:,} | ${t['cost_median_usd']:.4f} | ${t['cost_total_usd']:.4f} | "
            f"${t['cost_billed_total_usd']:.4f} ({t['billed_over_primary']}x) | "
            f"{tab.get('resolved')}/{tab.get('n')} | {ten.get('resolved')}/{ten.get('n')} |"
        )
    L.append("")
    L.append("## H5 forward transfer (a − b, paired by task)\n")
    L.append(r["cluster_ci_note"] + "\n")
    for pair, metrics in r["H5"].items():
        L.append(f"### {pair}\n")
        L.append("| metric | slice | n | diff | task-level 95% CI | cluster 95% CI | W/T/L or medians |")
        L.append("|---|---|---|---|---|---|---|")
        for m, slices in metrics.items():
            for sl, e in slices.items():
                if "diff" not in e:
                    L.append(f"| {m} | {sl} | {e.get('n', 0)} | – | – | – | – |")
                    continue
                nd = {"resolved": 2, "calls": 2, "tokens": 0, "cost": 4, "cost_billed": 4}[m]
                if m == "resolved":
                    extra = f"{e['win']}/{e['tie']}/{e['loss']}"
                    unit = "pp"
                else:
                    extra = f"{e['median_a']:,.{nd}f} vs {e['median_b']:,.{nd}f} (ratio {e['ratio_of_medians']})"
                    unit = ""
                L.append(
                    f"| {m} | {sl} | {e['n']} | {e['diff']:+,.{nd}f}{unit} | {fmt_ci(e['task_ci95'], nd)} | "
                    f"{fmt_ci(e['cluster_ci95'], nd)} | {extra} |"
                )
        L.append("")
    L.append("## Resolve rate by sequence third\n")
    L.append("| arm | first | middle | last |")
    L.append("|---|---|---|---|")
    for arm in r["arms"]:
        b = r["by_third"][arm]
        L.append(f"| {arm} | " + " | ".join(f"{b[k]['resolved']}/{b[k]['n']}" for k in THIRDS) + " |")
    L.append("")
    L.append("## Retry sensitivity (three rules)\n")
    s = r["sensitivity"]
    L.append(f"Ledger rows with an error: {len(s['rows_with_error'])}; retries by type: {s['retries_by_type']}.")
    L.append(f"Resolved per arm — all retries: { {a: r['table'][a]['resolved'] for a in r['arms']} }; "
             f"first attempt: {s['first_attempt_table']}; wall-clock counts: {s['wall_clock_table']}.")
    L.append(f"Flips (first attempt): {s['flips_first_attempt']}; flips (wall clock): {s['flips_wall_clock']}.")
    L.append(f"Any conclusion changes across rules: {s['conclusions_change']}.\n")
    L.append("## Memory state\n")
    L.append("G-persist graph carried in (bytes) by task: " + json.dumps(r["memory"]["G-persist_graph_in_bytes"]))
    L.append("")
    mu = r["memory"]["M-notes_usage"]
    L.append(f"M-notes: {mu['tasks_that_read_notes']}/{mu['tasks']} tasks read /memory/NOTES.md, "
             f"{mu['tasks_that_wrote_notes']}/{mu['tasks']} wrote it, largest NOTES.md after any task "
             f"{mu['max_notes_bytes_after_any_task']} bytes. {mu['note']}")
    L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
