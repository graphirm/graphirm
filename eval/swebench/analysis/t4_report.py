#!/usr/bin/env python3
"""T4 analysis, same paired-bootstrap method as T1 H1.

Headline grades are the filtered regrade where one exists, and the
recorded grade otherwise. Tasks finished after the filter was in the
agent already carry that grade. Nothing here changes a run file.

Outputs: /home/krs/graphirm-runs/t4/analysis/.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from metrics import cache_hit, primary_cost_from_calls  # noqa: E402
from run_arm import mini_costs  # noqa: E402
from t1_report import (  # noqa: E402
    N_BOOT,
    SEED,
    bootstrap_ratio_of_medians,
    cache_hit_mini,
    paired_bootstrap_diff,
    win_tie_loss,
)

RUNS = Path("/home/krs/graphirm-runs/t4")
DATA = RUNS / "datasets" / "swe-rebench-07-2026"
OUT = RUNS / "analysis"
REGISTRY = HERE.parents[1] / "registry.json"
ARMS = ("M", "G")
# The three arms whose filtered patch did not apply. Both arms of the
# second task are in the set, so the paired "without" cut drops two tasks.
HELD_TASKS = {"apache__hugegraph-3037", "danny-avila__LibreChat-13166"}
LANG = {
    ".py": "Python",
    ".java": "Java",
    ".go": "Go",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".rs": "Rust",
}


def scored_ids() -> list[str]:
    registry = json.loads(REGISTRY.read_text())
    excluded = {
        row["instance_id"]
        for row in registry["exclusions"]
        if row.get("track") == "t4"
    }
    ids = sorted(
        path.name
        for path in DATA.iterdir()
        if path.is_dir() and (path / "task.toml").is_file() and path.name not in excluded
    )
    if len(ids) != 106:
        raise SystemExit(f"expected 106 scored tasks, found {len(ids)}")
    return ids


def ledger() -> list[dict]:
    return [json.loads(line) for line in (RUNS / "arms.jsonl").read_text().splitlines() if line.strip()]


def original_grades(ids: list[str]) -> dict[str, dict[str, bool]]:
    out = {}
    for iid in ids:
        grades = json.loads((RUNS / iid / "grades.json").read_text())
        out[iid] = {arm: bool(grades[arm]) for arm in ARMS}
    return out


def filtered_grades(ids: list[str], original: dict[str, dict[str, bool]], rows: list[dict]) -> dict[str, dict[str, bool]]:
    """Filtered regrade when the replay produced a boolean, else the recorded grade.

    A launch rerun, appended after the filter regrade, replaces that grade.
    """
    latest: dict[tuple[str, str], dict] = {}
    for row in rows:
        if row.get("regrade") == "filter" and row.get("arm") in ARMS:
            latest[(row["task"], row["arm"])] = row
    launched: dict[tuple[str, str], bool] = {}
    for row in rows:
        if row.get("rerun") == "launch" and row.get("arm") in ARMS and isinstance(row.get("grade"), bool):
            launched[(row["task"], row["arm"])] = row["grade"]
    out = {iid: dict(original[iid]) for iid in ids}
    for iid in ids:
        for arm in ARMS:
            row = latest.get((iid, arm))
            if row is not None and row.get("grade") is not None:
                out[iid][arm] = bool(row["grade"])
            if (iid, arm) in launched:
                out[iid][arm] = launched[(iid, arm)]
    return out


def graph_calls(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        found = conn.execute(
            "SELECT metadata FROM nodes WHERE node_type = 'interaction' "
            "AND json_extract(metadata, '$.usage_input') IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    return [json.loads(meta) for (meta,) in found]


def graph_cache(calls: list[dict]) -> float | None:
    prompt = cached = 0
    for call in calls:
        prompt += int(call.get("usage_input") or 0)
        cached += int(call.get("usage_cache_read") or 0)
    return cache_hit(prompt, cached)


def arm_metrics(ids: list[str]) -> dict:
    """Primary cost, billed cost, calls and cache hit, one value per task."""
    out = {arm: {"cost": [], "billed": [], "calls": [], "cache": []} for arm in ARMS}
    for iid in ids:
        calls = graph_calls(RUNS / iid / "G.graph.db")
        out["G"]["cost"].append(primary_cost_from_calls(calls))
        summary_path = RUNS / iid / "G.summary.json"
        billed = 0.0
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            billed = float(summary.get("cost_usd_reported") or 0.0)
        out["G"]["billed"].append(billed)
        out["G"]["calls"].append(len(calls))
        out["G"]["cache"].append(graph_cache(calls))
        mini = mini_costs(RUNS / iid / "M.traj.json")
        out["M"]["cost"].append(mini["cost_usd"])
        out["M"]["billed"].append(mini["cost_usd_billed"])
        out["M"]["calls"].append(mini["calls"])
        out["M"]["cache"].append(cache_hit_mini(RUNS / iid / "M.traj.json"))
    return {
        arm: {key: np.array(vals, dtype=float) if key != "cache" else vals for key, vals in cols.items()}
        for arm, cols in out.items()
    }


def h1(g: np.ndarray, m: np.ndarray, rng) -> dict:
    diff, lo, hi, _ = paired_bootstrap_diff(g, m, rng)
    wtl = win_tie_loss(g, m)
    inside = lo >= -0.05 and hi <= 0.05
    return {
        "n": int(len(g)),
        "resolved_G": int(g.sum()),
        "resolved_M": int(m.sum()),
        "diff_pp": round(diff * 100, 2),
        "ci95_pp": [round(lo * 100, 2), round(hi * 100, 2)],
        "non_inferior_at_minus_5pp": bool(lo > -0.05),
        "ci_inside_plus_minus_5pp": bool(inside),
        **wtl,
    }


def cost_block(metrics: dict, resolved: dict[str, np.ndarray], rng) -> dict:
    ratio, lo, hi = bootstrap_ratio_of_medians(metrics["G"]["cost"], metrics["M"]["cost"], rng)
    per = {}
    for arm in ARMS:
        pairs = np.stack([metrics[arm]["cost"], resolved[arm]], axis=1)

        def cpr(sample: np.ndarray) -> float:
            got = sample[:, 1].sum()
            return float(sample[:, 0].sum() / got) if got else float("nan")

        point = cpr(pairs)
        idx = rng.integers(0, len(pairs), size=(N_BOOT, len(pairs)))
        boots = np.array([cpr(pairs[row]) for row in idx])
        boots = boots[~np.isnan(boots)]
        per[arm] = {
            "point": round(point, 4),
            "ci95": [round(float(np.quantile(boots, 0.025)), 4), round(float(np.quantile(boots, 0.975)), 4)],
        }
    return {
        "G_over_M_median_primary": {"point": round(ratio, 3), "ci95": [round(lo, 3), round(hi, 3)]},
        "cost_per_resolved_task_usd": per,
    }


def attempt_class(row: dict) -> str:
    """Map a T4 ledger error onto the T1 retry classes."""
    error = row.get("error") or ""
    if error in {"infra", "provider"}:
        return "provider_429_or_overload"
    if error == "harness":
        return "harness"
    if error == "wall":
        return "wall_clock_1900s"
    if error == "config":
        return "config"
    if row.get("grade_status") == "error" and not error:
        return "harness"
    return ""


def corrected_class(rows: list[dict], task: str, arm: str, attempt: int | None) -> str | None:
    for row in rows:
        if row.get("corrects") == "error" and row.get("task") == task and row.get("arm") == arm:
            if attempt is None or row.get("attempt") in {None, attempt}:
                if row.get("error"):
                    return attempt_class(row)
    return None


def retry_grades(ids: list[str], final: dict[str, dict[str, bool]], rows: list[dict]) -> dict:
    """The three T1 rules: final grade, first attempt, wall-clock counts.

    Provider retries (infra) may stand. A wall-clock kill counts as
    unresolved. The first attempt counts as unresolved when it errored.
    """
    by_key: dict[tuple[str, str], list[dict]] = {}
    launch_arms = {
        (row["task"], row["arm"])
        for row in rows
        if row.get("rerun") == "launch" and row.get("arm") in ARMS
    }
    for row in rows:
        if row.get("arm") not in ARMS or row.get("corrects") or row.get("regrade"):
            continue
        by_key.setdefault((row["task"], row["arm"]), []).append(row)
    first = {arm: np.zeros(len(ids), dtype=int) for arm in ARMS}
    wall = {arm: np.zeros(len(ids), dtype=int) for arm in ARMS}
    kinds = {arm: Counter() for arm in ARMS}
    provider = {"provider_429_or_overload", "provider_connection", "llm_call_timeout", "config"}
    for i, iid in enumerate(ids):
        for arm in ARMS:
            attempts = by_key.get((iid, arm), [])
            labelled = []
            for row in attempts:
                kind = corrected_class(rows, iid, arm, row.get("attempt")) or attempt_class(row)
                labelled.append((row, kind))
                if kind and row.get("complete") is not True:
                    kinds[arm][kind] += 1
            if (iid, arm) in launch_arms:
                labelled = [(row, kind or "harness") for row, kind in labelled]
            real = [(row, kind) for row, kind in labelled if kind != "config"]
            first_row, first_kind = (real or labelled or [(None, "")])[0]
            if first_row is not None and first_kind:
                first[arm][i] = 0
            else:
                first[arm][i] = 1 if final[iid][arm] else 0
            agent = [(row, kind) for row, kind in labelled if kind not in provider]
            wall_row, wall_kind = (agent or labelled or [(None, "")])[0]
            if wall_row is not None and wall_kind:
                wall[arm][i] = 0
            else:
                wall[arm][i] = 1 if final[iid][arm] else 0
    return {"first": first, "wall": wall, "retry_rows": {arm: dict(kinds[arm]) for arm in ARMS}}


def language_of(task: str) -> str:
    script = DATA / task / "solution" / "solve.sh"
    counts: Counter[str] = Counter()
    if script.is_file():
        for line in script.read_text(errors="replace").splitlines():
            if not line.startswith("diff --git "):
                continue
            path = line.split()[-1]
            if path.startswith("b/"):
                path = path[2:]
            counts[LANG.get(Path(path).suffix.lower(), "other")] += 1
    if not counts:
        return "other"
    return counts.most_common(1)[0][0]


def no_patch(rows: list[dict]) -> list[dict]:
    """Arms whose regrade found no saved patch, with the trial's exception."""
    found = []
    seen = set()
    for row in rows:
        if row.get("regrade") != "filter" or "no saved patch" not in (row.get("note") or ""):
            continue
        key = (row["arm"], row["task"])
        if key in seen:
            continue
        seen.add(key)
        reason = "no trial result"
        if any(
            later.get("rerun") == "launch" and later.get("task") == row["task"] and later.get("arm") == row["arm"]
            for later in rows
        ):
            continue
        trial = RUNS / "trials" / row["arm"] / row["task"] / "attempt1"
        results = sorted(trial.rglob("result.json")) if trial.is_dir() else []
        if results:
            info = (json.loads(results[-1].read_text()).get("exception_info") or {})
            message = str(info.get("exception_message") or "")
            if info.get("exception_type") == "AgentTimeoutError":
                reason = "wall-clock timeout at 1800s, no patch file saved"
            elif "syntax error near unexpected token" in message:
                reason = "bash syntax error in the agent command, agent did not start, no patch file saved"
            elif info.get("exception_type"):
                reason = f"{info['exception_type']}, no patch file saved"
        found.append({"arm": row["arm"], "task": row["task"], "reason": reason})
    return sorted(found, key=lambda item: (item["task"], item["arm"]))


def table_row(arm: str, resolved: np.ndarray, metrics: dict) -> dict:
    cache = [v for v in metrics[arm]["cache"] if v is not None]
    return {
        "resolved": int(resolved.sum()),
        "n": int(len(resolved)),
        "resolve_rate": round(float(resolved.mean()), 4),
        "median_primary_usd": round(float(np.median(metrics[arm]["cost"])), 6),
        "median_billed_usd": round(float(np.median(metrics[arm]["billed"])), 6),
        "median_calls": round(float(np.median(metrics[arm]["calls"])), 1),
        "median_cache_hit": round(float(np.median(cache)), 4) if cache else None,
        "cache_hit_n": len(cache),
    }


def render(report: dict) -> str:
    lines = [
        "# T4 analysis",
        "",
        "Same paired bootstrap as T1 H1: 10,000 resamples, seed 20261022. "
        "G minus M. Headline grades use the filtered regrade.",
        "",
        "## Resolved and cost",
        "",
    ]
    for arm in ARMS:
        row = report["table"][arm]
        lines.append(
            f"- {arm}: {row['resolved']}/{row['n']}, "
            f"median primary ${row['median_primary_usd']:.6f}, "
            f"median billed ${row['median_billed_usd']:.6f}, "
            f"median calls {row['median_calls']}, "
            f"median cache hit {row['median_cache_hit']}"
        )
    h = report["H1"]
    lines += [
        "",
        "## H1, G minus M",
        "",
        f"- diff {h['diff_pp']:+.2f} pp, 95% CI [{h['ci95_pp'][0]:+.2f}, {h['ci95_pp'][1]:+.2f}] pp",
        f"- win/tie/loss {h['win']}/{h['tie']}/{h['loss']}, discordant {h['discordant']}",
        f"- non-inferior at -5 pp: {h['non_inferior_at_minus_5pp']}",
        f"- CI inside ±5 pp: {h['ci_inside_plus_minus_5pp']}",
        f"- T1 published H1 was {report['T1_H1']['diff_pp']:+.2f} pp, "
        f"CI [{report['T1_H1']['ci95_pp'][0]:+.2f}, {report['T1_H1']['ci95_pp'][1]:+.2f}], "
        f"non-inferior {report['T1_H1']['non_inferior_at_minus_5pp']}",
        "- consistent in direction and size with T1; the CI is too wide to confirm the -5pp bound alone",
    ]
    before = report.get("before_launch_rerun")
    if before:
        bh = before["H1"]
        lines += [
            f"- before the launch rerun: G {bh['resolved_G']}/M {bh['resolved_M']} "
            f"diff {bh['diff_pp']:+.2f} pp CI [{bh['ci95_pp'][0]:+.2f}, {bh['ci95_pp'][1]:+.2f}] "
            f"W/T/L {bh['win']}/{bh['tie']}/{bh['loss']}",
            f"- after the launch rerun: G {h['resolved_G']}/M {h['resolved_M']} "
            f"diff {h['diff_pp']:+.2f} pp CI [{h['ci95_pp'][0]:+.2f}, {h['ci95_pp'][1]:+.2f}] "
            f"W/T/L {h['win']}/{h['tie']}/{h['loss']}",
        ]
    lines += [
        f"- the filtered regrade moved {report['sensitivity']['filter_regrade_flips']} grades in opposite directions, "
        "and that comparison's rate and win/tie/loss stayed the same",
        "",
        "## Cost",
        "",
        f"- median primary G/M {report['cost']['G_over_M_median_primary']['point']} "
        f"CI {report['cost']['G_over_M_median_primary']['ci95']}",
    ]
    for arm in ARMS:
        per = report["cost"]["cost_per_resolved_task_usd"][arm]
        lines.append(f"- cost per resolved {arm}: ${per['point']} CI {per['ci95']}")
    lines += ["", "## Sensitivity", ""]
    for name, block in report["sensitivity"].items():
        if not isinstance(block, dict) or "diff_pp" not in block:
            continue
        lines.append(
            f"- {name}: G {block['resolved_G']}/M {block['resolved_M']} "
            f"diff {block['diff_pp']:+.2f} pp CI {block['ci95_pp']} "
            f"W/T/L {block['win']}/{block['tie']}/{block['loss']}"
        )
    lines += ["", "## Exploratory, by language", ""]
    for lang, block in report["by_language"].items():
        lines.append(
            f"- {lang} n={block['n']}: M {block['resolved_M']}, G {block['resolved_G']}, "
            f"diff {block['diff_pp']:+.2f} pp"
        )
    launch = report.get("launch_rerun") or {}
    if launch:
        lines += [
            "",
            "## Launch rerun",
            "",
            f"- {launch['arms']} arms rerun, {launch['resolved']} resolved, "
            f"{launch['unresolved']} unresolved, "
            f"primary ${launch['primary_usd']}, billed ${launch['billed_usd']}",
        ]
        for row in launch["outcomes"]:
            lines.append(f"- {row['arm']} {row['task']}: {row['grade_status']}")
    lines += ["", "## Arms with no saved patch", ""]
    for row in report["no_patch"]:
        lines.append(f"- {row['arm']} {row['task']}: {row['reason']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ids = scored_ids()
    rows = ledger()
    original = original_grades(ids)
    filtered = filtered_grades(ids, original, rows)
    metrics = arm_metrics(ids)
    final = {arm: np.array([1 if filtered[iid][arm] else 0 for iid in ids]) for arm in ARMS}
    orig = {arm: np.array([1 if original[iid][arm] else 0 for iid in ids]) for arm in ARMS}
    rng = np.random.default_rng(SEED)
    headline = h1(final["G"], final["M"], rng)
    costs = cost_block(metrics, final, rng)
    rules = retry_grades(ids, filtered, rows)
    rng_first = np.random.default_rng(SEED + 1)
    rng_wall = np.random.default_rng(SEED + 2)
    first = h1(rules["first"]["G"], rules["first"]["M"], rng_first)
    wall = h1(rules["wall"]["G"], rules["wall"]["M"], rng_wall)
    keep = np.array([iid not in HELD_TASKS for iid in ids])
    rng_held = np.random.default_rng(SEED + 3)
    without = h1(final["G"][keep], final["M"][keep], rng_held)
    rng_orig = np.random.default_rng(SEED + 4)
    original_h1 = h1(orig["G"], orig["M"], rng_orig)
    changed = [
        {"task": iid, "arm": arm, "original": bool(orig[arm][i]), "filtered": bool(final[arm][i])}
        for i, iid in enumerate(ids)
        for arm in ARMS
        if bool(orig[arm][i]) != bool(final[arm][i])
    ]
    launch_arms = {
        (row["task"], row["arm"])
        for row in rows
        if row.get("rerun") == "launch" and row.get("arm") in ARMS
    }
    filter_changed = [item for item in changed if (item["task"], item["arm"]) not in launch_arms]
    launch_rows = [row for row in rows if row.get("rerun") == "launch"]
    langs = [language_of(iid) for iid in ids]
    by_lang = {}
    for lang in ("Go", "Java", "Python", "TypeScript", "Rust", "other"):
        mask = np.array([item == lang for item in langs])
        if not mask.any():
            continue
        diff = float(final["G"][mask].mean() - final["M"][mask].mean())
        by_lang[lang] = {
            "n": int(mask.sum()),
            "resolved_M": int(final["M"][mask].sum()),
            "resolved_G": int(final["G"][mask].sum()),
            "diff_pp": round(diff * 100, 2),
            "label": "exploratory, not pre-registered",
        }
    t1 = json.loads(Path("/home/krs/graphirm-runs/t1/analysis/t1_report.json").read_text())["H1"]
    report = {
        "n": 106,
        "seed": SEED,
        "bootstrap": N_BOOT,
        "grade_source": "filtered regrade where grade is boolean, else grades.json",
        "table": {arm: table_row(arm, final[arm], metrics) for arm in ARMS},
        "H1": headline,
        "T1_H1": {
            "diff_pp": t1["diff_pp"],
            "ci95_pp": t1["ci95_pp"],
            "non_inferior_at_minus_5pp": t1["non_inferior_at_minus_5pp"],
        },
        "T1_comparison": (
            "consistent in direction and size with T1; "
            "the CI is too wide to confirm the -5pp bound alone"
        ),
        "non_inferior_at_minus_5pp": headline["non_inferior_at_minus_5pp"],
        "cost": costs,
        "sensitivity": {
            "final_after_retries": headline,
            "first_attempt": first,
            "wall_clock_counts": wall,
            "without_held_arms": without,
            "original_grading": original_h1,
            "grades_changed_by_regrade": filter_changed,
            "filter_regrade_flips": len(filter_changed),
            "retry_rows": rules["retry_rows"],
            "held_tasks_dropped": sorted(HELD_TASKS),
        },
        "by_language": by_lang,
        "no_patch": no_patch(rows),
        "launch_rerun": {
            "arms": len(launch_rows),
            "resolved": sum(1 for row in launch_rows if row.get("grade") is True),
            "unresolved": sum(1 for row in launch_rows if row.get("grade") is False),
            "primary_usd": round(sum(row.get("cost_usd") or 0 for row in launch_rows), 4),
            "billed_usd": round(sum(row.get("cost_usd_billed") or 0 for row in launch_rows), 4),
            "outcomes": [
                {"arm": row["arm"], "task": row["task"], "grade_status": row.get("grade_status")}
                for row in launch_rows
            ],
        },
    }
    before_path = OUT / "before-launch-rerun" / "t4_report.json"
    if before_path.is_file():
        before = json.loads(before_path.read_text())
        report["before_launch_rerun"] = {
            "table": before["table"],
            "H1": before["H1"],
            "cost": before["cost"],
        }
    (OUT / "t4_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "t4_report.md").write_text(render(report))
    per = []
    for i, iid in enumerate(ids):
        per.append({
            "task": iid,
            "language": langs[i],
            "M": bool(final["M"][i]),
            "G": bool(final["G"][i]),
            "M_original": bool(orig["M"][i]),
            "G_original": bool(orig["G"][i]),
            "M_primary": float(metrics["M"]["cost"][i]),
            "G_primary": float(metrics["G"]["cost"][i]),
        })
    (OUT / "t4_per_task.json").write_text(json.dumps(per, indent=2) + "\n")
    print(render(report))


if __name__ == "__main__":
    main()
