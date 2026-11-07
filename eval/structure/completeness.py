#!/usr/bin/env python3
"""Check finished Graphirm session databases for gaps in the record (step 0.4).

Reads each graph.db read-only and counts, per check, how many items the record
covers. Checks and thresholds: docs/plans/2026-11-07-structure-completeness.md.

    eval/structure/completeness.py --label t4 DB [DB ...]
    eval/structure/completeness.py --label t4 --glob '/runs/t4/*/G.graph.db'

Prints one JSON object: per-check totals for the corpus, whether each check
meets its threshold, and up to five example node ids per miss.
"""

import argparse
import glob
import json
import sqlite3
import sys

GATED_TOOLS = ("write", "edit", "bash")
FILE_TOOLS = ("write", "edit")
CHECKS = ("C1_reply_parts", "C2_tool_results", "C3_file_changes", "C4_approvals",
          "C5_produced_by_agent")
EXAMPLES = 5


def load(path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        nodes = {}
        for nid, ntype, data, meta in con.execute(
                "SELECT id, node_type, data, metadata FROM nodes"):
            nodes[nid] = {"type": ntype, "data": json.loads(data), "meta": json.loads(meta)}
        edges = [{"type": t, "src": s, "dst": d, "meta": json.loads(m)}
                 for t, s, d, m in con.execute(
                     "SELECT edge_type, source_id, target_id, metadata FROM edges")]
    finally:
        con.close()
    return nodes, edges


def role(node):
    return node["data"].get("role") if node["type"] == "interaction" else None


def tool_calls(node):
    calls = node["meta"].get("tool_calls") or []
    out = []
    for call in calls:
        args = call.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        out.append((call.get("id"), call.get("name"), args if isinstance(args, dict) else {}))
    return out


def path_matches(recorded, requested):
    if not recorded or not requested:
        return False
    requested = requested[2:] if requested.startswith("./") else requested
    return recorded == requested or recorded.endswith("/" + requested.lstrip("/"))


class Tally:
    def __init__(self):
        self.total = 0
        self.ok = 0
        self.misses = []

    def add(self, ok, node_id):
        self.total += 1
        if ok:
            self.ok += 1
        elif len(self.misses) < EXAMPLES:
            self.misses.append(node_id)


def check(nodes, edges):
    """Return {check: Tally} plus counts reported without a threshold."""
    out_edges, in_edges = {}, {}
    for e in edges:
        out_edges.setdefault(e["src"], []).append(e)
        in_edges.setdefault(e["dst"], []).append(e)
    results = {}
    for nid, node in nodes.items():
        if role(node) == "tool" and node["meta"].get("tool_call_id"):
            results[node["meta"]["tool_call_id"]] = nid

    tallies = {name: Tally() for name in CHECKS}
    info = {"dangling_edges": 0, "bash_calls": 0, "approvals": 0,
            "approvals_with_metadata": 0, "final_replies": 0, "final_replies_with_parts": 0}

    for e in edges:
        if e["src"] not in nodes or e["dst"] not in nodes:
            info["dangling_edges"] += 1
        if e["type"] == "approved_by":
            info["approvals"] += 1
            if e["meta"]:
                info["approvals_with_metadata"] += 1

    for nid, node in nodes.items():
        if node["type"] == "interaction":
            session = node["meta"].get("session_id")
            produced = any(e["type"] == "produces"
                           and nodes.get(e["src"], {}).get("type") == "agent"
                           and (session is None or e["src"] == session)
                           for e in in_edges.get(nid, []))
            tallies["C5_produced_by_agent"].add(produced, nid)
        if role(node) != "assistant":
            continue
        mine = out_edges.get(nid, [])
        calls = tool_calls(node)
        # Edges carry no call id, so each Modifies edge or rejection node
        # satisfies one call only: two edits of one file need two edges.
        used = set()
        if (node["data"].get("content") or "").strip():
            has_parts = any(e["type"] == "contains" for e in mine)
            tallies["C1_reply_parts"].add(has_parts or bool(node["meta"].get("tool_call_leak")), nid)
            if not calls:
                info["final_replies"] += 1
                info["final_replies_with_parts"] += has_parts
        for call_id, name, args in calls:
            result_id = results.get(call_id)
            tallies["C2_tool_results"].add(result_id is not None, nid)
            result = nodes.get(result_id)
            if name == "bash":
                info["bash_calls"] += 1
            if name in FILE_TOOLS and result and not result["meta"].get("is_error"):
                # A HITL Modify decision runs with new arguments; headless runs have none.
                edge = next((i for i, e in enumerate(mine) if i not in used
                             and e["type"] == "modifies"
                             and nodes.get(e["dst"], {}).get("data", {}).get("content_type") == "file"
                             and path_matches(nodes[e["dst"]]["data"].get("path"),
                                              args.get("path"))), None)
                if edge is not None:
                    used.add(edge)
                tallies["C3_file_changes"].add(edge is not None, nid)
            if name in GATED_TOOLS:
                tallies["C4_approvals"].add(
                    approved_or_rejected(result_id, name, mine, nodes, out_edges, used), nid)
    return tallies, info


def approved_or_rejected(result_id, name, mine, nodes, out_edges, used):
    if result_id is None:
        return False
    if any(e["type"] == "approved_by" for e in out_edges.get(result_id, [])):
        return True
    if not (nodes[result_id]["data"].get("content") or "").startswith("rejected by user:"):
        return False
    for i, e in enumerate(mine):
        target = nodes.get(e["dst"], {}).get("data", {})
        if (i not in used and e["type"] == "produces"
                and target.get("content_type") == "tool_rejection"
                and f"'{name}'" in (target.get("body") or "")
                and any(r["type"] == "rejected_by" for r in out_edges.get(e["dst"], []))):
            used.add(i)
            return True
    return False


def run(paths, label):
    pooled = {name: {"total": 0, "ok": 0, "examples": []} for name in CHECKS}
    info_total = {}
    for path in paths:
        nodes, edges = load(path)
        tallies, info = check(nodes, edges)
        for name, t in tallies.items():
            pooled[name]["total"] += t.total
            pooled[name]["ok"] += t.ok
            for miss in t.misses:
                if len(pooled[name]["examples"]) < EXAMPLES:
                    pooled[name]["examples"].append(f"{path}:{miss}")
        for k, v in info.items():
            info_total[k] = info_total.get(k, 0) + v
    for name, p in pooled.items():
        p["share"] = round(p["ok"] / p["total"], 4) if p["total"] else None
        p["passes"] = p["total"] == 0 or p["ok"] == p["total"]
    integrity = info_total.get("dangling_edges", 0) == 0
    return {
        "label": label,
        "databases": len(paths),
        "checks": pooled,
        "reported": info_total,
        "passes": integrity and all(p["passes"] for p in pooled.values()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", default="corpus")
    ap.add_argument("--glob", action="append", default=[])
    ap.add_argument("dbs", nargs="*")
    args = ap.parse_args(argv)
    paths = list(args.dbs)
    for pattern in args.glob:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        ap.error("no databases given")
    json.dump(run(paths, args.label), sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
