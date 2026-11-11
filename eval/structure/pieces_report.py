#!/usr/bin/env python3
"""Per-session numbers from one or more graph.db files (read-only).

For every session: model, status, workspace, assistant turns, user messages, tokens and cost, and for
the assistant replies with text how they were cut (`pieces_source`), whether the cut failed and how many
citations resolved. Prints a table and the totals P-0002's "tuned" bar is stated in. With `--tsv` it
prints one machine-read row per session for the P-0008 session log instead (the columns a person cannot
fill in: task, how it was chosen, solved, stepped in, friction, are added by hand).

    python3 eval/structure/pieces_report.py ~/.graphirm/graph.db [more.db ...]
    python3 eval/structure/pieces_report.py --tsv [--session ID_PREFIX] ~/.graphirm/graph.db
"""
import argparse
import json
import sqlite3
from collections import defaultdict

SOURCES = ("html", "markdown", "over_cap", "failed", "none")
TSV_COLUMNS = (
    "session", "model", "ran", "status", "workspace", "turns", "user_msgs", "input_tokens", "output_tokens",
    "cost_usd", "replies_with_text", "html", "markdown", "over_cap", "failed", "none",
    "cite_ok", "cite_bad",
)


def sessions_in(path):
    """Yield (session id, summary dict) for each session in one database."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    agents = {}
    for nid, data in con.execute("SELECT id, data FROM nodes WHERE node_type = 'agent'"):
        data = json.loads(data)
        agents[nid] = {"model": data.get("model", ""), "status": data.get("status", "")}
    for nid, meta in con.execute("SELECT id, metadata FROM nodes WHERE node_type = 'agent'"):
        agents[nid]["workspace"] = json.loads(meta).get("workspace", "")
    sessions = defaultdict(lambda: defaultdict(int, ran=defaultdict(int)))
    rows = con.execute(
        "SELECT json_extract(metadata,'$.session_id'), data, metadata FROM nodes "
        "WHERE node_type = 'interaction'"
    )
    for session, data, meta in rows:
        data, meta = json.loads(data), json.loads(meta)
        s = sessions[session]
        if data.get("role") == "user":
            s["user_msgs"] += 1
            continue
        if data.get("role") != "assistant":
            continue
        s["turns"] += 1
        if meta.get("model_selected"):
            s["ran"][meta["model_selected"]] += 1
        s["input_tokens"] += meta.get("usage_input") or 0
        s["output_tokens"] += meta.get("usage_output") or 0
        s["cost_micro"] += meta.get("usage_cost_micro") or 0
        s["costed"] += "usage_cost_micro" in meta
        s["resolved"] += meta.get("citations_resolved") or 0
        s["unresolved"] += meta.get("citations_unresolved") or 0
        if (data.get("content") or "").strip() or "pieces_source" in meta:
            s["replies"] += 1  # a tool-call-only turn with no text is not a reply
            s[meta.get("pieces_source", "none")] += 1
    for session, s in sessions.items():
        s["info"] = agents.get(session, {})
        yield session, s


def collect(paths, prefix=None):
    out = []
    for path in paths:
        for session, s in sessions_in(path):
            if session and (prefix is None or str(session).startswith(prefix)):
                out.append((path, session, s))
    return sorted(out, key=lambda r: (str(r[0]), str(r[1])))


def ran(s):
    """The model(s) the replies actually ran on: routing can differ from the agent node's model."""
    return ",".join(sorted(s["ran"])) or "unknown"


def tsv_row(session, s):
    info = s["info"]
    values = (
        session, info.get("model", ""), ran(s), info.get("status", ""), info.get("workspace", ""),
        s["turns"], s["user_msgs"], s["input_tokens"], s["output_tokens"],
        f"{s['cost_micro'] / 1e6:.4f}" if s["costed"] else "", s["replies"], *(s[k] for k in SOURCES),
        s["resolved"], s["unresolved"],
    )
    return "\t".join(str(v) for v in values)


def cost(s):
    return f"{s['cost_micro'] / 1e6:.4f}" if s["costed"] else "n/a"


def table(found):
    print(f"{'session':10} {'turns':>5} {'replies':>7} " + " ".join(f"{k:>8}" for k in SOURCES)
          + f" {'cite ok':>8} {'cite bad':>8} {'cost $':>8}")
    total = defaultdict(int)
    for _path, session, s in found:
        print(f"{str(session)[:8]:10} {s['turns']:>5} {s['replies']:>7} "
              + " ".join(f"{s[k]:>8}" for k in SOURCES)
              + f" {s['resolved']:>8} {s['unresolved']:>8} {cost(s):>8}")
        for k, v in s.items():
            if k not in ("info", "ran"):
                total[k] += v
    n = total["replies"]
    cites = total["resolved"] + total["unresolved"]
    print(f"\n{len(found)} sessions, {n} replies with text")
    if n:
        print(f"cut as html: {total['html'] / n:.0%}; failed: {total['failed']}; "
              f"cost per reply: ${total['cost_micro'] / 1e6 / n:.4f}")
    print("citations resolved: "
          + (f"{total['resolved'] / cites:.0%} of {cites}" if cites else "no citations yet"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("db", nargs="+")
    ap.add_argument("--tsv", action="store_true", help="one machine-read row per session")
    ap.add_argument("--session", help="only sessions whose id starts with this")
    args = ap.parse_args(argv)
    found = collect(args.db, args.session)
    if args.tsv:
        print("\t".join(TSV_COLUMNS))
        for _path, session, s in found:
            print(tsv_row(session, s))
    else:
        table(found)


if __name__ == "__main__":
    main()
