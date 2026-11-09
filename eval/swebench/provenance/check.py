"""Check segment source citations against a Graphirm graph.

Reads Content nodes that stored `segment_item_src`. For each citation:

- exists: a `tool:<node_id>` node is in the graph; a `file:` citation has a
  read, write, or edit tool node for that path, or a bash view command
  (`cat`, `sed`, `grep`, `head`, and the same kind of command) whose command
  or output contains the path; a `web:` citation is not a graph check.
- type_match: a tool citation is a tool-result interaction, and a file
  citation's node used read, write, or edit, or a bash view, on that path.

Writes a CSV. The `supports` column is left blank for a person to fill.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

FILE_TOOLS = {"read", "write", "edit"}
VIEW_COMMANDS = {"cat", "sed", "grep", "head", "tail", "less", "more", "nl", "awk", "rg", "bat"}
STRICT_COMMANDS = {"cat", "sed", "head", "less"}


def _nodes(conn: sqlite3.Connection) -> list[dict]:
    rows = []
    for node_id, node_type, data, metadata in conn.execute(
        "SELECT id, node_type, data, metadata FROM nodes"
    ):
        rows.append(
            {
                "id": node_id,
                "node_type": node_type,
                "data": json.loads(data) if data else {},
                "metadata": json.loads(metadata) if metadata else {},
            }
        )
    return rows


def _is_tool(node: dict) -> bool:
    if node["metadata"].get("tool_name"):
        return True
    data = node["data"]
    if isinstance(data, dict) and data.get("role") == "tool":
        return True
    return False


def _path_forms(path: str) -> list[str]:
    wanted = path.lstrip("./")
    forms = [wanted]
    for prefix in ("/testbed/", "testbed/"):
        if wanted.startswith(prefix):
            forms.append(wanted[len(prefix) :])
    return [form for form in forms if form]


def _mentions_path(text: str, path: str) -> bool:
    for form in _path_forms(path):
        if form in text or text.rstrip("/").endswith(form):
            return True
    return False


def _bash_commands(nodes: list[dict]) -> dict[str, str]:
    """Map a tool-call id to the bash command string."""
    commands: dict[str, str] = {}
    for node in nodes:
        for call in node["metadata"].get("tool_calls") or []:
            if not isinstance(call, dict) or call.get("name") != "bash":
                continue
            arguments = call.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"command": arguments}
            command = str(arguments.get("command") or "")
            call_id = str(call.get("id") or "")
            if call_id and command:
                commands[call_id] = command
    return commands


def _is_view_command(command: str) -> bool:
    tokens = []
    token = []
    for char in command:
        if char.isalnum() or char in {"_", "-"}:
            token.append(char)
        elif token:
            tokens.append("".join(token))
            token = []
    if token:
        tokens.append("".join(token))
    return any(token in VIEW_COMMANDS for token in tokens)


def _bash_views_path(nodes: list[dict], path: str) -> bool:
    """True when cat, sed, grep, head, or a similar command shows this path.

    A directory listing that only names the file does not count.
    """
    commands = _bash_commands(nodes)
    for node in nodes:
        if node["metadata"].get("tool_name") != "bash":
            continue
        call_id = str(node["metadata"].get("tool_call_id") or "")
        command = commands.get(call_id, "")
        if not _is_view_command(command):
            continue
        output = ""
        data = node["data"]
        if isinstance(data, dict):
            output = str(data.get("content") or "")
        if _mentions_path(command, path) or _mentions_path(output, path):
            return True
    return False


def _file_nodes(nodes: list[dict], path: str) -> list[dict]:
    wanted = path.lstrip("./")
    found = []
    for node in nodes:
        if node["metadata"].get("tool_name") not in FILE_TOOLS:
            continue
        recorded = str(node["metadata"].get("path") or "").lstrip("./")
        if recorded == wanted or recorded.endswith("/" + wanted):
            found.append(node)
    return found


def _tool_nodes_for(cite: str, nodes: list[dict]) -> list[dict]:
    exact = [node for node in nodes if node["id"] == cite and _is_tool(node)]
    if exact:
        return exact
    prefix = cite.replace("-", "")
    if len(prefix) < 8:
        return []
    hits = []
    for node in nodes:
        if not _is_tool(node):
            continue
        compact = node["id"].replace("-", "")
        if compact.startswith(prefix):
            hits.append(node)
    if len(hits) == 1:
        return hits
    return []


def _bash_named(nodes: list[dict], path: str, commands: set[str] | None, command_only: bool) -> bool:
    """Lenient: any bash command or its output names the path.

    Strict: the command itself is cat, sed, head, or less and names the file.
    """
    bash_commands = _bash_commands(nodes)
    for node in nodes:
        if node["metadata"].get("tool_name") != "bash":
            continue
        call_id = str(node["metadata"].get("tool_call_id") or "")
        command = bash_commands.get(call_id, "")
        if commands is not None and not any(token in commands for token in _command_tokens(command)):
            continue
        output = ""
        data = node["data"]
        if isinstance(data, dict):
            output = str(data.get("content") or "")
        if _mentions_path(command, path):
            return True
        if not command_only and _mentions_path(output, path):
            return True
    return False


def _command_tokens(command: str) -> list[str]:
    tokens = []
    token = []
    for char in command:
        if char.isalnum() or char in {"_", "-"}:
            token.append(char)
        elif token:
            tokens.append("".join(token))
            token = []
    if token:
        tokens.append("".join(token))
    return tokens


def _file_matches(nodes: list[dict], path: str, rule: str) -> bool:
    if _file_nodes(nodes, path):
        return True
    if rule == "strict":
        return _bash_named(nodes, path, STRICT_COMMANDS, command_only=True)
    return _bash_named(nodes, path, None, command_only=False)


def check_src(src: str, nodes: list[dict]) -> tuple[str, str]:
    """Widened rule kept for the CSV: file tools, or a bash view of the path."""
    if src.startswith("tool:"):
        matches = _tool_nodes_for(src.removeprefix("tool:"), nodes)
        if len(matches) != 1:
            return "no", "no"
        return "yes", "yes"
    if src.startswith("file:"):
        path = src.removeprefix("file:")
        if _file_nodes(nodes, path) or _bash_views_path(nodes, path):
            return "yes", "yes"
        return "no", "no"
    if src.startswith("web:"):
        return "n/a", "n/a"
    return "no", "no"


def rule_match(src: str, nodes: list[dict], rule: str) -> str:
    if src.startswith("tool:"):
        matches = _tool_nodes_for(src.removeprefix("tool:"), nodes)
        return "yes" if len(matches) == 1 else "no"
    if src.startswith("file:"):
        return "yes" if _file_matches(nodes, src.removeprefix("file:"), rule) else "no"
    if src.startswith("web:"):
        return "n/a"
    return "no"


def claims_from_graph(nodes: list[dict]) -> tuple[list[dict], dict]:
    claims = []
    items_total = 0
    items_with_src = 0
    for node in nodes:
        meta = node["metadata"]
        items = meta.get("segment_items") or []
        src_rows = meta.get("segment_item_src") or []
        if not items and not src_rows:
            continue
        for index, item in enumerate(items):
            items_total += 1
            sources = src_rows[index] if index < len(src_rows) else []
            if isinstance(sources, str):
                sources = [sources]
            sources = [str(src) for src in sources if src]
            if sources:
                items_with_src += 1
            for src in sources:
                exists, type_match = check_src(src, nodes)
                claims.append(
                    {
                        "node_id": node["id"],
                        "item": item,
                        "src": src,
                        "exists": exists,
                        "type_match": type_match,
                        "supports": "",
                    }
                )
    summary = {
        "items": items_total,
        "items_with_src": items_with_src,
        "citations": len(claims),
    }
    return claims, summary


def write_csv(claims: list[dict], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["node_id", "item", "src", "exists", "type_match", "supports"],
        )
        writer.writeheader()
        writer.writerows(claims)


def summarize(graph: Path) -> dict:
    conn = sqlite3.connect(f"file:{graph}?mode=ro", uri=True)
    try:
        nodes = _nodes(conn)
    finally:
        conn.close()
    claims, summary = claims_from_graph(nodes)
    strict = 0
    lenient = 0
    for row in claims:
        src = row["src"]
        if rule_match(src, nodes, "strict") == "yes":
            strict += 1
        if rule_match(src, nodes, "lenient") == "yes":
            lenient += 1
    summary.update(
        {
            "strict_match": strict,
            "lenient_match": lenient,
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    conn = sqlite3.connect(args.graph)
    try:
        claims, summary = claims_from_graph(_nodes(conn))
    finally:
        conn.close()
    write_csv(claims, Path(args.out))
    exists = sum(1 for row in claims if row["exists"] == "yes")
    typed = sum(1 for row in claims if row["type_match"] == "yes")
    both = summarize(Path(args.graph))
    summary.update(
        {
            "exists_yes": exists,
            "type_match_yes": typed,
            "strict_match": both["strict_match"],
            "lenient_match": both["lenient_match"],
            "out": args.out,
        }
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
