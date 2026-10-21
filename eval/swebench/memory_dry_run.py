"""One mock task per memory arm. No paid model calls.

G-fresh and G-persist use `graphirm run --model mock/patch` on the debug
binary. M and M-notes have no mock provider in mini-swe-agent, so this
script checks the notes carry itself: M leaves no file, M-notes copies
NOTES.md forward and drops the oldest characters past 4k tokens.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from adapters.memory_notes import carry, token_count

ROOT = Path(__file__).resolve().parents[2]
DEBUG_BIN = ROOT / "target" / "debug" / "graphirm"
CONFIG = ROOT / "eval" / "swebench" / "adapters" / "eval-agent.toml"
OUT = Path("/tmp/memory-dry")


def graphirm(task: str, cwd: Path, graph: Path) -> None:
    cwd.mkdir(parents=True, exist_ok=True)
    task_path = cwd / "task.md"
    task_path.write_text(task)
    graph.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(DEBUG_BIN),
            "run",
            "--task",
            str(task_path),
            "--cwd",
            str(cwd),
            "--graph",
            str(graph),
            "--config",
            str(CONFIG),
            "--model",
            "mock/patch",
            "--reply-format",
            "md",
            "--max-turns",
            "2",
        ],
        check=True,
    )


def task_texts(graph: Path) -> list[str]:
    conn = sqlite3.connect(graph)
    rows = conn.execute(
        "SELECT data, metadata FROM nodes WHERE node_type = 'interaction'"
    ).fetchall()
    conn.close()
    texts = []
    for data, metadata in rows:
        body = json.loads(data)
        meta = json.loads(metadata)
        if body.get("role") == "user" and meta.get("task") is True:
            texts.append(body["content"])
    return texts


def agent_count(graph: Path) -> int:
    conn = sqlite3.connect(graph)
    count = conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE node_type = 'agent'"
    ).fetchone()[0]
    conn.close()
    return int(count)


def run_graph_arms() -> dict:
    fresh_a = OUT / "g-fresh" / "a" / "graph.db"
    fresh_b = OUT / "g-fresh" / "b" / "graph.db"
    persist = OUT / "g-persist" / "graph.db"
    graphirm("TASK-ONE persist me", OUT / "g-fresh" / "a", fresh_a)
    graphirm("TASK-TWO only", OUT / "g-fresh" / "b", fresh_b)
    graphirm("TASK-ONE persist me", OUT / "g-persist" / "t1", persist)
    graphirm("TASK-TWO current", OUT / "g-persist" / "t2", persist)
    fresh_texts = task_texts(fresh_b)
    persist_texts = task_texts(persist)
    return {
        "G-fresh": {
            "agents_on_second_graph": agent_count(fresh_b),
            "second_graph_has_task_one": any("TASK-ONE" in text for text in fresh_texts),
            "second_graph_has_task_two": any("TASK-TWO" in text for text in fresh_texts),
        },
        "G-persist": {
            "agents": agent_count(persist),
            "has_task_one": any("TASK-ONE" in text for text in persist_texts),
            "has_task_two": any("TASK-TWO" in text for text in persist_texts),
            "marked_tasks": len(persist_texts),
        },
    }


def run_notes_arms() -> dict:
    host = OUT / "notes"
    host.mkdir(parents=True, exist_ok=True)
    absent = host / "m" / "NOTES.md"
    assert not absent.exists()

    carried = host / "between" / "NOTES.md"
    carried.parent.mkdir(parents=True, exist_ok=True)
    old = "OLD " * 8000
    new = "\nNEW-TAIL keeps this line\n"
    carried.write_text(old + new)
    mounted = host / "next" / "NOTES.md"
    log = carry(carried, mounted)
    text = mounted.read_text()
    return {
        "M": {"notes_present": absent.exists()},
        "M-notes": {
            **log,
            "tokens_after_at_most_4000": token_count(text) <= 4000,
            "keeps_newest": "NEW-TAIL" in text,
            "drops_oldest": not text.startswith("OLD"),
        },
    }


def main() -> int:
    if not DEBUG_BIN.is_file():
        print(f"missing {DEBUG_BIN}", file=sys.stderr)
        return 1
    report = {"graph": run_graph_arms(), "notes": run_notes_arms()}
    dest = OUT / "report.json"
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(dest.read_text())
    graph = report["graph"]
    notes = report["notes"]
    ok = (
        graph["G-fresh"]["agents_on_second_graph"] == 1
        and graph["G-fresh"]["second_graph_has_task_one"] is False
        and graph["G-fresh"]["second_graph_has_task_two"] is True
        and graph["G-persist"]["agents"] == 1
        and graph["G-persist"]["has_task_one"] is True
        and graph["G-persist"]["has_task_two"] is True
        and graph["G-persist"]["marked_tasks"] == 2
        and notes["M"]["notes_present"] is False
        and notes["M-notes"]["truncated"] is True
        and notes["M-notes"]["keeps_newest"] is True
        and notes["M-notes"]["drops_oldest"] is True
        and notes["M-notes"]["tokens_after_at_most_4000"] is True
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
