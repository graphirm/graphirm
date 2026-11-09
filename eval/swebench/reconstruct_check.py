"""Rebuild packed context for 10 turns and diff it against the logged request.

Two modes:

- re-selection: `graphirm replay-pack --stateless` runs selection again with
  an empty already-sent list. A budget note on the last logged message is
  removed before the diff. Tool schemas are not part of the rebuild.
- sequential: the same replay, carrying each call's kept node ids into the
  next call as `pack_prefix_ids`.
- recorded: rebuild from the stored list of what was sent. The graph stores
  dropped thread-node ids, not that ordered list and not the budget note.
  The sent text then lives only in the external pack-trace file, and this
  mode compares that file to itself.
"""

from __future__ import annotations

import json
import os
import random
import re
import sqlite3
import subprocess
from pathlib import Path

SYSTEM_PROMPT = "Fix the failing test by editing product code. Do not edit tests."
REMINDER_MARK = "\n\nReply with only this JSON shape:"
BUDGET_NOTE = re.compile(r"\n\n\[Budget\] Token usage at \d+% of limit\.[\s\S]*$")
NATIVE = Path(__file__).resolve().parents[2] / "target/release/graphirm"
MUSL = (
    Path(__file__).resolve().parents[2]
    / "target/x86_64-unknown-linux-musl/release/graphirm"
)
CONFIG = Path(__file__).resolve().parent / "adapters/eval-agent.toml"


def sampled(instance_id: str, arm: str) -> bool:
    import hashlib

    digest = hashlib.sha256(f"{instance_id}\n{arm}".encode()).digest()
    return int.from_bytes(digest[:2], "big") % 100 < 5


def _binary() -> Path | None:
    if os.environ.get("GRAPHIRM_REPLAY_BIN"):
        path = Path(os.environ["GRAPHIRM_REPLAY_BIN"])
        if path.is_file():
            return path
    if NATIVE.is_file():
        return NATIVE
    if MUSL.is_file():
        return MUSL
    return None


def _logged_messages(blocks: list[dict]) -> str:
    text = "\n".join(block.get("text") or "" for block in blocks if block.get("kind") == "message")
    return BUDGET_NOTE.sub("", text)


def _rebuilt_messages(window: str, system: str = SYSTEM_PROMPT) -> str:
    if window.startswith(system):
        rest = window[len(system) :]
        if rest.startswith("\n"):
            rest = rest[1:]
        return rest
    return window


def budget_system(lines: list[dict]) -> str:
    """System text counted by the packer, before the post-pack JSON reminder."""
    for block in lines[0].get("blocks") or []:
        if block.get("kind") == "system":
            text = block.get("text") or ""
            if REMINDER_MARK in text:
                return text.split(REMINDER_MARK, 1)[0]
            return text
    return SYSTEM_PROMPT


def _config_for(arm: str, dest: Path, system: str | None = None) -> Path:
    text = CONFIG.read_text()
    if arm == "G-b16-recency" and "recency_only" not in text:
        text = text.replace("[agent]\n", "[agent]\nrecency_only = true\n", 1)
    if system is not None:
        start = text.index('system_prompt = "')
        end = text.index('"\n', start)
        text = text[:start] + "system_prompt = " + json.dumps(system) + text[end + 1 :]
    dest.write_text(text)
    return dest


def graph_selection_record(graph: Path) -> dict:
    """What each assistant turn stores about the pack."""
    uri = f"file:{graph}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    turns = 0
    dropped_nonempty = 0
    ordered = False
    budget = False
    fields: set[str] = set()
    try:
        rows = connection.execute(
            "SELECT data, metadata FROM nodes WHERE node_type = 'interaction'"
        )
        for data, metadata in rows:
            body = json.loads(data)
            if body.get("role") != "assistant":
                continue
            turns += 1
            record = json.loads(metadata)
            for key, value in record.items():
                if key.startswith("context_") or "budget" in key or "kept" in key or "pack" in key:
                    fields.add(key)
                if key in {
                    "context_kept_node_ids",
                    "kept_node_ids",
                    "pack_node_ids",
                    "packed_node_ids",
                    "context_sent_node_ids",
                } and isinstance(value, list) and value:
                    ordered = True
                if "budget" in key or (isinstance(value, str) and "[Budget]" in value):
                    budget = True
            dropped = record.get("context_dropped_node_ids") or []
            if isinstance(dropped, list) and dropped:
                dropped_nonempty += 1
    finally:
        connection.close()
    return {
        "assistant_turns": turns,
        "stores_ordered_ids": ordered,
        "stores_budget_note": budget,
        "dropped_turns_nonempty": dropped_nonempty,
        "context_fields": sorted(fields),
    }


def _sample_indexes(n_lines: int, n_windows: int, seed: str) -> list[int]:
    count = min(10, n_lines, n_windows)
    if count <= 0:
        return []
    return sorted(random.Random(seed).sample(range(min(n_lines, n_windows)), count))


def reselection_matches(
    lines: list[dict],
    windows: list[dict],
    indexes: list[int],
    system: str = SYSTEM_PROMPT,
) -> int:
    """Byte-equal turns after the budget note is removed. Selection is re-run."""
    equal = 0
    for index in indexes:
        logged = _logged_messages(lines[index].get("blocks") or [])
        rebuilt = _rebuilt_messages(windows[index].get("text") or "", system)
        if logged == rebuilt:
            equal += 1
    return equal


def recorded_matches(lines: list[dict], indexes: list[int], record: dict) -> dict:
    """Rebuild from a stored selection. The graph has no ordered id list."""
    if record.get("stores_ordered_ids"):
        return {
            "turns_checked": len(indexes),
            "byte_equal": None,
            "source": "graph",
            "independent": True,
            "reason": "ordered ids are stored; this checker does not render them yet",
        }
    equal = 0
    for index in indexes:
        blocks = lines[index].get("blocks") or []
        if _logged_messages(blocks) == _logged_messages(blocks):
            equal += 1
    return {
        "turns_checked": len(indexes),
        "byte_equal": equal,
        "source": "pack-trace.jsonl",
        "independent": False,
        "reason": "graph has no ordered sent-id list; the pack trace is the logged request",
    }


def _replay(
    graph: Path,
    trace: Path,
    arm: str,
    *,
    sequential: bool,
    system: str | None = None,
) -> tuple[int, list[dict]]:
    """Run replay-pack. Sequential carries kept ids forward; stateless does not."""
    binary = _binary()
    if binary is None:
        return 127, []
    suffix = ".sequential.csv" if sequential else ".replay.csv"
    out = trace.with_suffix(suffix)
    config = _config_for(arm, trace.with_suffix(".replay.toml"), system if sequential else None)
    budget = "16000" if arm in {"G-b16", "G-b16-recency"} else "100000"
    env = os.environ.copy()
    env["GRAPHIRM_REPLAY_DUMP"] = "1"
    command = [
        str(binary),
        "replay-pack",
        "--graph",
        str(graph),
        "--config",
        str(config),
        "--out",
        str(out),
        "--packing",
        "stable",
        "--context-max-tokens",
        budget,
    ]
    if not sequential:
        command.append("--stateless")
    proc = subprocess.run(
        command,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    windows_path = out.with_suffix(".windows.jsonl")
    windows = []
    if windows_path.is_file():
        windows = [json.loads(line) for line in windows_path.read_text().splitlines() if line.strip()]
    return proc.returncode, windows


def check(graph: Path, trace: Path, arm: str, seed: str) -> dict:
    if not trace.is_file():
        return {"turns_checked": 0, "byte_equal": None, "reason": "no pack trace"}
    lines = []
    for raw in trace.read_text().splitlines():
        if raw.strip():
            lines.append(json.loads(raw))
    if not lines:
        return {"turns_checked": 0, "byte_equal": None, "reason": "empty pack trace"}
    if _binary() is None:
        return {"turns_checked": 0, "byte_equal": None, "reason": "no graphirm binary"}
    code, windows = _replay(graph, trace, arm, sequential=False)
    if not windows:
        return {
            "turns_checked": 0,
            "byte_equal": None,
            "logged_calls": len(lines),
            "replay_exit": code,
            "reason": "replay did not dump window text",
        }
    system = budget_system(lines)
    _, sequential_windows = _replay(graph, trace, arm, sequential=True, system=system)
    indexes = _sample_indexes(len(lines), len(windows), seed)
    equal = reselection_matches(lines, windows, indexes)
    sequential_indexes = _sample_indexes(len(lines), len(sequential_windows), seed)
    sequential_equal = (
        reselection_matches(lines, sequential_windows, sequential_indexes, system)
        if sequential_windows
        else None
    )
    record = graph_selection_record(graph)
    recorded = recorded_matches(lines, indexes, record)
    return {
        "turns_checked": len(indexes),
        "byte_equal": equal,
        "logged_calls": len(lines),
        "replay_calls": len(windows),
        "reselection": {
            "turns_checked": len(indexes),
            "byte_equal": equal,
            "budget_note": "removed",
        },
        "sequential": {
            "turns_checked": len(sequential_indexes),
            "byte_equal": sequential_equal,
            "budget_note": "removed",
        },
        "recorded": recorded,
        "graph_record": record,
    }


def sequential_sample_rate(root: Path) -> tuple[int, int]:
    """Sequential byte-equal rate over every pack trace under root."""
    equal = 0
    checked = 0
    for trace in sorted(root.glob("*/*.pack-trace.jsonl")):
        arm = trace.name.split(".pack-trace.jsonl")[0]
        graph = trace.parent / f"{arm}.graph.db"
        if not graph.is_file():
            continue
        lines = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
        if not lines:
            continue
        system = budget_system(lines)
        _code, windows = _replay(graph, trace, arm, sequential=True, system=system)
        indexes = _sample_indexes(len(lines), len(windows), f"{trace.parent.name}::{arm}")
        if not indexes:
            continue
        equal += reselection_matches(lines, windows, indexes, system)
        checked += len(indexes)
    return equal, checked


def score_saved(graph: Path, trace: Path, seed: str) -> dict:
    """Score a finished run from its pack trace and an existing replay dump."""
    lines = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    windows_path = trace.with_suffix(".replay.windows.jsonl")
    windows = [
        json.loads(line) for line in windows_path.read_text().splitlines() if line.strip()
    ]
    indexes = _sample_indexes(len(lines), len(windows), seed)
    record = graph_selection_record(graph)
    return {
        "reselection_byte_equal": reselection_matches(lines, windows, indexes),
        "recorded": recorded_matches(lines, indexes, record),
        "graph_record": record,
        "turns_checked": len(indexes),
        "logged_calls": len(lines),
        "replay_calls": len(windows),
    }
