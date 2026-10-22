"""Frozen eval settings from prereg.md.

Environment variables must not change the model, the reasoning effort,
or the provider pin. A mismatch is invalid_config and is not scored.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sqlite3
from pathlib import Path

GRAPHIRM_MODEL = "openrouter/deepseek/deepseek-v4.1-flash"
SUMMARY_MODEL = "deepseek/deepseek-v4.1-flash"
MINI_MODEL = "deepseek/deepseek-v4.1-flash"
REASONING = "high"
PROVIDER = "morph/fp8"
CONFIG_HASH = "f791cc2803a1219088716944750698551dd6bd9bb38bbacc76a975e567086487"
CONFIG_PATH = Path(__file__).resolve().parent / "adapters" / "eval-agent.toml"
GRAPH_ARMS = ("G", "G-md", "G-b16", "G-b16-recency")
TRACK_TASKS = 210


def toml_problem() -> str | None:
    if not CONFIG_PATH.is_file():
        return f"missing {CONFIG_PATH}"
    text = CONFIG_PATH.read_text()
    if 'openrouter_provider = "morph/fp8"' not in text:
        return "eval-agent.toml provider pin is not morph/fp8"
    digest = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    if digest != CONFIG_HASH:
        return f"config hash {digest} does not match prereg"
    return None


def _model_ok(value: object) -> bool:
    return value in {SUMMARY_MODEL, GRAPHIRM_MODEL, MINI_MODEL}


def graphirm_problem(summary: dict | None, metadata: dict | None) -> str | None:
    src = metadata or summary
    if not src:
        return "missing run_metadata"
    if not _model_ok(src.get("model")):
        return f"model {src.get('model')}"
    if src.get("reasoning_effort") != REASONING:
        return f"reasoning {src.get('reasoning_effort')}"
    if src.get("config_hash") != CONFIG_HASH:
        return "config_hash does not match prereg"
    return None


def mini_problem(model_cfg: dict | None) -> str | None:
    if not model_cfg:
        return "missing trajectory config"
    if model_cfg.get("model_name") != MINI_MODEL:
        return f"model {model_cfg.get('model_name')}"
    kwargs = model_cfg.get("model_kwargs") or {}
    effort = (kwargs.get("reasoning") or {}).get("effort")
    if effort != REASONING:
        return f"reasoning {effort}"
    provider = kwargs.get("provider") or {}
    if provider.get("only") != [PROVIDER] or provider.get("allow_fallbacks") is not False:
        return f"provider {provider.get('only')}"
    return None


_STATUS_404 = re.compile(
    r"(?:OpenRouter|HTTP|status(?:_code)?|code|Error)\W{0,4}404\b"  # "OpenRouter 404 Not Found: ...", "status_code=404", '"code": 404'
    r"|\b404 Not Found\b"
    r"|NotFoundError",  # litellm / openai exception class
    re.IGNORECASE,
)


def provider_error(text: str) -> bool:
    """A provider-side refusal: OpenRouter has no allowed provider for the pinned
    model, or the model endpoint returned 404. Matches the status in its error
    context only; a bare "404" substring also appears inside container ids and
    node UUIDs in log tails and must not count (T1 preflight, T2 first start)."""
    lowered = text.lower()
    return (
        "no allowed providers" in lowered
        or bool(_STATUS_404.search(text))
        or "provider error" in lowered
    )


def read_run_metadata(graph_path: Path) -> dict | None:
    if not graph_path.is_file() or graph_path.stat().st_size == 0:
        return None
    try:
        conn = sqlite3.connect(f"file:{graph_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute(
            "SELECT data FROM nodes WHERE data LIKE '%run_metadata%' LIMIT 1"
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    if not row:
        return None
    try:
        data = json.loads(row[0])
        body = data.get("body")
        if isinstance(body, str):
            return json.loads(body)
    except json.JSONDecodeError:
        return None
    return None


def arm_problem(arm: str, out: Path, error_text: str = "") -> str | None:
    patch_path = out / f"{arm}.patch"
    patch = patch_path.read_text() if patch_path.is_file() else ""
    if arm == "M":
        traj_path = out / "M.traj.json"
        model_cfg = None
        if traj_path.is_file():
            try:
                traj = json.loads(traj_path.read_text())
                model_cfg = traj["info"]["config"]["model"]
            except (json.JSONDecodeError, KeyError, TypeError):
                model_cfg = None
        problem = mini_problem(model_cfg) if model_cfg is not None else None
        if model_cfg is None and provider_error(error_text):
            return "provider error"
        if problem:
            return problem
    else:
        summary = None
        summary_path = out / f"{arm}.summary.json"
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text())
            except json.JSONDecodeError:
                summary = None
        metadata = read_run_metadata(out / f"{arm}.graph.db")
        if summary is None and metadata is None:
            if provider_error(error_text):
                return "provider error"
            return None
        problem = graphirm_problem(summary, metadata)
        if problem:
            return problem
    if not patch.strip() and provider_error(error_text):
        return "empty patch from provider error"
    return None


def gzip_size(path: Path) -> int:
    return len(gzip.compress(path.read_bytes(), mtime=0))


def graph_size_report(root: Path, task_ids: list[str]) -> dict:
    sizes: list[int] = []
    for iid in task_ids:
        for arm in GRAPH_ARMS:
            db = root / iid / f"{arm}.graph.db"
            if db.is_file() and db.stat().st_size > 0:
                sizes.append(gzip_size(db))
    sizes.sort()
    total = sum(sizes)
    median = sizes[len(sizes) // 2] if sizes else 0
    per_task = (total / len(task_ids)) if task_ids else 0
    return {
        "tasks": len(task_ids),
        "graphs": len(sizes),
        "median_gz": median,
        "total_gz": total,
        "track_gz": int(per_task * TRACK_TASKS),
    }
