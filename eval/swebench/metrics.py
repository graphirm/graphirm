"""The one definition of each eval metric.

Prices come from the registry's price file after its sha256 is checked.
Prompt tokens already include cached tokens, so the uncached count is
prompt minus cached. Cache hit is cached divided by prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
import hashlib

REGISTRY = Path(__file__).resolve().parents[1] / "registry.json"


def prices() -> dict:
    registry = json.loads(REGISTRY.read_text())
    entry = registry["prices"]
    path = Path(__file__).resolve().parents[2] / entry["path"]
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != entry["sha256"]:
        raise SystemExit(f"prices file {path} sha256 {digest} != registry {entry['sha256']}")
    return json.loads(data)


def primary_cost(prompt: int, cached: int, output: int, table: dict | None = None) -> float:
    """Uncached prompt times input price, plus cached times cache price, plus output times output price."""
    table = table or prices()
    uncached = prompt - cached
    return (
        uncached * table["input_cost_per_token"]
        + cached * table["input_cache_read_per_token"]
        + output * table["output_cost_per_token"]
    )


def primary_cost_micro(prompt: int, cached: int, output: int, table: dict | None = None) -> int:
    """primary_cost rounded to the nearest millionth of a dollar.

    The binary does this on every call, then sums. Cached tokens are capped
    at the prompt size, which is what the binary does.
    """
    cached = min(cached, prompt)
    usd = primary_cost(prompt, cached, output, table)
    if usd <= 0.0:
        return 0
    return int(usd * 1_000_000 + 0.5)


def primary_cost_from_calls(calls: list[dict], table: dict | None = None) -> float:
    """Sum of per-call primary_cost_micro values, in dollars.

    A Graphirm summary's cost_usd is this sum for the calls of that run.
    """
    table = table or prices()
    micros = 0
    for call in calls:
        micros += primary_cost_micro(
            int(call.get("usage_input") or 0),
            int(call.get("usage_cache_read") or 0),
            int(call.get("usage_output") or 0),
            table,
        )
    return micros / 1_000_000


def recorded_primary(summary: dict) -> float:
    """A Graphirm summary's cost_usd, the sum of primary_cost_micro for that run."""
    return float(summary.get("cost_usd") or 0.0)


def recorded_billed(summary: dict) -> float:
    """OpenRouter's bill as stored on a run summary."""
    return float(summary.get("cost_usd_reported") or summary.get("cost_usd_billed") or 0.0)


def billed_cost(usage: dict) -> float:
    """OpenRouter's usage.cost. This is the bill, not the primary cost."""
    return float(usage.get("cost") or 0.0)


def cache_hit(prompt: int, cached: int) -> float | None:
    """Cached tokens divided by prompt tokens. Prompt tokens already include the cached ones."""
    if prompt <= 0:
        return None
    return cached / prompt


def tokens(prompt: int, output: int) -> int:
    """Prompt tokens plus completion tokens."""
    return prompt + output
