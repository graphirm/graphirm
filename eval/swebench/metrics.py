"""The one definition of each eval metric.

Prices come from the registry's price file after its sha256 is checked.
Prompt tokens already include cached tokens, so the uncached count is
prompt minus cached. Cache hit is cached divided by prompt.
"""

from __future__ import annotations

import json
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[1] / "registry.json"


def prices() -> dict:
    registry = json.loads(REGISTRY.read_text())
    entry = registry["prices"]
    path = Path(__file__).resolve().parents[2] / entry["path"]
    data = path.read_bytes()
    import hashlib

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
