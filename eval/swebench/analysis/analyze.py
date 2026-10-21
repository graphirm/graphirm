"""Primary cost uses the provider's cached and uncached prompt counts.

`cost_usd` is uncached times the input rate, plus cached times the
cache-read rate, plus output times the output rate. `cost_usd_list`
bills every prompt token at the input rate. `cost_usd_reported` is the
provider figure. A row is flagged when primary and reported differ by
more than 5 percent. After 2026-10-19 that flag is expected: Morph's live
prompt and cache-read prices fell below prices.lock.json, while primary
cost stays on the frozen rates.

Graphirm's `--max-cost-usd` reads this table when `--prices` points at
`prices.lock.json`. mini-swe-agent's `cost_limit` reads
`litellm_model_registry.json`, which this script checks against the lock.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
LOCK = HERE / "prices.lock.json"
REGISTRY = HERE / "litellm_model_registry.json"
FLAG = 0.05


def load_prices(path: Path = LOCK) -> dict:
    return json.loads(path.read_text())


def registry_matches_lock(prices: dict | None = None, registry_path: Path = REGISTRY) -> None:
    prices = prices or load_prices()
    registry = json.loads(registry_path.read_text())
    row = registry[prices["litellm_model"]]
    if row["input_cost_per_token"] != prices["input_cost_per_token"]:
        raise SystemExit("liteLLM input price does not match prices.lock.json")
    if row["output_cost_per_token"] != prices["output_cost_per_token"]:
        raise SystemExit("liteLLM output price does not match prices.lock.json")
    if row.get("cache_read_input_token_cost") != prices["input_cache_read_per_token"]:
        raise SystemExit(
            "liteLLM cache reads must use input_cache_read_per_token so the cost limit matches the primary formula"
        )


def real_usd(
    tokens_in: int, cache_read: int, tokens_out: int, prices: dict | None = None
) -> float:
    prices = prices or load_prices()
    cached = min(max(cache_read, 0), max(tokens_in, 0))
    uncached = max(tokens_in, 0) - cached
    return (
        uncached * prices["input_cost_per_token"]
        + cached * prices["input_cache_read_per_token"]
        + tokens_out * prices["output_cost_per_token"]
    )


def list_usd(tokens_in: int, tokens_out: int, prices: dict | None = None) -> float:
    prices = prices or load_prices()
    return tokens_in * prices["input_cost_per_token"] + tokens_out * prices["output_cost_per_token"]


def diverges(frozen: float, reported: float | None) -> bool:
    if reported is None:
        return True
    base = abs(reported) if reported else abs(frozen)
    if base == 0:
        return False
    return abs(frozen - reported) / base > FLAG


def score_row(row: dict, prices: dict | None = None) -> dict:
    prices = prices or load_prices()
    tokens_in = int(row["tokens_in"])
    tokens_out = int(row["tokens_out"])
    cache_read = int(row.get("cache_read_tokens") or 0)
    primary = real_usd(tokens_in, cache_read, tokens_out, prices)
    listed = list_usd(tokens_in, tokens_out, prices)
    reported = row.get("cost_usd_reported")
    reported_f = None if reported is None else float(reported)
    return {
        **row,
        "cost_usd": primary,
        "cost_usd_list": listed,
        "cost_usd_reported": reported_f,
        "cost_diverges": diverges(primary, reported_f),
    }


def main() -> None:
    registry_matches_lock()
    raise SystemExit(
        "pass rows to score_row(); this file checks that the liteLLM registry matches the lock"
    )


if __name__ == "__main__":
    main()
