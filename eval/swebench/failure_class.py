"""Split a failed eval arm into infrastructure or a request the provider rejected.

Infrastructure is retried after a wait. A request-caused failure is unresolved
and is not retried. A config failure, such as a 404 for the wrong model, stops
the driver and is not retried by the watchdog.
"""

from __future__ import annotations

import re
from pathlib import Path

INFRASTRUCTURE = "infrastructure"
REQUEST = "request"
CONFIG = "config"
OTHER = "other"


def _has_status(low: str, code: str) -> bool:
    return re.search(rf"(?<![0-9]){code}(?![0-9])", low) is not None


def classify(text: str) -> str:
    low = text.lower()
    if any(
        marker in low
        for marker in (
            "context length",
            "context_length",
            "maximum context",
            "too many tokens",
            "malformed",
            "invalid_request",
            "validation error",
        )
    ) or _has_status(low, "400"):
        return REQUEST
    if _has_status(low, "404") or "no allowed providers" in low:
        return CONFIG
    if any(
        marker in low
        for marker in (
            "service_overloaded",
            "upstream_provider_shared_pool",
            "error sending request",
            "timed out",
            "timeout",
            "connection reset",
            "connection refused",
            "temporarily unavailable",
        )
    ) or any(_has_status(low, code) for code in ("429", "500", "502", "503", "504")):
        return INFRASTRUCTURE
    if "provider error" in low:
        return INFRASTRUCTURE
    return OTHER


def classify_arm(out: Path, arm: str, error: str) -> str:
    parts = [error or ""]
    retries = out / f"{arm}.retries.log"
    if retries.is_file():
        parts.append(retries.read_text(errors="replace")[-4000:])
    return classify("\n".join(parts))
