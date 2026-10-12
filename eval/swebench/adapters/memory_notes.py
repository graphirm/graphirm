"""Carry a NOTES.md file between tasks in one repository.

The cap is 4_000 tokens at the Phase C fitted factor of 3.8 characters
per token. When the file is over the cap, the oldest characters go first.
"""

from __future__ import annotations

from pathlib import Path

TOKEN_CAP = 4_000
CHARS_PER_TOKEN = 3.8


def token_count(text: str) -> int:
    if not text:
        return 0
    return int(len(text) / CHARS_PER_TOKEN)


def truncate_oldest(text: str, cap: int = TOKEN_CAP) -> str:
    """Keep the newest characters that fit in `cap` tokens."""
    if token_count(text) <= cap:
        return text
    keep = int(cap * CHARS_PER_TOKEN)
    return text[-keep:]


def carry(src: Path, dest: Path, cap: int = TOKEN_CAP) -> dict:
    """Copy `src` to `dest`, truncating oldest content first. Returns a size log."""
    raw = src.read_text() if src.is_file() else ""
    kept = truncate_oldest(raw, cap)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(kept)
    return {
        "tokens_before": token_count(raw),
        "tokens_after": token_count(kept),
        "chars_before": len(raw),
        "chars_after": len(kept),
        "truncated": kept != raw,
    }
