"""Shared SWE-bench arm helpers.

Loads the repo `.env` itself. Does not read a pre-exported shell for the key.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = re.compile(r"(^|/)tests?(/|$)")
TEST_FILE = re.compile(r"(^|/)(test_.+\.py|.+_test\.py)$")


def load_dotenv(path: Path | None = None) -> None:
    """Set missing environment variables from a `.env` file."""
    env_path = path or (REPO_ROOT / ".env")
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def render_task(problem_statement: str, template: Path | None = None) -> str:
    """Fill the shared issue template. Both arms must use this text."""
    path = template or (Path(__file__).resolve().parent / "adapters" / "task_template.md")
    text = path.read_text()
    return text.replace("{{problem_statement}}", problem_statement.rstrip()) + "\n"


def render_notes_task(problem_statement: str) -> str:
    """Shared issue text plus the one line that tells M-notes about NOTES.md."""
    path = Path(__file__).resolve().parent / "adapters" / "notes_task.md"
    text = path.read_text()
    return text.replace("{{problem_statement}}", problem_statement.rstrip()) + "\n"


def _touched_paths(test_patch: str) -> set[str]:
    paths: set[str] = set()
    for line in test_patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4 and parts[3].startswith("b/"):
                paths.add(parts[3][2:])
        elif line.startswith("+++ b/"):
            paths.add(line[6:])
    return paths


def is_test_path(path: str, test_patch_files: set[str] | None = None) -> bool:
    name = path.replace("\\", "/")
    if TEST_DIR.search(name) or TEST_FILE.search(name):
        return True
    return name in (test_patch_files or set())


def strip_test_hunks(patch: str, test_patch: str = "") -> str:
    """Drop test-file hunks. Use this on every arm's model_patch."""
    banned = _touched_paths(test_patch)
    if not patch.strip():
        return ""
    chunks = re.split(r"(?=^diff --git )", patch, flags=re.M)
    kept = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        path = ""
        for line in chunk.splitlines():
            if line.startswith("diff --git "):
                parts = line.split()
                if len(parts) >= 4 and parts[3].startswith("b/"):
                    path = parts[3][2:]
                break
        if path and is_test_path(path, banned):
            continue
        kept.append(chunk if chunk.endswith("\n") else chunk + "\n")
    return "".join(kept)


def main() -> None:
    load_dotenv()
    raise SystemExit(
        "run_arm.py loads .env and owns strip_test_hunks. "
        "It does not launch an arm yet."
    )


if __name__ == "__main__":
    main()
