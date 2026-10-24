"""Shared SWE-bench arm helpers.

Loads the repo `.env` itself. Does not read a pre-exported shell for the key.
"""

from __future__ import annotations

import os
import json
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


def render_notes_task(problem_statement: str, template: str = "notes_task.md") -> str:
    """Issue text plus the M-notes instruction about /memory/NOTES.md.

    notes_task_v1.md is the one-line "you may read and update" text T3 ran
    with (0 of 19 tasks wrote a note). notes_task.md is the explicit T2 text.
    """
    path = Path(__file__).resolve().parent / "adapters" / template
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


def load_prices() -> dict:
    from metrics import prices

    return prices()


def usage_cost(usage: dict, prices: dict | None = None) -> float:
    """Primary cost of one call from an OpenRouter usage block (prereg §4)."""
    from metrics import primary_cost

    prices = prices or load_prices()
    prompt = int(usage.get("prompt_tokens") or 0)
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    output = int(usage.get("completion_tokens") or 0)
    return primary_cost(prompt, cached, output, prices)


def mini_costs(traj: Path) -> dict:
    """Primary cost (frozen prices x provider-reported tokens, prereg §4) and
    OpenRouter's billed cost from one mini-swe-agent trajectory.

    mini's own `agent.cost` is the billed `usage.cost`, which moved with the
    upstream price during the study. Graphirm's summary already reports both.
    """
    from metrics import billed_cost, primary_cost

    table = load_prices()
    out = {"prompt": 0, "cached": 0, "output": 0, "calls": 0, "cost_usd": 0.0, "cost_usd_billed": 0.0}
    if not traj.is_file():
        return out
    data = json.loads(traj.read_text())
    for msg in data.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        usage = ((msg.get("extra") or {}).get("response") or {}).get("usage") or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        output = int(usage.get("completion_tokens") or 0)
        out["prompt"] += prompt
        out["cached"] += cached
        out["output"] += output
        out["calls"] += 1
        out["cost_usd_billed"] += billed_cost(usage)
    out["cost_usd"] = primary_cost(out["prompt"], out["cached"], out["output"], table)
    return out


def read_patch(path: Path) -> str:
    """Read a patch without newline translation, so CR bytes survive."""
    if not path.is_file():
        return ""
    with open(path, "r", encoding="utf-8", errors="surrogateescape", newline="") as handle:
        return handle.read()


def write_patch(path: Path, text: str) -> None:
    """Write a patch without newline translation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", errors="surrogateescape", newline="") as handle:
        handle.write(text)


def capture_diff(container_id: str, cwd: str, paths: list[str] | None = None) -> str:
    """`git diff` from a container as bytes, decoded without newline translation.

    Python text-mode pipes turn CRLF into LF, which made every arm's patch on a
    CRLF file unapplicable (pydap-624). This path keeps the bytes.
    """
    import subprocess

    command = ["docker", "exec", container_id, "git", "-C", cwd, "diff"]
    if paths:
        command += ["--", *paths]
    proc = subprocess.run(command, capture_output=True, timeout=300, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace")[-400:])
    return proc.stdout.decode("utf-8", errors="surrogateescape")


def patch_paths(patch: str) -> list[str]:
    """File paths named by `diff --git` headers, in order."""
    found: list[str] = []
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4 and parts[3].startswith("b/"):
                found.append(parts[3][2:])
    return found


def crlf_safe_submission(submission: str, container_id: str, cwd: str) -> tuple[str, str]:
    """Return mini-swe-agent's submission with CR bytes restored, and how.

    mini's own `cat patch.txt` goes through a text pipe. If the submission names
    files, the same `git diff -- <files>` is read from the container as bytes.
    The byte version is used only when it covers exactly the same files, so the
    agent's choice of what to submit is unchanged. Otherwise the text is kept.
    """
    paths = patch_paths(submission)
    if not paths or not container_id:
        return submission, "text"
    try:
        binary = capture_diff(container_id, cwd, paths)
    except Exception:
        return submission, "text_capture_failed"
    if patch_paths(binary) != paths:
        return submission, "text_file_set_differs"
    return binary, "binary"


def main() -> None:
    load_dotenv()
    raise SystemExit(
        "run_arm.py loads .env and owns strip_test_hunks. "
        "It does not launch an arm yet."
    )


if __name__ == "__main__":
    main()
