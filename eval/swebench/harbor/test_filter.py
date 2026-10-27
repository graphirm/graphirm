"""Drop test-file edits from a Harbor worktree before the verifier runs.

T1, T2 and T3 pass every model patch through strip_test_hunks before grading.
Harbor grades the worktree the agent left behind, so the same filter has to
undo those paths in place. The saved patch is the agent's full diff. The
verifier then applies the task's own test patch onto the product edits.
"""

from __future__ import annotations

import os
import shlex


def revert_shell(patch_file: str) -> str:
    """Shell that records the full diff, then restores test paths.

    Test directories and test_*.py / *_test.py match the pre-registered
    patterns. Paths from T4_TEST_PATCH_PATHS are the hidden test patch, which
    the host sets for this trial.
    """
    extra = [
        line.strip()
        for line in os.environ.get("T4_TEST_PATCH_PATHS", "").splitlines()
        if line.strip()
    ]
    listed = " ".join(shlex.quote(path) for path in extra)
    dest = shlex.quote(patch_file)
    # A bare ";" is a syntax error in bash (`}; ; git` does not parse).
    # Emit the extra-path loop only when the host passed paths.
    extra_loop = ""
    if listed:
        extra_loop = f"for p in {listed}; do revert_test_path \"$p\"; done; "
    return (
        "if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then "
        "git add -A >/dev/null 2>&1 || true; "
        f"git diff --cached > {dest} || true; "
        "git reset >/dev/null 2>&1 || true; "
        "revert_test_path() { "
        "p=\"$1\"; "
        "[ -n \"$p\" ] || return 0; "
        "if git cat-file -e \"HEAD:$p\" >/dev/null 2>&1; then "
        "git checkout HEAD -- \"$p\" >/dev/null 2>&1 || true; "
        "else rm -rf -- \"$p\"; fi; "
        "}; "
        f"{extra_loop}"
        "git status --porcelain | while IFS= read -r line; do "
        "p=\"${line:3}\"; "
        "case \"$p\" in "
        "tests/*|*/tests/*|test/*|*/test/*|test_*.py|*/test_*.py|*_test.py) "
        "revert_test_path \"$p\" ;; "
        "esac; "
        "done; "
        f"else : > {dest}; fi"
    )


def graphirm_shell(
    patch_file: str,
    summary: str = "/logs/agent/summary.json",
    model: str = "openrouter/deepseek/deepseek-v4.1-flash",
    max_turns: int = 100,
    cost_limit: float = 0.50,
    logs_dir: str = "/logs/agent",
) -> str:
    """The Graphirm launch command. The task text stays in /tmp/issue.md."""
    graph = shlex.quote(logs_dir.rstrip("/") + "/graph.db")
    return (
        "set -eu; "
        "wd=$(pwd); "
        f"mkdir -p {shlex.quote(logs_dir)}; "
        "set +e; "
        "graphirm run "
        "--task /tmp/issue.md "
        "--cwd \"$wd\" "
        f"--graph {graph} "
        "--config /tmp/eval-agent.toml "
        f"--model {shlex.quote(model)} "
        "--reasoning-effort high "
        "--reply-format segments "
        f"--max-turns {int(max_turns)} "
        f"--max-cost-usd {float(cost_limit)} "
        f"--prices /tmp/prices.lock.json "
        f"--summary {shlex.quote(summary)}; "
        "code=$?; "
        + revert_shell(patch_file)
        + "; "
        "if [ \"$code\" -eq 2 ]; then exit 0; fi; "
        "exit \"$code\""
    )
