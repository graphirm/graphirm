#!/usr/bin/env python3
"""Prove the seven launch-failure tasks parse and reach the agent command.

The failed trials died in bash before either agent started. This rebuilds
both launch commands with the live filter, checks them with bash -n, and
runs them with a stand-in agent on PATH. A stand-in that prints STARTED
means bash accepted the task text and invoked the agent.
"""

from __future__ import annotations

import importlib.util
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from run_arm import _touched_paths  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FILTER = _load("t4_test_filter_check", HERE / "test_filter.py")
MINI = _load("t4_mini_check", HERE / "mini_swe_agent.py")

DATA = Path("/home/krs/graphirm-runs/t4/datasets/swe-rebench-07-2026")
TASKS = [
    "ivov__lisette-491",
    "ivov__lisette-627",
    "ivov__lisette-657",
    "kenn-io__agentsview-602",
    "kestra-io__kestra-16067",
    "kestra-io__kestra-16119",
    "kestra-io__kestra-16450",
]
PAIRS = [
    (arm, task)
    for task in TASKS
    for arm in ("G", "M")
    if not (arm == "G" and task == "kestra-io__kestra-16450")
]


def paths_for(task: str) -> str:
    script = DATA / task / "tests" / "test.sh"
    return "\n".join(sorted(_touched_paths(script.read_text(errors="replace"))))


def bash_n(command: str) -> None:
    proc = subprocess.run(["bash", "-n", "-c", command], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.stderr or proc.stdout)


def mini_command(instruction: str, patch: str, trajectory: str, tee: str) -> str:
    escaped = shlex.quote(instruction)
    base = (
        'if [ -f "$HOME/.local/bin/env" ]; then . "$HOME/.local/bin/env"; '
        'else export PATH="$HOME/.local/bin:$PATH"; fi; '
        f"mini-swe-agent --yolo --model=deepseek/deepseek-v4.1-flash --task={escaped} "
        f"--output={shlex.quote(trajectory)} --cost-limit 0.5 "
        f"--exit-immediately 2>&1 </dev/null | tee {shlex.quote(tee)}"
    )
    return MINI.prepare_mini_command(base, patch)


def run_standin(arm: str, command: str, work: Path) -> None:
    bindir = work / "bin"
    bindir.mkdir()
    marker = work / "started.txt"
    token = "GRAPHIRM_STARTED" if arm == "G" else "MINI_STARTED"
    (bindir / "graphirm").write_text(f"#!/bin/sh\necho {token} > {shlex.quote(str(marker))}\nexit 0\n")
    (bindir / "python3").write_text(
        "#!/bin/sh\n"
        "if [ -n \"$1\" ]; then echo MINI_STARTED > "
        + shlex.quote(str(marker))
        + "\nexit 0\nfi\ncat >/dev/null\nexit 0\n"
    )
    (bindir / "mini-swe-agent").write_text("#!/usr/bin/env python3\n")
    for path in bindir.iterdir():
        path.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
    env["HOME"] = str(work / "home")
    (work / "home").mkdir()
    repo = work / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "launch@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "launch"], cwd=repo, check=True)
    (repo / "README").write_text("x\n")
    subprocess.run(["git", "add", "README"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
    proc = subprocess.run(["bash", "-c", command], cwd=repo, env=env, capture_output=True, text=True)
    text = marker.read_text() if marker.is_file() else ""
    if proc.returncode != 0 or token not in text:
        raise SystemExit(
            f"{arm} stand-in failed rc={proc.returncode} started={text!r}\n"
            f"{proc.stdout[-800:]}\n{proc.stderr[-800:]}"
        )


def main() -> None:
    os.environ["T4_TEST_PATCH_PATHS"] = "weird'name\nhas space\nhas`tick\nhas$(echo hi)"
    bash_n("set -o pipefail; " + FILTER.graphirm_shell("/tmp/launch-nasty.patch"))
    os.environ["T4_TEST_PATCH_PATHS"] = ""
    bash_n("set -o pipefail; " + FILTER.graphirm_shell("/tmp/launch-empty.patch"))
    for arm, task in PAIRS:
        os.environ["T4_TEST_PATCH_PATHS"] = paths_for(task)
        instruction = (DATA / task / "instruction.md").read_text()
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / "logs"
            logs.mkdir()
            if arm == "G":
                command = FILTER.graphirm_shell(
                    str(logs / "G.patch"),
                    summary=str(logs / "summary.json"),
                    logs_dir=str(logs),
                )
            else:
                command = mini_command(
                    instruction,
                    str(logs / "M.patch"),
                    str(logs / "traj.json"),
                    str(logs / "mini.txt"),
                )
            bash_n("set -o pipefail; " + command)
            work = Path(tmp) / "work"
            work.mkdir()
            run_standin(arm, command, work)
        print(f"STARTED {arm} {task}", flush=True)
    print(f"CHECK_OK pairs={len(PAIRS)}", flush=True)


if __name__ == "__main__":
    main()
