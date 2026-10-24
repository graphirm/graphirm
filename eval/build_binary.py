#!/usr/bin/env python3
"""Build an eval binary from a clean commit, with the command the study used.

Refuses a dirty tree. Prints the commit, the toolchain, and the output sha256.
Does not overwrite target/x86_64-unknown-linux-musl/release/graphirm; that path
is the frozen T2 binary.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGET = Path("/home/krs/graphirm-frozen-target")
BIN = TARGET / "x86_64-unknown-linux-musl/release/graphirm"
COMMAND = [
    "cargo",
    "zigbuild",
    "--release",
    "--target",
    "x86_64-unknown-linux-musl",
    "--bin",
    "graphirm",
]


def porcelain() -> str:
    out = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def main() -> int:
    dirty = porcelain()
    if dirty:
        print("refusing to build an eval binary from a dirty tree:", file=sys.stderr)
        print(dirty, file=sys.stderr)
        return 2
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()
    rustc = subprocess.run(["rustc", "--version"], check=True, capture_output=True, text=True).stdout.strip()
    zig = subprocess.run(["zig", "version"], check=True, capture_output=True, text=True).stdout.strip()
    print(f"commit {commit}")
    print(f"rustc {rustc}")
    print(f"zig {zig}")
    print("command " + " ".join(COMMAND))
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(TARGET)
    subprocess.run(COMMAND, cwd=REPO, check=True, env=env)
    digest = hashlib.sha256(BIN.read_bytes()).hexdigest()
    print(f"sha256 {digest}")
    print(f"binary {BIN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
