#!/usr/bin/env python3
"""Smoke run for [agent.reply_pieces]: three prompts on a tiny repo, then check 0.4.

Starts `graphirm serve` from a scratch folder whose config/default.toml has
reply pieces on, sends three prompts in one session, stops the server with
SIGTERM, and prints the completeness report and pieces_source counts.

    set -a; . ./.env; set +a
    python3 eval/structure/smoke_reply_pieces.py --binary target/release/graphirm --dir /tmp/rp-smoke

Needs OPENROUTER_API_KEY and GRAPHIRM_API_KEY in the environment. The agent
runs bash inside --dir/work with auto-approve on. Cost: a few cents on
deepseek-v4.1-flash.
"""

import argparse
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PORT = 3917
MODEL = "openrouter/deepseek/deepseek-v4.1-flash"
PROMPTS = [
    "The tests in the work/ folder fail. Run them with `cd work && python3 -m pytest -q`, "
    "find the bug, and fix it.",
    "In work/stats.py add a function mean(xs) that raises ValueError on an empty list, with "
    "tests in work/test_stats.py. Run the tests.",
    "Summarize what you changed in this session and why.",
]
STATS = """def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2]
"""
TESTS = """from stats import median

def test_odd():
    assert median([3, 1, 2]) == 2

def test_even():
    assert median([4, 1, 3, 2]) == 2.5
"""


def setup(root):
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "work").mkdir(exist_ok=True)
    config = (REPO / "config/default.toml").read_text()
    config = config.replace("[agent.reply_pieces]\nenabled = false",
                            "[agent.reply_pieces]\nenabled = true", 1)
    config = re.sub(r'(\[agent\]\n(?:.*\n)*?)model = "[^"]*"',
                    lambda m: m.group(1) + f'model = "{MODEL}"', config, count=1)
    if "[agent.reply_pieces]\nenabled = true" not in config:
        sys.exit("could not turn reply pieces on in the copied config")
    (root / "config/default.toml").write_text(config)
    (root / "work/stats.py").write_text(STATS)
    (root / "work/test_stats.py").write_text(TESTS)


def call(key, method, path, body=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}", method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read() or b"{}")


def wait_healthy():
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/health", timeout=2)
            return
        except OSError:
            time.sleep(1)
    sys.exit("server did not start")


def drive(key):
    sid = call(key, "POST", "/api/sessions", {"auto_approve": True})["id"]
    print("session", sid, flush=True)
    for prompt in PROMPTS:
        call(key, "POST", f"/api/sessions/{sid}/prompt", {"content": prompt})
        start = time.time()
        status = "running"
        while status == "running" and time.time() - start < 600:
            time.sleep(3)
            status = call(key, "GET", f"/api/sessions/{sid}").get("status")
        print(f"prompt done status={status} seconds={round(time.time() - start)}", flush=True)


def report(db):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT json_extract(metadata,'$.pieces_source'), "
        "json_extract(metadata,'$.tool_calls') IS NOT NULL, count(*) FROM nodes "
        "WHERE json_extract(data,'$.role')='assistant' "
        "AND trim(json_extract(data,'$.content'))<>'' GROUP BY 1, 2").fetchall()
    con.close()
    print("pieces_source by reply (source, has_tool_calls, count):", rows)
    subprocess.run([sys.executable, str(HERE / "completeness.py"), "--label", "smoke", str(db)],
                   check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    for name in ("OPENROUTER_API_KEY", "GRAPHIRM_API_KEY"):
        if not os.environ.get(name):
            sys.exit(f"{name} is not set")
    root = Path(args.dir).resolve()
    setup(root)
    db = root / "graph.db"
    log = open(root / "serve.log", "w")
    server = subprocess.Popen([str(Path(args.binary).resolve()), "--db", str(db), "serve",
                               "-p", str(PORT)], cwd=root, stdout=log, stderr=log)
    try:
        wait_healthy()
        drive(os.environ["GRAPHIRM_API_KEY"])
    finally:
        server.send_signal(signal.SIGTERM)
        server.wait(timeout=30)
        log.close()
    report(db)


if __name__ == "__main__":
    main()
