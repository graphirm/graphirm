#!/bin/sh
# Host-side adapter. Runs one SWE-bench instance container.
#
# Environment matched to mini-swe-agent's SWE-bench run:
#   image: the instance image (caller starts CONTAINER from it)
#   workdir: /testbed
#   python: the image PATH already points at the testbed env; this script
#           does not activate a second venv
#   budgets: GRAPHIRM_MAX_TURNS and GRAPHIRM_MAX_COST, same numbers as M
# Difference: Graphirm is a multi-tool process inside that container.
# mini-swe-agent's only tool is bash. Both see the same image env.
#
# Required env: CONTAINER, TASK_FILE, WORKDIR
# Optional: GRAPH_OUT, SUMMARY_OUT, CONFIG, TEST_PATCH_FILE
# The binary path defaults to ./target/release/graphirm.
set -eu

# .env holds API keys and the everyday chat model. Eval ignores both
# GRAPHIRM_MODEL and GRAPHIRM_REASONING. The flags below are the
# pre-registered model and reasoning effort.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi
unset GRAPHIRM_MODEL
unset GRAPHIRM_REASONING

if [ -z "${CONTAINER:-}" ] || [ -z "${TASK_FILE:-}" ] || [ -z "${WORKDIR:-}" ]; then
  echo "graphirm_runner.sh needs CONTAINER, TASK_FILE, and WORKDIR" >&2
  exit 1
fi

# 16k arms can drop the issue text until the task node is pinned.
# M, G, and G-md do not set these, so they keep running.
if [ -f /tmp/phase-d/pause-b16 ]; then
  if [ -n "${GRAPHIRM_CONTEXT_MAX_TOKENS:-}" ] || [ "${GRAPHIRM_SCORING:-}" = "recency" ]; then
    echo "paused: 16k arm waiting for the task-node pin" >&2
    exit 4
  fi
fi

BIN="${GRAPHIRM_BIN:-./target/release/graphirm}"
CONFIG="${CONFIG:-./eval/swebench/adapters/eval-agent.toml}"
if ! grep -q 'openrouter_provider = "morph/fp8"' "$CONFIG"; then
  echo "refusing to run: $CONFIG provider pin is not morph/fp8" >&2
  exit 5
fi
PRICES="${GRAPHIRM_PRICES:-./eval/swebench/prices.lock.json}"
GRAPH_OUT="${GRAPH_OUT:-./graph.db}"
SUMMARY_OUT="${SUMMARY_OUT:-./summary.json}"
export GRAPH_OUT
export SUMMARY_OUT
if [ -n "${TEST_PATCH_FILE:-}" ]; then
  export TEST_PATCH_FILE
fi

docker cp "$BIN" "$CONTAINER:/usr/local/bin/graphirm"
docker cp "$CONFIG" "$CONTAINER:/tmp/eval-agent.toml"
docker cp "$PRICES" "$CONTAINER:/tmp/prices.lock.json"
docker cp "$TASK_FILE" "$CONTAINER:/tmp/issue.md"
# Memory tracks: GRAPH_IN is the repo's carried graph (G-persist). Graphirm
# reopens it and appends this task. Unset or missing means a fresh graph.
# A docker-cp'd file keeps the host uid and SQLite then refuses to write it,
# so the copy is re-created by the container's own user.
if [ -n "${GRAPH_IN:-}" ] && [ -s "$GRAPH_IN" ]; then
  docker cp "$GRAPH_IN" "$CONTAINER:/tmp/graph.in.db"
  docker exec "$CONTAINER" sh -c 'cp /tmp/graph.in.db /tmp/graph.db && rm -f /tmp/graph.in.db'
fi

mkdir -p "$(dirname "$SUMMARY_OUT")"
docker exec "$CONTAINER" git -C "$WORKDIR" rev-parse HEAD > "${SUMMARY_OUT}.start_commit"

set +e
pack_env=""
if [ -n "${GRAPHIRM_CONTEXT_MAX_TOKENS:-}" ]; then
  pack_env="$pack_env -e GRAPHIRM_CONTEXT_MAX_TOKENS"
fi
if [ -n "${GRAPHIRM_PREFIX_STABLE:-}" ]; then
  pack_env="$pack_env -e GRAPHIRM_PREFIX_STABLE"
fi
if [ -n "${GRAPHIRM_EXCERPT_AFTER:-}" ]; then
  pack_env="$pack_env -e GRAPHIRM_EXCERPT_AFTER"
fi
if [ -n "${PACK_TRACE_OUT:-}" ]; then
  pack_env="$pack_env -e GRAPHIRM_PACK_TRACE=/tmp/pack-trace.jsonl"
fi
if [ -n "${RETRY_LOG_OUT:-}" ]; then
  pack_env="$pack_env -e GRAPHIRM_RETRY_LOG=/tmp/provider-retries.log"
fi
# shellcheck disable=SC2086
docker exec -e OPENROUTER_API_KEY $pack_env "$CONTAINER" graphirm run \
  --task /tmp/issue.md \
  --cwd "$WORKDIR" \
  --graph /tmp/graph.db \
  --config /tmp/eval-agent.toml \
  --model openrouter/deepseek/deepseek-v4.1-flash \
  --reasoning-effort high \
  --scoring "${GRAPHIRM_SCORING:-default}" \
  --reply-format "${GRAPHIRM_REPLY:-segments}" \
  --max-turns "${GRAPHIRM_MAX_TURNS:-100}" \
  --max-cost-usd "${GRAPHIRM_MAX_COST:-0.50}" \
  --prices /tmp/prices.lock.json \
  --summary /tmp/summary.json
code=$?
set -e
# Exit 2 is a budget stop. Copy the graph and the patch either way,
# then return the real status so a provider failure can be retried.

docker exec "$CONTAINER" git -C "$WORKDIR" diff > "${GRAPH_OUT}.diff.raw" || true
python3 - <<'PY'
import os
from pathlib import Path
import sys
sys.path.insert(0, "eval/swebench")
from run_arm import strip_test_hunks
raw = Path(os.environ["GRAPH_OUT"] + ".diff.raw").read_text()
test_patch = ""
extra = os.environ.get("TEST_PATCH_FILE")
if extra:
    test_patch = Path(extra).read_text()
Path(os.environ["GRAPH_OUT"] + ".diff").write_text(strip_test_hunks(raw, test_patch))
PY
docker cp "$CONTAINER:/tmp/graph.db" "$GRAPH_OUT" || true
docker cp "$CONTAINER:/tmp/summary.json" "$SUMMARY_OUT" || true
python3 - <<'PY'
import json
import os
from pathlib import Path

summary = Path(os.environ["SUMMARY_OUT"])
commit_path = Path(str(summary) + ".start_commit")
if summary.is_file() and commit_path.is_file():
    data = json.loads(summary.read_text())
    data["start_commit"] = commit_path.read_text().strip()
    data["image"] = os.environ.get("GRAPHIRM_IMAGE", "")
    summary.write_text(json.dumps(data, indent=2) + "\n")
PY
if [ -n "${PACK_TRACE_OUT:-}" ]; then
  docker cp "$CONTAINER:/tmp/pack-trace.jsonl" "$PACK_TRACE_OUT" || true
fi
if [ -n "${RETRY_LOG_OUT:-}" ]; then
  docker cp "$CONTAINER:/tmp/provider-retries.log" "$RETRY_LOG_OUT" || true
fi
if [ "$code" -ne 0 ] && [ "$code" -ne 2 ]; then
  echo "graphirm run exited $code" >&2
  exit "$code"
fi
