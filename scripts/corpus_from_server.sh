#!/usr/bin/env bash
# Generate a corpus of assistant turns by driving Graphirm non-interactively via the HTTP API.
#
# Prerequisites:
#   - graphirm built (cargo build --release or cargo build)
#   - jq
#   - DEEPSEEK_API_KEY (or GRAPHIRM_MODEL + corresponding API key) for real LLM responses
#
# Usage:
#   ./scripts/corpus_from_server.sh [--db PATH] [--out PATH] [--limit N] [--prompts FILE]
#
# Default: DB=/tmp/graphirm_corpus_gen.db, out=corpus.jsonl, limit=50, prompts=embedded list.
# The server is started with --db and stopped after all prompts are sent; then export-corpus runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DB="${DB:-/tmp/graphirm_corpus_gen.db}"
OUT="${OUT:-corpus.jsonl}"
LIMIT="${LIMIT:-50}"
PORT="${PORT:-5555}"
BASE_URL="http://127.0.0.1:${PORT}"

# Default prompts chosen to elicit varied segments (reasoning, code, observation, plan, answer).
DEFAULT_PROMPTS=(
  "What is the difference between a Vec and an array in Rust? Give a one-paragraph answer."
  "Write a Rust function that returns the nth Fibonacci number. Use a loop, not recursion."
  "I have a bug: my program panics with 'index out of bounds'. What should I check first?"
  "List three best practices for error handling in Rust. Be concise."
  "Explain in one sentence what the Option type is used for."
  "Write a tiny Rust test that checks 2 + 2 equals 4. No need for a full crate."
  "What does the question mark operator (?) do in Rust? One sentence."
  "Suggest a simple refactor: I have a function that returns Result<T, String>. What could I use instead of String for the error type?"
  "How would you debug a deadlock in a multi-threaded program? Short answer."
  "Write one line of Rust that creates a HashMap with keys 'a' and 'b' and integer values 1 and 2."
)

usage() {
  echo "Usage: $0 [--db PATH] [--out PATH] [--limit N] [--port P] [--prompts FILE]"
  echo "  --db PATH     Graph DB path for server and export (default: $DB)"
  echo "  --out PATH    Output JSONL path (default: $OUT)"
  echo "  --limit N     Max assistant turns to export (default: $LIMIT)"
  echo "  --port P      Server port (default: $PORT)"
  echo "  --prompts F   One prompt per line; if omitted, uses embedded defaults."
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --db)    DB="$2"; shift 2 ;;
    --out)   OUT="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --port)  PORT="$2"; BASE_URL="http://127.0.0.1:${PORT}"; shift 2 ;;
    --prompts) PROMPTS_FILE="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

GRAPHIRM_BIN="${GRAPHIRM_BIN:-$REPO_ROOT/target/release/graphirm}"
if [[ ! -x "$GRAPHIRM_BIN" ]]; then
  GRAPHIRM_BIN="$REPO_ROOT/target/debug/graphirm"
fi
if [[ ! -x "$GRAPHIRM_BIN" ]]; then
  echo "Error: graphirm binary not found. Run: cargo build --release (or cargo build)"
  exit 1
fi

if ! command -v jq &>/dev/null; then
  echo "Error: jq is required."
  exit 1
fi

# Ensure DB directory exists
mkdir -p "$(dirname "$DB")"

echo "Using DB=$DB OUT=$OUT LIMIT=$LIMIT PORT=$PORT"
echo "Starting server in background..."

"$GRAPHIRM_BIN" serve --db "$DB" --host 127.0.0.1 --port "$PORT" &
SERVER_PID=$!
cleanup() {
  echo "Stopping server (PID $SERVER_PID)..."
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for server to be up
for i in {1..30}; do
  if curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/health" 2>/dev/null | grep -q 200; then
    break
  fi
  if [[ $i -eq 30 ]]; then
    echo "Error: server did not become ready in time."
    exit 1
  fi
  sleep 0.5
done
echo "Server ready."

# Create session with auto_approve so tools run without HITL
SESSION_RESP=$(curl -s -X POST "$BASE_URL/api/sessions" \
  -H "Content-Type: application/json" \
  -d '{"auto_approve": true}')
SESSION_ID=$(echo "$SESSION_RESP" | jq -r '.id')
if [[ -z "$SESSION_ID" || "$SESSION_ID" == "null" ]]; then
  echo "Error: failed to create session: $SESSION_RESP"
  exit 1
fi
echo "Session: $SESSION_ID"

wait_until_idle() {
  local sid="$1"
  while true; do
    local status
    status=$(curl -s "$BASE_URL/api/sessions/$sid" | jq -r '.status')
    if [[ "$status" != "running" ]]; then
      return 0
    fi
    sleep 2
  done
}

if [[ -n "${PROMPTS_FILE:-}" ]]; then
  if [[ ! -f "$PROMPTS_FILE" ]]; then
    echo "Error: prompts file not found: $PROMPTS_FILE"
    exit 1
  fi
  echo "Sending prompts from $PROMPTS_FILE..."
  while IFS= read -r line || [[ -n "$line" ]]; do
    line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    [[ -z "$line" ]] && continue
    content=$(jq -n --arg c "$line" '{ content: $c }')
    curl -s -X POST "$BASE_URL/api/sessions/$SESSION_ID/prompt" \
      -H "Content-Type: application/json" -d "$content" -o /dev/null -w "%{http_code}\n" | grep -q 202 || true
    wait_until_idle "$SESSION_ID"
    echo "  Turn done."
  done < "$PROMPTS_FILE"
else
  echo "Sending ${#DEFAULT_PROMPTS[@]} default prompts..."
  for p in "${DEFAULT_PROMPTS[@]}"; do
    content=$(jq -n --arg c "$p" '{ content: $c }')
    curl -s -X POST "$BASE_URL/api/sessions/$SESSION_ID/prompt" \
      -H "Content-Type: application/json" -d "$content" -o /dev/null -w "%{http_code}\n" | grep -q 202 || true
    wait_until_idle "$SESSION_ID"
    echo "  Turn done."
  done
fi

# Server is stopped by cleanup trap; then export
echo "Exporting corpus to $OUT (limit $LIMIT)..."
"$GRAPHIRM_BIN" export-corpus --db "$DB" --out "$OUT" --limit "$LIMIT"
echo "Done. Exported to $OUT"
