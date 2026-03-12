#!/usr/bin/env bash
# Generate a corpus of assistant turns by driving Graphirm non-interactively via the HTTP API.
#
# Prerequisites:
#   - graphirm built (cargo build --release or cargo build)
#   - jq
#   - DEEPSEEK_API_KEY (or GRAPHIRM_MODEL + corresponding API key) for real LLM responses
#
# Usage:
#   ./scripts/corpus_from_server.sh [--db PATH] [--out PATH] [--limit N] [--max-prompts N] [--batch-size N] [--turn-gap N] [--prompts FILE]
#
# Default: DB=~/.graphirm/corpus_gen.db, out=corpus.jsonl, limit=50, prompts=embedded list.
# Rate limiting: use --turn-gap N (default 8) to sleep N seconds after each turn to avoid 429.
# Memory: use --batch-size 10 (or 15) to restart the server every N prompts; script sends SIGINT
# so the server actually exits. Use --batch-gap 5 to sleep between batches. Optional
# --server-memory-mb 2048 caps the server process (ulimit -v). Or run: ulimit -v 4194304
# The server is started with --db and stopped after each batch (or once). Export runs at the end
# and also on EXIT (Ctrl+C or error) so the corpus is always saved to --out.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DB="${DB:-$HOME/.graphirm/corpus_gen.db}"
OUT="${OUT:-corpus.jsonl}"
LIMIT="${LIMIT:-50}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
BATCH_SIZE="${BATCH_SIZE:-0}"
PORT="${PORT:-5555}"
BASE_URL="http://127.0.0.1:${PORT}"
# MAX_PROMPTS 0 = no limit; BATCH_SIZE 0 = single session (no restart)

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
  echo "Usage: $0 [--db PATH] [--out PATH] [--limit N] [--max-prompts N] [--batch-size N] [--batch-gap N] [--turn-gap N] [--server-memory-mb N] [--port P] [--prompts FILE]"
  echo "  --db PATH        Graph DB path for server and export (default: $DB)"
  echo "  --out PATH       Output JSONL path (default: $OUT)"
  echo "  --limit N        Max assistant turns to export (default: $LIMIT)"
  echo "  --max-prompts N  Stop after sending N prompts (default: no limit)"
  echo "  --batch-size N   Restart server every N prompts to limit memory; same DB, export once at end (default: 0 = off)"
  echo "  --batch-gap N    Seconds to sleep between batches so memory can be reclaimed (default: 3)"
  echo "  --turn-gap N     Seconds to sleep after each turn to avoid LLM rate limits (default: 8)"
  echo "  --server-memory-mb N  Cap server process at N MB virtual memory (ulimit -v); 0 = no cap (default: 0)"
  echo "  --port P         Server port (default: $PORT)"
  echo "  --prompts F      One prompt per line; if omitted, uses embedded defaults."
  echo "  --all-roles      Export user prompts and assistant turns (for annotating both); limit becomes 2×limit."
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --db)    DB="$2"; shift 2 ;;
    --out)   OUT="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --max-prompts) MAX_PROMPTS="$2"; shift 2 ;;
    --batch-size)  BATCH_SIZE="$2"; shift 2 ;;
    --batch-gap)   BATCH_GAP="$2"; shift 2 ;;
    --turn-gap)    TURN_GAP="$2"; shift 2 ;;
    --server-memory-mb) SERVER_MEMORY_MB="$2"; shift 2 ;;
    --port)  PORT="$2"; BASE_URL="http://127.0.0.1:${PORT}"; shift 2 ;;
    --prompts) PROMPTS_FILE="$2"; shift 2 ;;
    --all-roles) ALL_ROLES=1; shift 1 ;;
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

# Use Qwen via OpenRouter by default to avoid provider rate limits (override with GRAPHIRM_MODEL).
export GRAPHIRM_MODEL="${GRAPHIRM_MODEL:-openrouter/qwen/qwen3-coder-next}"
# Disable per-turn knowledge extraction during corpus generation — we run predict-spans separately.
export GRAPHIRM_EXTRACTION="${GRAPHIRM_EXTRACTION:-false}"
# Ensure DB and OUT directories exist; resolve OUT to absolute so export always writes where expected
mkdir -p "$(dirname "$DB")"
TURN_GAP="${TURN_GAP:-8}"
if [[ -n "$OUT" ]]; then
  OUT_DIR="$(dirname "$OUT")"
  mkdir -p "$OUT_DIR"
  # Resolve to absolute path so export works even if we cd or script is invoked from elsewhere
  [[ "$OUT" != /* ]] && OUT="$(cd "$OUT_DIR" && pwd)/$(basename "$OUT")"
fi
EXPORT_DONE=
DID_RUN=

# Load prompts into array
PROMPTS_ARR=()
if [[ -n "${PROMPTS_FILE:-}" ]]; then
  if [[ ! -f "$PROMPTS_FILE" ]]; then
    echo "Error: prompts file not found: $PROMPTS_FILE"
    exit 1
  fi
  while IFS= read -r line || [[ -n "$line" ]]; do
    line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    [[ -z "$line" ]] && continue
    PROMPTS_ARR+=("$line")
  done < "$PROMPTS_FILE"
  echo "Loaded ${#PROMPTS_ARR[@]} prompts from $PROMPTS_FILE"
else
  for p in "${DEFAULT_PROMPTS[@]}"; do
    PROMPTS_ARR+=("$p")
  done
  echo "Using ${#PROMPTS_ARR[@]} default prompts"
fi
if [[ "${MAX_PROMPTS:-0}" -gt 0 && "${#PROMPTS_ARR[@]}" -gt "$MAX_PROMPTS" ]]; then
  PROMPTS_ARR=("${PROMPTS_ARR[@]:0:$MAX_PROMPTS}")
  echo "Trimmed to $MAX_PROMPTS prompts (--max-prompts)"
fi
TOTAL_PROMPTS=${#PROMPTS_ARR[@]}
[[ "$TOTAL_PROMPTS" -eq 0 ]] && { echo "Error: no prompts to send."; exit 1; }

SERVER_PID=
# Server only shuts down on SIGINT (Ctrl+C). SIGTERM is ignored.
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    echo "Stopping server (PID $SERVER_PID)..."
    kill -INT "$SERVER_PID" 2>/dev/null || true
    for _ in {1..15}; do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Server did not exit; sending SIGKILL."
      kill -9 "$SERVER_PID" 2>/dev/null || true
      wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=
  fi
}

do_export() {
  [[ -n "$EXPORT_DONE" ]] && return 0
  [[ -z "$DID_RUN" ]] && return 0
  [[ ! -f "$DB" ]] && return 0
  local export_limit="$LIMIT"
  local extra_args=()
  if [[ -n "${ALL_ROLES:-}" ]]; then
    export_limit=$(( LIMIT * 2 ))
    extra_args=(--all-roles)
    echo "Exporting corpus to $OUT (all roles, limit $export_limit)..."
  else
    echo "Exporting corpus to $OUT (assistant only, limit $LIMIT)..."
  fi
  if "$GRAPHIRM_BIN" export-corpus --db "$DB" --out "$OUT" --limit "$export_limit" "${extra_args[@]}" 2>&1; then
    echo "Done. Exported to $OUT"
  else
    echo "Warning: export failed (see above)."
  fi
  EXPORT_DONE=1
}

on_exit() {
  cleanup
  do_export
}
trap on_exit EXIT
# Seconds to sleep between batches so the OS can reclaim memory (default 3)
BATCH_GAP="${BATCH_GAP:-3}"
SERVER_MEMORY_MB="${SERVER_MEMORY_MB:-0}"

start_server() {
  if [[ "${SERVER_MEMORY_MB:-0}" -gt 0 ]]; then
    ( ulimit -v $(( SERVER_MEMORY_MB * 1024 )); exec "$GRAPHIRM_BIN" serve --db "$DB" --host 127.0.0.1 --port "$PORT" ) &
  else
    "$GRAPHIRM_BIN" serve --db "$DB" --host 127.0.0.1 --port "$PORT" &
  fi
  SERVER_PID=$!
  for i in {1..30}; do
    if curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/health" 2>/dev/null | grep -q 200; then
      return 0
    fi
    [[ $i -eq 30 ]] && { echo "Error: server did not become ready in time."; exit 1; }
    sleep 0.5
  done
}

create_session() {
  SESSION_RESP=$(curl -s -X POST "$BASE_URL/api/sessions" \
    -H "Content-Type: application/json" \
    -d '{"auto_approve": true}')
  SESSION_ID=$(echo "$SESSION_RESP" | jq -r '.id')
  if [[ -z "$SESSION_ID" || "$SESSION_ID" == "null" ]]; then
    echo "Error: failed to create session: $SESSION_RESP"
    exit 1
  fi
}

wait_until_idle() {
  local sid="$1"
  while true; do
    local status
    status=$(curl -s "$BASE_URL/api/sessions/$sid" | jq -r '.status')
    [[ "$status" != "running" ]] && return 0
    sleep 2
  done
}

send_one() {
  local msg="$1"
  local sid="$2"
  local code
  content=$(jq -n --arg c "$msg" '{ content: $c }')
  code=$(curl -s -X POST "$BASE_URL/api/sessions/$sid/prompt" \
    -H "Content-Type: application/json" -d "$content" -o /dev/null -w "%{http_code}")
  if [[ "$code" == "429" ]]; then
    echo "  Rate limited (429); sleeping 60s then retrying once..."
    sleep 60
    code=$(curl -s -X POST "$BASE_URL/api/sessions/$sid/prompt" \
      -H "Content-Type: application/json" -d "$content" -o /dev/null -w "%{http_code}")
  fi
  if [[ "$code" != "202" ]]; then
    echo "  Warning: prompt returned HTTP $code (expected 202)."
  fi
  DID_RUN=1
  wait_until_idle "$sid"
  if [[ "${TURN_GAP:-0}" -gt 0 ]]; then
    sleep "$TURN_GAP"
  fi
}

echo "Using DB=$DB OUT=$OUT LIMIT=$LIMIT PORT=$PORT model=$GRAPHIRM_MODEL batch_size=${BATCH_SIZE:-0} batch_gap=${BATCH_GAP:-3} turn_gap=${TURN_GAP:-8} server_memory_mb=${SERVER_MEMORY_MB:-0}"

if [[ "${BATCH_SIZE:-0}" -gt 0 ]]; then
  # Batch mode: restart server every BATCH_SIZE prompts
  total_sent=0
  batch_num=0
  while [[ $total_sent -lt $TOTAL_PROMPTS ]]; do
    (( batch_num++ )) || true
    end=$(( total_sent + BATCH_SIZE ))
    [[ $end -gt $TOTAL_PROMPTS ]] && end=$TOTAL_PROMPTS
    echo "--- Batch $batch_num (prompts $(( total_sent + 1 ))-$end of $TOTAL_PROMPTS) ---"
    start_server
    echo "Server ready."
    create_session
    echo "Session: $SESSION_ID"
    batch_sent=0
    while [[ $batch_sent -lt $BATCH_SIZE && $total_sent -lt $TOTAL_PROMPTS ]]; do
      send_one "${PROMPTS_ARR[$total_sent]}" "$SESSION_ID"
      (( total_sent++ )) || true
      (( batch_sent++ )) || true
      echo "  Turn $total_sent done."
    done
    cleanup
    if [[ $total_sent -lt $TOTAL_PROMPTS && "$BATCH_GAP" -gt 0 ]]; then
      echo "Sleeping ${BATCH_GAP}s before next batch (memory reclaim)..."
      sleep "$BATCH_GAP"
    fi
  done
  echo "All $total_sent prompts sent in $batch_num batch(es)."
else
  # Single session
  start_server
  echo "Server ready."
  create_session
  echo "Session: $SESSION_ID"
  for (( i = 0; i < TOTAL_PROMPTS; i++ )); do
    send_one "${PROMPTS_ARR[$i]}" "$SESSION_ID"
    echo "  Turn $(( i + 1 )) done."
  done
  cleanup
fi
# Export runs in on_exit trap (so it runs on Ctrl+C or error too). Ensure we mark that we ran.
do_export
