#!/usr/bin/env bash
# Fake `pi` for offline tests: replays a Pi `--mode json` fixture line by line.
#
# Knobs are read from the environment and, additionally, from argv as
# `--fake-knob KEY=VALUE` pairs (argv wins over env). The argv form exists
# because `spawn_pi` deliberately does not accept arbitrary env: tests carry
# knobs in `PiConfig.extra_args`, which keeps them parallel-safe (no
# `std::env::set_var`). Every other argument is ignored; everything after
# `--` is the task.
#
#   FAKE_PI_FIXTURE   path to JSONL (default: hello-run.jsonl next to this script)
#   FAKE_PI_DELAY_MS  per-line delay in ms (default 5; 0 → no sleep at all)
#   FAKE_PI_EXIT      exit code after replay (default 0)
#   FAKE_PI_HANG_AT   line number after which to sleep forever (cancel/timeout tests)
#   FAKE_PI_GARBAGE   1 → emit a non-JSON line after every 3rd event
#   FAKE_PI_NO_END    1 → stop before the last assistant message_end
#   FAKE_PI_PIDFILE   write $$ here (kill tests)
#   FAKE_PI_STDERR    1 → write "warn: fake stderr" to stderr once
#   FAKE_PI_ARGV      path → write "$@" (one per line) here (argv tests)
#   FAKE_PI_CHILD     1 → also fork `sleep 3600 &` (a grandchild in the same
#                     process group) and, when FAKE_PI_PIDFILE is set, write
#                     its pid to ${FAKE_PI_PIDFILE}.child before the main
#                     pidfile, so a test that sees the main pidfile can rely
#                     on the child pidfile. The child's stdout/stderr go to
#                     /dev/null unless …
#   FAKE_PI_CHILD_HOLDS_STDOUT
#                     1 → the sleep child inherits stdout/stderr, so the pipes
#                     stay open after this script exits (post-exit grace test)
#   FAKE_PI_ENV_DUMP  path → write `export -p` there (env policy test)
#
# When the task (first positional after `--`) starts with `@`, the byte length
# of the referenced file is written to stderr as `task-file-bytes: N`
# (file-route tests). Also answers `--version` with 0.85.1-fake and exits 0.
#
# Uses only bash builtins plus `sleep` and `wc` (coreutils).
set -u
[ "${1:-}" = "--version" ] && { echo 0.85.1-fake; exit 0; }

argv_all=("$@")
while [ $# -gt 0 ]; do
  case "$1" in
    --) shift; break ;;
    --fake-knob)
      if [ $# -ge 2 ]; then export "$2"; shift 2; else shift; fi ;;
    *) shift ;;
  esac
done
task="${1:-}"

if [ -n "${FAKE_PI_ARGV:-}" ]; then
  printf '%s\n' "${argv_all[@]}" > "$FAKE_PI_ARGV"
fi
if [ -n "${FAKE_PI_ENV_DUMP:-}" ]; then
  export -p > "$FAKE_PI_ENV_DUMP"
fi
if [ "${task:0:1}" = "@" ]; then
  task_file=${task#@}
  if [ -f "$task_file" ]; then
    echo "task-file-bytes: $(wc -c < "$task_file")" >&2
  else
    echo "task-file-missing: $task_file" >&2
  fi
fi

if [ "${FAKE_PI_CHILD:-0}" = 1 ]; then
  if [ "${FAKE_PI_CHILD_HOLDS_STDOUT:-0}" = 1 ]; then
    sleep 3600 &
  else
    sleep 3600 >/dev/null 2>&1 &
  fi
  [ -n "${FAKE_PI_PIDFILE:-}" ] && echo $! > "${FAKE_PI_PIDFILE}.child"
fi
[ -n "${FAKE_PI_PIDFILE:-}" ] && echo $$ > "$FAKE_PI_PIDFILE"
[ "${FAKE_PI_STDERR:-0}" = 1 ] && echo "warn: fake stderr" >&2

here=$(cd "$(dirname "$0")" && pwd)
fixture=${FAKE_PI_FIXTURE:-$here/hello-run.jsonl}

ms=$(( ${FAKE_PI_DELAY_MS:-5} ))
delay=""
if [ "$ms" -gt 0 ]; then
  printf -v delay '%d.%03d' $((ms / 1000)) $((ms % 1000))
fi

is_assistant_end() {
  [[ $1 == *'"type":"message_end"'* && $1 == *'"role":"assistant"'* ]]
}

# FAKE_PI_NO_END: find the line number of the *last* assistant message_end in
# a pre-pass, then stop the replay right before it.
stop_at=0
if [ "${FAKE_PI_NO_END:-0}" = 1 ]; then
  n=0
  while IFS= read -r line || [ -n "$line" ]; do
    n=$((n + 1))
    is_assistant_end "$line" && stop_at=$n
  done < "$fixture"
fi

n=0
while IFS= read -r line || [ -n "$line" ]; do
  n=$((n + 1))
  if [ "$stop_at" -gt 0 ] && [ "$n" -ge "$stop_at" ]; then
    break
  fi
  printf '%s\n' "$line"
  if [ "${FAKE_PI_GARBAGE:-0}" = 1 ] && [ $((n % 3)) = 0 ]; then
    echo "not json $n"
  fi
  if [ -n "${FAKE_PI_HANG_AT:-}" ] && [ "$n" -ge "$FAKE_PI_HANG_AT" ]; then
    sleep 3600
  fi
  if [ -n "$delay" ]; then
    sleep "$delay"
  fi
done < "$fixture"
exit "${FAKE_PI_EXIT:-0}"
