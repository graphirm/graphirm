# Pi `--mode json` fixtures

Recorded output of a real `pi` (`@earendil-works/pi-coding-agent` 0.85.1) run,
used by the `pi_delegate` parser/process tests so the workspace tests need no
network and no `pi` binary.

## `hello-run.jsonl`

Recorded 2026-09-28 (Task A2.1 of `docs/plans/2026-09-28-pi-delegate-executor.md`):

```bash
D=$(mktemp -d) && cd "$D" && git init -q
PI_SKIP_VERSION_CHECK=1 pi --mode json -p --no-session --no-approve \
  --no-extensions --no-skills --no-prompt-templates \
  --provider openrouter --model deepseek/deepseek-v4-flash \
  -- "Create hello.txt containing exactly 'hi', then run: cat hello.txt. Reply with the file content." \
  > run.jsonl 2> run.stderr
```

Exit code `0`, 18 s wall clock, empty stderr, `hello.txt` contained `hi`.

Event-type histogram (173 lines):

| type | count |
|---|---|
| `session` | 1 |
| `agent_start` / `agent_end` | 1 / 1 |
| `agent_settled` | 1 |
| `turn_start` / `turn_end` | 4 / 4 |
| `message_start` / `message_end` | 9 / 9 |
| `message_update` | 128 |
| `tool_execution_start` / `tool_execution_end` | 4 / 4 |
| `tool_execution_update` | 7 |

Tool calls: `write` (hello.txt) and three `bash` calls. The first `bash cat`
ran concurrently with the `write` and failed (`isError: true`), so the fixture
contains a real error result as well as three successes.

## Scrub rules

- Recording cwd (`/tmp/tmp.*`) and `$HOME` → `/workspace`.
- The local username → `user` (appeared in `ls -la` output).
- Verified with `rg -i 'sk-|api[_-]?key|authorization|bearer|/home/|/tmp/tmp'` → no matches.
- Session id and timestamps are kept; they carry no secrets.

## Observed shape vs `docs/json.md` (Pi 0.85.1)

The parser in `crates/agent/src/pi_delegate/events.rs` follows the recording:

- `message_end.message.role` takes three values: `user`, `assistant`,
  `toolResult`. Only `assistant` is turned into `PiEvent::AssistantMessage`.
- `message.content` is always an **array of parts**: `{"type":"text","text"}`,
  `{"type":"thinking","thinking"}`, `{"type":"toolCall",...}`. Assistant text
  is the concatenation of the `text` parts; `thinking` parts are dropped.
- `message.stopReason` is `"toolUse"` for turns that call tools and `"stop"`
  for the final answer; `message.usage` is present on assistant messages.
- `message_update.assistantMessageEvent.type` takes `text_start | text_delta |
  text_end | thinking_start | thinking_delta | thinking_end | toolcall_start |
  toolcall_delta | toolcall_end`. Only `text_delta` carries user-visible text.
- `tool_execution_end.result` is `{"content":[{"type":"text","text"}],"details":{}}`
  (`details` may be absent).
- `tool_execution_update` carries `partialResult` with the same shape as
  `result`; ignored in v1.
- `agent_settled` (not in `docs/json.md`) is emitted after `agent_end`;
  treated as a lifecycle event.
- `agent_end.messages` repeats the whole conversation; the parser does not
  read it (the Task result comes from the last assistant `message_end`).
- `session` carries `version`, `id`, `timestamp`, `cwd`.
