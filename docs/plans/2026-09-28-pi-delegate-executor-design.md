# Pi as delegate executor — design (Track A, phase A0)

> **For Claude:** This is a design doc, not an implementation plan. After approval,
> use `writing-plans` to produce `docs/plans/2026-09-28-pi-delegate-executor.md`,
> then `executing-plans` / `subagent-driven-development` per phase A1–A4.

**Status:** DRAFT — awaiting approval. No code written.
**Scope:** Track A only. Track B (`phone-first-decision-chat`) has its own design doc after A4.
**Parent brief:** "Graphirm as director: integrate Pi as the coding executor" (2026-09-28).

---

## Goal

Graphirm stays the director (Jev routing, HITL judge, graph memory, pinned rules).
Pi (`@earendil-works/pi-coding-agent` 0.85.1) becomes a delegated coding executor,
run as a subprocess in `--mode json`, whose every tool call is recorded in the graph
under a `Task` node exactly as in-process delegation is, visible live in the
whiteboard and TUI, scored observe-only by the HITL judge, and killed when the
session aborts.

## Locked (from the brief — not re-opened here)

- Graphirm directs; Pi is the only external executor. No Hermes.
- `--mode json`, observe-only in v1. Gating Pi's calls (`--mode rpc` or a Pi
  extension consulting graphirm's judge) is a follow-up; the process model must
  not require rewriting the event mapping to get there.
- Pi's work is graph-native: tool nodes under the delegated `Task`, final message as
  the Task result, parent Interaction linked as for in-process delegation.
- Jev seats observe-first, fail-soft, raw probabilities in metadata, thresholds in
  readers, versioned constants.
- No change to context selection, compaction, memory ranking, knowledge extraction.

---

## What the code says today (findings that shaped the design)

1. **`ToolContext` has no emitter.** Live events come only from `workflow.rs`
   (`ToolStart` before dispatch, `ToolEnd` after recording, `GraphUpdate` once per
   turn via `emit_graph_update`). A tool that runs for minutes is invisible until it
   returns. `ToolContext` already carries two optional trait objects
   (`knowledge_retriever`, `impact_provider`), so a third follows precedent.
   The struct literal appears 21 times (15 in test helpers) — adding a field is a
   compile-only sweep with no behaviour change.
2. **The HTTP server and TUI never register `delegate`.** `SubagentTool` is only
   wired by `Coordinator::run_primary`, which is used by tests. `routes.rs:628` and
   `chat.rs:37` pass `state.tools` / `build_tool_registry()` straight to
   `run_agent_loop`. So `delegate_pi` will be the first delegation reachable from
   the web-app and TUI. Consequence: `delegate_pi` must not need per-session
   construction (parent id, cancel token, event bus) the way `SubagentTool` does —
   everything it needs must come from `ToolContext`.
3. **`EventBus` is created per prompt** (`routes.rs:584`), not per session or
   process. A tool in the global registry cannot capture it at construction; it
   must be handed in per call → via `ToolContext`.
4. **In-process tool result node shape** (`workflow.rs:1147–1160`):
   `Interaction { role: "tool", content }`, metadata `tool_call_id`, `tool_name`,
   `is_error`, optional `hitl_judge`, `session_id`; label `interaction_{turn}_{pos}_1`;
   edges `Agent --Produces--> node`, `node --RespondsTo--> previous`,
   `node --ApprovedBy--> Agent` when HITL-approved. The web-app's
   `InteractionNode.tsx` highlights `role === 'tool'` with `tool_name ∈ {write, edit, bash}`
   and `NodePopover.tsx` renders `role === 'tool'`. Matching this shape means zero
   UI/eval special-casing.
5. **In-process delegation graph** (`multi.rs:170–262`):
   `Parent Agent --DelegatesTo--> Task --SpawnedBy--> Subagent Agent --Produces--> …`.
   Task status is patched to Completed/Failed at the end. There is no "task result"
   field; `SubagentTool::collect_subagent_output` reads the subagent's `Produces`.
6. **HITL judge is reusable as-is.** `DestructiveJudge::judge(tool_name, args)`
   returns `JudgeVerdict`; `JudgeOutcome::to_metadata()` produces the stored block.
   `build_judge(&AgentConfig)` gives `Option<Arc<DestructiveJudge>>`, `None` when
   disabled or keyless. Tests use `DecisionsTransport` fakes — no network.
7. **Pi's JSON mode is documented** (`docs/json.md` in the package): first line
   `{"type":"session",…}`, then `agent_start / turn_start / message_start /
   message_update{assistantMessageEvent} / message_end{message} / tool_execution_start
   {toolCallId,toolName,args} / tool_execution_update / tool_execution_end
   {toolCallId,toolName,result,isError} / turn_end / agent_end{messages}`, plus
   `error`, `extension_error`, `extension_ui_request`, `queue_update`,
   `compaction_*`. `message_update` is delta-only. Non-interactive modes
   **never show a trust prompt**; `--approve` trusts project resources for the run.
8. **`pi_agent.py`** (codeporate) tolerates two event shapes (top-level and
   nested `assistantMessageEvent`), prefers the last `message_end` as final text,
   writes the prompt to `.pi/oneshot_prompt.txt` and passes `@file`, merges stderr
   into stdout (which pollutes the JSONL), and kills on a total timeout only.
9. **`bash.rs` cancels by `task.abort()`** without `kill_on_drop` — the child
   survives cancellation. Pi must not repeat this: a leaked Pi keeps editing the
   workspace after the user pressed abort.

---

## Design

### D1. Process model

One `pi` subprocess per `delegate_pi` call, via `tokio::process::Command`:

```text
<binary> --mode json -p --no-session --approve
         [--provider <provider>] [--model <model>] [<extra_args>…]
         [--] <task text>            # or @<tmpfile> when task > 64 KiB
cwd    = ctx.working_dir            # the session workspace (or its subagent dir)
stdin  = null                       # Pi must never wait on us
stdout = piped                      # JSONL
stderr = piped                      # kept separate; last 4 KiB retained for errors
env    = inherited unchanged        # Pi's provider key lives in Pi's env/auth.json
kill_on_drop(true); process_group(0) on unix
```

- **Task delivery:** positional argument (Linux `MAX_ARG_STRLEN` is 128 KiB). Above
  64 KiB write to `std::env::temp_dir()/graphirm-pi-<task_id>.md`, pass `@path`,
  delete on exit. Nothing is written into the user's workspace (unlike
  `pi_agent.py`'s `.pi/oneshot_prompt.txt`).
- **Draining without blocking:** `BufReader` over `child.stdout` read with
  `read_until(b'\n')` into a `Vec<u8>` capped at 4 MiB (oversized line → dropped,
  warned, counted). A separate `tokio::spawn` drains stderr into a bounded ring
  buffer. The main loop is a `tokio::select!` over `{ next line, ctx.signal.cancelled(),
  deadline }` — no blocking reads on the runtime, no `spawn_blocking` for I/O.
- **Cancel:** on `ctx.signal.cancelled()` → `killpg(SIGKILL)` on unix
  (`libc`, already transitive; add as a direct dep behind `cfg(unix)`), else
  `child.start_kill()`; then `child.wait()` with a 5 s cap. Task → `Failed`
  (`metadata.failure = "cancelled"`), return `ToolError::Cancelled`. Process group
  kill is needed because Pi's own `bash` spawns grandchildren that
  `start_kill()` on the node process would orphan.
- **Timeout:** `timeout_seconds` (default 900) total wall-clock. Same kill path,
  `ToolError::Timeout`, Task `Failed` (`failure = "timeout"`), summary of what was
  recorded so far is still returned in the error text so the parent can reason
  about partial work. No idle timeout in v1 (total covers a hung network call);
  revisit if real runs show long silent stretches.
- **Exit:** wait for `agent_end` **or** EOF, then `child.wait()`. Non-zero exit
  with no `message_end` → tool error carrying the stderr tail. Non-zero exit *with*
  a final message → success with `exit_code` recorded in Task metadata (Pi exits 1
  on some provider errors after producing text; the parent should still see it).

**Why this survives the switch to `--mode rpc`:** RPC mode emits the *same*
`AgentEvent` records on stdout and adds a stdin command channel. The event parser
(D2) is a pure `&str → PiEvent` function fed by a line source; the process wrapper
owns spawn/drain/kill. Switching means adding a stdin writer and a `prompt`
command, not touching the mapping or the graph writes. The parser is therefore a
separate module (`pi_delegate/events.rs`) with no `tokio::process` import.

### D2. Event mapping (Pi JSONL → graph)

Parsed per line into `enum PiEvent` via `serde_json::Value` + `match type`; any
unknown `type` → `PiEvent::Unknown(String)` (debug-logged, counted, never fatal).
Malformed JSON → skipped, warned, counted. `\r` stripped.

| Pi event | Graph / event action |
|---|---|
| `session` | Store `pi_session_id`, `pi_cwd`, `pi_version` (from `--version` at spawn) in the **Pi Agent node** metadata. |
| `agent_start`, `turn_start`, `turn_end`, `queue_update`, `compaction_*` | Counters only. |
| `message_start` | Nothing. (Pi's text is not the director's text; do **not** emit `MessageStart/Delta` on the parent's stream — the web-app would render it as the assistant speaking.) |
| `message_update` (`assistantMessageEvent.type == text_delta`) | Append to an in-memory buffer for the current assistant message. Not forwarded in v1 (see D3 follow-up). |
| `message_end`, `message.role == assistant` | New `Interaction { role: "assistant", content: flattened text parts }`, `Pi Agent --Produces--> node`, `RespondsTo` chain among Pi's nodes; metadata `executor: "pi"`, `stop_reason`, `usage`. Remember as `last_assistant_text`. |
| `message_end`, `role == user` / `toolResult` | Skip (prompt echo; tool results are recorded from `tool_execution_end`). |
| `tool_execution_start {toolCallId, toolName, args}` | Emit `ToolStart { response_node_id: ctx.interaction_id, call_id: toolCallId, tool_name: toolName }` via the sink. If `toolName ∈ {bash, write, edit}` and a judge exists: spawn `judge(toolName, args)` (non-blocking, keyed by `toolCallId`). |
| `tool_execution_update` | Ignore. |
| `tool_execution_end {toolCallId, toolName, result, isError}` | Create tool node (shape below); attach judge verdict if its future has resolved (await it up to the judge's own timeout — it started seconds ago); emit `ToolEnd { node_id, is_error }` and `GraphUpdate` via the sink. |
| `error`, `extension_error` | Append to an `errors` list; surfaced in the Task metadata and in the tool error text if the run fails. |
| `extension_ui_request` | Should not occur in `-p` mode; if it does, log at warn and continue (Pi will time out its own UI request). |
| `agent_end` | Mark run complete; break the read loop after draining. |

**Pi tool node shape** — identical to in-process (finding 4) plus three keys:

```json
{
  "node_type": { "Interaction": { "role": "tool", "content": "<result text, ≤ max_result_chars>" } },
  "metadata": {
    "session_id": "<pi agent node id>",
    "parent_session_id": "<ctx.agent_id>",
    "tool_call_id": "<Pi toolCallId>",
    "tool_name": "bash",
    "is_error": false,
    "executor": "pi",
    "arguments": "<compact JSON, ≤ 1500 chars>",
    "hitl_judge": { "version": "v1", "p_irreversible": 0.07, "threshold": 0.8, "action": "observed", "latency_ms": 310 }
  }
}
```

- `tool_name` is Pi's real name (`bash`/`read`/`write`/`edit`/…) so the existing
  UI destructive highlight and the judge's instructions apply unchanged.
  `executor: "pi"` is the discriminator.
- `arguments` is stored because, unlike in-process calls, there is no assistant
  Interaction in *our* graph holding Pi's tool-call parts. Truncated with the same
  `MAX_ARGS_CHARS` rule the judge uses.
- Label `interaction_{ctx.turn}_{ctx.turn_pos_counter++}_1` — same counter the
  parent's tools use, so ordering inside the turn is preserved.

**Graph structure** (mirrors `spawn_subagent`):

```text
Parent Agent      --DelegatesTo--> Task("Delegated to pi")
Parent Interaction --Produces-->   Task            # the assistant turn that called delegate_pi
Task              --SpawnedBy-->   Pi Agent(name="pi", model=<cfg.model>, status)
Pi Agent          --Produces-->    tool nodes, assistant nodes  (RespondsTo-chained)
```

- `Interaction --Produces--> Task` is the "parent Interaction links to it" edge.
  In-process delegation has only the Agent edge; adding the Interaction edge is
  what lets the chat UI (B2 STEPS) find the Task from the turn without a BFS from
  the Agent. Cheap, additive, and the whiteboard already draws `produces`.
- **Task result:** `TaskData` has no result field. Rather than change the graph
  schema (and `web-app/src/types/graph.ts`), store `metadata.result` =
  `last_assistant_text` (≤ `max_result_chars`), plus `metadata.executor = "pi"`,
  `tool_calls`, `exit_code`, `duration_ms`, `pi_session_id`, `failure` (when
  failed). Status → `Completed` / `Failed` as `spawn_subagent` does.
- The `delegate_pi` tool's **return value** (the parent's own tool-result node,
  recorded by `workflow.rs` as usual) is a summary:
  `"Pi completed.\nTask ID: …\nTool calls: N (bash 4, edit 2, read 7)\nJudge: 2 calls ≥ 0.8 (observed, not gated)\n\nFinal message:\n<last_assistant_text>"`.
  Returned via `ToolOutput::success_with_node(summary, task_id)` (the `node_id`
  field is unused by `workflow.rs` today but is the right hook for B2).

### D3. Live visibility — option (a): emitter on `ToolContext`

**Chosen: (a).** Add to `graphirm-tools`:

```rust
/// Lets a long-running tool report progress through the agent loop's event
/// stream without depending on `graphirm-agent`.
pub trait ToolEventSink: Send + Sync {
    fn tool_started(&self, response_node_id: &NodeId, call_id: &str, tool_name: &str);
    fn tool_finished(&self, node_id: &NodeId, is_error: bool);
    /// `anchor` is the node the update is about; `touched` are nodes created since the last call.
    fn graph_changed(&self, anchor: &NodeId, touched: &[NodeId]);
}
// ToolContext gains: pub event_sink: Option<Arc<dyn ToolEventSink>>,
```

`graphirm-agent` provides `EventBusSink { bus: Arc<EventBus>, graph: Arc<GraphStore> }`
mapping the three methods to `AgentEvent::ToolStart / ToolEnd / GraphUpdate` (the
last by `tokio::spawn`-ing the existing `emit_graph_update` payload builder, factored
to take `&GraphStore` instead of `&Session`). `workflow.rs` sets
`event_sink: Some(Arc::new(EventBusSink::new(events.clone(), session.graph.clone())))`
at the one production `ToolContext` literal (`workflow.rs:925`).

**Blast radius (a):** 21 struct literals gain `event_sink: None` (compile-only; 15
are test helpers — `make_test_context()` in tools and delegate tests cover most
sites); one new trait + one adapter; `emit_graph_update` signature tweak. No
existing tool reads the field; behaviour of every current tool is unchanged;
existing tool tests stay green with `None`.

**Rejected (b): `GraphUpdate` on node insertion.** There is nothing to emit *from*:
the `EventBus` is per prompt (finding 3) and `GraphStore` (crate `graphirm-graph`)
has no event channel. Making (b) work means either hooks in the graph crate (blast
radius: every `add_node` caller, a new dependency direction) or per-prompt registry
rebuilding in `routes.rs` and `chat.rs` to hand the bus to the tool — which is (a)
with worse ergonomics. (b) also fires on *every* node the whole process inserts
(knowledge extraction, segments), not just the tool's, so the whiteboard would
re-layout on unrelated writes.

**Emission cadence:** one `ToolStart` per Pi call at start; one `ToolEnd` +
`GraphUpdate` per Pi call at end; one `GraphUpdate` after Task/Agent creation and
one at completion. `GraphUpdate` builds a ≤ 50-node payload (`list_recent_nodes(50)`)
— at Pi's rate (~1 call / few seconds) this is fine; if a run shows > 5 calls/s,
coalesce with a 250 ms debounce in the sink (follow-up, not v1).

**Follow-up hook (not v1):** `ToolEventSink::progress(&self, text: &str)` for Pi's
text deltas, mapped to a new `AgentEvent::DelegateProgress` that B2's STEPS row
can render. Left out because no consumer exists yet.

### D4. HITL judge on Pi's calls (observe-only)

- Reuse `DestructiveJudge` unchanged. `delegate_pi` holds `Option<Arc<DestructiveJudge>>`
  built by `build_judge(&config)` at registration (same `[agent.hitl_judge]` and
  `[agent.adaptive_routing.jev]` as the in-process gate).
- Judge only Pi's `bash`, `write`, `edit`. Start the call at `tool_execution_start`
  (arguments are complete there), attach at `tool_execution_end`.
- Verdict stored as `metadata.hitl_judge` with `action: "observed"` — a new action
  value so read-outs can separate "scored but not gated" from `paused` / `approved`.
  Add `JUDGE_ACTION_OBSERVED: &str = "observed"` next to `JUDGE_VERSION`.
- Fail-soft: `Err` → no `hitl_judge` key, `tracing::warn!` once per run (not per
  call), run continues. Judge never blocks Pi, never delays node creation by more
  than the judge's own `timeout_ms`.
- Tool description and the delegation summary both state: "Pi's tool calls are
  recorded and scored but cannot be paused in this mode."

### D5. Where `delegate_pi` sits — a second tool

**Chosen: a separate tool `delegate_pi`.** Rejected: an `executor` parameter on
`delegate`.

- `delegate` requires an `agent` name from `AgentRegistry` and is constructed per
  session with `llm_factory`, `base_tools`, `events`, `parent_agent_id`, `cancel`
  (finding 2). Bolting an `executor` switch onto it would force `delegate_pi` to
  inherit that construction and stay unreachable from the server and TUI.
- The model sees two clearly different tools: `delegate(agent, task)` = "an
  in-process subagent with scoped tools and graph context", `delegate_pi(task,
  context_paths?)` = "an external coding agent in the workspace; one bounded,
  well-specified implementation task; returns its final message; its tool calls are
  recorded and judged but not gated". Distinct names make the routing decision
  legible in the graph (`tool_name` on the parent's result node) and in B9's
  `PlanStep.executor`.
- **Parameters:** `task: string` (required), `context_paths: string[]` (optional;
  rendered into the task text as "Relevant files: …" — Pi reads them itself),
  `timeout_seconds: integer` (optional, capped by config).
- `is_destructive() == true` — the parent's HITL gate pauses on `delegate_pi` like
  on `bash`, so a human approves *sending work to Pi* even though Pi's internal calls
  are not gated. Confirm card (B5) shows the task brief.
- **System prompt:** one paragraph appended when the tool is registered (same
  idempotent pattern as `apply_disable_bash_system_notice`): when to use it
  (multi-file implementation with a clear brief, tests to run), when not to (reads,
  questions, anything needing graph memory or the user's approval per step), and
  that the result must be verified by graphirm's own tools before claiming done.
- **`disable_bash` interplay:** Pi runs shell. If `ctx.disable_bash`, `execute`
  returns `ExecutionFailed("delegate_pi is disabled: bash is locked down on this server")`,
  and `stream_and_record` omits it from the tool list alongside `bash`
  (`workflow.rs:245` filter gains `|| t.name == "delegate_pi"`).

**Registration:** stateless w.r.t. session — `PiDelegateTool { config: PiConfig, judge: Option<Arc<DestructiveJudge>>, pi_version: Option<String> }`.
Registered in `src/commands/serve.rs` and `src/commands/chat.rs` after
`AgentConfig` is loaded (today `build_tool_registry()` runs *before* config in
`serve.rs`; reorder so `register_pi_delegate(&mut registry, &agent_config)` can run).
Only registered when `[agent.pi].enabled = true`; when disabled the model never sees it.

### D6. Config

```toml
[agent.pi]
enabled = false                       # flip to true in its own commit after A4 passes
binary = "pi"                         # resolved on PATH; or absolute, e.g. "~/.nvm/versions/node/v24.18.0/bin/pi" (tilde expanded)
provider = "openrouter"               # passed as --provider
model = "deepseek/deepseek-v4-flash"  # passed as --model (Pi's own model string, not graphirm's); approved 2026-09-28
timeout_seconds = 900                 # total wall-clock per delegation; per-call override is capped here
extra_args = []                       # appended verbatim before the task, e.g. ["--thinking", "low", "--no-extensions"]
max_result_chars = 16000              # truncation for tool-result content and Task result
```

`PiConfig` in `crates/agent/src/config.rs` with `#[serde(default)]` and a `Default`
impl matching the values above; `AgentConfig.pi: Option<PiConfig>`. Provider key
handling: none — graphirm passes its environment through and never reads
`~/.pi/agent/auth.json` or provider variables. The binary is probed once at
registration (`<binary> --version`, 5 s timeout) to record `pi_version` and warn
early if missing; a missing binary at registration does **not** fail startup (the
tool still registers so the failure is visible to the model as a tool error).

### D7. Failure modes

| Failure | Detection | Result to parent (never a panic, never a stuck turn) |
|---|---|---|
| Pi not installed / not executable | `spawn()` → `ErrorKind::NotFound` / `PermissionDenied` | `ExecutionFailed("pi not found at '<binary>'; set [agent.pi].binary or install @earendil-works/pi-coding-agent")`. No Task node created. |
| Provider key missing / auth error | Pi exits non-zero, emits `error` event or stderr text, no `message_end` | `ExecutionFailed("pi exited <code> without a result: <stderr tail ≤ 1 KiB>")`. Task `Failed`, `failure = "exit"`. |
| Non-zero exit *with* a final message | `exit_code != 0 && last_assistant_text.is_some()` | Success summary with `exit_code` noted; Task `Completed`, `exit_code` in metadata. |
| Malformed JSONL line | `serde_json` error | Skip, `warn!` (first 3 per run, then count), `malformed_lines` in Task metadata. |
| Unknown event type | no match arm | `debug!`, counted in `unknown_events`. |
| Line > 4 MiB | reader cap | Drop line, `warn!`, counted. |
| Trust prompt | cannot happen: non-interactive modes never prompt; `--approve` added anyway | — |
| `extension_ui_request` | event | `warn!`; continue; Pi's own timeout resolves it. |
| Pi hangs (no output, no exit) | `timeout_seconds` deadline | Kill group; `ToolError::Timeout`; Task `Failed`, `failure = "timeout"`; partial summary in error text. |
| Session aborted | `ctx.signal.cancelled()` | Kill group within 5 s; `ToolError::Cancelled`; Task `Failed`, `failure = "cancelled"`. |
| Graph write fails mid-run | `GraphError` | `error!`, keep draining Pi (do not kill it for our own bug), count `graph_write_errors`; summary notes it. |
| Judge unavailable / errors | `build_judge → None` / `judge() → Err` | No `hitl_judge` metadata; one `warn!` per run. |
| `disable_bash` session | `ctx.disable_bash` | `ExecutionFailed(...)` before spawn; tool hidden from model. |
| Temp prompt file cannot be written | `io::Error` | `ExecutionFailed`; no spawn. |

---

## Module layout

```text
crates/tools/src/lib.rs                 + trait ToolEventSink; ToolContext.event_sink
crates/agent/src/event_sink.rs          EventBusSink (ToolEventSink → AgentEvent)
crates/agent/src/pi_delegate/mod.rs     PiDelegateTool (Tool impl), register_pi_delegate(), system-prompt notice
crates/agent/src/pi_delegate/process.rs spawn / drain / kill / timeout (tokio::process)
crates/agent/src/pi_delegate/events.rs  PiEvent parser (pure; no process imports) ← reused by --mode rpc later
crates/agent/src/pi_delegate/graph.rs   Task / Pi Agent / tool-node writes, summary builder
crates/agent/src/config.rs              + PiConfig
crates/agent/src/hitl_judge.rs          + JUDGE_ACTION_OBSERVED
crates/agent/src/workflow.rs            ToolContext.event_sink wiring; tool-list filter for delegate_pi
src/commands/{serve,chat}.rs            registration after config load
config/default.toml                     [agent.pi] block (enabled = false)
crates/agent/tests/fixtures/pi/         recorded JSONL (scrubbed) + fake_pi.sh
```

## Testing (all offline, `cargo test` needs no network and no real `pi`)

- **Parser (`events.rs`):** one test per event type from the recorded fixture;
  unknown type; malformed line; `\r\n`; oversized line; `message.content` as string
  vs typed parts; `role == user` skipped.
- **Fake Pi:** `crates/agent/tests/fixtures/pi/fake_pi.sh` — `cat`s a fixture JSONL
  with a small per-line delay; env knobs `FAKE_PI_EXIT` (exit code),
  `FAKE_PI_HANG=1` (sleep forever after N lines, for cancel/timeout tests),
  `FAKE_PI_GARBAGE=1` (interleave non-JSON lines), `FAKE_PI_NO_END=1` (exit before
  `message_end`). Tests point `[agent.pi].binary` at it.
- **Fixture:** one real run recorded in A2 (`pi --mode json -p --no-session --approve
  --provider openrouter --model … "create hello.txt containing 'hi', then cat it"`
  in a `mktemp -d`), scrubbed: `cwd` → `/workspace`, any `/home/…` → `/workspace`,
  no keys (Pi never prints them, verify with `rg -i 'sk-|key'`). Committed as
  `crates/agent/tests/fixtures/pi/hello-run.jsonl`.
- **Tool tests (`pi_delegate/mod.rs`):** happy path builds the exact graph shape
  (Task, Pi Agent, N tool nodes with `executor: "pi"`, `RespondsTo` chain, Task
  `metadata.result`, status Completed); not-installed error; non-zero exit without
  result; non-zero with result; malformed lines counted; cancel kills within 5 s
  (assert child pid gone); timeout; `disable_bash` refusal; sink receives
  `ToolStart/ToolEnd` per Pi call in order and `GraphUpdate` per end.
- **Judge (A3):** `ReplyTransport` fake → verdict on `bash`/`write`/`edit` nodes
  only, `action == "observed"`; `HangingTransport` → node still created, no key;
  `build_judge → None` → no key, no error.
- **Regression:** existing `graphirm-tools`, `graphirm-agent`, `graphirm-server`
  tests unchanged except the mechanical `event_sink: None` additions.
- **A4 live check** (recorded in this doc's "A4 findings" section when done):
  session id, whiteboard screenshot, TUI observation, one abort, judge verdicts.

## Out of scope for Track A

Gating Pi's calls; forwarding Pi's text deltas; per-call idle timeout; `GraphUpdate`
debounce; any web-app change (Track B); `delegate` (in-process) wiring into the
server/TUI (pre-existing gap, noted for the backlog); fixing `bash.rs`'s
cancel-without-kill (pre-existing, noted for the backlog).

---

## Decisions

| # | Decision | Alternatives considered | Why |
|---|---|---|---|
| 1 | Live visibility via `ToolContext.event_sink: Option<Arc<dyn ToolEventSink>>` (option a) | (b) `GraphUpdate` on insert from the graph store; (c) per-prompt registry rebuild to hand the tool the `EventBus` | (b) has no emitter to hook and would fire on unrelated writes; (c) is (a) with more churn in `routes.rs`/`chat.rs`. (a) follows the existing `knowledge_retriever` / `impact_provider` precedent; 21 compile-only edits, zero behaviour change for current tools. |
| 2 | Separate tool `delegate_pi`, not an `executor` param on `delegate` | `delegate(executor="pi")` | `delegate` is constructed per session and is not registered in server/TUI; a separate tool is reachable everywhere, gives the model a legible choice, and maps 1:1 to `PlanStep.executor`. |
| 3 | Tool nodes are `Interaction{role:"tool"}` with in-process metadata keys + `executor:"pi"`, `arguments`, `parent_session_id` | Dedicated `Content{content_type:"pi_tool_call"}` nodes | Whiteboard (`InteractionNode.tsx`), popover, `infer_task_phase`, and eval already read `role == "tool"` + `tool_name`; Content nodes would need special cases everywhere. |
| 4 | Pi gets its own `Agent` node (`SpawnedBy` from the Task), mirroring `spawn_subagent` | Hang tool nodes directly off the Task | Same shape as in-process delegation → same BFS depth, same `collect_subagent_results`-style reads, same whiteboard grouping. The Agent node also carries `pi_session_id`/`pi_version`. |
| 5 | Task result in `Task.metadata.result` (+ final assistant Interaction node) | Add `result: Option<String>` to `TaskData` | No graph schema or `types/graph.ts` change; `metadata` is already free-form on both sides. Revisit if B9 needs typed results. |
| 6 | `Interaction --Produces--> Task` in addition to `Agent --DelegatesTo--> Task` | Only the Agent edge (as in-process today) | Lets the chat UI find the Task from the turn without a BFS from the Agent; additive; `produces` is already drawn. |
| 7 | Task text via positional argv; temp file outside the workspace only above 64 KiB | `.pi/oneshot_prompt.txt` in the workspace (pi_agent.py) | Do not write artefacts into the user's repo; briefs are small. |
| 8 | stderr piped separately, ring-buffered | Merge into stdout (pi_agent.py) | Merged stderr corrupts the JSONL stream. |
| 9 | Kill by process group (`libc::killpg`, unix) with `kill_on_drop(true)` | `start_kill()` on the node process only; `task.abort()` as `bash.rs` does | Pi's `bash` grandchildren would survive; `bash.rs`'s pattern leaks the child entirely. |
| 10 | Total timeout only (900 s default) | + idle timeout | YAGNI until a real run shows silent hangs shorter than the total. |
| 11 | Pi text deltas not forwarded in v1 | Emit `MessageDelta` on the parent's node | Would render as the director speaking in every client; the right consumer (STEPS) doesn't exist until B2. Hook named for the follow-up. |
| 12 | `hitl_judge.action = "observed"` on Pi nodes | Reuse `"approved"` | Read-outs must separate scored-not-gated from gated; a new constant costs nothing. |
| 13 | `delegate_pi.is_destructive() == true` | Non-destructive (Pi's calls are judged anyway) | The human should approve handing the workspace to an ungated executor; also gives B5 a natural confirm card with the brief. |
| 14 | Registered only when `[agent.pi].enabled`; binary probed at registration but absence is non-fatal | Fail startup when misconfigured | Spoke image has no node; the model should see a clear tool error rather than the server refusing to start. |
| 15 | Parser in its own module with no process imports | Inline in the tool | Same parser serves `--mode rpc` later (same event records on stdout). |
| 16 | Default Pi model `openrouter` / `deepseek/deepseek-v4-flash` (approved) | `deepseek/deepseek-v3.2` (graphirm's chat default) | Matches the codeporate Pi setup that is already known to work; Pi's model string is independent of graphirm's. |
| 17 | `[agent] default_auto_approve = true` (approved) | Keep per-request `auto_approve` defaulting to `false` | With `delegate_pi` destructive (13), a default of `false` would pause every delegation; `hitl_judge` ≥ 0.8 still pauses graphirm's own risky calls. |
| 18 | `bash.rs` cancel fix folded into A1 (approved); `delegate` server/TUI wiring proposed as A1b pending size confirmation | Backlog both | User asked to fix now; the bash fix is S and touches the same file family. The delegate wiring is M (no registry, no subagent configs, no factory in server/TUI) and deserves its own section. |
| 19 | **Proposed:** `--no-approve` by default (`trust_project = false`) | `--approve` (brief's A0 text, codeporate precedent) | Pi's security doc: `--approve` loads project `.pi/extensions` that run inside Pi's process. We pass the task via argv and need nothing from `.pi/`; `AGENTS.md` loads regardless. Awaiting approval (Q5). |
| 20 | **Proposed:** `extra_args` default `["--no-extensions", "--no-skills", "--no-prompt-templates"]`; child env `PI_SKIP_VERSION_CHECK=1` | Inherit user's Pi extensions; let Pi ping pi.dev | Keeps the JSONL stream to documented built-ins in v1; no network ping per delegation. Awaiting approval (Q6). |

## Pi dos and don'ts (from pi.dev and the `pi-mono` docs, read 2026-09-28)

What Pi's own documentation says about running it unattended, and what each point
means for this design.

| Pi says | Consequence here |
|---|---|
| "Pi does not ask before every tool call. Treat model-generated commands and code as untrusted." (security.md) | Matches observe-only v1. The **session workspace is not a security boundary**; Pi runs with graphirm's OS user. On a shared or public spoke `delegate_pi` must stay disabled (it already is under `disable_bash`). Real isolation = container/VM (Pi's containerization.md); follow-up for the spoke. |
| "Watching the transcript, using project trust, and reviewing changes do not create a security boundary." | The judge's `p_irreversible` is telemetry and UI signal, not protection. The design doc and tool description say so. |
| Project trust gates `.pi/settings.json`, `.pi/extensions|skills|prompts|themes`, `.pi/SYSTEM.md`, project `.agents/skills`. `--approve` trusts them for the run; `--no-approve` skips them. **`AGENTS.md`/`CLAUDE.md` load regardless.** | **Proposed change to D1:** default to `--no-approve`, not `--approve`. A workspace repo can ship `.pi/extensions/*.ts` that execute inside Pi's process; the director should not grant that by default. New config `trust_project = false` (→ `--no-approve`; `true` → `--approve`). Codeporate used `--approve` because its worktrees carried `.pi/oneshot_prompt.txt`; we pass the task via argv, so we need nothing from `.pi/`. Pi still reads the repo's `AGENTS.md`, which is what we want. |
| Non-interactive modes (`-p`, `--mode json`, `--mode rpc`) never show the trust prompt. | Confirms D7: the "asks for trust interactively" failure cannot occur. |
| Extensions are TypeScript modules running **inside the Pi process** with its permissions; user-level ones load from `~/.pi/agent/extensions` regardless of project trust and can add events (`extension_ui_request`, custom tools). | **Proposed default** `extra_args = ["--no-extensions", "--no-skills", "--no-prompt-templates"]` so the JSONL stream and tool set are the documented built-ins. Users who want their extensions remove the flags. Keeps the parser's assumptions true in v1. |
| `PI_SKIP_VERSION_CHECK` disables the `pi.dev` latest-version request; `PI_OFFLINE` disables all startup network activity (also model-catalog refresh). | Set `PI_SKIP_VERSION_CHECK=1` in the child env (no version ping per delegation, no "update available" noise on stderr). Do **not** set `PI_OFFLINE` — it may block provider catalog data OpenRouter models need. |
| Pi sets `AI_AGENT=pi` and `PI_CODING_AGENT=true` in child processes; its `bash` tool gets `PI_SESSION_ID`, `PI_PROVIDER`, `PI_MODEL`. | Useful for the A4 live check (`printf '%s/%s' "$PI_PROVIDER" "$PI_MODEL"` proves which model ran). Nothing to build. |
| Sessions auto-save under `~/.pi/agent/sessions/` by cwd; `--no-session` makes the run ephemeral. | Keep `--no-session`: graphirm's graph is the record. Without it every delegation would leave a session file on disk outside the graph. |
| "Files, comments, instructions, command output … can steer the model through prompt injection." | Pi's final message and tool results enter the director's context as tool output. The director already treats tool results as data; the summary must not be re-injected as a system or user message. Noted in D2. |
| "Use snapshots, backups, or version control before substantial changes." | The system-prompt paragraph for `delegate_pi` (D5) tells the director to work on a clean git state or a branch and to verify with its own tools afterwards. |
| "Review sessions before exporting or sharing them. They can contain … credentials exposed during the conversation." | Applies to our recorded fixture (scrub) and to graph exports of sessions that used Pi. |
| Pi's core skips sub-agents and plan mode by design; the RPC mode and the `tool_call` extension event (`{ block: true, reason }`) are the documented ways to steer or gate. | Confirms the follow-up path for gating: either `--mode rpc` (same events + stdin commands) or a small Pi extension calling graphirm's judge. D1's parser split keeps both open. |

## Open questions for the approver

1. ~~Default Pi model~~ **Answered 2026-09-28:** `provider = "openrouter"`,
   `model = "deepseek/deepseek-v4-flash"`.
2. ~~Decision 6~~ **Answered 2026-09-28: yes** — add `Interaction --Produces--> Task`.
3. ~~Decision 13~~ **Answered 2026-09-28: yes, and auto-approve should be on by
   default.** Today `auto_approve` is a per-request flag on `POST /api/sessions`
   defaulting to `false`. Add `[agent] default_auto_approve = true` (used when the
   request omits the flag; the TUI reads the same setting). The `hitl_judge` still
   adds a pause for graphirm's own calls scoring ≥ 0.8, so the safety net stays.
   Scheduled in A2 (config phase).
4. ~~Pre-existing gaps~~ **Answered 2026-09-28: fix now.** Split by size:
   - `bash.rs` cancel leaks the child — **S**, folded into A1 (`kill_on_drop(true)`
     + explicit kill on cancel/timeout; one test asserting the pid is gone).
   - `delegate` unreachable from server/TUI — **M, not a fix**: the server and TUI
     have no `AgentRegistry`, no subagent TOMLs (`config/agents/` does not exist),
     and no `LlmFactory`; `SubagentTool` also needs the per-prompt `EventBus`, so
     the registry must be composed per prompt in `routes.rs` and `chat.rs`. Doing it
     means deciding which subagents exist and which models they run. Proposed as
     **A1b** (own plan section, own commits) — confirm or leave in the backlog.
5. **New — trust default:** switch D1 from `--approve` to `--no-approve`
   (`trust_project = false`) per the dos/don'ts table? Recommended.
6. **New — extension defaults:** `extra_args` default
   `["--no-extensions", "--no-skills", "--no-prompt-templates"]`? Recommended for v1.

## A4 findings

_(to be filled after the live check: session id, whiteboard/TUI observations,
abort timing, judge verdicts, anything that forces a change to the Decisions table)_
