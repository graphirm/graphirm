# Pi Delegate Executor — Implementation Plan (Track A)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or
> superpowers:subagent-driven-development when run in-session) to implement this
> plan task-by-task. Work in a git worktree (`using-git-worktrees`). Run
> `cargo fmt && cargo clippy --workspace -- -D warnings && cargo test --workspace`
> before every commit. Tick checkboxes here as tasks complete; update
> `docs/backlog.md` / `docs/completion-log.md` / `docs/journal.md` in the same
> commits (see `AGENTS.md` → Governance docs).

**Goal:** The director agent can call `delegate_pi(task)`; a `pi --mode json`
subprocess does the coding in the session workspace; every Pi tool call becomes a
graph node under a `Task` as it happens, visible in the whiteboard and TUI, scored
observe-only by the HITL judge, and killed when the session aborts.

**Architecture:** A `ToolEventSink` trait on `ToolContext` lets any long-running
tool emit `ToolStart / ToolEnd / GraphUpdate` through the agent loop's `EventBus`.
`PiDelegateTool` (in `graphirm-agent`) spawns Pi, parses its JSONL with a pure
`PiEvent` parser, writes `Task → Pi Agent → Interaction{role:"tool"}` nodes in the
in-process shape, judges Pi's `bash/write/edit` arguments fail-soft, and returns
Pi's final message as the tool result. Registered only when `[agent.pi].enabled`.

**Tech stack:** Rust 2024 / MSRV 1.88, `tokio::process`, `serde_json`, existing
`DestructiveJudge` + `DecisionsClient` (fake transport in tests), `libc::killpg`
behind `cfg(unix)`, a shell-script fake `pi` for offline tests.

**Key decisions** (full table in the design doc
`docs/plans/2026-09-28-pi-delegate-executor-design.md`):
- Live visibility via `ToolContext.event_sink: Option<Arc<dyn ToolEventSink>>` (D3).
- Separate `delegate_pi` tool, destructive, stateless w.r.t. session (D5).
- Pi tool nodes are `Interaction{role:"tool"}` with in-process metadata + `executor:"pi"` (D2).
- `--no-approve`, `--no-extensions --no-skills --no-prompt-templates`, `PI_SKIP_VERSION_CHECK=1` (decisions 19–20).
- `[agent] default_auto_approve = true` (decision 17). Server only — the TUI has no HITL gate today.

**Success criteria (Track A acceptance):**
- From the whiteboard or TUI the agent delegates a bounded task to Pi; Pi's tool
  calls appear as nodes under the Task while Pi runs; the Task carries Pi's final
  message in `metadata.result`; each Pi `bash`/`write`/`edit` node carries
  `metadata.hitl_judge` with `action: "observed"`; aborting the session kills Pi
  within 5 s; with Pi missing or misconfigured the tool returns an error and the
  turn continues; `cargo test --workspace` is green with no network and no real `pi`.

**Risks / blockers:**
- Real Pi JSONL may differ from `docs/json.md` in small ways (e.g. `message.content`
  shape). Task A2.1 records a real run first and the parser is written against it.
- `libc::killpg` is unix-only; the plan gates it with `cfg(unix)` and falls back
  to `start_kill()` elsewhere. CI is Linux.
- `ToolContext` literal sweep (Task A1.1) touches 20 sites; compile errors are the
  guide — do not hand-count.

---

## Phase A1 — Event plumbing + bash cancel fix

### Task A1.1: `ToolEventSink` trait and `ToolContext.event_sink`

**Files:**
- Modify: `crates/tools/src/lib.rs:44-65` (struct), add trait above it
- Modify (add `event_sink: None`): `crates/tools/src/lib.rs:206`, `crates/tools/tests/integration.rs:47`,
  `crates/tools/src/script.rs:174`, `crates/agent/src/trace_analysis_tool.rs:97`,
  `crates/agent/src/delegate.rs:229`, `crates/agent/src/workflow.rs:925` (set to `None` here for now; A1.3 wires it).
  Actual literal count is 6 — the per-tool `make_ctx_with_dir` helpers in
  `crates/tools/src/{bash,find,grep,edit,read_many,write,diff,read,ls}.rs` reuse `make_test_context()`
  and need no change.
- Test: `crates/tools/src/lib.rs` (tests module)

**Step 1: Write the failing test** (in `crates/tools/src/lib.rs` tests):

```rust
#[test]
fn tool_context_event_sink_defaults_to_none_and_accepts_a_sink() {
    use std::sync::Mutex;
    struct Recording(Mutex<Vec<String>>);
    impl ToolEventSink for Recording {
        fn tool_started(&self, _r: &NodeId, call_id: &str, tool_name: &str) {
            self.0.lock().unwrap().push(format!("start:{call_id}:{tool_name}"));
        }
        fn tool_finished(&self, node_id: &NodeId, is_error: bool) {
            self.0.lock().unwrap().push(format!("end:{node_id}:{is_error}"));
        }
        fn graph_changed(&self, anchor: &NodeId, touched: &[NodeId]) {
            self.0.lock().unwrap().push(format!("graph:{anchor}:{}", touched.len()));
        }
    }
    let mut ctx = make_test_context();
    assert!(ctx.event_sink.is_none());
    let sink = Arc::new(Recording(Mutex::new(vec![])));
    ctx.event_sink = Some(sink.clone());
    let s = ctx.event_sink.as_ref().unwrap();
    s.tool_started(&ctx.interaction_id, "c1", "bash");
    s.tool_finished(&ctx.interaction_id, false);
    s.graph_changed(&ctx.interaction_id, &[]);
    assert_eq!(sink.0.lock().unwrap().len(), 3);
}
```

**Step 2:** `cargo test -p graphirm-tools tool_context_event_sink` → FAIL (no field / trait).

**Step 3: Implement** in `crates/tools/src/lib.rs`, above `ToolContext`:

```rust
/// Lets a long-running tool report progress through the agent loop's event
/// stream without depending on `graphirm-agent`. All methods are synchronous
/// and must not block; implementations forward to a channel or `tokio::spawn`.
pub trait ToolEventSink: Send + Sync {
    /// A sub-step began. `response_node_id` is the assistant Interaction that
    /// owns the current turn; `call_id` is unique within the run.
    fn tool_started(&self, response_node_id: &NodeId, call_id: &str, tool_name: &str);
    /// A sub-step's result node was written.
    fn tool_finished(&self, node_id: &NodeId, is_error: bool);
    /// Nodes were inserted; `anchor` is the node the update is about.
    fn graph_changed(&self, anchor: &NodeId, touched: &[NodeId]);
}
```

Add to `ToolContext`:

```rust
    /// Optional progress emitter for long-running tools (e.g. `delegate_pi`).
    /// `None` for tools that return promptly; existing tools ignore it.
    pub event_sink: Option<Arc<dyn ToolEventSink>>,
```

Add `event_sink: None,` to every literal listed above. Export `ToolEventSink` from
`lib.rs` (`pub use` not needed — it is defined there).

**Step 4:** `cargo test --workspace` → PASS (compile errors point to any missed literal).

**Step 5: Commit**

```bash
git add crates/tools crates/agent/src/{workflow,delegate,trace_analysis_tool}.rs
git commit -m "tools: add ToolEventSink trait and ToolContext.event_sink (None everywhere)"
```

- [x] A1.1 done

### Task A1.2: `EventBusSink` adapter in `graphirm-agent`

**Files:**
- Create: `crates/agent/src/event_sink.rs`
- Modify: `crates/agent/src/lib.rs` (`pub mod event_sink; pub use event_sink::EventBusSink;`)
- Modify: `crates/agent/src/workflow.rs:1398-1458` — split `emit_graph_update` into
  `pub(crate) async fn emit_graph_update_for(graph: Arc<GraphStore>, node_id, tool_result_node_ids, events: &EventBus)`
  plus the existing thin wrapper that passes `session.graph.clone()`.

**Step 1: Failing test** (`crates/agent/src/event_sink.rs` tests):

```rust
#[tokio::test]
async fn sink_maps_to_agent_events() {
    let graph = Arc::new(GraphStore::open_memory().unwrap());
    let mut bus = EventBus::new();
    let mut rx = bus.subscribe();
    let sink = EventBusSink::new(Arc::new(bus), graph.clone());
    let n = graph.add_node(GraphNode::new(NodeType::Interaction(InteractionData {
        role: "assistant".into(), content: "x".into(), token_count: None }))).unwrap();

    sink.tool_started(&n, "pi:1", "bash");
    sink.tool_finished(&n, true);
    sink.graph_changed(&n, &[n.clone()]);

    let e1 = rx.recv().await.unwrap();
    assert!(matches!(e1, AgentEvent::ToolStart { ref call_id, ref tool_name, .. } if call_id == "pi:1" && tool_name == "bash"));
    let e2 = rx.recv().await.unwrap();
    assert!(matches!(e2, AgentEvent::ToolEnd { is_error: true, .. }));
    let e3 = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv()).await.unwrap().unwrap();
    assert!(matches!(e3, AgentEvent::GraphUpdate { .. }));
}
```

**Step 2:** run → FAIL (module missing).

**Step 3: Implement**

```rust
//! Bridges `graphirm_tools::ToolEventSink` to the agent loop's `EventBus`.
use std::sync::Arc;
use graphirm_graph::GraphStore;
use graphirm_graph::nodes::NodeId;
use graphirm_tools::ToolEventSink;
use crate::event::{AgentEvent, EventBus};

pub struct EventBusSink {
    bus: Arc<EventBus>,
    graph: Arc<GraphStore>,
}

impl EventBusSink {
    pub fn new(bus: Arc<EventBus>, graph: Arc<GraphStore>) -> Self { Self { bus, graph } }
}

impl ToolEventSink for EventBusSink {
    fn tool_started(&self, response_node_id: &NodeId, call_id: &str, tool_name: &str) {
        self.bus.emit(AgentEvent::ToolStart {
            response_node_id: response_node_id.clone(),
            call_id: call_id.to_string(),
            tool_name: tool_name.to_string(),
        });
    }
    fn tool_finished(&self, node_id: &NodeId, is_error: bool) {
        self.bus.emit(AgentEvent::ToolEnd { node_id: node_id.clone(), is_error });
    }
    fn graph_changed(&self, anchor: &NodeId, touched: &[NodeId]) {
        let (bus, graph, anchor, touched) =
            (self.bus.clone(), self.graph.clone(), anchor.clone(), touched.to_vec());
        tokio::spawn(async move {
            crate::workflow::emit_graph_update_for(graph, &anchor, touched, &bus).await;
        });
    }
}
```

Implementation note: GraphUpdate emission is serialised and coalesced through a single worker task fed by an unbounded channel, rather than a per-call `tokio::spawn` (preserves snapshot ordering and bounds task count); `graph_changed` is a sync `send`, so no runtime `Handle` is needed.

`emit_graph_update_for` is the body of today's `emit_graph_update` with `session.graph`
replaced by the `graph` parameter; keep `emit_graph_update(session, …)` as a one-line
wrapper so the three existing call sites are untouched.

**Step 4:** `cargo test -p graphirm-agent event_sink` → PASS; `cargo test --workspace` green.

**Step 5: Commit** — `feat(agent): EventBusSink bridges ToolEventSink to AgentEvent`

- [x] A1.2 done

### Task A1.3: Wire the sink into the agent loop

**Files:**
- Modify: `crates/agent/src/workflow.rs:925-937` — `event_sink: Some(Arc::new(EventBusSink::new(events_arc, session.graph.clone())))`.
  `execute_tools_parallel` receives `events: &EventBus`, not an `Arc`. Check the
  caller (`run_agent_loop` gets `events: &Arc<EventBus>`? — verify at
  `workflow.rs` signature of `run_agent_loop`, line ~1500). If only `&EventBus` is
  available at the ToolContext site, change `execute_tools_parallel`'s parameter to
  `events: &Arc<EventBus>` and adjust its callers (all inside `workflow.rs`).
- Test: `crates/agent/src/workflow.rs` tests — extend an existing tool-execution
  test (e.g. the one asserting `ToolEnd` around line 2356) with a custom tool that
  calls `ctx.event_sink.as_ref().unwrap().tool_started(...)` and assert the extra
  `ToolStart` reaches the bus.

**Steps:** failing test → wire → `cargo test -p graphirm-agent` → commit
`feat(agent): pass EventBusSink to tools via ToolContext.event_sink`.

- [ ] A1.3 done

### Task A1.4: `bash.rs` — kill the child on cancel/timeout (approved fix)

**Files:**
- Modify: `crates/tools/src/bash.rs:82-112`
- Test: `crates/tools/src/bash.rs` tests

**Step 1: Failing test**

```rust
#[tokio::test]
async fn cancel_kills_the_child_process() {
    let dir = TempDir::new().unwrap();
    let ctx = make_ctx_with_dir(&dir);
    let signal = ctx.signal.clone();
    let pidfile = dir.path().join("pid");
    let cmd = format!("echo $$ > {} && sleep 30", pidfile.display());
    let fut = BashTool.execute(json!({"command": cmd}), &ctx);
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(300)).await;
        signal.cancel();
    });
    let err = fut.await.unwrap_err();
    assert!(matches!(err, ToolError::Cancelled));
    let pid: i32 = std::fs::read_to_string(&pidfile).unwrap().trim().parse().unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    // ESRCH when gone. `kill -0` via std::process to avoid a libc dep in tools.
    let alive = std::process::Command::new("kill").arg("-0").arg(pid.to_string())
        .status().unwrap().success();
    assert!(!alive, "bash child {pid} still alive after cancel");
}
```

**Step 2:** run → FAIL (child survives; `task.abort()` drops the future without killing).

**Step 3: Implement** — replace the `tokio::spawn(child.wait_with_output())` +
`abort()` pattern with `cmd.kill_on_drop(true)` and a `select!` over
`child.wait_with_output()` held directly; on timeout/cancel call
`child.start_kill()` before returning. Because `wait_with_output` consumes the
child, structure as: spawn → `let mut child`; `let stdout/stderr` readers →
`tokio::select! { out = read_both => …, _ = sleep => { child.start_kill().ok(); return Err(Timeout) }, _ = cancelled => { child.start_kill().ok(); return Err(Cancelled) } }`.
Keep output semantics identical (stdout + "stderr:\n…").

**Step 4:** `cargo test -p graphirm-tools bash` → PASS; whole workspace green.

**Step 5: Commit** — `fix(tools): bash kills its child on cancel/timeout instead of leaking it`.
Add a `## 2026-09-xx: bash cancel leak fixed` line to `docs/completion-log.md` and
tick the item under "Pre-existing gaps" in `docs/backlog.md` in the same commit.

- [ ] A1.4 done

**Phase A1 checkpoint:** `cargo fmt --check && cargo clippy --workspace -- -D warnings && cargo test --workspace` green. Report progress.

---

## Phase A2 — `delegate_pi` tool

### Task A2.1: Record a real Pi run as the test fixture

**Files:**
- Create: `crates/agent/tests/fixtures/pi/hello-run.jsonl`
- Create: `crates/agent/tests/fixtures/pi/README.md` (how it was recorded, scrub rules)

**Steps:**

```bash
D=$(mktemp -d) && cd "$D" && git init -q
PI=~/.nvm/versions/node/v24.18.0/bin/pi
PI_SKIP_VERSION_CHECK=1 $PI --mode json -p --no-session --no-approve \
  --no-extensions --no-skills --no-prompt-templates \
  --provider openrouter --model deepseek/deepseek-v4-flash \
  "Create hello.txt containing exactly 'hi', then run: cat hello.txt. Reply with the file content." \
  > run.jsonl 2> run.stderr; echo "exit=$?"
wc -l run.jsonl; jq -c '.type' run.jsonl | sort | uniq -c
# scrub: cwd and any /home/... → /workspace ; verify no secrets
sed -E "s#$D#/workspace#g; s#/home/[a-z]+#/workspace#g" run.jsonl > hello-run.jsonl
rg -i 'sk-|api[_-]?key|authorization|bearer' hello-run.jsonl && echo "SECRET FOUND — fix before commit" || echo clean
```

Requires `OPENROUTER_API_KEY` in Pi's environment (Pi reads it; graphirm does not).
Copy `hello-run.jsonl` into the fixtures dir. Record the exit code and event-type
histogram in `README.md`. Confirm the fixture contains at least: `session`,
`agent_start`, `tool_execution_start`/`_end` for `write` and `bash`,
`message_update` with `text_delta`, `message_end` with `role: "assistant"`, `agent_end`.
If the observed shape deviates from `docs/json.md`, note it in the README — the
parser in A2.2 follows the recording.

**Commit:** `test(agent): record scrubbed Pi --mode json fixture (hello-run)`

- [ ] A2.1 done

### Task A2.2: `PiEvent` parser (pure)

**Files:**
- Create: `crates/agent/src/pi_delegate/mod.rs` (module skeleton: `pub mod events; pub mod process; pub mod graph;`)
- Create: `crates/agent/src/pi_delegate/events.rs`
- Modify: `crates/agent/src/lib.rs` (`pub mod pi_delegate;`)

**Step 1: Failing tests** (in `events.rs`), driven by the fixture:

```rust
fn fixture() -> Vec<String> {
    std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/pi/hello-run.jsonl"))
        .unwrap().lines().map(str::to_string).collect()
}

#[test]
fn parses_every_fixture_line_without_error() {
    for line in fixture() {
        assert!(parse_line(&line).is_ok(), "line failed: {line}");
    }
}

#[test]
fn tool_execution_events_carry_ids_names_and_payloads() {
    let evs: Vec<PiEvent> = fixture().iter().map(|l| parse_line(l).unwrap()).collect();
    let start = evs.iter().find_map(|e| match e { PiEvent::ToolStart { tool_call_id, tool_name, args } if tool_name == "write" => Some((tool_call_id.clone(), args.clone())), _ => None }).expect("write start");
    let end = evs.iter().find_map(|e| match e { PiEvent::ToolEnd { tool_call_id, is_error, .. } if *tool_call_id == start.0 => Some(*is_error), _ => None }).expect("write end");
    assert!(!end);
    assert!(start.1.get("path").is_some());
}

#[test]
fn message_end_assistant_flattens_text_parts() {
    let evs: Vec<PiEvent> = fixture().iter().map(|l| parse_line(l).unwrap()).collect();
    let last = evs.iter().rev().find_map(|e| match e { PiEvent::AssistantMessage { text, .. } => Some(text.clone()), _ => None }).unwrap();
    assert!(last.to_lowercase().contains("hi"));
}

#[test]
fn user_and_tool_result_message_end_are_skipped() {
    let l = r#"{"type":"message_end","message":{"role":"user","content":"x"}}"#;
    assert!(matches!(parse_line(l).unwrap(), PiEvent::Ignored));
}

#[test]
fn unknown_type_is_unknown_not_error() {
    assert!(matches!(parse_line(r#"{"type":"banana"}"#).unwrap(), PiEvent::Unknown(t) if t == "banana"));
}

#[test]
fn malformed_json_is_error() {
    assert!(parse_line("{not json").is_err());
}

#[test]
fn crlf_is_tolerated() {
    assert!(matches!(parse_line("{\"type\":\"agent_start\"}\r").unwrap(), PiEvent::Lifecycle(_)));
}

#[test]
fn text_delta_extracted_from_message_update() {
    let l = r#"{"type":"message_update","usage":{},"assistantMessageEvent":{"type":"text_delta","contentIndex":0,"delta":"He"}}"#;
    assert!(matches!(parse_line(l).unwrap(), PiEvent::TextDelta(d) if d == "He"));
}
```

**Step 2:** run → FAIL.

**Step 3: Implement**

```rust
//! Pure parser for Pi's `--mode json` (and `--mode rpc`) event lines.
//! No process or graph imports — reused unchanged when RPC mode lands.
use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
pub enum PiEvent {
    Session { id: Option<String>, cwd: Option<String> },
    Lifecycle(String),                    // agent_start, turn_start, turn_end, agent_end, …
    TextDelta(String),
    AssistantMessage { text: String, stop_reason: Option<String>, usage: Option<Value> },
    ToolStart { tool_call_id: String, tool_name: String, args: Value },
    ToolEnd { tool_call_id: String, tool_name: String, result: Value, is_error: bool },
    Error(String),
    Ignored,
    Unknown(String),
}

pub fn parse_line(line: &str) -> Result<PiEvent, serde_json::Error> {
    let v: Value = serde_json::from_str(line.trim_end_matches(['\r', '\n']))?;
    let ty = v.get("type").and_then(Value::as_str).unwrap_or("");
    Ok(match ty {
        "session" => PiEvent::Session {
            id: v.get("id").and_then(Value::as_str).map(str::to_string),
            cwd: v.get("cwd").and_then(Value::as_str).map(str::to_string),
        },
        "agent_start" | "turn_start" | "turn_end" | "agent_end" | "message_start"
        | "queue_update" | "compaction_start" | "compaction_end" | "tool_execution_update" => {
            PiEvent::Lifecycle(ty.to_string())
        }
        "message_update" => match v.pointer("/assistantMessageEvent/type").and_then(Value::as_str) {
            Some("text_delta") => PiEvent::TextDelta(
                v.pointer("/assistantMessageEvent/delta").and_then(Value::as_str).unwrap_or("").to_string()),
            _ => PiEvent::Ignored,
        },
        "message_end" => {
            let msg = v.get("message").cloned().unwrap_or(Value::Null);
            if msg.get("role").and_then(Value::as_str) != Some("assistant") { return Ok(PiEvent::Ignored); }
            PiEvent::AssistantMessage {
                text: flatten_content(msg.get("content")),
                stop_reason: msg.get("stopReason").and_then(Value::as_str).map(str::to_string),
                usage: msg.get("usage").cloned(),
            }
        }
        "tool_execution_start" => PiEvent::ToolStart {
            tool_call_id: str_field(&v, "toolCallId"),
            tool_name: str_field(&v, "toolName"),
            args: v.get("args").cloned().unwrap_or(Value::Null),
        },
        "tool_execution_end" => PiEvent::ToolEnd {
            tool_call_id: str_field(&v, "toolCallId"),
            tool_name: str_field(&v, "toolName"),
            result: v.get("result").cloned().unwrap_or(Value::Null),
            is_error: v.get("isError").and_then(Value::as_bool).unwrap_or(false),
        },
        "error" | "extension_error" => PiEvent::Error(
            v.get("message").or(v.get("error")).map(|e| e.to_string()).unwrap_or_else(|| line.to_string())),
        other => PiEvent::Unknown(other.to_string()),
    })
}

/// Pi `content` is a string or an array of typed parts; keep only `text` parts.
pub fn flatten_content(content: Option<&Value>) -> String { /* string → itself; array → join text parts; else "" */ }

/// Tool `result` is usually `{content:[{type:"text",text}], details}`; flatten the same way, fall back to compact JSON.
pub fn flatten_result(result: &Value) -> String { /* … */ }

fn str_field(v: &Value, k: &str) -> String { v.get(k).and_then(Value::as_str).unwrap_or("").to_string() }
```

**Step 4:** `cargo test -p graphirm-agent pi_delegate::events` → PASS.

**Step 5: Commit** — `feat(agent): PiEvent parser for pi --mode json lines (fixture-driven)`

- [ ] A2.2 done

### Task A2.3: `PiConfig` + `default_auto_approve` in config

**Files:**
- Modify: `crates/agent/src/config.rs` — add `PiConfig`, `AgentConfig.pi: Option<PiConfig>`,
  `AgentConfig.default_auto_approve: bool` (`#[serde(default = "default_true")]`), thread
  both through the `[agent]` file struct (~line 593) and the merge (~line 666) and
  `Default` (~line 514).
- Modify: `config/default.toml` — `[agent.pi]` block (from design D6) and
  `default_auto_approve = true` under `[agent]` with a comment.
- Test: `crates/agent/src/config.rs` tests

**Step 1: Failing tests**

```rust
#[test]
fn pi_config_parses_and_defaults() {
    let toml = r#"
[agent]
name = "a"
model = "m"
system_prompt = "s"
max_turns = 1
[agent.pi]
enabled = true
model = "deepseek/deepseek-v4-flash"
"#;
    let cfg = AgentConfig::from_str(toml).unwrap();  // or the existing parse helper used by other config tests
    let pi = cfg.pi.expect("pi");
    assert!(pi.enabled);
    assert_eq!(pi.binary, "pi");
    assert_eq!(pi.provider, "openrouter");
    assert_eq!(pi.timeout_seconds, 900);
    assert!(!pi.trust_project);
    assert_eq!(pi.extra_args, vec!["--no-extensions", "--no-skills", "--no-prompt-templates"]);
    assert_eq!(pi.max_result_chars, 16_000);
}

#[test]
fn pi_absent_is_none_and_default_auto_approve_is_true() {
    let cfg = AgentConfig::default();
    assert!(cfg.pi.is_none());
    assert!(cfg.default_auto_approve);
}

#[test]
fn default_toml_has_pi_disabled() {
    let cfg = AgentConfig::from_file(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/default.toml"))).unwrap();
    assert!(!cfg.pi.expect("pi block present").enabled);
    assert!(cfg.default_auto_approve);
}
```

**Step 3: Implement**

```rust
/// `[agent.pi]` — Pi (`@earendil-works/pi-coding-agent`) as an external delegate executor.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PiConfig {
    #[serde(default)] pub enabled: bool,
    #[serde(default = "default_pi_binary")] pub binary: String,
    #[serde(default = "default_pi_provider")] pub provider: String,
    #[serde(default = "default_pi_model")] pub model: String,
    #[serde(default = "default_pi_timeout")] pub timeout_seconds: u64,
    /// `false` → `--no-approve` (ignore the workspace's `.pi/` resources); `true` → `--approve`.
    #[serde(default)] pub trust_project: bool,
    #[serde(default = "default_pi_extra_args")] pub extra_args: Vec<String>,
    #[serde(default = "default_pi_max_result_chars")] pub max_result_chars: usize,
}
```

`config/default.toml` additions (verbatim from design D6, `enabled = false`).

**Step 5: Commit** — `feat(agent): [agent.pi] PiConfig and [agent] default_auto_approve`

- [ ] A2.3 done

### Task A2.4: Server honours `default_auto_approve`

**Files:**
- Modify: `crates/server/src/routes.rs:393` — `let headless = body.auto_approve.unwrap_or(config.default_auto_approve);`
- Test: `crates/server/tests/integration.rs` — create a session without `auto_approve`
  with a config whose `default_auto_approve = true` and assert `SessionResponse`
  (or the handle) reports auto-approve on; and the inverse with `false`.

**Commit** — `feat(server): sessions default to auto-approve per [agent] default_auto_approve`

- [ ] A2.4 done

### Task A2.5: Process wrapper — spawn, drain, kill, timeout

**Files:**
- Create: `crates/agent/src/pi_delegate/process.rs`
- Create: `crates/agent/tests/fixtures/pi/fake_pi.sh` (executable; `chmod +x`, commit mode 755)
- Modify: `crates/agent/Cargo.toml` — `[target.'cfg(unix)'.dependencies] libc = "0.2"`

**`fake_pi.sh`:**

```bash
#!/usr/bin/env bash
# Replays a Pi --mode json fixture. Knobs (env):
#   FAKE_PI_FIXTURE   path to JSONL (default: hello-run.jsonl next to this script)
#   FAKE_PI_DELAY_MS  per-line delay (default 5)
#   FAKE_PI_EXIT      exit code after replay (default 0)
#   FAKE_PI_HANG_AT   line number after which to sleep forever (cancel/timeout tests)
#   FAKE_PI_GARBAGE   1 → emit a non-JSON line after every 3rd event
#   FAKE_PI_NO_END    1 → stop before the last message_end
#   FAKE_PI_PIDFILE   write $$ here (kill tests)
# Also answers `--version` with 0.85.1-fake and exits 0.
set -u
[ "${1:-}" = "--version" ] && { echo 0.85.1-fake; exit 0; }
[ -n "${FAKE_PI_PIDFILE:-}" ] && echo $$ > "$FAKE_PI_PIDFILE"
here=$(cd "$(dirname "$0")" && pwd)
fixture=${FAKE_PI_FIXTURE:-$here/hello-run.jsonl}
delay=$(( ${FAKE_PI_DELAY_MS:-5} ))
n=0
while IFS= read -r line; do
  n=$((n+1))
  if [ "${FAKE_PI_NO_END:-0}" = 1 ] && printf '%s' "$line" | grep -q '"type":"message_end"' && printf '%s' "$line" | grep -q '"role":"assistant"'; then break; fi
  printf '%s\n' "$line"
  [ "${FAKE_PI_GARBAGE:-0}" = 1 ] && [ $((n % 3)) = 0 ] && echo "not json $n"
  [ -n "${FAKE_PI_HANG_AT:-}" ] && [ "$n" -ge "$FAKE_PI_HANG_AT" ] && sleep 3600
  sleep "$(awk "BEGIN{print $delay/1000}")"
done < "$fixture"
exit "${FAKE_PI_EXIT:-0}"
```

**Step 1: Failing tests** (in `process.rs`), all pointing `binary` at `fake_pi.sh`:

- `runs_fixture_to_completion_and_yields_events`: collect `Vec<PiEvent>`, exit code 0, `agent_end` seen.
- `nonzero_exit_is_reported`: `FAKE_PI_EXIT=3` → `PiRunOutcome.exit_code == Some(3)`.
- `garbage_lines_are_counted_not_fatal`: `FAKE_PI_GARBAGE=1` → `malformed_lines > 0`, events still complete.
- `cancel_kills_child_within_5s`: `FAKE_PI_HANG_AT=3`, `FAKE_PI_PIDFILE`, cancel after 300 ms → `Err(PiProcessError::Cancelled)`, pid dead (`kill -0` fails) within 5 s.
- `timeout_kills_child`: `FAKE_PI_HANG_AT=3`, timeout 1 s → `Err(PiProcessError::Timeout)`, pid dead.
- `missing_binary_is_not_found`: `binary = "/nonexistent/pi"` → `Err(PiProcessError::NotFound(_))`.
- `stderr_tail_is_captured`: fixture script writes to stderr (add `echo warn >&2` via a `FAKE_PI_STDERR=1` knob) → `outcome.stderr_tail.contains("warn")`.

**Step 3: Implement** the public surface:

```rust
pub struct PiSpawnSpec<'a> {
    pub config: &'a PiConfig,
    pub cwd: &'a Path,
    pub task: &'a str,
    pub timeout: Duration,
}

#[derive(Debug, thiserror::Error)]
pub enum PiProcessError {
    #[error("pi not found at '{0}'")] NotFound(String),
    #[error("failed to spawn pi: {0}")] Spawn(String),
    #[error("pi timed out after {0:?}")] Timeout(Duration),
    #[error("cancelled")] Cancelled,
}

pub struct PiRunOutcome {
    pub exit_code: Option<i32>,
    pub stderr_tail: String,
    pub malformed_lines: u32,
    pub oversized_lines: u32,
    pub duration: Duration,
}

/// Runs Pi, calling `on_event` for every parsed line as it arrives.
/// Returns when Pi exits, the deadline passes, or `cancel` fires.
pub async fn run_pi(
    spec: PiSpawnSpec<'_>,
    cancel: &CancellationToken,
    mut on_event: impl FnMut(PiEvent) -> BoxFuture<'static, ()> + Send,   // or a plain async closure via `impl AsyncFnMut`
) -> Result<PiRunOutcome, PiProcessError>
```

Simpler alternative to the async callback: `run_pi` returns an `mpsc::Receiver<PiEvent>`
plus a `JoinHandle<Result<PiRunOutcome, PiProcessError>>`; the tool consumes the
receiver and awaits graph writes between events. **Prefer this** — it keeps
`process.rs` free of graph/judge concerns and makes the cancel/timeout tests
trivial. `build_argv(&PiConfig, task_arg: &str) -> Vec<String>` is a pure fn with
its own test asserting the exact flag order from design D1 (`--no-approve` when
`trust_project == false`, `--approve` when true, `extra_args` after `--model`, `--`
before the task).

Kill path (unix): `cmd.process_group(0)`; on cancel/timeout
`unsafe { libc::killpg(child.id() as i32, libc::SIGKILL) }` then `child.wait()`
under a 5 s `timeout`; non-unix: `child.start_kill()`. Env: `cmd.env("PI_SKIP_VERSION_CHECK", "1")`.
Task arg: if `task.len() > 64 * 1024` write to `temp_dir()/graphirm-pi-<uuid>.md`,
pass `@<path>`, remove in a `defer`-style guard (a small `TempTask` struct with `Drop`).

**Step 5: Commit** — `feat(agent): Pi subprocess wrapper — spawn, JSONL drain, group kill, timeout (fake pi tests)`

- [ ] A2.5 done

### Task A2.6: Graph writes — Task, Pi Agent, tool nodes, result

**Files:**
- Create: `crates/agent/src/pi_delegate/graph.rs`
- Test: same file

**Step 1: Failing tests** with `GraphStore::open_memory()`:

- `creates_task_and_pi_agent_with_delegation_edges`: after `PiRun::begin(ctx, task_text, model)`:
  `ctx.agent_id --DelegatesTo--> task`, `ctx.interaction_id --Produces--> task`,
  `task --SpawnedBy--> pi_agent`, Task title `"Delegated to pi"`, status Pending,
  Pi Agent `name == "pi"`, `status == "running"`.
- `tool_node_matches_in_process_shape`: `record_tool_call(...)` → node is
  `Interaction{role:"tool"}`; metadata has `tool_call_id`, `tool_name`, `is_error`,
  `executor == "pi"`, `arguments` (string ≤ 1500 chars + `…` when truncated),
  `session_id == pi_agent`, `parent_session_id == ctx.agent_id`; label
  `interaction_{turn}_{pos}_1`; edge `pi_agent --Produces--> node`; second node
  `RespondsTo` the first.
- `assistant_message_node_and_task_result`: `record_assistant_message` creates
  `Interaction{role:"assistant"}` under the Pi Agent; `finish(Completed, summary)`
  sets Task status Completed, `metadata.result`, `tool_calls`, `duration_ms`,
  `exit_code`; Pi Agent status `"completed"`.
- `finish_failed_records_failure_kind`: `finish(Failed{kind:"timeout"})` →
  `metadata.failure == "timeout"`, Pi Agent status `"failed"`.
- `result_truncated_to_max_chars`.

**Step 3: Implement** a `PiRun` struct holding `graph`, `ctx` clones, `task_id`,
`pi_agent_id`, `last_node: Option<NodeId>`, counters, `max_result_chars`. All
graph calls inside `tokio::task::spawn_blocking` (the store is sync; never block
the runtime — the existing `record_content_node` is the template).

**Step 5: Commit** — `feat(agent): Pi delegation graph writes mirror spawn_subagent shape`

- [ ] A2.6 done

### Task A2.7: `PiDelegateTool` — assemble, register, system-prompt notice

**Files:**
- Modify: `crates/agent/src/pi_delegate/mod.rs` — `PiDelegateTool`, `register_pi_delegate`,
  `apply_pi_delegate_system_notice`
- Modify: `crates/agent/src/workflow.rs:245` — filter `|| (disable_bash && t.name == "delegate_pi")`
- Modify: `src/commands/serve.rs` — load `agent_config` **before** building tools; then
  `let mut registry = super::build_tool_registry(); graphirm_agent::pi_delegate::register_pi_delegate(&mut registry, &agent_config); let tools = Arc::new(registry);`
- Modify: `src/commands/chat.rs` — same after `config` is loaded
- Modify: `crates/agent/src/lib.rs` — `pub use pi_delegate::{PiDelegateTool, register_pi_delegate};`
- Modify: `crates/agent/src/hitl_judge.rs` — `pub const JUDGE_ACTION_OBSERVED: &str = "observed";`
- Test: `crates/agent/src/pi_delegate/mod.rs` tests + one integration test
  `crates/agent/tests/pi_delegate_integration.rs`

**Step 1: Failing tests**

```rust
#[test]
fn tool_name_schema_and_destructive() {
    let t = PiDelegateTool::new(PiConfig::default(), None);
    assert_eq!(t.name(), "delegate_pi");
    assert!(t.is_destructive());
    let p = t.parameters();
    assert_eq!(p["required"], serde_json::json!(["task"]));
    assert!(p["properties"]["context_paths"].is_object());
    assert!(t.description().contains("cannot be paused"));
}

#[tokio::test]
async fn happy_path_builds_graph_and_returns_summary() {
    // config.binary = fake_pi.sh; ctx with Recording sink
    let out = tool.execute(json!({"task": "make hello.txt"}), &ctx).await.unwrap();
    assert!(out.content.contains("Pi completed"));
    assert!(out.content.contains("Tool calls:"));
    assert!(out.node_id.is_some());                       // the Task
    // graph: Task Completed with metadata.result; ≥ 2 tool nodes with executor == "pi"
    // sink: ToolStart count == ToolEnd count == number of tool_execution_end in fixture; ≥ 1 graph_changed
}

#[tokio::test]
async fn disable_bash_refuses_before_spawn() { /* ctx.disable_bash = true → ExecutionFailed, no Task node */ }

#[tokio::test]
async fn missing_binary_is_tool_error_and_no_task_node() { /* binary = /nonexistent */ }

#[tokio::test]
async fn nonzero_exit_without_result_is_error_with_stderr_tail() { /* FAKE_PI_NO_END=1 FAKE_PI_EXIT=2 FAKE_PI_STDERR=1 → Err containing "exited 2" and "warn"; Task Failed, failure == "exit" */ }

#[tokio::test]
async fn nonzero_exit_with_result_is_success_with_exit_code() { /* FAKE_PI_EXIT=1 → Ok; Task metadata.exit_code == 1 */ }

#[tokio::test]
async fn cancel_marks_task_failed_and_returns_cancelled() { /* FAKE_PI_HANG_AT=3; cancel; Err(Cancelled); Task failure == "cancelled" */ }

#[test]
fn register_only_when_enabled_and_notice_is_idempotent() {
    let mut reg = ToolRegistry::new();
    let mut cfg = AgentConfig::default();
    register_pi_delegate(&mut reg, &cfg);
    assert!(reg.get("delegate_pi").is_err());
    cfg.pi = Some(PiConfig { enabled: true, ..Default::default() });
    register_pi_delegate(&mut reg, &cfg);
    assert!(reg.get("delegate_pi").is_ok());
    let mut prompt = String::from("base");
    apply_pi_delegate_system_notice(&mut prompt);
    apply_pi_delegate_system_notice(&mut prompt);
    assert_eq!(prompt.matches("delegate_pi").count(), 1 /* or the notice's own count */);
}
```

Plus a `workflow.rs` test: with `disable_bash = true` and `delegate_pi` registered,
the tool definitions sent to the mock provider exclude both `bash` and `delegate_pi`.

**Step 3: Implement** `execute`:

1. `ctx.disable_bash` → `Err(ExecutionFailed(...))`.
2. Parse `task`, optional `context_paths` (append "Relevant files:\n- …" to the task text), optional `timeout_seconds` (min with config).
3. `PiRun::begin(...)` → Task + Pi Agent; `sink.graph_changed(task_id, [task, pi_agent])`.
4. `run_pi(spec, &ctx.signal)` → `(rx, handle)`.
5. Loop `rx.recv()`: `ToolStart` → `sink.tool_started(ctx.interaction_id, id, name)`; if judge && name ∈ {bash,write,edit} → spawn `judge.judge(name, &args)` into `HashMap<String, JoinHandle<…>>`. `ToolEnd` → await judge handle (if any) with `timeout(judge_timeout)`; `run.record_tool_call(...)` → node; `sink.tool_finished(node, is_error)`; `sink.graph_changed(node, [node])`. `AssistantMessage` → `run.record_assistant_message`. `TextDelta` → buffer (unused in v1). `Error` → `run.errors.push`. Others → counters.
6. `handle.await` → `PiRunOutcome` or `PiProcessError`:
   - `Ok(outcome)`: if `last_assistant_text.is_none() && exit_code != Some(0)` → `run.finish(Failed{"exit"})`, `Err(ExecutionFailed(format!("pi exited {code} without a result: {stderr_tail}")))`; else `run.finish(Completed, summary)`, `Ok(ToolOutput::success_with_node(summary, task_id))`.
   - `Err(Cancelled)` → `finish(Failed{"cancelled"})`, `Err(ToolError::Cancelled)`.
   - `Err(Timeout)` → `finish(Failed{"timeout"})`, `Err(ToolError::Timeout(secs))` with partial summary in the message.
   - `Err(NotFound|Spawn)` → `finish(Failed{"spawn"})` (Task already exists at this point — acceptable; it records the attempt) → `Err(ExecutionFailed(...))`. *Alternative:* probe `binary` with `--version` before `begin` so no Task is created when Pi is absent — do this; it matches the design's "no Task node created" for not-installed.
7. Every `graph_changed` after the final `finish` so the Task status flips live.

`register_pi_delegate(registry, config)`: if `config.pi.enabled` → build
`Option<Arc<DestructiveJudge>>` via `build_judge(config)`, probe `--version`
(5 s, warn if missing), `registry.register(Arc::new(PiDelegateTool::new(pi_cfg, judge)))`.

System-prompt notice (appended by `Session::new` when `pi.enabled`, same place as
`apply_disable_bash_system_notice`): when to delegate (multi-file implementation
with a clear brief and a verification command), when not to (reads, questions,
per-step approval, anything needing graph memory), that Pi's calls are recorded and
scored but not paused, and that the director must verify Pi's result with its own
tools before claiming done. Prefer a clean git state or a branch.

**Step 5: Commits** (split):
- `feat(agent): delegate_pi tool — Pi as delegated executor (observe-only, fake-pi tests)`
- `feat(cli): register delegate_pi in serve and chat when [agent.pi].enabled`
- Governance: mark A2 in `docs/backlog.md`, entry in `docs/completion-log.md`.

- [ ] A2.7 done

**Phase A2 checkpoint:** workspace fmt/clippy/test green; `config/default.toml` still `enabled = false`. Report progress.

---

## Phase A3 — Judge on Pi's calls

A2.7 already threads the judge. A3 is the dedicated test coverage and the
metadata contract.

### Task A3.1: Judge metadata on Pi tool nodes

**Files:**
- Modify: `crates/agent/src/pi_delegate/mod.rs` (only if A2.7 left gaps)
- Test: `crates/agent/src/pi_delegate/mod.rs` tests using `ReplyTransport` /
  `HangingTransport` copied from `hitl_judge.rs` tests (make them `pub(crate)`
  test helpers in `hitl_judge.rs` to avoid duplication).

**Tests:**
- `judge_verdict_recorded_on_bash_write_edit_only`: judge replying `noul 0.93` →
  Pi `bash`/`write` nodes carry `hitl_judge.{version, p_irreversible: 0.93, threshold, action: "observed", latency_ms}`; `read` node has no `hitl_judge`.
- `judge_error_is_fail_soft`: `HangingTransport` with 200 ms timeout → nodes created, no `hitl_judge`, run completes.
- `no_judge_means_no_key`: `judge = None` → no `hitl_judge` anywhere.
- `summary_counts_over_threshold`: summary text contains `"Judge: N calls ≥ 0.8 (observed, not gated)"` (0 when none).

**Commit** — `feat(agent): hitl_judge observe-only on Pi bash/write/edit (action=observed)`.
Governance: A3 ticked in backlog, completion-log entry.

- [ ] A3.1 done

---

## Phase A4 — Live check, then enable

### Task A4.1: Live check with the current whiteboard and TUI

**Preconditions:** `OPENROUTER_API_KEY` in the shell (Pi reads it); local build.

**Steps:**

```bash
# temporary local enable — do NOT commit this edit
sed -i 's/^enabled = false\(.*# flip\)/enabled = true\1/' config/default.toml   # or edit [agent.pi] enabled by hand
cargo build --release
mkdir -p /tmp/pi-live && cd /tmp/pi-live && git init -q && cd -
GRAPHIRM_API_KEY=dev OPENROUTER_API_KEY=$OPENROUTER_API_KEY ./target/release/graphirm serve --db /tmp/pi-live-graph.db
# other terminal: web-app dev or built dist at http://localhost:3000
```

In the web-app: create a session with workspace `/tmp/pi-live`; prompt:
"Delegate to Pi: create `hello.py` printing 'hi' and run it with python3. Then
verify the file exists yourself." Observe:
1. Confirm card for `delegate_pi` (auto-approve is on by default → none; toggle it
   off once to see the card, approve).
2. Task node + Pi Agent node appear; tool nodes appear one by one while Pi runs.
3. Task result carries Pi's final message; `hitl_judge` present on `bash`/`write` nodes.
4. Second run: prompt a longer task, press Abort mid-run; `pgrep -f 'pi --mode json'` is empty within 5 s; Task `failure == "cancelled"`.
5. TUI: `./target/release/graphirm chat` in `/tmp/pi-live`, same delegation; graph panel shows the nodes.
6. Negative: `[agent.pi].binary = "/nonexistent"` → tool error in chat, turn continues.

Record session ids, timings, screenshots (paths), and any deviation in the design
doc's "A4 findings" section. Revert the temporary `enabled = true`.

- [ ] A4.1 done (findings recorded)

### Task A4.2: Enable

**Files:** `config/default.toml` — `[agent.pi] enabled = true` only.

**Commit** — `config: enable delegate_pi by default after A4 live check (see design doc A4 findings)`.
Governance: Track A ticked `### ✅` in `docs/backlog.md`; `docs/completion-log.md`
entry with key files; `docs/journal.md` entry only if a locked decision changed.
Also update `AGENTS.md` Key Conventions with one line: `delegate_pi` — Pi as
external executor, `[agent.pi]`, observe-only; and `crates/agent/AGENTS.md` Key
Components table (`pi_delegate/`, `event_sink.rs`); `crates/tools/AGENTS.md` (`ToolEventSink`).

- [ ] A4.2 done

---

## Critical path and dependencies

```text
A1.1 → A1.2 → A1.3 ──┐
A1.4 (independent) ──┤
A2.1 (needs network once) → A2.2 ──┐
A2.3 → A2.4                        ├→ A2.7 → A3.1 → A4.1 → A4.2
A2.5 (needs fixture from A2.1) ────┤
A2.6 (needs A1.1 for the sink) ────┘
```

Parallelisable: A1.4, A2.1, A2.3/A2.4 can run alongside A1.1–A1.3.

## Review gates (per `000-skills-first`)

After each phase: spec-review subagent (does the code match this plan and the
design doc's Decisions?), then code-quality subagent (clippy-clean, no `unwrap()`
outside tests, `tracing` not `println!`, no blocking I/O on the runtime, no
`Arc<RwLock>` held across `.await`).
