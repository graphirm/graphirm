# `trace_analysis` built-in tool — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development` (or `executing-plans`) task-by-task.

**Goal:** Expose `graphirm_agent::trace_analysis::build_trace_report` as a non-destructive `Tool` so the primary agent (and subagents using the base registry) can run cross-session pattern analysis without leaving the session.

**Constraint:** `graphirm-tools` must **not** depend on `graphirm-agent` (cycle: `agent` → `tools`). Therefore the `Tool` implementation lives in **`crates/agent/src/trace_analysis_tool.rs`**, same pattern as `SubagentTool` in `delegate.rs`.

**Tech:** `async_trait`, `tokio::task::spawn_blocking`, `serde_json::to_string_pretty` for output. No new crate dependencies.

---

## Task 1: `TraceAnalysisTool` + tests

**Files:**
- Create: `crates/agent/src/trace_analysis_tool.rs`
- Modify: `crates/agent/src/lib.rs` — `pub mod trace_analysis_tool;` and `pub use trace_analysis_tool::TraceAnalysisTool;`

**Behavior:**
- `name()` → `"trace_analysis"`
- `description()` — read-only cross-session failure-pattern report (over_tooling, doom_loops, etc.)
- `parameters()` — `{ "type": "object", "properties": { "max_sessions": { "type": "integer", "description": "...", "default": 50 } }, "required": [] }`
- `execute` — `max_sessions` from params, default `50`; `spawn_blocking(|| build_trace_report(&graph, max_sessions))`; return `ToolOutput::success` with pretty-printed JSON (optionally wrapped in a short markdown header + fenced `json` block for readability)
- `is_destructive()` — default `false` (inherit from trait default)

**Tests:** `tool_name`, `parameters` include optional `max_sessions`, `execute` on empty in-memory graph returns success with `sessions_analyzed == 0` (parse JSON from content or assert substring).

**Commit:** `feat(agent): add trace_analysis built-in tool`

---

## Task 2: Register in `build_tool_registry`

**Files:**
- Modify: `src/commands/mod.rs` — `registry.register(Arc::new(graphirm_agent::TraceAnalysisTool::new()));` next to other non-destructive tools (e.g. after `context_report`).

**Verify:** `cargo build`, `cargo test -p graphirm-agent`, `cargo test --workspace` (or at least agent + root binary tests).

**Commit:** `feat(cli): register trace_analysis tool`

---

## Task 3: Docs (minimal)

**Files:**
- Modify: `AGENTS.md` — extend Phase 53 row or add a bullet under harness tools: `trace_analysis` tool delegates to `build_trace_report`.
- Modify: `docs/backlog.md` — if there is a line about optional tool, mark done or add one-line “Done” note.

**Commit:** `docs: trace_analysis built-in tool`

---

## Success criteria

- [x] `trace_analysis` appears in tool list for `graphirm chat` / `graphirm serve` sessions
- [x] Calling the tool returns JSON report without mutating the graph
- [x] Tests pass; `cargo clippy -D warnings` clean
