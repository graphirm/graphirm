# `trace_analysis` Built-in Tool — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wrap `build_trace_report` as a non-destructive agent tool so the agent can self-diagnose session quality without leaving the chat.

**Architecture:** Thin `TraceAnalysisTool` struct in `crates/tools/src/trace_analysis.rs` implementing the `Tool` trait. Delegates to `graphirm_agent::trace_analysis::build_trace_report` via `spawn_blocking`. Formats the report as readable markdown for the agent (same structure as the CLI markdown output). Registered in `build_tool_registry()` alongside existing non-destructive tools.

**Tech Stack:** `graphirm-tools` crate (`Tool` trait, `ToolContext`, `ToolError`), `graphirm-agent` crate (`trace_analysis::build_trace_report`, `TraceReport`), `async_trait`, `serde_json`. No new dependencies.

**Key decisions:**
- **Tool, not ad-hoc code in the agent loop**: keeps all tool implementations in `graphirm-tools`; the agent calls it like any other tool
- **Markdown output, not JSON**: the agent reads markdown naturally; JSON is available via the HTTP endpoint
- **No new `ToolContext` fields**: `TraceAnalysisTool` only needs `ctx.graph` which is already available

**Success criteria:**
- [ ] `trace_analysis` tool is registered and appears in the agent's tool list
- [ ] Agent can call it with optional `max_sessions` param, gets markdown report
- [ ] Non-destructive (no HITL gating)
- [ ] 4+ unit tests (name/params, empty graph output, formatting)
- [ ] All existing tests pass, clippy clean

**Risks:** None — this is a thin wrapper around tested code.

---

## Task 1: Create `TraceAnalysisTool` and register it

**Files:**
- Create: `crates/tools/src/trace_analysis.rs`
- Modify: `crates/tools/src/lib.rs` (add `pub mod trace_analysis;`)
- Modify: `src/commands/mod.rs` → `build_tool_registry()` (register the tool)

**Step 1: Write the failing test**

```rust
// In crates/tools/src/trace_analysis.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_name_and_description() {
        let t = TraceAnalysisTool::new();
        assert_eq!(t.name(), "trace_analysis");
        assert!(!t.description().is_empty());
    }

    #[test]
    fn parameters_has_max_sessions() {
        let t = TraceAnalysisTool::new();
        let p = t.parameters();
        assert!(p["properties"]["max_sessions"].is_object());
    }

    #[test]
    fn parameters_has_no_required_fields() {
        let t = TraceAnalysisTool::new();
        let p = t.parameters();
        // No required fields — max_sessions has a default
        let required = p.get("required");
        assert!(required.is_none() || required.unwrap().as_array().unwrap().is_empty());
    }

    #[test]
    fn is_not_destructive() {
        let t = TraceAnalysisTool::new();
        assert!(!t.is_destructive());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p graphirm-tools trace_analysis -- --nocapture`
Expected: FAIL — module doesn't exist yet.

**Step 3: Implement the tool**

```rust
// crates/tools/src/trace_analysis.rs

use async_trait::async_trait;
use serde_json::json;

use crate::{Tool, ToolContext, ToolError, ToolOutput};

pub struct TraceAnalysisTool;

impl TraceAnalysisTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TraceAnalysisTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for TraceAnalysisTool {
    fn name(&self) -> &str {
        "trace_analysis"
    }

    fn description(&self) -> &str {
        "Analyze recent sessions for failure patterns (over-tooling, doom loops, \
         token waste, unrecovered errors, premature completion). Returns a structured \
         report with per-session findings and suggested harness parameter changes."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "max_sessions": {
                    "type": "integer",
                    "description": "Maximum number of recent sessions to analyze (default: 20)",
                    "default": 20
                }
            }
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let max_sessions = params
            .get("max_sessions")
            .and_then(|v| v.as_u64())
            .unwrap_or(20) as usize;

        let graph = ctx.graph.clone();

        let output = tokio::task::spawn_blocking(move || {
            let report = graphirm_agent::trace_analysis::build_trace_report(&graph, max_sessions);
            format_report(&report)
        })
        .await
        .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        Ok(ToolOutput::success(output))
    }
}

fn format_report(report: &graphirm_agent::trace_analysis::TraceReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "## Trace Analysis Report\n\nSessions analyzed: {}\n\n",
        report.sessions_analyzed
    ));

    if !report.patterns.is_empty() {
        out.push_str("### Patterns Detected\n\n");
        for p in &report.patterns {
            out.push_str(&format!(
                "**{}** ({:?}) — {} occurrence(s)\n{}\n\n",
                p.pattern, p.severity, p.occurrences, p.description
            ));
        }
    }

    if !report.per_session.is_empty() {
        out.push_str("### Per-Session Summary\n\n");
        for s in &report.per_session {
            out.push_str(&format!(
                "- **{}** ({}) — {} turns, {} tokens, {} finding(s)\n",
                s.agent_name,
                s.status,
                s.turn_count,
                s.token_total,
                s.findings.len()
            ));
        }
        out.push('\n');
    }

    if !report.suggestions.is_empty() {
        out.push_str("### Suggestions\n\n");
        for s in &report.suggestions {
            out.push_str(&format!("- {s}\n"));
        }
    }

    if report.sessions_analyzed == 0 {
        out.push_str("No sessions found to analyze.\n");
    }

    out
}
```

**Step 4: Add module to `crates/tools/src/lib.rs`**

Add `pub mod trace_analysis;` in alphabetical order (after `submit`, before `write`).

**Step 5: Register in `build_tool_registry()` in `src/commands/mod.rs`**

Add alongside other non-destructive tools (e.g. after `context_report`):

```rust
registry.register(Arc::new(graphirm_tools::trace_analysis::TraceAnalysisTool::new()));
```

**Step 6: Run tests**

Run: `cargo test -p graphirm-tools trace_analysis -- --nocapture`
Expected: PASS (4 tests)

Run: `cargo test --workspace 2>&1 | grep -E "(FAILED|error\[)"` — should be empty.

**Step 7: Run clippy**

Run: `cargo clippy -- -D warnings`

**Step 8: Commit**

```bash
git add crates/tools/src/trace_analysis.rs crates/tools/src/lib.rs src/commands/mod.rs
git commit -m "feat(tools): add trace_analysis built-in tool for agent self-diagnosis"
```

---

## Task 2: Integration test with real graph data

**Files:**
- Modify: `crates/tools/src/trace_analysis.rs` (add async test)

**Step 1: Write the integration test**

```rust
#[tokio::test]
async fn execute_on_empty_graph_returns_zero_sessions() {
    use graphirm_graph::GraphStore;
    use graphirm_graph::nodes::NodeId;
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;

    let graph = Arc::new(GraphStore::open_memory().unwrap());
    let ctx = ToolContext {
        graph,
        agent_id: NodeId::new(),
        interaction_id: NodeId::new(),
        working_dir: std::path::PathBuf::from("/tmp"),
        signal: CancellationToken::new(),
        turn: 1,
        turn_pos_counter: Arc::new(std::sync::atomic::AtomicU32::new(0)),
        knowledge_retriever: None,
        impact_provider: None,
        disable_bash: false,
        auto_link_write_to_planning: false,
    };

    let t = TraceAnalysisTool::new();
    let result = t.execute(serde_json::json!({}), &ctx).await.unwrap();
    assert!(!result.is_error);
    assert!(result.content.contains("Sessions analyzed: 0"));
    assert!(result.content.contains("No sessions found"));
}
```

**Step 2: Run test**

Run: `cargo test -p graphirm-tools trace_analysis -- --nocapture`
Expected: PASS (5 tests)

**Step 3: Commit**

```bash
git add crates/tools/src/trace_analysis.rs
git commit -m "test(tools): integration test for trace_analysis on empty graph"
```

---

## Task 3: Update docs and backlog

**Files:**
- Modify: `docs/backlog.md` — add note under the trace analysis entry that the tool is now also available
- Modify: `AGENTS.md` — update Phase 53 description to mention the tool

**Step 1: Update backlog**

In the existing `### ✅ Automated trace analysis loop` entry, append a line:
`Built-in `trace_analysis` tool also registered for agent self-diagnosis.`

**Step 2: Update AGENTS.md Phase 53 row**

Change the phase 53 description to include `trace_analysis` tool:
```
| 53 | Automated trace analysis — `trace_analysis.rs`: `SessionDigest` extractor, 5 pattern detectors (...), `build_trace_report` aggregation + suggestions; `graphirm trace-analysis` CLI; `GET /api/trace-analysis` endpoint; `trace_analysis` built-in tool; 23 new tests | ✅ done |
```

**Step 3: Commit**

```bash
git add docs/backlog.md AGENTS.md
git commit -m "docs: trace_analysis tool added to Phase 53 and backlog"
```

---

## Execution Order

```
1 → 2 → 3
```

All tasks are sequential (Task 2 depends on Task 1; Task 3 is docs-last).
