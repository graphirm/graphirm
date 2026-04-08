# Automated Trace Analysis Loop — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a `trace_analysis` CLI command and built-in tool that mines completed sessions for systematic failure patterns and outputs a structured report with suggested harness improvements.

**Architecture:** New module `crates/agent/src/trace_analysis.rs` with pure functions that accept `&GraphStore` and return structured `TraceReport`. Exposed as (a) `graphirm trace-analysis` CLI subcommand in `src/commands/trace_analysis.rs`, (b) `GET /api/trace-analysis` HTTP endpoint, and (c) optionally a `trace_analysis` built-in tool for the agent to self-diagnose. All read-only — no graph mutations.

**Tech Stack:** `graphirm-graph` (GraphStore queries), `serde` (report serialization), existing metadata on Interaction nodes (Phase 36/37/42–49 stamps). No new crate dependencies.

**Key decisions:**
- **Pure analysis functions over tool-first**: the core logic lives in `graphirm-agent` (not `graphirm-tools`) because it needs `GraphStore` directly and doesn't run in a tool context. A thin tool wrapper can delegate later.
- **Heuristic pattern detectors over LLM classification**: keeps it deterministic, fast, and testable. LLM-based clustering is a future extension.
- **CLI + HTTP first, tool second**: the primary consumer is a human reviewing session quality. The agent tool is optional (Task 6).

**Success criteria:**
- [x] `graphirm trace-analysis` prints a JSON/markdown report from graph data
- [x] `GET /api/trace-analysis` returns the same report as JSON
- [x] Report includes: per-session scores, top failure patterns, suggested parameter changes
- [x] At least 5 pattern detectors implemented and tested
- [x] All existing tests pass, clippy clean

**Risks:**
- Empty/sparse graph data on dev machine — use `app.graphirm.ai` DB or write synthetic test fixtures
- Performance on large graphs — cap session scan to most recent N (default 50)

---

## Data Available (what we mine)

Each assistant `Interaction` node carries this metadata (from Phases 36–49):

| Key | Type | Source |
|-----|------|--------|
| `usage_input` / `usage_output` | u32 | Every turn |
| `model_tier` | string | Router (cheap/smart) |
| `model_selected` | string | Router |
| `routing_strategy` / `routing_reason` | string | Router |
| `routing_confidence` / `routing_decision_ms` | f64/u64 | Router |
| `tool_calls` | array | When model emitted tool calls |
| `context_stats` | object | Phase 37 telemetry |
| `fallback_chain` | array | When primary model failed |
| `tools_gated` | bool | Phase 52 tool gate |
| `session_token_cap_exceeded` | bool | Token cap hit |

Each tool-result `Interaction` has: `tool_call_id`, `tool_name`, `is_error`.

Agent nodes have: `status` (completed/error/token_cap_exceeded/cancelled).

---

## Task 1: `TraceReport` types and `SessionDigest` extractor

**Files:**
- Create: `crates/agent/src/trace_analysis.rs`
- Modify: `crates/agent/src/lib.rs` (add `pub mod trace_analysis;`)

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use graphirm_graph::GraphStore;
    use std::sync::Arc;

    #[test]
    fn digest_empty_session_returns_none() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let digest = build_session_digest(&graph, "nonexistent");
        assert!(digest.is_none());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p graphirm-agent trace_analysis::tests::digest_empty -- --nocapture`
Expected: FAIL — module doesn't exist yet.

**Step 3: Implement types and `build_session_digest`**

```rust
use serde::{Deserialize, Serialize};
use graphirm_graph::GraphStore;
use graphirm_graph::nodes::{GraphNode, NodeType};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionDigest {
    pub session_id: String,
    pub agent_name: String,
    pub status: String,
    pub turn_count: u32,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub tool_call_count: u32,
    pub tool_error_count: u32,
    pub unique_tools_used: Vec<String>,
    pub tools_gated_count: u32,
    pub fallback_count: u32,
    pub model_tiers: Vec<String>,  // ordered per turn
}

pub fn build_session_digest(graph: &GraphStore, session_id: &str) -> Option<SessionDigest> {
    let chain = graph.get_session_chain(session_id).ok()?;
    if chain.is_empty() {
        return None;
    }
    // ... extract from chain metadata
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p graphirm-agent trace_analysis::tests -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/agent/src/trace_analysis.rs crates/agent/src/lib.rs
git commit -m "feat(trace-analysis): SessionDigest type + build_session_digest extractor"
```

---

## Task 2: Pattern detectors — 5 heuristic analyzers

**Files:**
- Modify: `crates/agent/src/trace_analysis.rs`

Each detector takes `&SessionDigest` and returns `Option<PatternMatch>`.

**Detectors to implement:**

1. **`detect_over_tooling`** — `tool_call_count / turn_count > threshold` (default 3.0). Sessions where the agent called tools excessively relative to turns.

2. **`detect_doom_loops`** — scan tool-result nodes for repeated `tool_name` + `is_error: true` on the same file path. Flag when ≥3 consecutive errors on the same tool+path.

3. **`detect_token_waste`** — sessions where `total_output_tokens / turn_count > threshold` (default 2000) but `status != "completed"`. High token spend with no success.

4. **`detect_tool_errors_without_recovery`** — `tool_error_count > 0` and no subsequent successful tool call on the same tool. Agent hit an error and gave up or switched approach without fixing.

5. **`detect_premature_completion`** — `status == "completed"` but `turn_count <= 2` and `tool_call_count == 0`. Agent declared done without doing any work (conversational turns only).

**Step 1: Write failing tests for each detector**

```rust
#[test]
fn detect_over_tooling_flags_high_ratio() {
    let digest = SessionDigest {
        tool_call_count: 30,
        turn_count: 5,
        ..default_digest()
    };
    let result = detect_over_tooling(&digest, 3.0);
    assert!(result.is_some());
}

#[test]
fn detect_over_tooling_passes_normal_ratio() {
    let digest = SessionDigest {
        tool_call_count: 8,
        turn_count: 5,
        ..default_digest()
    };
    assert!(detect_over_tooling(&digest, 3.0).is_none());
}
```

Write similar paired tests (positive + negative) for each of the 5 detectors.

**Step 2: Run tests to verify they fail**

Run: `cargo test -p graphirm-agent trace_analysis::tests -- --nocapture`
Expected: FAIL (functions don't exist)

**Step 3: Implement detectors**

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatch {
    pub pattern: String,
    pub severity: Severity,
    pub description: String,
    pub suggestion: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Info,
    Warning,
    Critical,
}

pub fn detect_over_tooling(digest: &SessionDigest, threshold: f64) -> Option<PatternMatch> { ... }
pub fn detect_doom_loops(chain: &[GraphNode]) -> Option<PatternMatch> { ... }
pub fn detect_token_waste(digest: &SessionDigest, threshold: u64) -> Option<PatternMatch> { ... }
pub fn detect_tool_errors_without_recovery(chain: &[GraphNode]) -> Option<PatternMatch> { ... }
pub fn detect_premature_completion(digest: &SessionDigest) -> Option<PatternMatch> { ... }
```

Note: `detect_doom_loops` and `detect_tool_errors_without_recovery` need the raw `chain: &[GraphNode]` (not just the digest) because they inspect sequential tool-result metadata.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p graphirm-agent trace_analysis::tests -- --nocapture`
Expected: PASS (10+ tests)

**Step 5: Commit**

```bash
git add crates/agent/src/trace_analysis.rs
git commit -m "feat(trace-analysis): 5 pattern detectors with tests"
```

---

## Task 3: `build_trace_report` — aggregate across sessions

**Files:**
- Modify: `crates/agent/src/trace_analysis.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn build_report_from_empty_graph_returns_empty_patterns() {
    let graph = Arc::new(GraphStore::open_memory().unwrap());
    let report = build_trace_report(&graph, 50);
    assert!(report.patterns.is_empty());
    assert_eq!(report.sessions_analyzed, 0);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p graphirm-agent trace_analysis::tests::build_report -- --nocapture`
Expected: FAIL

**Step 3: Implement**

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceReport {
    pub sessions_analyzed: u32,
    pub patterns: Vec<AggregatePattern>,
    pub per_session: Vec<SessionSummary>,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatePattern {
    pub pattern: String,
    pub occurrences: u32,
    pub severity: Severity,
    pub affected_sessions: Vec<String>,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub session_id: String,
    pub agent_name: String,
    pub status: String,
    pub turn_count: u32,
    pub token_total: u64,
    pub findings: Vec<PatternMatch>,
}

/// Analyze up to `max_sessions` most recent completed sessions.
pub fn build_trace_report(graph: &GraphStore, max_sessions: usize) -> TraceReport {
    let agents = graph.get_agent_nodes().unwrap_or_default();
    // Sort by created_at DESC, take max_sessions
    // For each: build_session_digest, run all detectors, collect
    // Aggregate: group by pattern name, count, list affected sessions
    // Suggestions: map patterns to harness parameter recommendations
}
```

**Suggestion mapping (hardcoded v1):**
- `over_tooling` → "Consider enabling `tool_gate_enabled = true` or lowering `doom_loop_threshold`"
- `doom_loops` → "Consider reducing `doom_loop_threshold` from current value"
- `token_waste` → "Consider lowering `max_output_tokens` or enabling budget warnings"
- `tool_errors_without_recovery` → "Consider adding `error_recovery` routing rule if not present"
- `premature_completion` → "Check system prompt — agent may lack context to act"

**Step 4: Run tests**

Run: `cargo test -p graphirm-agent trace_analysis::tests -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/agent/src/trace_analysis.rs
git commit -m "feat(trace-analysis): build_trace_report aggregation + suggestions"
```

---

## Task 4: CLI subcommand `graphirm trace-analysis`

**Files:**
- Create: `src/commands/trace_analysis.rs`
- Modify: `src/commands/mod.rs` (add `pub mod trace_analysis;`)
- Modify: `src/main.rs` (add CLI variant + handler dispatch)

**Step 1: Check existing CLI pattern**

Read: `src/commands/mod.rs` for how other commands are structured.
Read: `src/main.rs` for the `Commands` enum pattern.

**Step 2: Implement CLI command**

```rust
// src/commands/trace_analysis.rs
use std::sync::Arc;
use graphirm_graph::GraphStore;

pub fn run(db_path: &str, max_sessions: usize, format: &str) -> Result<(), crate::GraphirmError> {
    let graph = Arc::new(GraphStore::open(db_path)?);
    let report = graphirm_agent::trace_analysis::build_trace_report(&graph, max_sessions);

    match format {
        "json" => println!("{}", serde_json::to_string_pretty(&report).unwrap()),
        "markdown" | _ => print_markdown_report(&report),
    }
    Ok(())
}

fn print_markdown_report(report: &graphirm_agent::trace_analysis::TraceReport) {
    println!("# Trace Analysis Report\n");
    println!("Sessions analyzed: {}\n", report.sessions_analyzed);
    for pattern in &report.patterns {
        println!("## {} ({:?}) — {} occurrences", pattern.pattern, pattern.severity, pattern.occurrences);
        println!("{}\n", pattern.description);
    }
    if !report.suggestions.is_empty() {
        println!("## Suggestions\n");
        for s in &report.suggestions {
            println!("- {}", s);
        }
    }
}
```

Add to `src/main.rs`:

```rust
// In Commands enum:
TraceAnalysis {
    #[arg(long, default_value = "50")]
    max_sessions: usize,
    #[arg(long, default_value = "markdown")]
    format: String,
},

// In match:
Commands::TraceAnalysis { max_sessions, format } => {
    commands::trace_analysis::run(&db_path, max_sessions, &format)?;
}
```

**Step 3: Build and smoke test**

Run: `cargo build` (verify compiles)
Run: `./target/debug/graphirm trace-analysis --format json` (verify runs against local DB)

**Step 4: Commit**

```bash
git add src/commands/trace_analysis.rs src/commands/mod.rs src/main.rs
git commit -m "feat(cli): add trace-analysis subcommand"
```

---

## Task 5: HTTP endpoint `GET /api/trace-analysis`

**Files:**
- Modify: `crates/server/src/routes.rs` (add handler + route)
- Modify: `crates/server/src/types.rs` (add query type if needed)

**Step 1: Write failing test**

Add a test in `routes::tests` that hits `/api/trace-analysis` and expects 200 + valid JSON:

```rust
#[tokio::test]
async fn test_trace_analysis_returns_200() {
    let (app, _state) = test_app().await;
    let resp = app
        .oneshot(authorized_request(Method::GET, "/api/trace-analysis"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p graphirm-server routes::tests::test_trace_analysis -- --nocapture`
Expected: FAIL (404)

**Step 3: Implement**

```rust
// In routes.rs:
async fn trace_analysis_report(
    State(state): State<AppState>,
    Query(params): Query<TraceAnalysisQuery>,
) -> Result<Json<graphirm_agent::trace_analysis::TraceReport>, ErrorResponse> {
    let graph = state.graph.clone();
    let max = params.max_sessions.unwrap_or(50);
    let report = tokio::task::spawn_blocking(move || {
        graphirm_agent::trace_analysis::build_trace_report(&graph, max)
    })
    .await
    .map_err(|e| ErrorResponse::internal(e.to_string()))?;
    Ok(Json(report))
}

// Query type:
#[derive(Debug, Deserialize)]
pub struct TraceAnalysisQuery {
    pub max_sessions: Option<usize>,
}

// Register:
.route("/api/trace-analysis", get(trace_analysis_report))
```

**Step 4: Run tests**

Run: `cargo test -p graphirm-server -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/server/src/routes.rs crates/server/src/types.rs
git commit -m "feat(server): GET /api/trace-analysis endpoint"
```

---

## Task 6 (optional): `trace_analysis` built-in tool

**Files:**
- Create: `crates/tools/src/trace_analysis.rs`
- Modify: `crates/tools/src/lib.rs` (add module)
- Modify: `src/main.rs` or `src/commands/serve.rs` (register in tool registry)

This wraps `build_trace_report` as a non-destructive `Tool` so the agent can self-diagnose during a session. Lower priority — the CLI + HTTP endpoints cover the primary use case.

Skip this task in v1 unless the agent needs self-reflection capability.

---

## Task 7: Update docs and backlog

**Files:**
- Modify: `docs/backlog.md` — mark item ✅ with summary
- Modify: `AGENTS.md` — add Phase 53 entry to Current State table

**Step 1: Update backlog**

Change `### Automated trace analysis loop — P3 · L` to `### ✅ Automated trace analysis loop — P3 · L` with a Done summary.

**Step 2: Update AGENTS.md**

Add a row to the Current State table with the phase number, description, and status.

**Step 3: Commit**

```bash
git add docs/backlog.md AGENTS.md
git commit -m "docs: mark trace analysis loop complete in backlog + AGENTS.md"
```

---

## Execution Order

Tasks 1–3 are sequential (each builds on the previous).
Tasks 4 and 5 are independent of each other (both depend on Task 3).
Task 6 is optional.
Task 7 is last.

```
1 → 2 → 3 → 4 (parallel with 5)
              → 5 (parallel with 4)
              → 6 (optional)
                → 7
```
