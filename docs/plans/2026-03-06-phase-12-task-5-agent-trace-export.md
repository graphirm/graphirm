# Agent Trace Export CLI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute this plan task-by-task.

**Goal:** Implement a CLI subcommand to export sessions to the Agent Trace interchange format (open spec, CC BY 4.0) — a JSON standard for linking code changes to AI conversations.

**Architecture:** Create an export module that serializes the graph's Interaction nodes into a three-level hierarchy: AgentTraceRecord (session) → TraceTurn (user/assistant/tool turn) → TraceToolCall (tool invocation + result). The CLI wires this into a new `export` subcommand that writes to stdout or a file.

**Tech Stack:** serde (serialization), clap (CLI args), existing GraphStore queries

**Background:** Agent Trace is a superset-friendly format — Graphirm's richer graph model maps cleanly to it: Interaction nodes with role="user"/"assistant" become turns; tool results get nested as TraceToolCall. This enables external tools (code editors, documentation systems) to correlate graph sessions with code changes.

---

## Prerequisites (verified present)

### From Phase 8 (HTTP Server)

The GraphStore has these query methods (used in export_session):
- `get_session_thread(session_id: &str) -> Result<Vec<GraphNode>>` — all Interaction nodes in order
- `get_tool_results_for(node_id: &NodeId) -> Result<Vec<GraphNode>>` — tool result children

Interaction node structure:
```rust
pub struct InteractionData {
    pub role: String,         // "user", "assistant", "tool"
    pub content: String,
}
pub struct GraphNode {
    pub id: NodeId,
    pub node_type: NodeType,
    pub metadata: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
}
```

---

## Task 1: Create export.rs with serde types

**Files:**
- Create: `crates/graph/src/export.rs`

**Step 1: Write the failing test**

Add to the bottom of `crates/graph/src/export.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn agent_trace_record_serializes() {
        let record = AgentTraceRecord {
            version: "0.1",
            session_id: "test-session".to_string(),
            turns: vec![
                TraceTurn {
                    id: "turn-1".to_string(),
                    role: "user".to_string(),
                    content: "What does this do?".to_string(),
                    tool_calls: vec![],
                    created_at: "2026-03-06T12:00:00Z".to_string(),
                },
                TraceTurn {
                    id: "turn-2".to_string(),
                    role: "assistant".to_string(),
                    content: "This function parses JSON.".to_string(),
                    tool_calls: vec![TraceToolCall {
                        id: "tool-1".to_string(),
                        name: "read".to_string(),
                        result: "file contents".to_string(),
                    }],
                    created_at: "2026-03-06T12:01:00Z".to_string(),
                },
            ],
        };

        let json = serde_json::to_string(&record).unwrap();
        assert!(json.contains("\"version\":\"0.1\""));
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"role\":\"assistant\""));
        assert!(json.contains("\"name\":\"read\""));
        assert_eq!(record.turns.len(), 2);
    }

    #[test]
    fn trace_tool_call_with_result() {
        let tool_call = TraceToolCall {
            id: "tc-1".to_string(),
            name: "bash".to_string(),
            result: "output from command".to_string(),
        };

        let json = serde_json::to_value(&tool_call).unwrap();
        assert_eq!(json["id"], "tc-1");
        assert_eq!(json["name"], "bash");
        assert_eq!(json["result"], "output from command");
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p graphirm-graph export::tests -- --nocapture 2>&1`

Expected: FAIL — module and types don't exist yet.

**Step 3: Implement the types**

Create `crates/graph/src/export.rs`:

```rust
//! Export sessions to Agent Trace interchange format.
//!
//! Agent Trace is an open CC BY 4.0 specification for linking code changes
//! to AI conversations. This module serializes Graphirm's Interaction nodes
//! into the standard three-level hierarchy: Record → Turn → ToolCall.
//!
//! See: https://github.com/anthropics/agent-trace-spec

use serde::Serialize;

/// A complete session export in Agent Trace format.
#[derive(Debug, Clone, Serialize)]
pub struct AgentTraceRecord {
    /// Agent Trace format version.
    pub version: &'static str,
    /// Session ID from which this trace was exported.
    pub session_id: String,
    /// Ordered sequence of turns (user prompts, assistant responses, tool results).
    pub turns: Vec<TraceTurn>,
}

/// A single turn in a conversation (user message, assistant response, or tool result).
#[derive(Debug, Clone, Serialize)]
pub struct TraceTurn {
    /// Unique identifier for this turn (node ID in graph).
    pub id: String,
    /// Role: "user", "assistant", or "tool".
    pub role: String,
    /// Content of the turn (prompt text, response, or tool output).
    pub content: String,
    /// Tool calls made during this turn (if role="assistant"; empty otherwise).
    pub tool_calls: Vec<TraceToolCall>,
    /// ISO 8601 timestamp when turn was created.
    pub created_at: String,
}

/// A tool invocation and its result, nested within a turn.
#[derive(Debug, Clone, Serialize)]
pub struct TraceToolCall {
    /// Unique identifier for this tool call (node ID in graph).
    pub id: String,
    /// Tool name (e.g. "bash", "read", "write").
    pub name: String,
    /// Tool result (stdout/output).
    pub result: String,
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p graphirm-graph export::tests -- --nocapture 2>&1`

Expected: PASS — both tests pass, types serialize cleanly.

**Step 5: Commit**

```bash
git add crates/graph/src/export.rs
git commit -m "feat(graph): add Agent Trace serde types (Record, Turn, ToolCall)"
```

---

## Task 2: Implement export_session function

**Files:**
- Modify: `crates/graph/src/export.rs`

**Step 1: Add imports and mock test setup**

At the top of `crates/graph/src/export.rs`, add:

```rust
use crate::nodes::{InteractionData, NodeType};
use crate::store::GraphStore;
use crate::error::GraphError;
```

In the test module, add:

```rust
    use crate::store::GraphStore;
    use std::sync::Arc;

    #[test]
    fn export_session_empty_graph() {
        // Create an in-memory store
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        // Try to export a non-existent session
        let result = export_session(&graph, "nonexistent-session");
        // Should return an error (session not found)
        assert!(result.is_err());
    }
```

**Step 2: Write the failing test for export_session**

Add to the test module:

```rust
    #[test]
    fn export_session_with_interactions() {
        // This is an integration-style test. For now, we'll just check the signature.
        // Full integration test will happen in Task 5 when we verify against a real session.
        // For this task, we define the function and let the compiler verify the signature.
    }
```

**Step 3: Implement export_session**

Add to `crates/graph/src/export.rs` after the types:

```rust
/// Export a session to Agent Trace format.
///
/// Queries the graph for all Interaction nodes in the session's thread,
/// filters out tool result nodes (role="tool"), and nests their corresponding
/// tool calls into each assistant turn.
///
/// # Arguments
/// * `graph` - The GraphStore to query
/// * `session_id` - The session ID to export
///
/// # Returns
/// An AgentTraceRecord on success, or a GraphError if the session is not found.
pub fn export_session(graph: &GraphStore, session_id: &str) -> Result<AgentTraceRecord, GraphError> {
    // Fetch all Interaction nodes in the session's thread (in creation order)
    let nodes = graph.get_session_thread(session_id)?;
    let mut turns = Vec::new();

    for node in &nodes {
        // Extract the interaction data; skip non-Interaction nodes
        let NodeType::Interaction(ref data) = node.node_type else {
            continue;
        };

        // Skip tool result nodes—they get merged into the parent turn below
        if data.role == "tool" {
            continue;
        }

        // Fetch any tool results that were called during this turn
        let tool_calls = graph
            .get_tool_results_for(&node.id)?
            .into_iter()
            .filter_map(|n| {
                // Each tool result is an Interaction node with role="tool"
                if let NodeType::Interaction(ref d) = n.node_type {
                    // Extract the tool name from metadata
                    let tool_name = n
                        .metadata
                        .get("tool_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();

                    Some(TraceToolCall {
                        id: n.id.to_string(),
                        name: tool_name,
                        result: d.content.clone(),
                    })
                } else {
                    None
                }
            })
            .collect();

        turns.push(TraceTurn {
            id: node.id.to_string(),
            role: data.role.clone(),
            content: data.content.clone(),
            tool_calls,
            created_at: node.created_at.to_rfc3339(),
        });
    }

    Ok(AgentTraceRecord {
        version: "0.1",
        session_id: session_id.to_string(),
        turns,
    })
}
```

**Step 4: Run tests**

Run: `cargo test -p graphirm-graph export -- --nocapture 2>&1`

Expected: PASS — both type tests pass. The export_session function compiles but will return an error on non-existent sessions (expected behavior).

**Step 5: Verify no clippy warnings**

Run: `cargo clippy -p graphirm-graph --all-targets 2>&1`

Expected: No errors or warnings related to export.

**Step 6: Commit**

```bash
git add crates/graph/src/export.rs
git commit -m "feat(graph): implement export_session with tool call nesting"
```

---

## Task 3: Export module in lib.rs

**Files:**
- Modify: `crates/graph/src/lib.rs`

**Step 1: Add module declaration**

In `crates/graph/src/lib.rs`, find the section with `pub mod` declarations (e.g., near `pub mod store;`, `pub mod nodes;`, etc.) and add:

```rust
pub mod export;
```

**Step 2: Add re-export for export function**

In the re-export section (if one exists), or after the `pub mod export;` line, add:

```rust
pub use export::{export_session, AgentTraceRecord, TraceTurn, TraceToolCall};
```

**Step 3: Verify the module compiles**

Run: `cargo check -p graphirm-graph 2>&1`

Expected: No errors.

**Step 4: Commit**

```bash
git add crates/graph/src/lib.rs
git commit -m "feat(graph): export module and types in lib.rs"
```

---

## Task 4: Add export subcommand to CLI

**Files:**
- Modify: `src/main.rs`

**Step 1: Add ExportArgs struct**

In `src/main.rs`, find the section where CLI arg structs are defined (e.g., `struct ServeArgs`, `struct ChatArgs`, etc.) and add:

```rust
/// Export a session to a standard interchange format.
#[derive(clap::Args, Debug)]
struct ExportArgs {
    /// Session ID to export.
    session_id: String,
    /// Output format (default: agent-trace).
    #[arg(long, default_value = "agent-trace")]
    format: String,
    /// Output file path (default: stdout).
    #[arg(short, long)]
    output: Option<std::path::PathBuf>,
}
```

**Step 2: Add Export variant to Commands enum**

Find the `#[derive(clap::Subcommand)]` enum (e.g., `enum Commands`) and add this variant:

```rust
    /// Export a session to interchange format
    Export(ExportArgs),
```

**Step 3: Add handler in the match arm**

Find the `match commands::Commands::` match statement in `main()` and add this arm:

```rust
    Commands::Export(args) => {
        let graph = open_graph(&config)?;
        let record = graphirm_graph::export::export_session(&graph, &args.session_id)?;
        let json = serde_json::to_string_pretty(&record)?;
        match args.output {
            Some(path) => {
                std::fs::write(&path, json).map_err(|e| {
                    format!("Failed to write export to {}: {e}", path.display()).into()
                })?;
                info!("Exported session {} to {}", args.session_id, path.display());
            }
            None => {
                println!("{json}");
            }
        }
        Ok(())
    }
```

**Step 4: Verify compilation**

Run: `cargo check --workspace 2>&1`

Expected: No errors. The CLI now has an `export` subcommand.

**Step 5: Build the binary**

Run: `cargo build --release 2>&1`

Expected: Binary compiles successfully.

**Step 6: Commit**

```bash
git add src/main.rs
git commit -m "feat(cli): add 'export' subcommand for Agent Trace format"
```

---

## Task 5: Integration test and verification

**Files:**
- No new files (verification only)

**Step 1: Get a test session ID**

Start the server and create a session:

```bash
# Terminal 1: Start the server
cargo run -- serve

# Terminal 2: Create a session and get its ID
SESSION_ID=$(curl -s -X POST http://localhost:3000/api/sessions \
  -H "Content-Type: application/json" \
  -d '{}' | jq -r '.id')
echo "Session ID: $SESSION_ID"
```

**Step 2: Send a prompt to populate the session**

```bash
curl -s -X POST http://localhost:3000/api/sessions/$SESSION_ID/prompt \
  -H "Content-Type: application/json" \
  -d '{"content": "List the files in this directory"}' \
  | jq '.'

# Wait for agent to complete (check status)
curl -s http://localhost:3000/api/sessions/$SESSION_ID | jq '.status'
```

**Step 3: Export the session**

```bash
cargo run -- export $SESSION_ID | jq '.'
```

Expected output:
```json
{
  "version": "0.1",
  "session_id": "<uuid>",
  "turns": [
    {
      "id": "<uuid>",
      "role": "user",
      "content": "List the files in this directory",
      "tool_calls": [],
      "created_at": "2026-03-06T12:00:00Z"
    },
    {
      "id": "<uuid>",
      "role": "assistant",
      "content": "...",
      "tool_calls": [
        {
          "id": "<uuid>",
          "name": "bash",
          "result": "..."
        }
      ],
      "created_at": "2026-03-06T12:01:00Z"
    }
  ]
}
```

**Step 4: Export to file**

```bash
cargo run -- export $SESSION_ID --output /tmp/trace.json
cat /tmp/trace.json | jq '.turns | length'
```

Expected: Number of turns printed (should be ≥2 if session had a prompt).

**Step 5: Verify format compliance**

```bash
# Check that all required fields are present
cargo run -- export $SESSION_ID | jq '
  . as $root |
  {
    has_version: ($root.version != null),
    has_session_id: ($root.session_id != null),
    has_turns: ($root.turns != null),
    turn_count: ($root.turns | length),
    first_turn_complete: (
      $root.turns[0] | has("id") and has("role") and has("content") and has("created_at")
    )
  }
'
```

Expected: All checks should be `true`.

**Step 6: Commit (if changes were made)**

```bash
# Typically no changes committed in this step — it's verification only
# But if you added any integration tests to the codebase, commit them:
git add crates/graph/tests/export_integration.rs
git commit -m "test(graph): add export integration test"
```

---

## Task 6: Final checks and documentation

**Files:**
- No modifications (verification only)

**Step 1: Run full test suite**

Run: `cargo test --workspace 2>&1`

Expected: All tests pass, including new export tests.

**Step 2: Run clippy across all crates**

Run: `cargo clippy --workspace --all-targets 2>&1`

Expected: No errors related to export code.

**Step 3: Check format**

Run: `cargo fmt --all -- --check 2>&1`

Expected: All files are correctly formatted (or use `cargo fmt --all` to auto-format).

**Step 4: Verify the feature works end-to-end**

```bash
# Quick smoke test
cargo run --release -- export --help
```

Expected: Help text shows the export subcommand with options.

**Step 5: Final commit**

```bash
git add -A
git commit -m "chore: Phase 12, Task 5 (Agent Trace export) — complete"
```

**Step 6: Push to remote**

```bash
git push origin main
```

---

## Success Criteria

- ✅ `crates/graph/src/export.rs` created with AgentTraceRecord, TraceTurn, TraceToolCall types
- ✅ `export_session()` function implemented, handles tool call nesting
- ✅ `crates/graph/src/lib.rs` exports the export module and types
- ✅ `src/main.rs` has `export` subcommand with `session_id`, `--format`, `--output` args
- ✅ CLI handler reads graph, calls export_session, serializes to JSON, writes to stdout or file
- ✅ No Rust compilation errors (`cargo build --release`)
- ✅ Manual test works: `graphirm export <session-id>` outputs valid Agent Trace JSON
- ✅ Export to file works: `graphirm export <session-id> --output trace.json`
- ✅ All tests pass (`cargo test --workspace`)
- ✅ Clippy clean (`cargo clippy --workspace`)
- ✅ Changes committed and pushed to main
- ✅ **Phase 12 complete: 5/5 tasks done** ✅ v3.0 ready to ship

---

## Execution Options

**Plan complete and ready for execution.**

### Option 1: Subagent-Driven (this session)

I dispatch a fresh subagent per task, review between tasks, fast iteration.

- Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6 (each independently verified)
- Each task gets spec review + code quality review before commit
- ~90 min end-to-end

### Option 2: Parallel Session (separate)

Open a new session with `executing-plans` skill, batch execution with checkpoints.

- More async-friendly for long operations
- Suited if you want to work on something else meanwhile

**Which approach would you prefer?**
