# Session Flow Traces Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `session_trace` tool that lets the agent query its own past decision flows across sessions.

**Architecture:** Knowledge-anchored trace retrieval — uses existing HNSW index to find relevant Knowledge nodes, then walks graph edges back to parent Interaction chains to reconstruct decision traces. Two modes: `search` (cross-session semantic query) and `replay` (single-session chain). Configurable detail level (`compact`/`full`).

**Tech Stack:** Rust, `graphirm-graph` (SQLite queries, graph traversal), `graphirm-tools` (Tool trait), existing `KnowledgeRetriever` trait.

**Key decisions:**
- Knowledge-anchored retrieval over direct Interaction embedding — reuses existing HNSW index, no new embedding pipeline
- Two modes (`search`/`replay`) — `replay` comes nearly free once formatting exists
- `compact`/`full` detail levels — agent usually needs patterns, not full text

**Design doc:** `docs/plans/2026-03-20-session-trace-design.md`

**Success criteria:**
- [ ] `session_trace` tool registered and callable by the agent
- [ ] `search` mode returns ranked decision traces via Knowledge-anchored HNSW (with keyword fallback)
- [ ] `replay` mode returns full chronological trace for a session
- [ ] `compact` and `full` output formatting works
- [ ] All unit tests pass (`cargo test -p graphirm-tools` and `cargo test -p graphirm-graph`)
- [ ] `cargo clippy` and `cargo fmt` clean

---

## Task 1: Add `get_session_chain` to GraphStore

**Files:**
- Modify: `crates/graph/src/store.rs`

**Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block at the bottom of `store.rs`:

```rust
#[test]
fn get_session_chain_returns_chronological_interactions() {
    let store = GraphStore::open_memory().unwrap();
    let session_id = "test-session-abc";

    let mut msg1 = GraphNode::new(NodeType::Interaction(InteractionData {
        role: "user".to_string(),
        content: "first message".to_string(),
        token_count: None,
    }));
    msg1.metadata["session_id"] = serde_json::json!(session_id);
    let id1 = store.add_node(msg1).unwrap();

    let mut msg2 = GraphNode::new(NodeType::Interaction(InteractionData {
        role: "assistant".to_string(),
        content: "response".to_string(),
        token_count: None,
    }));
    msg2.metadata["session_id"] = serde_json::json!(session_id);
    let id2 = store.add_node(msg2).unwrap();

    // Different session — should not appear
    let mut other = GraphNode::new(NodeType::Interaction(InteractionData {
        role: "user".to_string(),
        content: "other session".to_string(),
        token_count: None,
    }));
    other.metadata["session_id"] = serde_json::json!("other-session");
    store.add_node(other).unwrap();

    let chain = store.get_session_chain(session_id).unwrap();
    assert_eq!(chain.len(), 2);
    assert_eq!(chain[0].id, id1); // chronological — oldest first
    assert_eq!(chain[1].id, id2);
}

#[test]
fn get_session_chain_empty_for_unknown_session() {
    let store = GraphStore::open_memory().unwrap();
    let chain = store.get_session_chain("nonexistent").unwrap();
    assert!(chain.is_empty());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p graphirm-graph get_session_chain`
Expected: FAIL — `get_session_chain` method does not exist.

**Step 3: Write implementation**

Add to the `impl GraphStore` block in `store.rs`, near `conversation_thread`:

```rust
/// Return all Interaction nodes for a session, ordered chronologically (oldest first).
/// Uses `metadata.session_id` to filter. Returns empty vec for unknown sessions.
pub fn get_session_chain(&self, session_id: &str) -> Result<Vec<GraphNode>, GraphError> {
    let conn = self.pool.get()?;
    let mut stmt = conn.prepare(
        "SELECT id, node_type, data, metadata, created_at, updated_at
         FROM nodes
         WHERE node_type = 'interaction'
           AND json_extract(metadata, '$.session_id') = ?1
         ORDER BY created_at ASC",
    )?;

    let nodes: Vec<GraphNode> = stmt
        .query_map(params![session_id], |row| {
            let id: String = row.get(0)?;
            let _node_type: String = row.get(1)?;
            let data: String = row.get(2)?;
            let metadata: String = row.get(3)?;
            let created_at: String = row.get(4)?;
            let updated_at: String = row.get(5)?;
            Ok((id, data, metadata, created_at, updated_at))
        })?
        .filter_map(|r| r.ok())
        .filter_map(|(id, data, meta, created, updated)| {
            let node_type: NodeType = serde_json::from_str(&data).ok()?;
            let metadata: serde_json::Value = serde_json::from_str(&meta).ok()?;
            let created_at = chrono::DateTime::parse_from_rfc3339(&created)
                .ok()?
                .with_timezone(&chrono::Utc);
            let updated_at = chrono::DateTime::parse_from_rfc3339(&updated)
                .ok()?
                .with_timezone(&chrono::Utc);
            Some(GraphNode {
                id: NodeId(id),
                node_type,
                metadata,
                created_at,
                updated_at,
            })
        })
        .collect();

    Ok(nodes)
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p graphirm-graph get_session_chain`
Expected: PASS (both tests)

**Step 5: Commit**

```bash
git add crates/graph/src/store.rs
git commit -m "feat(graph): add get_session_chain helper for session trace"
```

---

## Task 2: Create `session_trace.rs` with tool skeleton and mode validation

**Files:**
- Create: `crates/tools/src/session_trace.rs`
- Modify: `crates/tools/src/lib.rs` — add `pub mod session_trace;`

**Step 1: Write the failing tests**

Create `crates/tools/src/session_trace.rs` with the test module first:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::make_test_context;
    use serde_json::json;

    #[tokio::test]
    async fn invalid_mode_returns_error() {
        let ctx = make_test_context();
        let tool = SessionTraceTool::new();
        let result = tool.execute(json!({"mode": "teleport"}), &ctx).await;
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[tokio::test]
    async fn search_mode_requires_query() {
        let ctx = make_test_context();
        let tool = SessionTraceTool::new();
        let result = tool.execute(json!({"mode": "search"}), &ctx).await;
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[tokio::test]
    async fn replay_mode_requires_session_id() {
        let ctx = make_test_context();
        let tool = SessionTraceTool::new();
        let result = tool.execute(json!({"mode": "replay"}), &ctx).await;
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[tokio::test]
    async fn tool_name_and_params() {
        let tool = SessionTraceTool::new();
        assert_eq!(tool.name(), "session_trace");
        assert!(!tool.is_destructive());
        let params = tool.parameters();
        assert!(params["properties"]["mode"].is_object());
        assert!(params["properties"]["query"].is_object());
        assert!(params["properties"]["session_id"].is_object());
        assert!(params["properties"]["detail"].is_object());
    }
}
```

**Step 2: Add module declaration**

In `crates/tools/src/lib.rs`, add `pub mod session_trace;` alongside the other module declarations.

**Step 3: Write the tool skeleton**

In `crates/tools/src/session_trace.rs`, add above the tests:

```rust
use async_trait::async_trait;
use serde_json::json;

use crate::{Tool, ToolContext, ToolError, ToolOutput};

pub struct SessionTraceTool;

impl SessionTraceTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SessionTraceTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for SessionTraceTool {
    fn name(&self) -> &str {
        "session_trace"
    }

    fn description(&self) -> &str {
        "Query the agent's past decision flows across sessions.

• search — Semantic (or keyword fallback) query across all sessions.
  Returns ranked decision traces showing what tools were called, what files
  were touched, and what the outcome was. Anchored by Knowledge nodes in the
  graph, walked back to their parent Interaction chains.

• replay — Full decision trace for a specific session.
  Returns the chronological chain of turns with tool calls and outcomes.

The tool is read-only — it never mutates the graph."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["search", "replay"],
                    "description": "search: query across sessions; replay: trace a specific session"
                },
                "query": {
                    "type": "string",
                    "description": "Natural language query (required for search mode)"
                },
                "session_id": {
                    "type": "string",
                    "description": "Session ID to replay (required for replay mode)"
                },
                "detail": {
                    "type": "string",
                    "enum": ["compact", "full"],
                    "description": "Output detail level (default: compact)"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Max session traces to return in search mode (default: 5)"
                },
                "context_turns": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Turns of context around each match in search mode (default: 3)"
                }
            },
            "required": ["mode"]
        })
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let mode = args["mode"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("'mode' is required".into()))?;

        match mode {
            "search" => execute_search(&args, ctx).await,
            "replay" => execute_replay(&args, ctx).await,
            other => Err(ToolError::InvalidArguments(format!(
                "unknown mode '{other}'; must be one of: search, replay"
            ))),
        }
    }
}

async fn execute_search(
    args: &serde_json::Value,
    _ctx: &ToolContext,
) -> Result<ToolOutput, ToolError> {
    let _query = args["query"]
        .as_str()
        .ok_or_else(|| ToolError::InvalidArguments("'query' is required for search mode".into()))?;
    if _query.trim().is_empty() {
        return Err(ToolError::InvalidArguments("'query' must not be empty".into()));
    }
    // Placeholder — implemented in Task 4
    Ok(ToolOutput::success("(search not yet implemented)"))
}

async fn execute_replay(
    args: &serde_json::Value,
    _ctx: &ToolContext,
) -> Result<ToolOutput, ToolError> {
    let _session_id = args["session_id"]
        .as_str()
        .ok_or_else(|| ToolError::InvalidArguments("'session_id' is required for replay mode".into()))?;
    // Placeholder — implemented in Task 5
    Ok(ToolOutput::success("(replay not yet implemented)"))
}
```

**Step 4: Run tests**

Run: `cargo test -p graphirm-tools session_trace`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add crates/tools/src/session_trace.rs crates/tools/src/lib.rs
git commit -m "feat(tools): add session_trace tool skeleton with mode validation"
```

---

## Task 3: Implement trace formatting helpers

**Files:**
- Modify: `crates/tools/src/session_trace.rs`

**Step 1: Write the failing tests**

Add to the test module in `session_trace.rs`:

```rust
#[test]
fn compact_format_user_turn() {
    let node = GraphNode::new(NodeType::Interaction(InteractionData {
        role: "user".to_string(),
        content: "fix the auth bug in jwt.rs".to_string(),
        token_count: None,
    }));
    let line = format_turn_compact(&node);
    assert!(line.contains("[user]"));
    assert!(line.contains("fix the auth bug"));
}

#[test]
fn compact_format_assistant_with_tool_calls() {
    let mut node = GraphNode::new(NodeType::Interaction(InteractionData {
        role: "assistant".to_string(),
        content: "Let me look at that file.".to_string(),
        token_count: None,
    }));
    node.metadata["tool_calls"] = json!([
        {"id": "tc1", "name": "read", "arguments": {"path": "src/auth/jwt.rs"}},
        {"id": "tc2", "name": "grep", "arguments": {"pattern": "refresh", "path": "src/"}}
    ]);
    let line = format_turn_compact(&node);
    assert!(line.contains("read src/auth/jwt.rs"));
    assert!(line.contains("grep"));
}

#[test]
fn compact_format_assistant_no_tools() {
    let node = GraphNode::new(NodeType::Interaction(InteractionData {
        role: "assistant".to_string(),
        content: "The fix is to change the comparison operator on line 42.".to_string(),
        token_count: None,
    }));
    let line = format_turn_compact(&node);
    assert!(line.contains("[assistant]"));
    assert!(line.contains("The fix is to change"));
}

#[test]
fn compact_format_truncates_long_content() {
    let long_content = "a".repeat(200);
    let node = GraphNode::new(NodeType::Interaction(InteractionData {
        role: "user".to_string(),
        content: long_content,
        token_count: None,
    }));
    let line = format_turn_compact(&node);
    assert!(line.len() < 200);
    assert!(line.contains('…'));
}

#[test]
fn full_format_includes_full_text() {
    let content = "This is a detailed assistant response with full reasoning.".to_string();
    let node = GraphNode::new(NodeType::Interaction(InteractionData {
        role: "assistant".to_string(),
        content: content.clone(),
        token_count: None,
    }));
    let out = format_turn_full(&node);
    assert!(out.contains(&content));
}

#[test]
fn full_format_includes_tool_args() {
    let mut node = GraphNode::new(NodeType::Interaction(InteractionData {
        role: "assistant".to_string(),
        content: "editing".to_string(),
        token_count: None,
    }));
    node.metadata["tool_calls"] = json!([
        {"id": "tc1", "name": "edit", "arguments": {"path": "src/main.rs", "old_string": "foo", "new_string": "bar"}}
    ]);
    let out = format_turn_full(&node);
    assert!(out.contains("edit"));
    assert!(out.contains("src/main.rs"));
    assert!(out.contains("old_string"));
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p graphirm-tools session_trace`
Expected: FAIL — `format_turn_compact` and `format_turn_full` do not exist.

**Step 3: Implement formatting helpers**

Add above the `execute_search` function in `session_trace.rs`:

```rust
use graphirm_graph::nodes::{GraphNode, NodeType};

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(max_chars).collect::<String>())
    }
}

/// Extract a compact tool call summary from metadata: "read src/auth.rs → grep 'pattern'"
fn format_tool_calls_compact(metadata: &serde_json::Value) -> Option<String> {
    let calls = metadata.get("tool_calls")?.as_array()?;
    if calls.is_empty() {
        return None;
    }
    let parts: Vec<String> = calls
        .iter()
        .map(|tc| {
            let name = tc["name"].as_str().unwrap_or("?");
            let args = &tc["arguments"];
            let path = args["path"].as_str()
                .or_else(|| args["file"].as_str())
                .or_else(|| args["file_a"].as_str());
            match path {
                Some(p) => format!("{name} {p}"),
                None => name.to_string(),
            }
        })
        .collect();
    Some(parts.join(" → "))
}

/// One-line summary of a turn: `[role] tool_chain or "truncated text"`
fn format_turn_compact(node: &GraphNode) -> String {
    let NodeType::Interaction(data) = &node.node_type else {
        return format!("[{}] (non-interaction node)", node.id);
    };

    let role = &data.role;
    if role == "tool" {
        return format!("  [tool] {}", truncate(&data.content, 80));
    }

    if let Some(tools) = format_tool_calls_compact(&node.metadata) {
        format!("  [{role}] {tools}")
    } else {
        format!("  [{role}] \"{}\"", truncate(&data.content, 80))
    }
}

/// Multi-line detailed turn: full text + tool arguments
fn format_turn_full(node: &GraphNode) -> String {
    let NodeType::Interaction(data) = &node.node_type else {
        return format!("[{}] (non-interaction node)\n", node.id);
    };

    let role = &data.role;
    let mut lines = Vec::new();
    lines.push(format!("  [{role}]"));

    if let Some(calls) = node.metadata.get("tool_calls").and_then(|v| v.as_array()) {
        for tc in calls {
            let name = tc["name"].as_str().unwrap_or("?");
            let args = &tc["arguments"];
            lines.push(format!("    tool: {name}"));
            if let Some(obj) = args.as_object() {
                for (k, v) in obj {
                    let val_str = match v.as_str() {
                        Some(s) => truncate(s, 120),
                        None => v.to_string(),
                    };
                    lines.push(format!("      {k}: {val_str}"));
                }
            }
        }
    }

    if !data.content.is_empty() && data.role != "tool" {
        lines.push(format!("    text: {}", &data.content));
    } else if data.role == "tool" {
        lines.push(format!("    result: {}", truncate(&data.content, 200)));
    }

    lines.join("\n")
}
```

**Step 4: Run tests**

Run: `cargo test -p graphirm-tools session_trace`
Expected: PASS (all tests including new formatting tests)

**Step 5: Commit**

```bash
git add crates/tools/src/session_trace.rs
git commit -m "feat(tools): add compact and full trace formatting helpers"
```

---

## Task 4: Implement `search` mode

**Files:**
- Modify: `crates/tools/src/session_trace.rs`

**Step 1: Write the failing tests**

Add to the test module:

```rust
use crate::retriever::{KnowledgeResult, KnowledgeRetriever};
use graphirm_graph::nodes::KnowledgeData;
use std::sync::Arc;

struct MockRetriever {
    results: Vec<(String, String, String, f64)>, // (entity, summary, session_id, score)
}

#[async_trait::async_trait]
impl KnowledgeRetriever for MockRetriever {
    async fn retrieve_semantic(
        &self,
        _query: &str,
        k: usize,
    ) -> Result<Vec<KnowledgeResult>, ToolError> {
        Ok(self.results.iter().take(k).map(|(entity, summary, session_id, score)| {
            let mut node = GraphNode::new(NodeType::Knowledge(KnowledgeData {
                entity: entity.clone(),
                entity_type: "concept".to_string(),
                summary: summary.clone(),
                confidence: 0.9,
            }));
            node.metadata["session_id"] = json!(session_id);
            KnowledgeResult {
                node_id: node.id.clone(),
                node,
                score: *score,
            }
        }).collect())
    }
}

#[tokio::test]
async fn search_with_retriever_returns_traces() {
    let mut ctx = make_test_context();

    // Seed a session with interactions
    let session_id = ctx.agent_id.to_string();
    let mut msg = GraphNode::new(NodeType::Interaction(InteractionData {
        role: "user".to_string(),
        content: "debug the auth middleware".to_string(),
        token_count: None,
    }));
    msg.metadata["session_id"] = json!(session_id);
    ctx.graph.add_node(msg).unwrap();

    ctx.knowledge_retriever = Some(Arc::new(MockRetriever {
        results: vec![(
            "auth_middleware".to_string(),
            "Debugged JWT validation in auth middleware".to_string(),
            session_id.clone(),
            0.87,
        )],
    }));

    let tool = SessionTraceTool::new();
    let out = tool.execute(json!({"mode": "search", "query": "auth debug"}), &ctx).await.unwrap();
    assert!(!out.is_error);
    assert!(out.content.contains("auth") || out.content.contains("debug"));
}

#[tokio::test]
async fn search_no_retriever_falls_back_to_keyword() {
    let ctx = make_test_context();
    let tool = SessionTraceTool::new();
    let out = tool.execute(json!({"mode": "search", "query": "auth"}), &ctx).await.unwrap();
    assert!(!out.is_error);
    // Should succeed (possibly empty) even without retriever
}

#[tokio::test]
async fn search_empty_results_returns_success() {
    let mut ctx = make_test_context();
    ctx.knowledge_retriever = Some(Arc::new(MockRetriever { results: vec![] }));
    let tool = SessionTraceTool::new();
    let out = tool.execute(json!({"mode": "search", "query": "nonexistent_xyz"}), &ctx).await.unwrap();
    assert!(!out.is_error);
    assert!(out.content.contains("no matching"));
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p graphirm-tools session_trace`
Expected: Some new tests fail (search returns placeholder).

**Step 3: Implement `execute_search`**

Replace the placeholder `execute_search` function:

```rust
async fn execute_search(
    args: &serde_json::Value,
    ctx: &ToolContext,
) -> Result<ToolOutput, ToolError> {
    let query = args["query"]
        .as_str()
        .ok_or_else(|| ToolError::InvalidArguments("'query' is required for search mode".into()))?;
    if query.trim().is_empty() {
        return Err(ToolError::InvalidArguments("'query' must not be empty".into()));
    }

    let detail = args["detail"].as_str().unwrap_or("compact");
    let limit = args["limit"].as_u64().unwrap_or(5) as usize;
    let context_turns = args["context_turns"].as_u64().unwrap_or(3) as usize;

    // Step 1: Find relevant Knowledge nodes (semantic or keyword fallback)
    let knowledge_results = match &ctx.knowledge_retriever {
        Some(retriever) => {
            retriever.retrieve_semantic(query, limit * 3).await?
        }
        None => {
            // Keyword fallback
            let graph = ctx.graph.clone();
            let query_owned = query.to_string();
            let k = limit * 3;
            tokio::task::spawn_blocking(move || {
                graph.search_knowledge(&query_owned, None, None, k)
            })
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?
            .into_iter()
            .map(|node| KnowledgeResult {
                node_id: node.id.clone(),
                node,
                score: 1.0,
            })
            .collect::<Vec<_>>()
        }
    };

    if knowledge_results.is_empty() {
        return Ok(ToolOutput::success(format!(
            "Session trace search for '{query}': (no matching traces found)"
        )));
    }

    // Step 2: Group by session_id, take best score per session
    let mut session_scores: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for kr in &knowledge_results {
        let sid = kr.node.metadata.get("session_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let entry = session_scores.entry(sid).or_insert(0.0);
        if kr.score > *entry {
            *entry = kr.score;
        }
    }

    // Step 3: Rank sessions by best score, take top `limit`
    let mut ranked: Vec<(String, f64)> = session_scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(limit);

    // Step 4: For each session, retrieve the interaction chain and format
    let graph = ctx.graph.clone();
    let mut output_lines = Vec::new();
    output_lines.push(format!(
        "Session trace search for '{}' ({} session{}):",
        query, ranked.len(), if ranked.len() == 1 { "" } else { "s" }
    ));

    for (session_id, score) in &ranked {
        let chain = {
            let g = graph.clone();
            let sid = session_id.clone();
            tokio::task::spawn_blocking(move || g.get_session_chain(&sid))
                .await
                .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?
                .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?
        };

        if chain.is_empty() {
            continue;
        }

        // Use the agent name from the session if available, else session_id
        let session_label = truncate(session_id, 40);
        output_lines.push(format!("\n=== Session \"{}\" (sim={:.2}) ===", session_label, score));

        // Show up to context_turns * 2 turns centered around the middle, or all if small
        let turns_to_show: Vec<&GraphNode> = if chain.len() <= context_turns * 2 {
            chain.iter().collect()
        } else {
            chain.iter().take(context_turns * 2).collect()
        };

        for (i, node) in turns_to_show.iter().enumerate() {
            let line = match detail {
                "full" => format!("  turn {}: {}", i + 1, format_turn_full(node)),
                _ => format!("  turn {}:{}", i + 1, format_turn_compact(node)),
            };
            output_lines.push(line);
        }

        if chain.len() > context_turns * 2 {
            output_lines.push(format!(
                "  ... ({} more turns, use replay mode for full trace)",
                chain.len() - context_turns * 2
            ));
        }
    }

    Ok(ToolOutput::success(output_lines.join("\n")))
}
```

**Step 4: Run tests**

Run: `cargo test -p graphirm-tools session_trace`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/tools/src/session_trace.rs
git commit -m "feat(tools): implement session_trace search mode with Knowledge-anchored retrieval"
```

---

## Task 5: Implement `replay` mode

**Files:**
- Modify: `crates/tools/src/session_trace.rs`

**Step 1: Write the failing tests**

Add to the test module:

```rust
#[tokio::test]
async fn replay_returns_full_session_trace() {
    let ctx = make_test_context();
    let session_id = ctx.agent_id.to_string();

    let mut msg1 = GraphNode::new(NodeType::Interaction(InteractionData {
        role: "user".to_string(),
        content: "read the config file".to_string(),
        token_count: None,
    }));
    msg1.metadata["session_id"] = json!(session_id);
    ctx.graph.add_node(msg1).unwrap();

    let mut msg2 = GraphNode::new(NodeType::Interaction(InteractionData {
        role: "assistant".to_string(),
        content: "Here is the config.".to_string(),
        token_count: None,
    }));
    msg2.metadata["session_id"] = json!(session_id);
    msg2.metadata["tool_calls"] = json!([
        {"id": "tc1", "name": "read", "arguments": {"path": "config/default.toml"}}
    ]);
    ctx.graph.add_node(msg2).unwrap();

    let tool = SessionTraceTool::new();
    let out = tool.execute(
        json!({"mode": "replay", "session_id": session_id}),
        &ctx,
    ).await.unwrap();

    assert!(!out.is_error);
    assert!(out.content.contains("turn 1"));
    assert!(out.content.contains("turn 2"));
    assert!(out.content.contains("read config/default.toml") || out.content.contains("read"));
}

#[tokio::test]
async fn replay_empty_session_returns_no_trace() {
    let ctx = make_test_context();
    let tool = SessionTraceTool::new();
    let out = tool.execute(
        json!({"mode": "replay", "session_id": "nonexistent-session"}),
        &ctx,
    ).await.unwrap();
    assert!(!out.is_error);
    assert!(out.content.contains("no trace"));
}

#[tokio::test]
async fn replay_full_detail_includes_text() {
    let ctx = make_test_context();
    let session_id = ctx.agent_id.to_string();

    let mut msg = GraphNode::new(NodeType::Interaction(InteractionData {
        role: "assistant".to_string(),
        content: "Detailed reasoning about the problem.".to_string(),
        token_count: None,
    }));
    msg.metadata["session_id"] = json!(session_id);
    ctx.graph.add_node(msg).unwrap();

    let tool = SessionTraceTool::new();
    let out = tool.execute(
        json!({"mode": "replay", "session_id": session_id, "detail": "full"}),
        &ctx,
    ).await.unwrap();

    assert!(!out.is_error);
    assert!(out.content.contains("Detailed reasoning about the problem"));
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p graphirm-tools session_trace`
Expected: Some fail (replay returns placeholder).

**Step 3: Implement `execute_replay`**

Replace the placeholder:

```rust
async fn execute_replay(
    args: &serde_json::Value,
    ctx: &ToolContext,
) -> Result<ToolOutput, ToolError> {
    let session_id = args["session_id"]
        .as_str()
        .ok_or_else(|| ToolError::InvalidArguments("'session_id' is required for replay mode".into()))?;
    let detail = args["detail"].as_str().unwrap_or("compact");

    let graph = ctx.graph.clone();
    let sid = session_id.to_string();
    let chain = tokio::task::spawn_blocking(move || g.get_session_chain(&sid))
        .await
        .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?
        .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

    if chain.is_empty() {
        return Ok(ToolOutput::success(format!(
            "Session '{session_id}': (no trace found — session may not exist or has no interactions)"
        )));
    }

    let mut lines = Vec::new();
    lines.push(format!(
        "=== Session \"{}\" ({} turn{}) ===",
        truncate(session_id, 40),
        chain.len(),
        if chain.len() == 1 { "" } else { "s" }
    ));

    for (i, node) in chain.iter().enumerate() {
        let line = match detail {
            "full" => format!("  turn {}: {}", i + 1, format_turn_full(node)),
            _ => format!("  turn {}:{}", i + 1, format_turn_compact(node)),
        };
        lines.push(line);
    }

    Ok(ToolOutput::success(lines.join("\n")))
}
```

Note: Fix the typo in the spawn_blocking closure — `g` should be `graph`:

```rust
let chain = tokio::task::spawn_blocking(move || graph.get_session_chain(&sid))
```

**Step 4: Run tests**

Run: `cargo test -p graphirm-tools session_trace`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add crates/tools/src/session_trace.rs
git commit -m "feat(tools): implement session_trace replay mode"
```

---

## Task 6: Register tool and verify end-to-end

**Files:**
- Modify: `src/main.rs` — register `SessionTraceTool` in `build_tool_registry()`

**Step 1: Find registration site**

Search for `build_tool_registry` in `src/main.rs`. Add `SessionTraceTool` alongside the other non-destructive tools (near `GraphQueryTool`, `GraphDiffTool`, `RepoBriefingTool`).

```rust
use graphirm_tools::session_trace::SessionTraceTool;
// ...
registry.register(Box::new(SessionTraceTool::new()));
```

**Step 2: Build and run full test suite**

Run: `cargo build && cargo test -p graphirm-tools && cargo test -p graphirm-graph`
Expected: PASS

**Step 3: Lint**

Run: `cargo fmt && cargo clippy --all-features -D warnings`
Expected: Clean

**Step 4: Commit**

```bash
git add src/main.rs
git commit -m "feat: register session_trace tool in build_tool_registry"
```

---

## Task 7: Update AGENTS.md and backlog

**Files:**
- Modify: `AGENTS.md` — add Phase 25 entry to Current State table, add session_trace to tool list
- Modify: `docs/backlog.md` — mark Session Flow Traces as ✅ done
- Modify: `crates/tools/AGENTS.md` — add `session_trace.rs` to key files table

**Step 1: Update all three files**

**Step 2: Commit**

```bash
git add AGENTS.md docs/backlog.md crates/tools/AGENTS.md
git commit -m "docs: mark session_trace as Phase 25 complete"
```
