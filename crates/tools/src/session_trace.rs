//! Query past agent decision flows: semantic search across sessions or full replay.

use async_trait::async_trait;
use serde_json::json;
use std::collections::HashMap;

use graphirm_graph::nodes::{GraphNode, NodeType};

use crate::retriever::KnowledgeResult;
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
                    "description": "Max turns to show per session in search mode (default: 6, i.e. 3*2)"
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

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(max_chars).collect::<String>())
    }
}

/// Extract a compact tool call summary: `read path → grep`
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
            let path = args["path"]
                .as_str()
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

/// One-line summary of a turn: `  [role] tool_chain` or `  [role] "truncated text"`
fn format_turn_compact(node: &GraphNode) -> String {
    let NodeType::Interaction(data) = &node.node_type else {
        return format!("  [{}] (non-interaction node)", node.id);
    };

    let role = data.role.as_str();
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
        return format!("  [{}] (non-interaction node)", node.id);
    };

    let role = data.role.as_str();
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

    if role == "tool" {
        lines.push(format!("    result: {}", truncate(&data.content, 200)));
    } else if !data.content.is_empty() {
        lines.push(format!("    text: {}", data.content));
    }

    lines.join("\n")
}

async fn execute_search(
    args: &serde_json::Value,
    ctx: &ToolContext,
) -> Result<ToolOutput, ToolError> {
    let query = args["query"]
        .as_str()
        .ok_or_else(|| ToolError::InvalidArguments("'query' is required for search mode".into()))?;
    if query.trim().is_empty() {
        return Err(ToolError::InvalidArguments(
            "'query' must not be empty".into(),
        ));
    }

    let detail = args["detail"].as_str().unwrap_or("compact");
    let limit = args["limit"].as_u64().unwrap_or(5) as usize;
    let max_turns = (args["context_turns"].as_u64().unwrap_or(3) as usize).saturating_mul(2);

    let keyword_fallback = ctx.knowledge_retriever.is_none();

    let knowledge_results: Vec<KnowledgeResult> = match &ctx.knowledge_retriever {
        Some(retriever) => retriever.retrieve_semantic(query, limit * 3).await?,
        None => {
            let graph = ctx.graph.clone();
            let query_owned = query.to_string();
            let k = limit * 3;
            let nodes = tokio::task::spawn_blocking(move || {
                graph.search_knowledge(&query_owned, None, None, k)
            })
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
            nodes
                .into_iter()
                .map(|node| {
                    let node_id = node.id.clone();
                    KnowledgeResult {
                        node,
                        node_id,
                        score: 1.0,
                    }
                })
                .collect()
        }
    };

    if knowledge_results.is_empty() {
        let mut msg = format!("Session trace search for '{query}': (no matching traces found)");
        if keyword_fallback {
            msg = format!("(Note: keyword fallback — no embedding provider configured.)\n{msg}");
        }
        return Ok(ToolOutput::success(msg));
    }

    let mut session_scores: HashMap<String, f64> = HashMap::new();
    for kr in &knowledge_results {
        let sid = kr
            .node
            .metadata
            .get("session_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let entry = session_scores.entry(sid).or_insert(0.0);
        if kr.score > *entry {
            *entry = kr.score;
        }
    }

    let mut ranked: Vec<(String, f64)> = session_scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(limit);

    let graph = ctx.graph.clone();
    let mut output_lines = Vec::new();
    if keyword_fallback {
        output_lines
            .push("(Note: keyword fallback — no embedding provider configured.)".to_string());
    }
    output_lines.push(format!(
        "Session trace search for '{}' ({} candidate session{}):",
        query,
        ranked.len(),
        if ranked.len() == 1 { "" } else { "s" }
    ));

    let mut sessions_shown = 0usize;
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
        sessions_shown += 1;

        let session_label = truncate(session_id, 40);
        output_lines.push(format!(
            "\n=== Session \"{}\" (sim={:.3}) ===",
            session_label, score
        ));

        let turns_to_show: Vec<&GraphNode> = if chain.len() <= max_turns {
            chain.iter().collect()
        } else {
            chain.iter().take(max_turns).collect()
        };

        for (i, node) in turns_to_show.iter().enumerate() {
            let line = match detail {
                "full" => format!("  turn {}:\n{}", i + 1, format_turn_full(node)),
                _ => format!("  turn {}:{}", i + 1, format_turn_compact(node)),
            };
            output_lines.push(line);
        }

        if chain.len() > max_turns {
            output_lines.push(format!(
                "  ... ({} more turns, use replay mode for full trace)",
                chain.len() - max_turns
            ));
        }
    }

    if sessions_shown == 0 {
        output_lines.push("\n(no interaction traces found for matched knowledge — sessions may lack stored interactions)".to_string());
    }

    Ok(ToolOutput::success(output_lines.join("\n")))
}

async fn execute_replay(
    args: &serde_json::Value,
    ctx: &ToolContext,
) -> Result<ToolOutput, ToolError> {
    let session_id = args["session_id"].as_str().ok_or_else(|| {
        ToolError::InvalidArguments("'session_id' is required for replay mode".into())
    })?;
    let detail = args["detail"].as_str().unwrap_or("compact");

    let graph = ctx.graph.clone();
    let sid = session_id.to_string();
    let chain = tokio::task::spawn_blocking(move || graph.get_session_chain(&sid))
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
            "full" => format!("  turn {}:\n{}", i + 1, format_turn_full(node)),
            _ => format!("  turn {}:{}", i + 1, format_turn_compact(node)),
        };
        lines.push(line);
    }

    Ok(ToolOutput::success(lines.join("\n")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retriever::{KnowledgeResult, KnowledgeRetriever};
    use crate::tests::make_test_context;
    use async_trait::async_trait;
    use graphirm_graph::nodes::{InteractionData, KnowledgeData};
    use serde_json::json;
    use std::sync::Arc;

    struct MockRetriever {
        results: Vec<(String, String, String, f64)>,
    }

    #[async_trait]
    impl KnowledgeRetriever for MockRetriever {
        async fn retrieve_semantic(
            &self,
            _query: &str,
            k: usize,
        ) -> Result<Vec<KnowledgeResult>, ToolError> {
            Ok(self
                .results
                .iter()
                .take(k)
                .map(|(entity, summary, session_id, score)| {
                    let mut node = GraphNode::new(NodeType::Knowledge(KnowledgeData {
                        entity: entity.clone(),
                        entity_type: "concept".to_string(),
                        summary: summary.clone(),
                        confidence: 0.9,
                    }));
                    node.metadata["session_id"] = json!(session_id);
                    let node_id = node.id.clone();
                    KnowledgeResult {
                        node,
                        node_id,
                        score: *score,
                    }
                })
                .collect())
        }
    }

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

    #[tokio::test]
    async fn search_with_retriever_returns_traces() {
        let mut ctx = make_test_context();
        let session_id = ctx.agent_id.to_string();

        let mut msg = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "debug the auth middleware".to_string(),
            token_count: None,
        }));
        msg.metadata["session_id"] = json!(session_id.clone());
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
        let out = tool
            .execute(json!({"mode": "search", "query": "auth debug"}), &ctx)
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(
            out.content.contains("auth") || out.content.contains("debug"),
            "content: {}",
            out.content
        );
    }

    #[tokio::test]
    async fn search_no_retriever_falls_back_to_keyword() {
        let ctx = make_test_context();
        let tool = SessionTraceTool::new();
        let out = tool
            .execute(json!({"mode": "search", "query": "auth"}), &ctx)
            .await
            .unwrap();
        assert!(!out.is_error);
    }

    #[tokio::test]
    async fn search_empty_results_returns_success() {
        let mut ctx = make_test_context();
        ctx.knowledge_retriever = Some(Arc::new(MockRetriever { results: vec![] }));
        let tool = SessionTraceTool::new();
        let out = tool
            .execute(json!({"mode": "search", "query": "nonexistent_xyz"}), &ctx)
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("no matching"));
    }

    #[tokio::test]
    async fn replay_returns_full_session_trace() {
        let ctx = make_test_context();
        let session_id = ctx.agent_id.to_string();

        let mut msg1 = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "read the config file".to_string(),
            token_count: None,
        }));
        msg1.metadata["session_id"] = json!(session_id.clone());
        ctx.graph.add_node(msg1).unwrap();

        let mut msg2 = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".to_string(),
            content: "Here is the config.".to_string(),
            token_count: None,
        }));
        msg2.metadata["session_id"] = json!(session_id.clone());
        msg2.metadata["tool_calls"] = json!([
            {"id": "tc1", "name": "read", "arguments": {"path": "config/default.toml"}}
        ]);
        ctx.graph.add_node(msg2).unwrap();

        let tool = SessionTraceTool::new();
        let out = tool
            .execute(json!({"mode": "replay", "session_id": session_id}), &ctx)
            .await
            .unwrap();

        assert!(!out.is_error);
        assert!(out.content.contains("turn 1"));
        assert!(out.content.contains("turn 2"));
        assert!(
            out.content.contains("read config/default.toml") || out.content.contains("read"),
            "{}",
            out.content
        );
    }

    #[tokio::test]
    async fn replay_empty_session_returns_no_trace() {
        let ctx = make_test_context();
        let tool = SessionTraceTool::new();
        let out = tool
            .execute(
                json!({"mode": "replay", "session_id": "nonexistent-session"}),
                &ctx,
            )
            .await
            .unwrap();
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
        msg.metadata["session_id"] = json!(session_id.clone());
        ctx.graph.add_node(msg).unwrap();

        let tool = SessionTraceTool::new();
        let out = tool
            .execute(
                json!({"mode": "replay", "session_id": session_id, "detail": "full"}),
                &ctx,
            )
            .await
            .unwrap();

        assert!(!out.is_error);
        assert!(out.content.contains("Detailed reasoning about the problem"));
    }
}
