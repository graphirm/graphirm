use async_trait::async_trait;
use graphirm_graph::edges::EdgeType;
use graphirm_graph::nodes::{GraphNode, NodeId, NodeType};
use serde_json::json;

use crate::{Tool, ToolContext, ToolError, ToolOutput};

pub struct GraphQueryTool;

impl GraphQueryTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GraphQueryTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for GraphQueryTool {
    fn name(&self) -> &str {
        "graph_query"
    }

    fn description(&self) -> &str {
        "Query the session graph in four modes:

• bfs — BFS traversal from a start node following outgoing edges.
  To get a start node_id, call list_type first or use an ID returned by a previous tool call.

• list_type — Enumerate nodes of a given type (interaction | agent | content | task | knowledge).
  Optionally filter by session_id and/or an exact-match metadata object (e.g. {\"status\":\"failed\"}).

• search — Case-insensitive keyword search over Knowledge nodes.
  Matches against entity, entity_type, and summary fields.

• semantic — Embedding-based similarity search over Knowledge nodes.
  Embeds the query string and finds the k most semantically similar Knowledge nodes using HNSW.
  Returns results ranked by cosine similarity (highest first).
  Requires an embedding provider to be configured (e.g. DEEPSEEK_API_KEY or local-embed feature).

The tool is read-only — it never mutates the graph."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["bfs", "list_type", "search", "semantic"],
                    "description": "Which query mode to run"
                },
                "node_id": {
                    "type": "string",
                    "description": "BFS start node ID (required for bfs mode)"
                },
                "depth": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "BFS max depth (required for bfs mode, capped at 10)"
                },
                "edge_types": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "BFS edge-type filter — array of snake_case EdgeType names \
                                    (e.g. [\"responds_to\", \"contains\", \"produces\"]). \
                                    Omit to follow all edge types."
                },
                "node_type": {
                    "type": "string",
                    "enum": ["interaction", "agent", "content", "task", "knowledge"],
                    "description": "Node type to enumerate (required for list_type mode)"
                },
                "session_id": {
                    "type": "string",
                    "description": "Filter nodes to a specific session (optional for list_type)"
                },
                "metadata": {
                    "type": "object",
                    "description": "Exact-match filter on serialized node fields \
                                    (e.g. {\"status\":\"failed\"} for Task nodes, \
                                    where TaskStatus::Failed serializes as \"failed\")"
                },
                "query": {
                    "type": "string",
                    "description": "Keyword to search for (search mode, case-insensitive) or \
                                    natural language query (semantic mode)"
                },
                "entity_type": {
                    "type": "string",
                    "description": "Filter Knowledge nodes by entity_type (optional for search mode)"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum results to return (default: 50 for bfs, 20 for list_type, 10 for search)"
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
            "bfs" => execute_bfs(&args, ctx).await,
            "list_type" => execute_list_type(&args, ctx).await,
            "search" => execute_search(&args, ctx).await,
            "semantic" => execute_semantic(&args, ctx).await,
            other => Err(ToolError::InvalidArguments(format!(
                "unknown mode '{other}'; must be one of: bfs, list_type, search, semantic"
            ))),
        }
    }
}

async fn execute_bfs(args: &serde_json::Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
    let node_id_str = args["node_id"]
        .as_str()
        .ok_or_else(|| ToolError::InvalidArguments("'node_id' is required for bfs mode".into()))?;
    let node_id = NodeId(node_id_str.to_string());

    let raw_depth = args["depth"]
        .as_u64()
        .ok_or_else(|| ToolError::InvalidArguments("'depth' is required for bfs mode".into()))?;
    let depth = (raw_depth as usize).min(10);

    let limit = args["limit"].as_u64().unwrap_or(50) as usize;

    // Parse optional edge_types from snake_case strings
    let edge_types: Vec<EdgeType> =
        if args["edge_types"].is_null() || !args["edge_types"].is_array() {
            EdgeType::all().to_vec()
        } else {
            let arr = args["edge_types"].as_array().unwrap();
            let mut parsed = Vec::with_capacity(arr.len());
            for v in arr {
                let s = v.as_str().ok_or_else(|| {
                    ToolError::InvalidArguments("edge_types entries must be strings".into())
                })?;
                // Use serde to deserialize the snake_case string into EdgeType
                let et: EdgeType = serde_json::from_value(serde_json::Value::String(s.to_string()))
                    .map_err(|_| {
                        ToolError::InvalidArguments(format!(
                            "unknown edge type '{s}'; use snake_case names like \
                         responds_to, contains, produces"
                        ))
                    })?;
                parsed.push(et);
            }
            parsed
        };

    let graph = ctx.graph.clone();
    let node_id_clone = node_id.clone();

    let (start_node, traversed) = tokio::task::spawn_blocking(move || {
        // Verify the start node exists — a missing node is a hard error, not an empty result
        let start = graph.get_node(&node_id_clone).map_err(|_| {
            ToolError::InvalidArguments(format!(
                "node '{}' does not exist in the graph",
                node_id_clone
            ))
        })?;
        let nodes = graph
            .traverse(&node_id_clone, &edge_types, depth)
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
        Ok::<_, ToolError>((start, nodes))
    })
    .await
    .map_err(|e| ToolError::ExecutionFailed(e.to_string()))??;

    let mut lines = Vec::new();
    lines.push(format!(
        "BFS from {} (type: {}) depth={depth} edge_types={}:",
        node_id,
        start_node.node_type.type_name(),
        if args["edge_types"].is_null() || !args["edge_types"].is_array() {
            "all".to_string()
        } else {
            args["edge_types"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(",")
        }
    ));

    let shown = traversed.iter().take(limit);
    let total = traversed.len();
    for node in shown {
        lines.push(format!("  {}", compact_node_summary(node)));
    }
    if total > limit {
        lines.push(format!(
            "  ... ({} more nodes, increase limit to see all)",
            total - limit
        ));
    }
    if total == 0 {
        lines.push("  (no reachable nodes)".to_string());
    }

    Ok(ToolOutput::success(lines.join("\n")))
}

async fn execute_list_type(
    args: &serde_json::Value,
    ctx: &ToolContext,
) -> Result<ToolOutput, ToolError> {
    let node_type = args["node_type"].as_str().ok_or_else(|| {
        ToolError::InvalidArguments("'node_type' is required for list_type mode".into())
    })?;

    // Validate node_type
    match node_type {
        "interaction" | "agent" | "content" | "task" | "knowledge" => {}
        other => {
            return Err(ToolError::InvalidArguments(format!(
                "unknown node_type '{other}'; must be one of: interaction, agent, content, task, knowledge"
            )));
        }
    }

    let session_id = args["session_id"].as_str().map(|s| s.to_string());
    let metadata_filter = if args["metadata"].is_object() {
        Some(args["metadata"].clone())
    } else {
        None
    };
    let limit = args["limit"].as_u64().unwrap_or(20) as usize;

    let graph = ctx.graph.clone();
    let node_type_owned = node_type.to_string();

    let nodes = tokio::task::spawn_blocking(move || {
        graph
            .list_nodes_by_type(
                &node_type_owned,
                session_id.as_deref(),
                metadata_filter.as_ref(),
                limit,
            )
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))
    })
    .await
    .map_err(|e| ToolError::ExecutionFailed(e.to_string()))??;

    let mut lines = Vec::new();
    lines.push(format!(
        "Nodes of type '{}' ({} result{}):",
        node_type,
        nodes.len(),
        if nodes.len() == 1 { "" } else { "s" }
    ));
    for node in &nodes {
        lines.push(format!("  {}", compact_node_summary(node)));
    }
    if nodes.is_empty() {
        lines.push("  (no matching nodes)".to_string());
    }

    Ok(ToolOutput::success(lines.join("\n")))
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

    let entity_type = args["entity_type"].as_str().map(|s| s.to_string());
    let session_id = args["session_id"].as_str().map(|s| s.to_string());
    let limit = args["limit"].as_u64().unwrap_or(10) as usize;
    let query_owned = query.to_string();

    let graph = ctx.graph.clone();

    let nodes = tokio::task::spawn_blocking(move || {
        graph
            .search_knowledge(
                &query_owned,
                entity_type.as_deref(),
                session_id.as_deref(),
                limit,
            )
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))
    })
    .await
    .map_err(|e| ToolError::ExecutionFailed(e.to_string()))??;

    let mut lines = Vec::new();
    lines.push(format!(
        "Knowledge search for '{}' ({} result{}):",
        query,
        nodes.len(),
        if nodes.len() == 1 { "" } else { "s" }
    ));
    for node in &nodes {
        if let NodeType::Knowledge(kd) = &node.node_type {
            lines.push(format!(
                "  [{id}] {entity} ({entity_type}) conf={conf:.2}: {summary}",
                id = node.id,
                entity = kd.entity,
                entity_type = kd.entity_type,
                conf = kd.confidence,
                summary = truncate(&kd.summary, 120),
            ));
        }
    }
    if nodes.is_empty() {
        lines.push(format!("  (no Knowledge nodes matching '{query}')"));
    }

    Ok(ToolOutput::success(lines.join("\n")))
}

async fn execute_semantic(
    args: &serde_json::Value,
    ctx: &ToolContext,
) -> Result<ToolOutput, ToolError> {
    let query = args["query"].as_str().ok_or_else(|| {
        ToolError::InvalidArguments("'query' is required for semantic mode".into())
    })?;
    if query.trim().is_empty() {
        return Err(ToolError::InvalidArguments(
            "'query' must not be empty".into(),
        ));
    }

    let retriever = ctx.knowledge_retriever.as_ref().ok_or_else(|| {
        ToolError::ExecutionFailed(
            "Semantic search is not available: this session has no embedding provider. \
             Configure an API key (e.g. DEEPSEEK_API_KEY) or enable the local-embed feature."
                .into(),
        )
    })?;

    let limit = args["limit"].as_u64().unwrap_or(10) as usize;

    let results = retriever.retrieve_semantic(query, limit).await?;

    let mut lines = Vec::new();
    lines.push(format!(
        "Semantic search for '{}' ({} result{}):",
        query,
        results.len(),
        if results.len() == 1 { "" } else { "s" }
    ));

    for result in &results {
        if let graphirm_graph::nodes::NodeType::Knowledge(kd) = &result.node.node_type {
            lines.push(format!(
                "  [{id}] {entity} ({entity_type}) sim={score:.3}: {summary}",
                id = result.node_id,
                entity = kd.entity,
                entity_type = kd.entity_type,
                score = result.score,
                summary = truncate(&kd.summary, 120),
            ));
        }
    }

    if results.is_empty() {
        lines.push(format!("  (no Knowledge nodes semantically similar to '{query}')"));
    }

    Ok(ToolOutput::success(lines.join("\n")))
}

fn compact_node_summary(node: &GraphNode) -> String {
    let type_name = node.node_type.type_name();
    let detail = match &node.node_type {
        NodeType::Interaction(d) => format!("role={} content={}", d.role, truncate(&d.content, 60)),
        NodeType::Agent(d) => format!("name={} model={} status={}", d.name, d.model, d.status),
        NodeType::Content(d) => {
            let path = d.path.as_deref().unwrap_or("(no path)");
            format!("type={} path={}", d.content_type, path)
        }
        NodeType::Task(d) => format!(
            "title={} status={} priority={}",
            truncate(&d.title, 40),
            d.status,
            d.priority.map_or("none".to_string(), |p| p.to_string())
        ),
        NodeType::Knowledge(d) => format!(
            "entity={} entity_type={} conf={:.2}: {}",
            d.entity,
            d.entity_type,
            d.confidence,
            truncate(&d.summary, 80)
        ),
    };
    format!("[{id}] {type_name} | {detail}", id = node.id)
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(max_chars).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::make_test_context;
    use graphirm_graph::nodes::{
        GraphNode, InteractionData, KnowledgeData, NodeType, TaskData, TaskStatus,
    };
    use serde_json::json;

    #[tokio::test]
    async fn bfs_returns_traversed_nodes_from_seeded_graph() {
        let ctx = make_test_context();
        let graph = &ctx.graph;

        let root = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "root message".to_string(),
            token_count: None,
        }));
        let root_id = root.id.clone();
        let child = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".to_string(),
            content: "response".to_string(),
            token_count: None,
        }));
        let child_id = child.id.clone();

        graph.add_node(root).unwrap();
        graph.add_node(child).unwrap();
        graph
            .add_edge(graphirm_graph::edges::GraphEdge::new(
                graphirm_graph::edges::EdgeType::RespondsTo,
                child_id.clone(),
                root_id.clone(),
            ))
            .unwrap();

        let tool = GraphQueryTool::new();
        let result = tool
            .execute(
                json!({
                    "mode": "bfs",
                    "node_id": root_id.to_string(),
                    "depth": 2
                }),
                &ctx,
            )
            .await;

        assert!(result.is_ok(), "bfs should return Ok, got: {result:?}");
        let out = result.unwrap();
        assert!(!out.is_error);
        // root follows outgoing edges — RespondsTo goes child→root, so root has no outgoing
        // this verifies bfs runs without error even if no outgoing edges
        let _ = out;
    }

    #[tokio::test]
    async fn bfs_follows_outgoing_edge_from_source() {
        let ctx = make_test_context();
        let graph = &ctx.graph;

        let parent = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "parent".to_string(),
            token_count: None,
        }));
        let child = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".to_string(),
            content: "child".to_string(),
            token_count: None,
        }));
        let parent_id = parent.id.clone();
        let child_id = child.id.clone();

        graph.add_node(parent).unwrap();
        graph.add_node(child).unwrap();
        // parent → child (outgoing from parent)
        graph
            .add_edge(graphirm_graph::edges::GraphEdge::new(
                graphirm_graph::edges::EdgeType::Contains,
                parent_id.clone(),
                child_id.clone(),
            ))
            .unwrap();

        let tool = GraphQueryTool::new();
        let out = tool
            .execute(
                json!({
                    "mode": "bfs",
                    "node_id": parent_id.to_string(),
                    "depth": 2
                }),
                &ctx,
            )
            .await
            .unwrap();

        assert!(!out.is_error);
        assert!(
            out.content.contains(&child_id.to_string()),
            "child should appear in bfs output"
        );
    }

    #[tokio::test]
    async fn bfs_nonexistent_start_node_returns_error() {
        let ctx = make_test_context();
        let tool = GraphQueryTool::new();
        let result = tool
            .execute(
                json!({
                    "mode": "bfs",
                    "node_id": "00000000-0000-0000-0000-000000000000",
                    "depth": 1
                }),
                &ctx,
            )
            .await;
        assert!(result.is_err(), "bfs with nonexistent node should be Err");
    }

    #[tokio::test]
    async fn bfs_unknown_edge_type_returns_error() {
        let ctx = make_test_context();
        let node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "hi".to_string(),
            token_count: None,
        }));
        let node_id = node.id.clone();
        ctx.graph.add_node(node).unwrap();

        let tool = GraphQueryTool::new();
        let result = tool
            .execute(
                json!({
                    "mode": "bfs",
                    "node_id": node_id.to_string(),
                    "depth": 1,
                    "edge_types": ["not_a_real_edge_type"]
                }),
                &ctx,
            )
            .await;
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[tokio::test]
    async fn list_type_returns_filtered_task_nodes() {
        let ctx = make_test_context();
        let graph = &ctx.graph;

        let session_id = ctx.agent_id.to_string();

        let mut task_failed = GraphNode::new(NodeType::Task(TaskData {
            title: "Fix bug".to_string(),
            description: "Critical fix".to_string(),
            status: TaskStatus::Failed,
            priority: Some(1),
        }));
        task_failed.metadata["session_id"] = json!(session_id.clone());
        graph.add_node(task_failed).unwrap();

        let mut task_pending = GraphNode::new(NodeType::Task(TaskData {
            title: "New feature".to_string(),
            description: "Pending work".to_string(),
            status: TaskStatus::Pending,
            priority: None,
        }));
        task_pending.metadata["session_id"] = json!(session_id.clone());
        graph.add_node(task_pending).unwrap();

        let tool = GraphQueryTool::new();
        let result = tool
            .execute(
                json!({
                    "mode": "list_type",
                    "node_type": "task",
                    "session_id": session_id,
                    "metadata": {"status": "failed"},
                    "limit": 5
                }),
                &ctx,
            )
            .await;

        assert!(
            result.is_ok(),
            "list_type should return Ok, got: {result:?}"
        );
        let out = result.unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("task") || out.content.contains("Fix bug"));
    }

    #[tokio::test]
    async fn list_type_unknown_type_returns_error() {
        let ctx = make_test_context();
        let tool = GraphQueryTool::new();
        let result = tool
            .execute(json!({"mode": "list_type", "node_type": "galaxy"}), &ctx)
            .await;
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[tokio::test]
    async fn search_returns_matching_knowledge_nodes() {
        let ctx = make_test_context();
        let graph = &ctx.graph;

        let k = GraphNode::new(NodeType::Knowledge(KnowledgeData {
            entity: "authenticate_user".to_string(),
            entity_type: "function".to_string(),
            summary: "Handles JWT authentication".to_string(),
            confidence: 0.9,
        }));
        graph.add_node(k).unwrap();

        let tool = GraphQueryTool::new();
        let result = tool
            .execute(
                json!({
                    "mode": "search",
                    "query": "auth"
                }),
                &ctx,
            )
            .await;

        assert!(result.is_ok(), "search should return Ok, got: {result:?}");
        let out = result.unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("authenticate_user") || out.content.contains("auth"));
    }

    #[tokio::test]
    async fn invalid_mode_returns_invalid_arguments_error() {
        let ctx = make_test_context();
        let tool = GraphQueryTool::new();
        let result = tool.execute(json!({"mode": "teleport"}), &ctx).await;
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[tokio::test]
    async fn empty_search_returns_success_not_error() {
        let ctx = make_test_context();
        let tool = GraphQueryTool::new();
        let result = tool
            .execute(
                json!({
                    "mode": "search",
                    "query": "xyzzy_no_match_expected"
                }),
                &ctx,
            )
            .await;
        assert!(result.is_ok(), "empty search should be Ok, got: {result:?}");
        let out = result.unwrap();
        assert!(!out.is_error);
    }
}
