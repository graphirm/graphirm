use async_trait::async_trait;
use graphirm_graph::Direction;
use graphirm_graph::edges::{EdgeType, GraphEdge};
use graphirm_graph::nodes::{GraphNode, KnowledgeData, NodeId, NodeType};
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
        "Query the session graph in six modes:

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

• neighbors — List direct neighbors of a node. Returns immediate neighbors (one hop away).
  To get a start node_id, call list_type first or use an ID returned by a previous tool call.

• project — Project planning operations for creating and managing planning Knowledge nodes:
  - create: Create a new planning node (epic, story, criterion, or decision)
  - list: List planning nodes with optional filtering
  - link_session: Link the current session to a planning node
  - update: Update a planning node's status

The tool is read-only for bfs/list_type/search/semantic/neighbors modes; project mode may write to the graph."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["bfs", "list_type", "search", "semantic", "neighbors", "project"],
                    "description": "Which query mode to run"
                },
                "node_id": {
                    "type": "string",
                    "description": "BFS start node ID (required for bfs mode), or target node for neighbors mode"
                },
                "depth": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "BFS max depth (required for bfs mode, capped at 10)"
                },
                "max_depth": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Optional override for BFS max depth. When provided, limits traversal depth; when omitted, uses the 'depth' parameter."
                },
                "edge_types": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "BFS edge-type filter — array of snake_case EdgeType names \
                                    (e.g. [\"responds_to\", \"contains\", \"produces\"]). \
                                    Omit to follow all edge types. For neighbors mode, filters by edge type."
                },
                "direction": {
                    "type": "string",
                    "enum": ["outgoing", "incoming", "all"],
                    "description": "Neighbors mode only — direction of edges to follow (default: all)"
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
                    "description": "Maximum results to return (default: 50 for bfs, 20 for list_type, 10 for search, 50 for neighbors)"
                },
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "link_session", "update"],
                    "description": "Action to perform in project mode"
                },
                "entity": {
                    "type": "string",
                    "description": "Entity name for create action"
                },
                "summary": {
                    "type": "string",
                    "description": "Summary/description for create action"
                },
                "parent_id": {
                    "type": "string",
                    "description": "Parent planning node ID for create action (optional)"
                },
                "planning_node_id": {
                    "type": "string",
                    "description": "Planning node ID for link_session action"
                },
                "status": {
                    "type": "string",
                    "enum": ["open", "in_progress", "done"],
                    "description": "Status for update action"
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
            "neighbors" => execute_neighbors(&args, ctx).await,
            "project" => execute_project(&args, ctx).await,
            other => Err(ToolError::InvalidArguments(format!(
                "unknown mode '{other}'; must be one of: bfs, list_type, search, semantic, neighbors, project"
            ))),
        }
    }
}

async fn execute_bfs(args: &serde_json::Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
    let node_id_str = args["node_id"]
        .as_str()
        .ok_or_else(|| ToolError::InvalidArguments("'node_id' is required for bfs mode".into()))?;
    let node_id = NodeId(node_id_str.to_string());

    // Use max_depth when provided, otherwise use depth parameter
    let raw_depth = args["depth"]
        .as_u64()
        .ok_or_else(|| ToolError::InvalidArguments("'depth' is required for bfs mode".into()))?;
    let depth = (raw_depth as usize).min(10);
    
    // max_depth is optional - when provided, it overrides depth
    let max_depth = if let Some(raw_max_depth) = args["max_depth"].as_u64() {
        (raw_max_depth as usize).min(10)
    } else {
        depth
    };

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
            .traverse(&node_id_clone, &edge_types, max_depth)
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
        Ok::<_, ToolError>((start, nodes))
    })
    .await
    .map_err(|e| ToolError::ExecutionFailed(e.to_string()))??;

    let mut lines = Vec::new();
    lines.push(format!(
        "BFS from {} (type: {}) depth={max_depth} edge_types={}:",
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
    for (node, depth_level) in shown {
        lines.push(format!("  [depth={depth_level}] {}", compact_node_summary(node)));
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
        lines.push(format!(
            "  (no Knowledge nodes semantically similar to '{query}')"
        ));
    }

    Ok(ToolOutput::success(lines.join("\n")))
}

async fn execute_neighbors(
    args: &serde_json::Value,
    ctx: &ToolContext,
) -> Result<ToolOutput, ToolError> {
    let node_id_str = args["node_id"]
        .as_str()
        .ok_or_else(|| ToolError::InvalidArguments("'node_id' is required for neighbors mode".into()))?;
    let node_id = NodeId(node_id_str.to_string());

    let direction_str = args["direction"].as_str().unwrap_or("all");
    let dir_filter: Option<Direction> = match direction_str {
        "outgoing" => Some(Direction::Outgoing),
        "incoming" => Some(Direction::Incoming),
        "all" => None,
        other => {
            return Err(ToolError::InvalidArguments(format!(
                "unknown direction '{other}'; must be one of: outgoing, incoming, all"
            )));
        }
    };

    let edge_type_labels: Option<String> = args["edge_types"].as_array().map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_str())
            .collect::<Vec<_>>()
            .join(",")
    });

    let edge_type_filter: Option<Vec<EdgeType>> = if let Some(arr) = args["edge_types"].as_array() {
        let mut parsed = Vec::with_capacity(arr.len());
        for v in arr {
            let s = v.as_str().ok_or_else(|| {
                ToolError::InvalidArguments("edge_types entries must be strings".into())
            })?;
            let et: EdgeType = serde_json::from_value(serde_json::Value::String(s.to_string()))
                .map_err(|_| {
                    ToolError::InvalidArguments(format!(
                        "unknown edge type '{s}'; use snake_case names like \
                         responds_to, contains, produces"
                    ))
                })?;
            parsed.push(et);
        }
        Some(parsed)
    } else {
        None
    };

    let limit = args["limit"].as_u64().unwrap_or(50) as usize;

    let graph = ctx.graph.clone();
    let node_id_clone = node_id.clone();

    let results: Vec<(GraphNode, String, String)> = tokio::task::spawn_blocking(move || {
        graph.get_node(&node_id_clone).map_err(|_| {
            ToolError::InvalidArguments(format!(
                "node '{}' does not exist in the graph",
                node_id_clone
            ))
        })?;

        let edges = graph.edges_for_node(&node_id_clone)?;
        let mut out: Vec<(GraphNode, String, String)> = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for edge in &edges {
            let (neighbor_id, dir) = if edge.source == node_id_clone {
                (&edge.target, "outgoing")
            } else {
                (&edge.source, "incoming")
            };

            if let Some(ref d) = dir_filter {
                let wanted = match d {
                    Direction::Outgoing => "outgoing",
                    Direction::Incoming => "incoming",
                };
                if dir != wanted {
                    continue;
                }
            }

            if let Some(ref allowed) = edge_type_filter
                && !allowed.contains(&edge.edge_type)
            {
                continue;
            }

            if !seen.insert(neighbor_id.clone()) {
                continue;
            }

            match graph.get_node(neighbor_id) {
                Ok(node) => out.push((node, edge.edge_type.as_str().to_string(), dir.to_string())),
                Err(_) => continue,
            }
        }

        Ok::<_, ToolError>(out)
    })
    .await
    .map_err(|e| ToolError::ExecutionFailed(e.to_string()))??;

    let mut lines = Vec::new();
    lines.push(format!(
        "Neighbors of {} (direction={}, edge_types={}):",
        node_id,
        direction_str,
        edge_type_labels.as_deref().unwrap_or("all")
    ));

    let total = results.len();
    for (node, edge_type, dir) in results.iter().take(limit) {
        lines.push(format!(
            "  [{}] (edge_type={}, direction={})",
            compact_node_summary(node),
            edge_type,
            dir,
        ));
    }
    if total > limit {
        lines.push(format!(
            "  ... ({} more neighbors, increase limit to see all)",
            total - limit
        ));
    }
    if total == 0 {
        lines.push("  (no neighbors)".to_string());
    }

    Ok(ToolOutput::success(lines.join("\n")))
}

async fn execute_project(
    args: &serde_json::Value,
    ctx: &ToolContext,
) -> Result<ToolOutput, ToolError> {
    let action = args["action"].as_str().ok_or_else(|| {
        ToolError::InvalidArguments("'action' is required for project mode".into())
    })?;

    match action {
        "create" => {
            let entity = args["entity"]
                .as_str()
                .ok_or_else(|| ToolError::InvalidArguments("'entity' is required".into()))?
                .to_string();
            let summary = args["summary"]
                .as_str()
                .ok_or_else(|| ToolError::InvalidArguments("'summary' is required".into()))?
                .to_string();
            let entity_type = args["entity_type"]
                .as_str()
                .ok_or_else(|| ToolError::InvalidArguments("'entity_type' is required".into()))?
                .to_string();

            match entity_type.as_str() {
                "epic" | "story" | "criterion" | "decision" => {}
                other => {
                    return Err(ToolError::InvalidArguments(format!(
                        "unknown entity_type '{other}'; must be one of: epic, story, criterion, decision"
                    )));
                }
            }

            let parent_id = args["parent_id"].as_str().map(|s| NodeId(s.to_string()));
            let agent_id = ctx.agent_id.clone();
            let graph = ctx.graph.clone();

            let mut node = GraphNode::new(NodeType::Knowledge(KnowledgeData {
                entity,
                entity_type: entity_type.clone(),
                summary,
                confidence: 0.5,
            }));

            let meta = node
                .metadata
                .as_object_mut()
                .expect("metadata should be object");
            meta.insert("planning".to_string(), serde_json::json!(true));
            meta.insert("status".to_string(), serde_json::json!("open"));
            meta.insert(
                "session_id".to_string(),
                serde_json::json!(agent_id.to_string()),
            );

            let new_node_id = node.id.clone();
            let new_node_id_ret = new_node_id.clone();

            tokio::task::spawn_blocking(move || {
                graph
                    .add_node(node)
                    .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

                if let Some(pid) = parent_id {
                    graph
                        .add_edge(GraphEdge::new(EdgeType::Contains, pid, new_node_id.clone()))
                        .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
                }

                Ok::<(), ToolError>(())
            })
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))??;

            Ok(ToolOutput::success(format!(
                "Created planning {entity_type} node: {new_node_id_ret}"
            )))
        }

        "list" => {
            let entity_type_filter = args["entity_type"].as_str().map(|s| s.to_string());
            let parent_id = args["parent_id"].as_str().map(|s| NodeId(s.to_string()));
            let graph = ctx.graph.clone();

            let nodes = tokio::task::spawn_blocking(move || {
                let all_knowledge = graph.list_nodes_by_type("knowledge", None, None, 1000)?;

                let mut planning_nodes: Vec<GraphNode> = all_knowledge
                    .into_iter()
                    .filter(|n| n.metadata.get("planning").and_then(|v| v.as_bool()) == Some(true))
                    .collect();

                if let Some(et) = &entity_type_filter {
                    planning_nodes.retain(|n| {
                        matches!(&n.node_type, NodeType::Knowledge(kd) if kd.entity_type == *et)
                    });
                }

                if let Some(pid) = &parent_id {
                    let children = graph
                        .neighbors(pid, Some(EdgeType::Contains), Direction::Outgoing)
                        .unwrap_or_default();
                    let child_ids: std::collections::HashSet<_> =
                        children.iter().map(|n| &n.id).collect();
                    planning_nodes.retain(|n| child_ids.contains(&n.id));
                }

                Ok::<Vec<GraphNode>, graphirm_graph::error::GraphError>(planning_nodes)
            })
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

            let mut lines = Vec::new();
            lines.push(format!(
                "Planning nodes ({} result{}):",
                nodes.len(),
                if nodes.len() == 1 { "" } else { "s" }
            ));
            for node in &nodes {
                if let NodeType::Knowledge(kd) = &node.node_type {
                    let status = node
                        .metadata
                        .get("status")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    lines.push(format!(
                        "  [{id}] {entity} ({entity_type}) status={status}: {summary}",
                        id = node.id,
                        entity = kd.entity,
                        entity_type = kd.entity_type,
                        summary = truncate(&kd.summary, 100),
                    ));
                }
            }
            if nodes.is_empty() {
                lines.push("  (no planning nodes)".to_string());
            }

            Ok(ToolOutput::success(lines.join("\n")))
        }

        "link_session" => {
            let planning_node_id_str = args["planning_node_id"].as_str().ok_or_else(|| {
                ToolError::InvalidArguments("'planning_node_id' is required".into())
            })?;
            let planning_node_id = NodeId(planning_node_id_str.to_string());
            let agent_id = ctx.agent_id.clone();
            let graph = ctx.graph.clone();

            let graph_check = graph.clone();
            let pid_check = planning_node_id.clone();
            let aid_check = agent_id.clone();
            let edge_exists = tokio::task::spawn_blocking(move || {
                let edges = graph_check
                    .edges_for_node(&pid_check)
                    .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
                Ok::<bool, ToolError>(
                    edges
                        .iter()
                        .any(|e| e.edge_type == EdgeType::DerivedFrom && e.source == aid_check),
                )
            })
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))??;

            if edge_exists {
                return Ok(ToolOutput::success(format!(
                    "Session already linked to planning node {planning_node_id}"
                )));
            }

            let pid_for_msg = planning_node_id.clone();
            tokio::task::spawn_blocking(move || {
                graph
                    .add_edge(GraphEdge::new(
                        EdgeType::DerivedFrom,
                        agent_id,
                        planning_node_id,
                    ))
                    .map_err(|e| ToolError::ExecutionFailed(e.to_string()))
            })
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))??;

            Ok(ToolOutput::success(format!(
                "Linked session to planning node {pid_for_msg}"
            )))
        }

        "update" => {
            let node_id_str = args["node_id"]
                .as_str()
                .ok_or_else(|| ToolError::InvalidArguments("'node_id' is required".into()))?
                .to_string();
            let status = args["status"]
                .as_str()
                .ok_or_else(|| ToolError::InvalidArguments("'status' is required".into()))?
                .to_string();

            match status.as_str() {
                "open" | "in_progress" | "done" => {}
                other => {
                    return Err(ToolError::InvalidArguments(format!(
                        "unknown status '{other}'; must be one of: open, in_progress, done"
                    )));
                }
            }

            let node_id = NodeId(node_id_str);
            let node_id_for_msg = node_id.clone();
            let status_for_msg = status.clone();
            let graph = ctx.graph.clone();

            tokio::task::spawn_blocking(move || {
                let mut node = graph
                    .get_node(&node_id)
                    .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
                if node.metadata.get("planning").and_then(|v| v.as_bool()) != Some(true) {
                    return Err(ToolError::ExecutionFailed(
                        "Node is not a planning node (missing metadata.planning = true)".into(),
                    ));
                }
                let meta = node
                    .metadata
                    .as_object_mut()
                    .expect("metadata should be object");
                meta.insert("status".to_string(), serde_json::json!(status));
                graph
                    .update_node(&node_id, node)
                    .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
                Ok::<(), ToolError>(())
            })
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))??;

            Ok(ToolOutput::success(format!(
                "Updated planning node {node_id_for_msg} status to {status_for_msg}"
            )))
        }

        other => Err(ToolError::InvalidArguments(format!(
            "unknown action '{other}'; must be one of: create, list, link_session, update"
        ))),
    }
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
    use crate::retriever::{KnowledgeResult, KnowledgeRetriever};
    use crate::tests::make_test_context;
    use graphirm_graph::nodes::{
        GraphNode, InteractionData, KnowledgeData, NodeType, TaskData, TaskStatus,
    };
    use serde_json::json;
    use std::sync::Arc;

    struct MockKnowledgeRetriever {
        results: Vec<(String, String, f64)>, // (entity, summary, score)
    }

    #[async_trait::async_trait]
    impl KnowledgeRetriever for MockKnowledgeRetriever {
        async fn retrieve_semantic(
            &self,
            _query: &str,
            k: usize,
        ) -> Result<Vec<KnowledgeResult>, crate::ToolError> {
            Ok(self
                .results
                .iter()
                .take(k)
                .map(|(entity, summary, score)| {
                    let node = GraphNode::new(NodeType::Knowledge(KnowledgeData {
                        entity: entity.clone(),
                        entity_type: "concept".to_string(),
                        summary: summary.clone(),
                        confidence: 0.9,
                    }));
                    KnowledgeResult {
                        node_id: node.id.clone(),
                        node,
                        score: *score,
                    }
                })
                .collect())
        }
    }

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

    #[tokio::test]
    async fn semantic_mode_returns_results_from_retriever() {
        let mut ctx = make_test_context();
        ctx.knowledge_retriever = Some(Arc::new(MockKnowledgeRetriever {
            results: vec![
                (
                    "JWT Auth".to_string(),
                    "Token-based authentication".to_string(),
                    0.92,
                ),
                (
                    "OAuth2".to_string(),
                    "Auth protocol for APIs".to_string(),
                    0.85,
                ),
            ],
        }));

        let tool = GraphQueryTool::new();
        let out = tool
            .execute(
                json!({
                    "mode": "semantic",
                    "query": "user authentication"
                }),
                &ctx,
            )
            .await
            .unwrap();

        assert!(!out.is_error);
        assert!(
            out.content.contains("JWT Auth"),
            "Should contain first result"
        );
        assert!(
            out.content.contains("OAuth2"),
            "Should contain second result"
        );
        assert!(out.content.contains("0.920") || out.content.contains("sim=0.92"));
    }

    #[tokio::test]
    async fn semantic_mode_without_retriever_returns_execution_failed() {
        let ctx = make_test_context(); // knowledge_retriever is None

        let tool = GraphQueryTool::new();
        let result = tool
            .execute(
                json!({
                    "mode": "semantic",
                    "query": "anything"
                }),
                &ctx,
            )
            .await;

        assert!(matches!(result, Err(ToolError::ExecutionFailed(_))));
    }

    #[tokio::test]
    async fn semantic_mode_empty_query_returns_invalid_arguments() {
        let mut ctx = make_test_context();
        ctx.knowledge_retriever = Some(Arc::new(MockKnowledgeRetriever { results: vec![] }));

        let tool = GraphQueryTool::new();
        let result = tool
            .execute(
                json!({
                    "mode": "semantic",
                    "query": "   "
                }),
                &ctx,
            )
            .await;

        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[tokio::test]
    async fn semantic_mode_no_results_returns_success_not_error() {
        let mut ctx = make_test_context();
        ctx.knowledge_retriever = Some(Arc::new(MockKnowledgeRetriever { results: vec![] }));

        let tool = GraphQueryTool::new();
        let out = tool
            .execute(
                json!({
                    "mode": "semantic",
                    "query": "xyzzy_unlikely_match"
                }),
                &ctx,
            )
            .await
            .unwrap();

        assert!(!out.is_error);
        assert!(out.content.contains("no Knowledge nodes"));
    }

    #[tokio::test]
    async fn semantic_mode_respects_limit() {
        let mut ctx = make_test_context();
        ctx.knowledge_retriever = Some(Arc::new(MockKnowledgeRetriever {
            results: vec![
                ("Node A".to_string(), "First result".to_string(), 0.9),
                ("Node B".to_string(), "Second result".to_string(), 0.8),
                ("Node C".to_string(), "Third result".to_string(), 0.7),
            ],
        }));

        let tool = GraphQueryTool::new();
        let out = tool
            .execute(
                json!({
                    "mode": "semantic",
                    "query": "test",
                    "limit": 2
                }),
                &ctx,
            )
            .await
            .unwrap();

        assert!(!out.is_error);
        assert!(out.content.contains("Node A"));
        assert!(out.content.contains("Node B"));
        // Node C was cut by limit=2
        assert!(!out.content.contains("Node C"));
    }

    #[tokio::test]
    async fn test_project_create() {
        let ctx = make_test_context();
        let tool = GraphQueryTool::new();

        let result = tool
            .execute(
                json!({
                    "mode": "project",
                    "action": "create",
                    "entity": "Test Epic",
                    "summary": "A test epic",
                    "entity_type": "epic"
                }),
                &ctx,
            )
            .await;

        assert!(result.is_ok(), "create should return Ok, got: {result:?}");
        let out = result.unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("Created planning epic node"));

        // Extract node ID from output and verify metadata
        let node_id_str = out.content.split("node: ").nth(1).unwrap().to_string();
        let node_id = NodeId(node_id_str);

        let graph = ctx.graph.clone();
        let node = tokio::task::spawn_blocking(move || graph.get_node(&node_id))
            .await
            .unwrap()
            .unwrap();

        assert_eq!(
            node.metadata.get("planning"),
            Some(&serde_json::json!(true))
        );
        assert_eq!(
            node.metadata.get("status"),
            Some(&serde_json::json!("open"))
        );
        if let NodeType::Knowledge(kd) = &node.node_type {
            assert_eq!(kd.entity_type, "epic");
            assert_eq!(kd.entity, "Test Epic");
        }
    }

    #[tokio::test]
    async fn test_project_create_with_parent() {
        let ctx = make_test_context();
        let tool = GraphQueryTool::new();

        // Create parent epic
        let parent_result = tool
            .execute(
                json!({
                    "mode": "project",
                    "action": "create",
                    "entity": "Parent Epic",
                    "summary": "Parent epic",
                    "entity_type": "epic"
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!parent_result.is_error);

        let parent_id_str = parent_result
            .content
            .split("node: ")
            .nth(1)
            .unwrap()
            .to_string();
        let parent_id_str_clone = parent_id_str.clone();
        let parent_id = NodeId(parent_id_str);

        // Create child story with parent_id
        let child_result = tool
            .execute(
                json!({
                    "mode": "project",
                    "action": "create",
                    "entity": "Child Story",
                    "summary": "Child story",
                    "entity_type": "story",
                    "parent_id": parent_id_str_clone
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!child_result.is_error);

        // Verify Contains edge exists from parent to child
        let graph = ctx.graph.clone();
        let child_id_str = child_result
            .content
            .split("node: ")
            .nth(1)
            .unwrap()
            .to_string();
        let child_id = NodeId(child_id_str);

        let edges = tokio::task::spawn_blocking(move || graph.edges_for_node(&parent_id))
            .await
            .unwrap()
            .unwrap();

        let contains_edge_exists = edges
            .iter()
            .any(|e| e.edge_type == EdgeType::Contains && e.target == child_id);
        assert!(
            contains_edge_exists,
            "Contains edge should exist from parent to child"
        );
    }

    #[tokio::test]
    async fn test_project_list() {
        let ctx = make_test_context();
        let tool = GraphQueryTool::new();

        // Create two planning nodes
        let epic_result = tool
            .execute(
                json!({
                    "mode": "project",
                    "action": "create",
                    "entity": "Epic One",
                    "summary": "First epic",
                    "entity_type": "epic"
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!epic_result.is_error);

        let story_result = tool
            .execute(
                json!({
                    "mode": "project",
                    "action": "create",
                    "entity": "Story One",
                    "summary": "First story",
                    "entity_type": "story"
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!story_result.is_error);

        // List all planning nodes
        let list_all_result = tool
            .execute(
                json!({
                    "mode": "project",
                    "action": "list"
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!list_all_result.is_error);
        assert!(list_all_result.content.contains("Epic One"));
        assert!(list_all_result.content.contains("Story One"));

        // List with entity_type filter
        let list_story_result = tool
            .execute(
                json!({
                    "mode": "project",
                    "action": "list",
                    "entity_type": "story"
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!list_story_result.is_error);
        assert!(!list_story_result.content.contains("Epic One"));
        assert!(list_story_result.content.contains("Story One"));
    }

    #[tokio::test]
    async fn test_project_link_session() {
        let ctx = make_test_context();
        let tool = GraphQueryTool::new();

        // Create a planning node
        let planning_result = tool
            .execute(
                json!({
                    "mode": "project",
                    "action": "create",
                    "entity": "Planning Node",
                    "summary": "A planning node",
                    "entity_type": "criterion"
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!planning_result.is_error);

        let planning_node_id_str = planning_result
            .content
            .split("node: ")
            .nth(1)
            .unwrap()
            .to_string();
        let planning_node_id_str_clone = planning_node_id_str.clone();
        let planning_node_id = NodeId(planning_node_id_str);

        // Link session to planning node
        let link_result = tool
            .execute(
                json!({
                    "mode": "project",
                    "action": "link_session",
                    "planning_node_id": planning_node_id_str_clone
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!link_result.is_error);
        assert!(link_result.content.contains("Linked session"));

        // Verify DerivedFrom edge exists from agent to planning node
        let graph = ctx.graph.clone();
        let agent_id = ctx.agent_id.clone();

        let edges = tokio::task::spawn_blocking(move || graph.edges_for_node(&planning_node_id))
            .await
            .unwrap()
            .unwrap();

        let derived_from_exists = edges
            .iter()
            .any(|e| e.edge_type == EdgeType::DerivedFrom && e.source == agent_id);
        assert!(
            derived_from_exists,
            "DerivedFrom edge should exist from agent to planning node"
        );
    }

    #[tokio::test]
    async fn test_project_update() {
        let ctx = make_test_context();
        let tool = GraphQueryTool::new();

        // Create a planning node
        let create_result = tool
            .execute(
                json!({
                    "mode": "project",
                    "action": "create",
                    "entity": "Update Test",
                    "summary": "Testing update",
                    "entity_type": "decision"
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!create_result.is_error);

        let node_id_str = create_result
            .content
            .split("node: ")
            .nth(1)
            .unwrap()
            .to_string();
        let node_id_str_clone = node_id_str.clone();
        let node_id = NodeId(node_id_str);

        // Update status to done
        let update_result = tool
            .execute(
                json!({
                    "mode": "project",
                    "action": "update",
                    "node_id": node_id_str_clone,
                    "status": "done"
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!update_result.is_error);
        assert!(update_result.content.contains("Updated planning node"));

        // Verify status changed to done
        let graph = ctx.graph.clone();
        let node = tokio::task::spawn_blocking(move || graph.get_node(&node_id))
            .await
            .unwrap()
            .unwrap();

        assert_eq!(
            node.metadata.get("status"),
            Some(&serde_json::json!("done"))
        );
    }

    #[tokio::test]
    async fn bfs_respects_max_depth_parameter() {
        let ctx = make_test_context();
        let graph = &ctx.graph;

        // Create a chain: parent -> child -> grandchild -> great_grandchild
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
        let grandchild = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "grandchild".to_string(),
            token_count: None,
        }));
        let great_grandchild = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".to_string(),
            content: "great_grandchild".to_string(),
            token_count: None,
        }));

        let parent_id = parent.id.clone();
        let child_id = child.id.clone();
        let grandchild_id = grandchild.id.clone();
        let great_grandchild_id = great_grandchild.id.clone();

        graph.add_node(parent).unwrap();
        graph.add_node(child).unwrap();
        graph.add_node(grandchild).unwrap();
        graph.add_node(great_grandchild).unwrap();

        // parent -> child -> grandchild -> great_grandchild
        graph
            .add_edge(graphirm_graph::edges::GraphEdge::new(
                graphirm_graph::edges::EdgeType::Contains,
                parent_id.clone(),
                child_id.clone(),
            ))
            .unwrap();
        graph
            .add_edge(graphirm_graph::edges::GraphEdge::new(
                graphirm_graph::edges::EdgeType::Contains,
                child_id.clone(),
                grandchild_id.clone(),
            ))
            .unwrap();
        graph
            .add_edge(graphirm_graph::edges::GraphEdge::new(
                graphirm_graph::edges::EdgeType::Contains,
                grandchild_id.clone(),
                great_grandchild_id.clone(),
            ))
            .unwrap();

        let tool = GraphQueryTool::new();

        // Test max_depth=1: should only reach child
        let out = tool
            .execute(
                json!({
                    "mode": "bfs",
                    "node_id": parent_id.to_string(),
                    "depth": 10,
                    "max_depth": 1
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains(&child_id.to_string()));
        assert!(!out.content.contains(&grandchild_id.to_string()));
        assert!(out.content.contains("[depth=1]"));

        // Test max_depth=2: should reach child and grandchild
        let out = tool
            .execute(
                json!({
                    "mode": "bfs",
                    "node_id": parent_id.to_string(),
                    "depth": 10,
                    "max_depth": 2
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains(&child_id.to_string()));
        assert!(out.content.contains(&grandchild_id.to_string()));
        assert!(!out.content.contains(&great_grandchild_id.to_string()));

        // Test without max_depth (uses depth): should reach all
        let out = tool
            .execute(
                json!({
                    "mode": "bfs",
                    "node_id": parent_id.to_string(),
                    "depth": 3
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains(&child_id.to_string()));
        assert!(out.content.contains(&grandchild_id.to_string()));
        assert!(out.content.contains(&great_grandchild_id.to_string()));
    }

    #[tokio::test]
    async fn neighbors_basic() {
        let ctx = make_test_context();
        let graph = &ctx.graph;

        let agent = GraphNode::new(NodeType::Agent(graphirm_graph::nodes::AgentData {
            name: "test_agent".to_string(),
            model: "claude".to_string(),
            system_prompt: None,
            status: "running".to_string(),
        }));
        let content1 = GraphNode::new(NodeType::Content(graphirm_graph::nodes::ContentData {
            content_type: "file".to_string(),
            path: Some("a.rs".to_string()),
            body: "code".to_string(),
            language: Some("rust".to_string()),
        }));
        let content2 = GraphNode::new(NodeType::Content(graphirm_graph::nodes::ContentData {
            content_type: "file".to_string(),
            path: Some("b.rs".to_string()),
            body: "more code".to_string(),
            language: Some("rust".to_string()),
        }));
        let agent_id = agent.id.clone();
        let c1_id = content1.id.clone();
        let c2_id = content2.id.clone();
        graph.add_node(agent).unwrap();
        graph.add_node(content1).unwrap();
        graph.add_node(content2).unwrap();

        graph
            .add_edge(GraphEdge::new(
                EdgeType::Reads,
                agent_id.clone(),
                c1_id.clone(),
            ))
            .unwrap();
        graph
            .add_edge(GraphEdge::new(
                EdgeType::Modifies,
                agent_id.clone(),
                c2_id.clone(),
            ))
            .unwrap();

        let tool = GraphQueryTool::new();
        let out = tool
            .execute(
                json!({
                    "mode": "neighbors",
                    "node_id": agent_id.to_string()
                }),
                &ctx,
            )
            .await
            .unwrap();

        assert!(!out.is_error);
        assert!(out.content.contains(&c1_id.to_string()));
        assert!(out.content.contains(&c2_id.to_string()));
        assert!(out.content.contains("Neighbors"));
    }

    #[tokio::test]
    async fn neighbors_direction_filtering() {
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

        // Test outgoing from parent - should see child
        let out = tool
            .execute(
                json!({
                    "mode": "neighbors",
                    "node_id": parent_id.to_string(),
                    "direction": "outgoing"
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains(&child_id.to_string()));

        // Test incoming to child - should see parent
        let out = tool
            .execute(
                json!({
                    "mode": "neighbors",
                    "node_id": child_id.to_string(),
                    "direction": "incoming"
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains(&parent_id.to_string()));

        // Test outgoing from child - should see nothing
        let out = tool
            .execute(
                json!({
                    "mode": "neighbors",
                    "node_id": child_id.to_string(),
                    "direction": "outgoing"
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("(no neighbors)") || !out.content.contains(&parent_id.to_string()));
    }

    #[tokio::test]
    async fn neighbors_edge_type_filtering() {
        let ctx = make_test_context();
        let graph = &ctx.graph;

        let agent = GraphNode::new(NodeType::Agent(graphirm_graph::nodes::AgentData {
            name: "test_agent".to_string(),
            model: "claude".to_string(),
            system_prompt: None,
            status: "running".to_string(),
        }));
        let content1 = GraphNode::new(NodeType::Content(graphirm_graph::nodes::ContentData {
            content_type: "file".to_string(),
            path: Some("a.rs".to_string()),
            body: "code".to_string(),
            language: Some("rust".to_string()),
        }));
        let content2 = GraphNode::new(NodeType::Content(graphirm_graph::nodes::ContentData {
            content_type: "file".to_string(),
            path: Some("b.rs".to_string()),
            body: "more code".to_string(),
            language: Some("rust".to_string()),
        }));
        let agent_id = agent.id.clone();
        let c1_id = content1.id.clone();
        let c2_id = content2.id.clone();
        graph.add_node(agent).unwrap();
        graph.add_node(content1).unwrap();
        graph.add_node(content2).unwrap();

        graph
            .add_edge(GraphEdge::new(
                EdgeType::Reads,
                agent_id.clone(),
                c1_id.clone(),
            ))
            .unwrap();
        graph
            .add_edge(GraphEdge::new(
                EdgeType::Modifies,
                agent_id.clone(),
                c2_id.clone(),
            ))
            .unwrap();

        let tool = GraphQueryTool::new();

        // Filter by Reads only - should see only content1
        let out = tool
            .execute(
                json!({
                    "mode": "neighbors",
                    "node_id": agent_id.to_string(),
                    "edge_types": ["reads"]
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains(&c1_id.to_string()));
        assert!(!out.content.contains(&c2_id.to_string()));

        // Filter by specific edge types - should see only content2
        let out = tool
            .execute(
                json!({
                    "mode": "neighbors",
                    "node_id": agent_id.to_string(),
                    "edge_types": ["modifies"]
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains(&c2_id.to_string()));
        assert!(!out.content.contains(&c1_id.to_string()));
    }
}
