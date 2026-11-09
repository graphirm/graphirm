//! Session restoration from the graph store.
//!
//! When the server starts, we query the graph for all Agent nodes and reconstruct
//! the sessions registry so users can list and resume previous sessions.

use std::collections::HashMap;
use std::sync::Arc;

use graphirm_agent::SessionMetadata;
use graphirm_agent::SessionStatus;
use graphirm_graph::{Direction, EdgeType, GraphError, GraphStore};

/// Query GraphStore for all Agent nodes and reconstruct session metadata.
/// Returns a HashMap of restored sessions indexed by session ID.
///
/// `workspaces_root` is the base directory under which per-session workspace
/// directories live. When `Some`, a restored session's `workspace_path` field
/// is set to `<workspaces_root>/<workspace>`. When `None`, only the workspace
/// name is stored; `workspace_path` stays `None`.
///
/// This function is called during server startup to restore previous sessions
/// from the persistent SQLite graph database.
pub async fn restore_sessions_from_graph(
    graph: &Arc<GraphStore>,
    workspaces_root: Option<&std::path::Path>,
) -> Result<HashMap<String, SessionMetadata>, GraphError> {
    tracing::debug!("Querying graph for Agent nodes to restore sessions");

    let graph_clone = graph.clone();
    let agent_nodes = tokio::task::spawn_blocking(move || director_agents(&graph_clone))
        .await
        .unwrap_or(Err(GraphError::LockPoisoned))?;

    let mut sessions = HashMap::new();

    for (node, agent_data) in agent_nodes {
        // Map agent status string to SessionStatus enum
        // Nothing is running at startup, so no restored session is `Running`:
        // a `Running` handle refuses new prompts.
        let status = match agent_data.status.as_str() {
            "completed" | "token_cap_exceeded" | "limit_reached" => SessionStatus::Completed,
            "failed" | "error" => SessionStatus::Failed,
            _ => SessionStatus::Idle,
        };

        let workspace = node
            .metadata
            .get("workspace")
            .and_then(|v| v.as_str())
            .map(String::from);

        let workspace_path = workspace
            .as_ref()
            .zip(workspaces_root)
            .map(|(ws, root)| root.join(ws));

        let mut metadata = SessionMetadata::from_agent_node_id(
            node.id.0.clone(),
            agent_data.name,
            agent_data.model,
            node.created_at,
            status,
        );
        metadata.workspace = workspace;
        metadata.workspace_path = workspace_path;
        metadata.reply_pieces = node.metadata.get("reply_pieces").and_then(|v| v.as_bool());

        sessions.insert(node.id.0.clone(), metadata);
    }

    tracing::info!(
        session_count = sessions.len(),
        "Session restoration complete"
    );

    Ok(sessions)
}

/// Graph status for a session whose run ended without the agent loop recording why.
pub const INTERRUPTED_STATUS: &str = "interrupted";

/// Mark every user session still `active` in the graph as `interrupted`, with a
/// `marked_interrupted_at` timestamp (when it was found, not when it stopped).
/// Called at startup (a previous process was killed mid-run) and after shutdown
/// (runs still in flight are abandoned). Returns how many sessions were marked.
///
/// Limit: every process sharing the database is treated as dead, so a `serve`
/// started next to a live `graphirm chat` on the same `graph.db` marks the chat's
/// running session too.
pub async fn mark_interrupted_sessions(graph: &Arc<GraphStore>) -> Result<usize, GraphError> {
    let graph = graph.clone();
    tokio::task::spawn_blocking(move || {
        let mut marked = 0;
        for (mut node, data) in director_agents(&graph)? {
            if data.status != "active" {
                continue;
            }
            if let graphirm_graph::nodes::NodeType::Agent(ref mut agent) = node.node_type {
                agent.status = INTERRUPTED_STATUS.to_string();
            }
            node.metadata["marked_interrupted_at"] =
                serde_json::json!(chrono::Utc::now().to_rfc3339());
            let id = node.id.clone();
            graph.update_node(&id, node)?;
            marked += 1;
        }
        Ok(marked)
    })
    .await
    .unwrap_or(Err(GraphError::LockPoisoned))
}

/// Agent nodes that are user sessions. A Pi or in-process subagent is
/// `Task --SpawnedBy--> Agent` and must not be restored as a promptable session
/// (its status `"running"` would otherwise look like a live director).
fn director_agents(
    graph: &GraphStore,
) -> Result<Vec<(graphirm_graph::GraphNode, graphirm_graph::nodes::AgentData)>, GraphError> {
    let nodes = graph.get_agent_nodes()?;
    let mut directors = Vec::with_capacity(nodes.len());
    for pair in nodes {
        let spawned_by =
            graph.neighbors(&pair.0.id, Some(EdgeType::SpawnedBy), Direction::Incoming)?;
        if spawned_by.is_empty() {
            directors.push(pair);
        } else {
            tracing::debug!(
                agent_id = %pair.0.id,
                name = %pair.1.name,
                "skipping subagent Agent during session restore"
            );
        }
    }
    Ok(directors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphirm_graph::GraphStore;
    use graphirm_graph::nodes::{AgentData, GraphNode, NodeType};
    use std::path::PathBuf;
    use std::sync::Arc;

    #[tokio::test]
    async fn restore_reads_workspace_from_metadata() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());

        let mut agent_node = GraphNode::new(NodeType::Agent(AgentData {
            name: "test".to_string(),
            model: "deepseek-chat".to_string(),
            system_prompt: None,
            status: "idle".to_string(),
        }));
        agent_node.metadata["workspace"] = serde_json::json!("myapp");
        graph.add_node(agent_node).unwrap();

        let root = PathBuf::from("/workspaces");
        let sessions = restore_sessions_from_graph(&graph, Some(root.as_path()))
            .await
            .unwrap();

        let meta = sessions.values().next().unwrap();
        assert_eq!(meta.workspace, Some("myapp".to_string()));
        assert_eq!(
            meta.workspace_path,
            Some(PathBuf::from("/workspaces/myapp"))
        );
    }

    fn add_agent(graph: &GraphStore, status: &str) -> graphirm_graph::NodeId {
        graph
            .add_node(GraphNode::new(NodeType::Agent(AgentData {
                name: "test".to_string(),
                model: "deepseek-chat".to_string(),
                system_prompt: None,
                status: status.to_string(),
            })))
            .unwrap()
    }

    #[tokio::test]
    async fn active_sessions_are_marked_interrupted_and_restore_idle() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let active = add_agent(&graph, "active");
        let done = add_agent(&graph, "completed");

        assert_eq!(mark_interrupted_sessions(&graph).await.unwrap(), 1);

        let node = graph.get_node(&active).unwrap();
        let NodeType::Agent(ref data) = node.node_type else {
            panic!("agent node");
        };
        assert_eq!(data.status, INTERRUPTED_STATUS);
        assert!(node.metadata.get("marked_interrupted_at").is_some());
        let sessions = restore_sessions_from_graph(&graph, None).await.unwrap();
        assert_eq!(sessions[&active.0].status, SessionStatus::Idle);
        assert_eq!(sessions[&done.0].status, SessionStatus::Completed);
        assert_eq!(mark_interrupted_sessions(&graph).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn restore_never_reports_a_session_as_running() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        for status in ["active", "cancelled", "error", "limit_reached", "unknown"] {
            add_agent(&graph, status);
        }
        let sessions = restore_sessions_from_graph(&graph, None).await.unwrap();
        assert_eq!(sessions.len(), 5);
        assert!(
            sessions
                .values()
                .all(|m| m.status != SessionStatus::Running)
        );
    }

    #[tokio::test]
    async fn restore_workspace_none_when_no_root() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());

        let mut agent_node = GraphNode::new(NodeType::Agent(AgentData {
            name: "test".to_string(),
            model: "deepseek-chat".to_string(),
            system_prompt: None,
            status: "idle".to_string(),
        }));
        agent_node.metadata["workspace"] = serde_json::json!("myapp");
        graph.add_node(agent_node).unwrap();

        let sessions = restore_sessions_from_graph(&graph, None).await.unwrap();

        let meta = sessions.values().next().unwrap();
        assert_eq!(meta.workspace, Some("myapp".to_string()));
        assert!(meta.workspace_path.is_none());
    }

    #[tokio::test]
    async fn restore_skips_agents_spawned_by_a_task() {
        use graphirm_graph::nodes::TaskData;
        use graphirm_graph::{GraphEdge, TaskStatus};

        let graph = Arc::new(GraphStore::open_memory().unwrap());

        let director = GraphNode::new(NodeType::Agent(AgentData {
            name: "graphirm".to_string(),
            model: "m".to_string(),
            system_prompt: None,
            status: "idle".to_string(),
        }));
        let director_id = director.id.0.clone();
        graph.add_node(director).unwrap();

        let task = GraphNode::new(NodeType::Task(TaskData {
            title: "Delegated to pi".to_string(),
            description: "x".to_string(),
            status: TaskStatus::Completed,
            priority: None,
        }));
        let task_id = task.id.clone();
        graph.add_node(task).unwrap();

        let mut pi = GraphNode::new(NodeType::Agent(AgentData {
            name: "pi".to_string(),
            model: "deepseek/deepseek-v4-flash".to_string(),
            system_prompt: None,
            status: "running".to_string(),
        }));
        pi.metadata["executor"] = serde_json::json!("pi");
        let pi_id = pi.id.clone();
        graph.add_node(pi).unwrap();
        graph
            .add_edge(GraphEdge::new(EdgeType::SpawnedBy, task_id, pi_id))
            .unwrap();

        let sessions = restore_sessions_from_graph(&graph, None).await.unwrap();
        assert_eq!(sessions.len(), 1);
        assert!(sessions.contains_key(&director_id));
    }
}
