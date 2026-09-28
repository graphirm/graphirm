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
        let status = match agent_data.status.as_str() {
            "active" => SessionStatus::Running,
            "idle" => SessionStatus::Idle,
            "completed" => SessionStatus::Completed,
            "failed" => SessionStatus::Failed,
            "token_cap_exceeded" => SessionStatus::Completed,
            _ => SessionStatus::Running, // Default to Running for unknown statuses
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

        sessions.insert(node.id.0.clone(), metadata);
    }

    tracing::info!(
        session_count = sessions.len(),
        "Session restoration complete"
    );

    Ok(sessions)
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
