//! Session restoration from the graph store.
//!
//! When the server starts, we query the graph for all Agent nodes and reconstruct
//! the sessions registry so users can list and resume previous sessions.

use std::collections::HashMap;
use std::sync::Arc;

use graphirm_agent::SessionStatus;
use graphirm_agent::SessionMetadata;
use graphirm_graph::{GraphError, GraphStore};

/// Query GraphStore for all Agent nodes and reconstruct session metadata.
/// Returns a HashMap of restored sessions indexed by session ID.
///
/// This function is called during server startup to restore previous sessions
/// from the persistent SQLite graph database.
pub async fn restore_sessions_from_graph(
    graph: &Arc<GraphStore>,
) -> Result<HashMap<String, SessionMetadata>, GraphError> {
    let agent_nodes = graph.get_agent_nodes()?;

    let mut sessions = HashMap::new();

    for (node, agent_data) in agent_nodes {
        // Map agent status string to SessionStatus enum
        let status = match agent_data.status.as_str() {
            "active" => SessionStatus::Running,
            "idle" => SessionStatus::Idle,
            "completed" => SessionStatus::Completed,
            "failed" => SessionStatus::Failed,
            _ => SessionStatus::Running, // Default to Running for unknown statuses
        };

        let metadata = SessionMetadata::from_agent_node_id(
            node.id.0.clone(),
            agent_data.name,
            agent_data.model,
            node.created_at,
            status,
        );

        sessions.insert(node.id.0.clone(), metadata);
    }

    Ok(sessions)
}
