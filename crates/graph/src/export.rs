//! Agent Trace export types for Graphirm.
//!
//! Agent Trace is an open specification (CC BY 4.0) for linking code changes to AI conversations.
//! This module serializes Graphirm's Interaction nodes into the standard Agent Trace format,
//! enabling external tools to correlate conversations with version control and file operations.
//!
//! The three-level hierarchy is:
//! - **AgentTraceRecord**: Top-level container for a session
//! - **TraceTurn**: Individual conversation turn (user message, assistant response, tool output)
//! - **TraceToolCall**: Tool invocation within an assistant turn

use serde::Serialize;

use crate::nodes::NodeType;
use crate::store::GraphStore;
use crate::error::GraphError;

/// Top-level Agent Trace record for a session.
///
/// Contains the session ID, trace version, and all turns in conversation order.
#[derive(Debug, Clone, Serialize)]
pub struct AgentTraceRecord {
    /// Agent Trace format version (currently "0.1")
    pub version: &'static str,
    /// Unique session identifier (UUID)
    pub session_id: String,
    /// Ordered list of conversation turns
    pub turns: Vec<TraceTurn>,
}

/// A single turn in the conversation.
///
/// A turn is either a user message, assistant response, or tool output.
/// Assistant turns may include tool calls.
#[derive(Debug, Clone, Serialize)]
pub struct TraceTurn {
    /// Unique identifier for this turn (node ID from graph)
    pub id: String,
    /// Role: "user", "assistant", or "tool"
    pub role: String,
    /// The message or output content
    pub content: String,
    /// Tool calls made by the assistant (only populated if role == "assistant")
    pub tool_calls: Vec<TraceToolCall>,
    /// ISO 8601 timestamp when this turn was created
    pub created_at: String,
}

/// A tool invocation within an assistant turn.
///
/// Records what tool was called, its parameters, and the result.
#[derive(Debug, Clone, Serialize)]
pub struct TraceToolCall {
    /// Unique identifier for this tool call (node ID from graph)
    pub id: String,
    /// Name of the tool ("bash", "read", "write", etc.)
    pub name: String,
    /// Result or output from the tool execution
    pub result: String,
}

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

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn agent_trace_record_serializes() {
        // Create a trace record with 2 turns: user message and assistant response with tool call
        let tool_call = TraceToolCall {
            id: "tool-001".to_string(),
            name: "bash".to_string(),
            result: "Hello from bash".to_string(),
        };

        let user_turn = TraceTurn {
            id: "turn-001".to_string(),
            role: "user".to_string(),
            content: "Run a test".to_string(),
            tool_calls: vec![],
            created_at: "2026-03-06T10:00:00Z".to_string(),
        };

        let assistant_turn = TraceTurn {
            id: "turn-002".to_string(),
            role: "assistant".to_string(),
            content: "I ran the bash command.".to_string(),
            tool_calls: vec![tool_call],
            created_at: "2026-03-06T10:00:01Z".to_string(),
        };

        let record = AgentTraceRecord {
            version: "0.1",
            session_id: "session-abc123".to_string(),
            turns: vec![user_turn, assistant_turn],
        };

        // Serialize to JSON
        let json = serde_json::to_string(&record).expect("serialization failed");

        // Assert all key fields are present
        assert!(json.contains("\"version\":\"0.1\""), "version field missing");
        assert!(json.contains("\"session_id\":\"session-abc123\""), "session_id field missing");
        assert!(json.contains("\"role\":\"user\""), "user role missing");
        assert!(json.contains("\"role\":\"assistant\""), "assistant role missing");
        assert!(json.contains("\"name\":\"bash\""), "tool name missing");
        assert!(json.contains("\"tool_calls\""), "tool_calls array missing");

        // Assert turn count by checking the JSON structure
        assert!(json.contains("turn-001"), "first turn missing");
        assert!(json.contains("turn-002"), "second turn missing");
    }

    #[test]
    fn trace_tool_call_with_result() {
        let tool_call = TraceToolCall {
            id: "tool-xyz".to_string(),
            name: "read".to_string(),
            result: "file contents".to_string(),
        };

        // Serialize to serde_json::Value for inspection
        let value = serde_json::to_value(&tool_call).expect("to_value failed");

        // Assert fields are correct
        assert_eq!(value["id"], "tool-xyz", "id mismatch");
        assert_eq!(value["name"], "read", "name mismatch");
        assert_eq!(value["result"], "file contents", "result mismatch");
    }
}
