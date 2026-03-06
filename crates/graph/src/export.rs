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

#[cfg(test)]
mod tests {
    use super::*;

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
