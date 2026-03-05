//! API request/response types for the Graphirm HTTP server.
//!
//! All types derive `Serialize` + `Deserialize` for JSON transport.
//! Request types derive `Deserialize` only where serialization is not needed.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use graphirm_graph::{GraphEdge, GraphNode};

// ── Newtypes ──────────────────────────────────────────────────────────────────

/// Opaque identifier for a server-managed session.
///
/// Wraps a UUID string. Using a newtype prevents accidentally passing a
/// `NodeId` or arbitrary string where a `SessionId` is expected.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub String);

impl SessionId {
    /// Create a new random session ID.
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for SessionId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for SessionId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

// ── Status enums ──────────────────────────────────────────────────────────────

/// Lifecycle status of a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    /// Session exists but no agent turn is running.
    Idle,
    /// An agent turn is currently in progress.
    Running,
    /// The agent finished successfully.
    Completed,
    /// The agent loop exited with an error.
    Failed,
    /// The agent loop was cancelled via [`CancellationToken`].
    Cancelled,
}

impl SessionStatus {
    /// Returns the snake_case string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            SessionStatus::Idle => "idle",
            SessionStatus::Running => "running",
            SessionStatus::Completed => "completed",
            SessionStatus::Failed => "failed",
            SessionStatus::Cancelled => "cancelled",
        }
    }
}

impl std::fmt::Display for SessionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Typed SSE event kinds emitted by the agent loop.
///
/// Using an enum rather than a bare `String` prevents typos and makes
/// exhaustive matching possible in SSE consumers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SseEventType {
    /// Agent loop started for a session.
    AgentStart,
    /// Agent loop finished for a session.
    AgentEnd,
    /// A new LLM turn began.
    TurnStart,
    /// An LLM turn completed.
    TurnEnd,
    /// Streaming text generation started.
    MessageStart,
    /// Streaming text delta received.
    MessageDelta,
    /// Streaming text generation ended.
    MessageEnd,
    /// A tool call was dispatched.
    ToolStart,
    /// A tool call returned.
    ToolEnd,
    /// The in-memory graph was updated; clients should refresh their view.
    GraphUpdate,
    /// An error occurred in the agent loop.
    Error,
    /// Server-sent keepalive tick.
    Heartbeat,
}

impl std::fmt::Display for SseEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Mirror the serde `snake_case` representation without allocating.
        let s = match self {
            Self::AgentStart => "agent_start",
            Self::AgentEnd => "agent_end",
            Self::TurnStart => "turn_start",
            Self::TurnEnd => "turn_end",
            Self::MessageStart => "message_start",
            Self::MessageDelta => "message_delta",
            Self::MessageEnd => "message_end",
            Self::ToolStart => "tool_start",
            Self::ToolEnd => "tool_end",
            Self::GraphUpdate => "graph_update",
            Self::Error => "error",
            Self::Heartbeat => "heartbeat",
        };
        f.write_str(s)
    }
}

// ── Request types ─────────────────────────────────────────────────────────────

/// Request body for `POST /api/sessions`.
#[derive(Debug, Deserialize)]
pub struct CreateSessionRequest {
    /// Optional agent profile name (defaults to `default_config.name`).
    pub agent: Option<String>,
    /// Optional model override (e.g. `"claude-opus-4-5"`).
    pub model: Option<String>,
}

/// Request body for `POST /api/sessions/:id/prompt`.
#[derive(Debug, Deserialize)]
pub struct PromptRequest {
    /// The user message content to submit to the agent.
    pub content: String,
}

/// Query parameters for `GET /api/sessions/:id/graph`.
#[derive(Debug, Deserialize)]
pub struct SubgraphQuery {
    /// Maximum traversal depth from the session root node (default: 3).
    pub depth: Option<usize>,
}

// ── Response types ────────────────────────────────────────────────────────────

/// Response body for session creation and retrieval endpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionResponse {
    /// Session UUID.
    pub id: String,
    /// Agent profile name.
    pub agent: String,
    /// LLM model identifier.
    pub model: String,
    /// UTC timestamp when the session was created.
    pub created_at: DateTime<Utc>,
    /// Current lifecycle status.
    pub status: SessionStatus,
}

/// Response body for `GET /api/health`.
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Always `"ok"` when the server is healthy.
    pub status: String,
    /// Server binary version from `CARGO_PKG_VERSION`.
    pub version: String,
}

/// Response body for graph query endpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphResponse {
    /// Nodes in the queried subgraph.
    pub nodes: Vec<GraphNode>,
    /// Edges connecting the nodes.
    pub edges: Vec<GraphEdge>,
}

/// A single SSE event pushed to connected clients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SseEvent {
    /// Session this event belongs to.
    pub session_id: SessionId,
    /// Discriminated event kind.
    pub event_type: SseEventType,
    /// Event-specific payload (varies by `event_type`).
    pub data: serde_json::Value,
}

/// JSON error body returned by all non-2xx responses.
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Human-readable error message.
    pub error: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn health_response_serde_roundtrip() {
        let health = HealthResponse {
            status: "ok".to_string(),
            version: "0.1.0".to_string(),
        };
        let json = serde_json::to_string(&health).unwrap();
        let back: HealthResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.status, "ok");
        assert_eq!(back.version, "0.1.0");
    }

    #[test]
    fn session_response_serde_roundtrip() {
        let now = Utc::now();
        let session = SessionResponse {
            id: "abc-123".to_string(),
            agent: "graphirm".to_string(),
            model: "claude-sonnet-4-20250514".to_string(),
            created_at: now,
            status: SessionStatus::Idle,
        };
        let json = serde_json::to_string(&session).unwrap();
        let back: SessionResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "abc-123");
        assert_eq!(back.agent, "graphirm");
        assert_eq!(back.model, "claude-sonnet-4-20250514");
        assert_eq!(back.status, SessionStatus::Idle);
        // Timestamps roundtrip through ISO-8601 — sub-nanosecond precision may differ
        assert_eq!(back.created_at.timestamp(), now.timestamp());
    }

    #[test]
    fn session_status_serializes_as_snake_case() {
        assert_eq!(
            serde_json::to_value(SessionStatus::Running).unwrap(),
            serde_json::json!("running")
        );
        assert_eq!(
            serde_json::to_value(SessionStatus::Completed).unwrap(),
            serde_json::json!("completed")
        );
        assert_eq!(
            serde_json::to_value(SessionStatus::Cancelled).unwrap(),
            serde_json::json!("cancelled")
        );
    }

    #[test]
    fn create_session_request_deserialize() {
        let json = r#"{"agent": "coder", "model": "gpt-4o"}"#;
        let req: CreateSessionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.agent, Some("coder".to_string()));
        assert_eq!(req.model, Some("gpt-4o".to_string()));
    }

    #[test]
    fn create_session_request_empty() {
        let json = r#"{}"#;
        let req: CreateSessionRequest = serde_json::from_str(json).unwrap();
        assert!(req.agent.is_none());
        assert!(req.model.is_none());
    }

    #[test]
    fn prompt_request_deserialize() {
        let json = r#"{"content": "Hello world"}"#;
        let req: PromptRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.content, "Hello world");
    }

    #[test]
    fn sse_event_serde_roundtrip() {
        let event = SseEvent {
            session_id: SessionId::from("s1"),
            event_type: SseEventType::TurnStart,
            data: serde_json::json!({"turn_index": 0}),
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: SseEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.session_id, SessionId::from("s1"));
        assert!(matches!(back.event_type, SseEventType::TurnStart));
        assert_eq!(back.data["turn_index"], 0);
    }

    #[test]
    fn sse_event_type_serializes_as_snake_case() {
        assert_eq!(
            serde_json::to_value(SseEventType::ToolStart).unwrap(),
            serde_json::json!("tool_start")
        );
        assert_eq!(
            serde_json::to_value(SseEventType::GraphUpdate).unwrap(),
            serde_json::json!("graph_update")
        );
    }

    #[test]
    fn session_id_roundtrip() {
        let id = SessionId::from("abc-123");
        let json = serde_json::to_string(&id).unwrap();
        let back: SessionId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, back);
    }

    #[test]
    fn subgraph_query_with_depth() {
        let json = r#"{"depth": 5}"#;
        let q: SubgraphQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.depth, Some(5));
    }

    #[test]
    fn subgraph_query_without_depth() {
        let json = r#"{}"#;
        let q: SubgraphQuery = serde_json::from_str(json).unwrap();
        assert!(q.depth.is_none());
    }
}
