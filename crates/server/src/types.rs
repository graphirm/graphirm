//! API request/response types for the Graphirm HTTP server.
//!
//! All types derive `Serialize` + `Deserialize` for JSON transport.
//! Request types derive `Deserialize` only where serialization is not needed.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use graphirm_graph::{GraphEdge, GraphNode, TaskStatus};

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
    /// Stopped because [`graphirm_agent::AgentConfig::max_session_tokens`] was exceeded.
    TokenCapExceeded,
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
            SessionStatus::TokenCapExceeded => "token_cap_exceeded",
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
    /// Agent is paused awaiting human approval on a tool call or manual hold.
    AwaitingApproval,
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
            Self::AwaitingApproval => "awaiting_approval",
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
    /// When true, skip the HITL approval gate so bash/write/edit run without
    /// human confirmation. Intended for programmatic clients and eval harnesses.
    pub auto_approve: Option<bool>,
    /// When true, enables structured response segmentation for this session.
    /// The agent will request segment-formatted JSON output from the LLM and
    /// persist each segment as a child Content node in the graph.
    #[serde(default)]
    pub enable_segments: Option<bool>,
    /// When set, restricts context window to only the listed segment types.
    /// Only takes effect when the session has segmented assistant responses.
    /// Example: `["reasoning", "code"]`
    #[serde(default)]
    pub segment_filter: Option<Vec<String>>,
    /// Optional workspace name. When omitted, falls back to `agent` (if set) or `"session"`.
    /// The server creates `<workspaces_root>/<workspace>/` if it does not exist.
    #[serde(default)]
    pub workspace: Option<String>,
    /// When true, enables post-hoc markdown outline extraction (`outline_item` nodes).
    #[serde(default)]
    pub enable_outline: Option<bool>,
}

/// Optional scoped context for prompts (e.g. elaborate one outline section).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SteerContext {
    /// Graph node id of an `outline_item` Content node.
    #[serde(default)]
    pub outline_node_id: Option<String>,
    /// Assistant `Interaction` node id that owns the outline.
    #[serde(default)]
    pub interaction_id: Option<String>,
}

/// Request body for `POST /api/sessions/:id/prompt`.
#[derive(Debug, Deserialize)]
pub struct PromptRequest {
    /// The user message content to submit to the agent.
    pub content: String,
    /// Optional graph node id: new user message `RespondsTo` this node (steer / fork from here).
    #[serde(default)]
    pub context_root: Option<String>,
    /// Optional outline / section scope (appended to the message for the model).
    #[serde(default)]
    pub steer_context: Option<SteerContext>,
}

/// Query for `GET /api/sessions/:id/outline`.
#[derive(Debug, Deserialize)]
pub struct OutlineQuery {
    /// Assistant interaction node id whose outline children to list.
    pub interaction_id: String,
}

/// Request body for `PATCH /api/graph/:session_id/node/:node_id`.
#[derive(Debug, Deserialize)]
pub struct PatchGraphNodeRequest {
    /// For Content nodes: replace body text.
    #[serde(default)]
    pub body: Option<String>,
    /// Shallow-merge into existing node metadata (outline_title, hidden, etc.).
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

/// Request body for `POST /api/sessions/:id/outline`.
#[derive(Debug, Deserialize)]
pub struct CreateOutlineItemRequest {
    /// Parent assistant interaction node id.
    pub parent_interaction_id: String,
    /// Display title (stored in metadata.outline_title).
    pub title: String,
    /// Body text under the title.
    #[serde(default)]
    pub body: String,
    /// Outline kind (e.g. epic, phase, misc).
    #[serde(default = "default_outline_kind_misc")]
    pub kind: String,
}

fn default_outline_kind_misc() -> String {
    "misc".to_string()
}

/// Query parameters for `GET /api/sessions/:id/export`.
#[derive(Debug, Deserialize)]
pub struct ExportQuery {
    /// Output format. Currently only `"markdown"` is supported (default: `"markdown"`).
    #[serde(default = "default_export_format")]
    pub format: String,
}

fn default_export_format() -> String {
    "markdown".to_string()
}

/// Query parameters for `GET /api/sessions/:id/graph`.
#[derive(Debug, Deserialize)]
pub struct SubgraphQuery {
    /// Maximum traversal depth from the session root node (default: 3).
    pub depth: Option<usize>,
}

/// Action a human can take on a pending HITL gate.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NodeAction {
    Approve,
    Reject,
    Modify,
}

/// Request body for `POST /api/graph/:session_id/node/:node_id/action`.
#[derive(Debug, Deserialize)]
pub struct NodeActionRequest {
    pub action: NodeAction,
    /// Required for Reject — injected back to the agent as a synthetic tool result.
    pub reason: Option<String>,
    /// Required for Modify — the new tool arguments as JSON.
    pub modified_args: Option<serde_json::Value>,
}

// ── Response types ────────────────────────────────────────────────────────────

/// Response body for session creation and retrieval endpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionResponse {
    /// Session UUID.
    pub id: String,
    /// Human-readable session name, mutable via `PATCH /api/sessions/:id`.
    pub name: String,
    /// Agent profile name (same value as `name` at creation; kept for API compatibility).
    pub agent: String,
    /// LLM model identifier.
    pub model: String,
    /// UTC timestamp when the session was created.
    pub created_at: DateTime<Utc>,
    /// Current lifecycle status.
    pub status: SessionStatus,
    /// Workspace name, if a per-session workspace was configured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace: Option<String>,
    /// Absolute path to the workspace directory (only set when `workspace` is `Some`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace_path: Option<String>,
    /// Cumulative LLM tokens (input + output from completions) used in this session.
    pub tokens_used: u64,
    /// Configured per-session LLM token cap, if any (`None` = unlimited).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_session_tokens: Option<u64>,
}

/// Request body for `PATCH /api/sessions/:id`.
#[derive(Debug, Deserialize)]
pub struct RenameSessionRequest {
    pub name: String,
}

/// Response body for `GET /api/health`.
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Always `"ok"` when the server is healthy.
    pub status: String,
    /// Server binary version from `CARGO_PKG_VERSION`.
    pub version: String,
    /// Number of active sessions currently loaded in memory.
    pub session_count: usize,
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

/// Canvas position hint for annotation nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationPosition {
    pub x: f64,
    pub y: f64,
}

/// `POST /api/sessions/{id}/auto-approve` request body.
#[derive(Debug, Deserialize)]
pub struct AutoApproveRequest {
    pub enabled: bool,
}

/// `POST /api/graph/{session_id}/annotate` request body.
#[derive(Debug, Deserialize)]
pub struct AnnotationRequest {
    pub entity: String,
    pub entity_type: String,
    pub summary: String,
    /// Optional canvas position hint stored in node metadata.
    pub position: Option<AnnotationPosition>,
    /// Optional interaction/tool node id: add `RelatesTo` from the new Knowledge node to this id.
    #[serde(default)]
    pub relates_to: Option<String>,
}

/// `PATCH /api/graph/{session_id}/tasks/{node_id}` — manual completion (popover).
#[derive(Debug, Deserialize)]
pub struct PatchTaskStatusRequest {
    pub status: TaskStatus,
}

/// `PATCH /api/knowledge/{id}` request body.
#[derive(Debug, Deserialize)]
pub struct PatchKnowledgeRequest {
    #[serde(default)]
    pub dismissed: Option<bool>,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub pinned: Option<bool>,
}

/// `PATCH /api/interactions/{id}/edit` — mark original user message as edited (audit).
#[derive(Debug, Deserialize)]
pub struct EditInteractionRequest {
    pub original_content: String,
}

/// `POST /api/knowledge` request body.
#[derive(Debug, Deserialize)]
pub struct CreateKnowledgeRequest {
    pub entity: String,
    pub entity_type: String,
    pub summary: String,
    #[serde(default)]
    pub confidence: Option<f64>,
    #[serde(default)]
    pub pinned: bool,
    #[serde(default)]
    pub session_id: Option<String>,
}

/// Query parameters for `GET /api/knowledge/pinned`.
#[derive(Debug, Deserialize)]
pub struct PinnedKnowledgeQuery {
    /// Maximum number of pinned knowledge nodes to return (default: 50).
    #[serde(default = "default_pinned_limit")]
    pub limit: usize,
}

fn default_pinned_limit() -> usize {
    50
}

/// Request body for `PATCH /api/sessions/:id/turns/:turn_id/rating`.
#[derive(Debug, Deserialize)]
pub struct RateTurnRequest {
    /// User satisfaction rating for this turn (1 = worst, 5 = best).
    pub rating: u8,
}

/// Per-strategy aggregated statistics returned by `GET /api/routing/report`.
#[derive(Debug, Serialize)]
pub struct StrategyReport {
    pub strategy_name: String,
    pub turn_count: u32,
    pub avg_input_tokens: f64,
    pub avg_output_tokens: f64,
    pub avg_latency_ms: f64,
    pub error_rate: f64,
    pub avg_user_rating: Option<f64>,
}

/// Graph context utilisation report returned by `GET /api/sessions/{id}/context-report`.
#[derive(Debug, Serialize)]
pub struct ContextReportRow {
    pub session_id: String,
    pub turns_with_stats: usize,
    pub avg_knowledge_count: f64,
    pub avg_graph_token_pct: f64,
    pub avg_pinned_count: f64,
    pub avg_cross_session_links: f64,
    pub compaction_triggered_count: usize,
    pub briefing_included_count: usize,
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
            session_count: 5,
        };
        let json = serde_json::to_string(&health).unwrap();
        let back: HealthResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.status, "ok");
        assert_eq!(back.version, "0.1.0");
        assert_eq!(back.session_count, 5);
    }

    #[test]
    fn session_response_serde_roundtrip() {
        let now = Utc::now();
        let session = SessionResponse {
            id: "abc-123".to_string(),
            name: "graphirm".to_string(),
            agent: "graphirm".to_string(),
            model: "claude-sonnet-4-20250514".to_string(),
            created_at: now,
            status: SessionStatus::Idle,
            workspace: None,
            workspace_path: None,
            tokens_used: 0,
            max_session_tokens: None,
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
        assert_eq!(
            serde_json::to_value(SessionStatus::TokenCapExceeded).unwrap(),
            serde_json::json!("token_cap_exceeded")
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
    fn create_session_request_enable_segments() {
        let json = r#"{"auto_approve": true, "enable_segments": true}"#;
        let req: CreateSessionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.auto_approve, Some(true));
        assert_eq!(req.enable_segments, Some(true));
    }

    #[test]
    fn test_create_session_request_segment_filter_deserialization() {
        let json = r#"{"segment_filter": ["code", "reasoning"]}"#;
        let req: CreateSessionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(
            req.segment_filter,
            Some(vec!["code".to_string(), "reasoning".to_string()])
        );

        let json_missing = r#"{}"#;
        let req2: CreateSessionRequest = serde_json::from_str(json_missing).unwrap();
        assert!(req2.segment_filter.is_none());
    }

    #[test]
    fn prompt_request_deserialize() {
        let json = r#"{"content": "Hello world"}"#;
        let req: PromptRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.content, "Hello world");
        assert!(req.context_root.is_none());

        let json2 = r#"{"content": "Hi", "context_root": "node-abc"}"#;
        let req2: PromptRequest = serde_json::from_str(json2).unwrap();
        assert_eq!(req2.context_root, Some("node-abc".to_string()));
        assert!(req2.steer_context.is_none());

        let json3 = r#"{"content": "Go", "steer_context": {"outline_node_id": "n1", "interaction_id": "i1"}}"#;
        let req3: PromptRequest = serde_json::from_str(json3).unwrap();
        assert_eq!(
            req3.steer_context
                .as_ref()
                .unwrap()
                .outline_node_id
                .as_deref(),
            Some("n1")
        );
        assert_eq!(
            req3.steer_context
                .as_ref()
                .unwrap()
                .interaction_id
                .as_deref(),
            Some("i1")
        );
    }

    #[test]
    fn patch_knowledge_request_deserialize() {
        let j = r#"{"dismissed": true}"#;
        let r: PatchKnowledgeRequest = serde_json::from_str(j).unwrap();
        assert_eq!(r.dismissed, Some(true));
        assert!(r.summary.is_none());
        assert!(r.pinned.is_none());

        let j2 = r#"{"pinned": true}"#;
        let r2: PatchKnowledgeRequest = serde_json::from_str(j2).unwrap();
        assert_eq!(r2.pinned, Some(true));
        assert!(r2.dismissed.is_none());
    }

    #[test]
    fn patch_task_status_request_deserialize() {
        use graphirm_graph::TaskStatus as TS;
        let j = r#"{"status": "completed"}"#;
        let r: PatchTaskStatusRequest = serde_json::from_str(j).unwrap();
        assert_eq!(r.status, TS::Completed);
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

    #[test]
    fn create_knowledge_request_deserialize() {
        // Full body with all fields
        let json = r#"{"entity": "TestEntity", "entity_type": "test", "summary": "Test summary", "confidence": 0.95, "pinned": true, "session_id": "abc-123"}"#;
        let req: CreateKnowledgeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.entity, "TestEntity");
        assert_eq!(req.entity_type, "test");
        assert_eq!(req.summary, "Test summary");
        assert_eq!(req.confidence, Some(0.95));
        assert!(req.pinned);
        assert_eq!(req.session_id, Some("abc-123".to_string()));

        // Minimal body (only required fields) — verify defaults
        let json = r#"{"entity": "Minimal", "entity_type": "test", "summary": "Minimal summary"}"#;
        let req: CreateKnowledgeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.entity, "Minimal");
        assert_eq!(req.entity_type, "test");
        assert_eq!(req.summary, "Minimal summary");
        assert!(req.confidence.is_none());
        assert!(!req.pinned);
        assert!(req.session_id.is_none());
    }

    #[test]
    fn pinned_knowledge_query_deserialize() {
        // With limit
        let json = r#"{"limit": 10}"#;
        let q: PinnedKnowledgeQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.limit, 10);

        // Without limit (uses default)
        let json = r#"{}"#;
        let q: PinnedKnowledgeQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.limit, 50);
    }

    #[test]
    fn rate_turn_request_deserializes() {
        let json = r#"{"rating": 4}"#;
        let req: RateTurnRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.rating, 4);
    }

    #[test]
    fn strategy_report_serializes() {
        let r = StrategyReport {
            strategy_name: "experiment:prompt_router".into(),
            turn_count: 10,
            avg_input_tokens: 500.0,
            avg_output_tokens: 200.0,
            avg_latency_ms: 1200.0,
            error_rate: 0.1,
            avg_user_rating: Some(4.2),
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("experiment:prompt_router"));
        assert!(json.contains("\"turn_count\":10"));
    }
}
