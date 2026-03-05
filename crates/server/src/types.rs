use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use graphirm_graph::{GraphEdge, GraphNode};

#[derive(Debug, Deserialize)]
pub struct CreateSessionRequest {
    pub agent: Option<String>,
    pub model: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct PromptRequest {
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct SubgraphQuery {
    pub depth: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionResponse {
    pub id: String,
    pub agent: String,
    pub model: String,
    pub created_at: DateTime<Utc>,
    pub status: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphResponse {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SseEvent {
    pub session_id: String,
    pub event_type: String,
    pub data: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
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
            status: "idle".to_string(),
        };
        let json = serde_json::to_string(&session).unwrap();
        let back: SessionResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "abc-123");
        assert_eq!(back.agent, "graphirm");
        assert_eq!(back.model, "claude-sonnet-4-20250514");
        assert_eq!(back.status, "idle");
        // Timestamps roundtrip through ISO-8601 — sub-nanosecond precision may differ
        assert_eq!(
            back.created_at.timestamp(),
            now.timestamp()
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
            session_id: "s1".to_string(),
            event_type: "turn_start".to_string(),
            data: serde_json::json!({"turn_index": 0}),
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: SseEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.session_id, "s1");
        assert_eq!(back.event_type, "turn_start");
        assert_eq!(back.data["turn_index"], 0);
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
