//! Integration tests for the Graphirm HTTP server.
//!
//! Tests the full request/response lifecycle using an in-memory graph and the
//! built-in `MockProvider` as the LLM backend, so no network calls are made.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tokio::sync::{RwLock, broadcast};
use tower::ServiceExt;

use graphirm_agent::AgentConfig;
use graphirm_graph::GraphStore;
use graphirm_llm::MockProvider;
use graphirm_server::types::{
    GraphResponse, HealthResponse, SessionResponse, SessionStatus, SseEvent,
};
use graphirm_server::{AppState, create_router};
use graphirm_tools::ToolRegistry;

fn test_app_state() -> AppState {
    let graph = Arc::new(GraphStore::open_memory().unwrap());
    let (event_tx, _) = broadcast::channel::<SseEvent>(256);

    AppState {
        graph,
        llm: Arc::new(MockProvider::fixed("I processed your request.")),
        tools: Arc::new(ToolRegistry::new()),
        event_tx,
        sessions: Arc::new(RwLock::new(HashMap::new())),
        default_config: AgentConfig::default(),
        memory_retriever: None,
    }
}

#[tokio::test]
async fn test_full_lifecycle() {
    let state = test_app_state();
    let event_tx = state.event_tx.clone();
    let mut event_rx = event_tx.subscribe();
    let app = create_router(state.clone());

    // 1. Health check
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let health: HealthResponse = serde_json::from_slice(&body).unwrap();
    assert_eq!(health.status, "ok");

    // 2. Create session
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/sessions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"agent": "test-agent", "model": "mock-model"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let session: SessionResponse = serde_json::from_slice(&body).unwrap();
    let session_id = session.id.clone();
    assert_eq!(session.agent, "test-agent");
    assert_eq!(session.model, "mock-model");
    assert_eq!(session.status, SessionStatus::Idle);

    // 3. List sessions — should include our session
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/sessions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let sessions: Vec<SessionResponse> = serde_json::from_slice(&body).unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].id, session_id);

    // 4. Get session by ID
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/api/sessions/{session_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // 5. Send prompt
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/api/sessions/{session_id}/prompt"))
                .header("content-type", "application/json")
                .body(Body::from(r#"{"content": "Hello, agent!"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::ACCEPTED);

    // 6. Wait for agent loop to complete
    tokio::time::sleep(Duration::from_millis(500)).await;

    // 7. Collect SSE events that were broadcast
    let mut events = vec![];
    while let Ok(event) = event_rx.try_recv() {
        events.push(event);
    }
    assert!(
        !events.is_empty(),
        "Expected SSE events to be broadcast during agent loop"
    );
    use graphirm_server::types::SseEventType;
    let event_types: Vec<&SseEventType> = events.iter().map(|e| &e.event_type).collect();
    assert!(
        event_types
            .iter()
            .any(|t| matches!(t, SseEventType::AgentStart)),
        "Missing agent_start event. Got: {:?}",
        event_types
    );
    assert!(
        event_types
            .iter()
            .any(|t| matches!(t, SseEventType::AgentEnd)),
        "Missing agent_end event. Got: {:?}",
        event_types
    );

    // 8. Get messages — should have user message + assistant response
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/api/sessions/{session_id}/messages"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let messages: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
    assert!(
        messages.len() >= 2,
        "Expected at least 2 messages (user + assistant), got {}",
        messages.len()
    );

    // 9. Query the session graph
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/api/graph/{session_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let graph: GraphResponse = serde_json::from_slice(&body).unwrap();
    assert!(
        graph.nodes.len() >= 3,
        "Expected at least 3 nodes (agent + user msg + assistant msg), got {}",
        graph.nodes.len()
    );
    assert!(!graph.edges.is_empty(), "Expected edges between nodes");

    // 10. Delete session
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/api/sessions/{session_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);

    // 11. Verify session is gone
    let resp = app
        .oneshot(
            Request::builder()
                .uri(format!("/api/sessions/{session_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_concurrent_sessions() {
    let state = test_app_state();
    let app = create_router(state.clone());

    // Create 5 sessions concurrently
    let mut handles = vec![];
    for i in 0..5 {
        let app = app.clone();
        handles.push(tokio::spawn(async move {
            let resp = app
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/api/sessions")
                        .header("content-type", "application/json")
                        .body(Body::from(format!(r#"{{"agent": "agent-{i}"}}"#)))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::CREATED);

            let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
                .await
                .unwrap();
            let session: SessionResponse = serde_json::from_slice(&body).unwrap();
            session.id
        }));
    }

    let mut ids = vec![];
    for h in handles {
        ids.push(h.await.unwrap());
    }

    // Verify all 5 exist and have unique IDs
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/api/sessions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let sessions: Vec<SessionResponse> = serde_json::from_slice(&body).unwrap();
    assert_eq!(sessions.len(), 5);

    let unique: std::collections::HashSet<_> = ids.iter().collect();
    assert_eq!(unique.len(), 5);
}
