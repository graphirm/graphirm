//! B-0001: a session created with its own `model` must run on it. The shipped config routes
//! every turn (static tiers plus the Jev router), which used to override the request.

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tokio::sync::{RwLock, broadcast};
use tower::ServiceExt;

use graphirm_agent::AgentConfig;
use graphirm_graph::GraphStore;
use graphirm_llm::MockProvider;
use graphirm_server::types::{SessionResponse, SseEvent};
use graphirm_server::{AppState, SessionId, create_router, next_test_rate_limit_shard};
use graphirm_tools::ToolRegistry;

fn routed_state() -> AppState {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/default.toml");
    let default_config =
        AgentConfig::from_file(std::path::Path::new(path)).expect("shipped default.toml");
    assert!(default_config.model_routing.is_some());
    assert!(default_config.adaptive_routing.is_some());

    let (event_tx, _) = broadcast::channel::<SseEvent>(256);
    AppState {
        graph: Arc::new(GraphStore::open_memory().unwrap()),
        llm: Arc::new(MockProvider::fixed("test")),
        tools: Arc::new(ToolRegistry::new()),
        event_tx,
        sessions: Arc::new(RwLock::new(HashMap::new())),
        default_config,
        memory_retriever: None,
        web_dir: None,
        api_key: String::new(),
        allowed_origins: vec![],
        rate_limit_shard: next_test_rate_limit_shard(),
    }
}

/// Create a session from `body` and return its stored config.
async fn session_config(state: &AppState, body: &str) -> AgentConfig {
    let response = create_router(state.clone())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/sessions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let session: SessionResponse = serde_json::from_slice(&bytes).unwrap();
    let sessions = state.sessions.read().await;
    sessions[&SessionId::from(session.id.as_str())]
        .session
        .agent_config
        .clone()
}

#[tokio::test]
async fn explicit_model_turns_routing_off() {
    let state = routed_state();
    let config = session_config(
        &state,
        r#"{"model": "openrouter/deepseek/deepseek-v4.1-flash"}"#,
    )
    .await;
    assert_eq!(config.model, "openrouter/deepseek/deepseek-v4.1-flash");
    assert!(config.model_routing.is_none());
    assert!(config.adaptive_routing.is_none());
}

#[tokio::test]
async fn no_model_keeps_the_servers_routing() {
    let state = routed_state();
    let config = session_config(&state, r#"{}"#).await;
    assert!(config.model_routing.is_some());
    assert!(config.adaptive_routing.is_some());
}
