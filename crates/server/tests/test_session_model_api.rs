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
use graphirm_server::{
    AppState, SessionId, build_restored_session_handles, create_router, next_test_rate_limit_shard,
    restore_sessions_from_graph,
};
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

/// B-0002: after a restart the session is rebuilt from the graph; it must keep its own model.
#[tokio::test]
async fn restored_session_keeps_its_explicit_model_and_stays_unrouted() {
    let state = routed_state();
    session_config(
        &state,
        r#"{"model": "openrouter/deepseek/deepseek-v4.1-flash"}"#,
    )
    .await;
    session_config(&state, r#"{}"#).await;

    let restored = restore_sessions_from_graph(&state.graph, None)
        .await
        .unwrap();
    let handles =
        build_restored_session_handles(&state.graph, &state.default_config, None, restored);
    assert_eq!(handles.len(), 2);

    let (explicit, plain): (Vec<_>, Vec<_>) = handles
        .values()
        .map(|handle| &handle.session.agent_config)
        .partition(|config| config.model == "openrouter/deepseek/deepseek-v4.1-flash");
    assert_eq!(explicit.len(), 1);
    assert!(explicit[0].model_routing.is_none());
    assert!(explicit[0].adaptive_routing.is_none());

    // A session created without a model follows the server's config again.
    assert_eq!(plain[0].model, state.default_config.model);
    assert!(plain[0].model_routing.is_some());
    assert!(plain[0].adaptive_routing.is_some());
}
