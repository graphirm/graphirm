//! Integration test: web UI static file serving alongside API routes.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tokio::sync::{RwLock, broadcast};
use tower::ServiceExt;

use graphirm_agent::AgentConfig;
use graphirm_graph::GraphStore;
use graphirm_llm::MockProvider;
use graphirm_server::types::SseEvent;
use graphirm_server::{AppState, create_router, next_test_rate_limit_shard};
use graphirm_tools::ToolRegistry;

fn test_app_state_with_web_dir(web_dir: PathBuf) -> AppState {
    let graph = Arc::new(GraphStore::open_memory().unwrap());
    let (event_tx, _) = broadcast::channel::<SseEvent>(256);

    AppState {
        graph,
        llm: Arc::new(MockProvider::fixed("test")),
        tools: Arc::new(ToolRegistry::new()),
        event_tx,
        sessions: Arc::new(RwLock::new(HashMap::new())),
        default_config: AgentConfig::default(),
        memory_retriever: None,
        web_dir: Some(web_dir),
        api_key: String::new(),
        allowed_origins: vec![],
        rate_limit_shard: next_test_rate_limit_shard(),
    }
}

#[tokio::test]
async fn test_web_ui_serves_index_html() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("index.html"),
        "<html><body>graphirm</body></html>",
    )
    .unwrap();

    let state = test_app_state_with_web_dir(dir.path().to_path_buf());
    let app = create_router(state);

    let resp = app
        .oneshot(Request::get("/").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("graphirm"));
}

#[tokio::test]
async fn test_web_ui_serves_static_assets() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("index.html"), "<html></html>").unwrap();
    std::fs::write(dir.path().join("styles.css"), "body { color: red; }").unwrap();
    std::fs::write(dir.path().join("main.js"), "console.log('ok');").unwrap();

    let state = test_app_state_with_web_dir(dir.path().to_path_buf());
    let app = create_router(state);

    // CSS
    let resp = app
        .clone()
        .oneshot(Request::get("/styles.css").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // JS
    let resp = app
        .oneshot(Request::get("/main.js").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_api_routes_take_precedence_over_static_files() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("index.html"), "<html></html>").unwrap();

    let state = test_app_state_with_web_dir(dir.path().to_path_buf());
    let app = create_router(state);

    // API route should still work
    let resp = app
        .oneshot(Request::get("/api/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("ok"));
}

#[tokio::test]
async fn test_client_config_returns_api_key_without_bearer() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("index.html"), "<html></html>").unwrap();

    let mut state = test_app_state_with_web_dir(dir.path().to_path_buf());
    state.api_key = "bootstrap-secret".to_string();
    let app = create_router(state);

    let resp = app
        .oneshot(
            Request::get("/api/client-config")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["api_key"], "bootstrap-secret");
}

#[tokio::test]
async fn test_no_web_dir_returns_404_for_root() {
    let graph = Arc::new(GraphStore::open_memory().unwrap());
    let (event_tx, _) = broadcast::channel::<SseEvent>(256);
    let state = AppState {
        graph,
        llm: Arc::new(MockProvider::fixed("test")),
        tools: Arc::new(ToolRegistry::new()),
        event_tx,
        sessions: Arc::new(RwLock::new(HashMap::new())),
        default_config: AgentConfig::default(),
        memory_retriever: None,
        web_dir: None,
        api_key: String::new(),
        allowed_origins: vec![],
        rate_limit_shard: next_test_rate_limit_shard(),
    };
    let app = create_router(state);

    let resp = app
        .oneshot(Request::get("/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    // Without web_dir, root should 404 (no fallback)
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}
