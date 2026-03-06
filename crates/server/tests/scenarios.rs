//! Usage discovery scenarios.
//!
//! Each test simulates a realistic developer workflow against MockProvider.
//! All tests write to a shared log file, which is analyzed post-run to
//! discover API usage patterns.
//!
//! Run all scenarios: `cargo test -p graphirm-server --test scenarios -- --nocapture`
//! Then analyze:      `cargo run --example analyze_request_log -- <log-path>`

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
use graphirm_server::request_log::RequestLogger;
use graphirm_server::types::{GraphResponse, SessionResponse, SseEvent};
use graphirm_server::{AppState, create_router};
use graphirm_tools::ToolRegistry;

fn scenario_state() -> AppState {
    let graph = Arc::new(GraphStore::open_memory().unwrap());
    let (event_tx, _) = broadcast::channel::<SseEvent>(256);

    AppState {
        graph,
        llm: Arc::new(MockProvider::fixed(
            "I'll help you with that. Here's my analysis of the code.",
        )),
        tools: Arc::new(ToolRegistry::new()),
        event_tx,
        sessions: Arc::new(RwLock::new(HashMap::new())),
        default_config: AgentConfig::default(),
    }
}

fn scenario_log_path(name: &str) -> std::path::PathBuf {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(format!("scenario_{name}.jsonl"))
}

fn build_app(name: &str) -> (axum::Router, std::path::PathBuf) {
    let state = scenario_state();
    let log_path = scenario_log_path(name);
    // Remove old log if exists
    let _ = std::fs::remove_file(&log_path);
    let (logger, _handle) = RequestLogger::new(log_path.clone());
    let app = create_router(state).layer(axum::Extension(logger));
    (app, log_path)
}

async fn create_session(app: &axum::Router) -> String {
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/sessions")
                .header("content-type", "application/json")
                .body(Body::from(r#"{}"#))
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
}

async fn send_prompt(app: &axum::Router, session_id: &str, content: &str) {
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/api/sessions/{session_id}/prompt"))
                .header("content-type", "application/json")
                .body(Body::from(format!(r#"{{"content": "{content}"}}"#)))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    // Wait for agent loop to complete
    tokio::time::sleep(Duration::from_millis(300)).await;
}

fn count_log_lines(path: &std::path::PathBuf) -> usize {
    // Wait for flush
    std::thread::sleep(std::time::Duration::from_millis(100));
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .count()
}

/// Scenario A: Developer asks a single question, checks the response, looks at the graph.
///
/// Expected pattern: create → prompt → wait → get_messages → get_graph → get_node
#[tokio::test]
async fn scenario_single_turn_chat() {
    let (app, log_path) = build_app("single_turn");

    // 1. Create session
    let sid = create_session(&app).await;

    // 2. Send one prompt
    send_prompt(&app, &sid, "What does this function do?").await;

    // 3. Check messages
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/api/sessions/{sid}/messages"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // 4. Look at the session graph
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/api/graph/{sid}"))
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

    // 5. Inspect one specific node
    if let Some(node) = graph.nodes.first() {
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/api/graph/{sid}/node/{}", node.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // Flush logger
    drop(app);
    tokio::time::sleep(Duration::from_millis(200)).await;

    let lines = count_log_lines(&log_path);
    assert!(lines >= 4, "Expected at least 4 log entries, got {lines}");

    eprintln!(
        "scenario_single_turn_chat: {lines} API calls logged to {}",
        log_path.display()
    );
}

/// Scenario B: Developer has a multi-turn conversation, checks messages after each turn.
///
/// Expected pattern: create → (prompt → wait → get_messages → get_graph) × N
#[tokio::test]
async fn scenario_multi_turn_conversation() {
    let (app, log_path) = build_app("multi_turn");

    let sid = create_session(&app).await;

    let prompts = [
        "Explain the auth module architecture",
        "Now refactor the middleware to use async",
        "Add error handling for the token expiry case",
        "Write tests for the new middleware",
    ];

    for prompt in &prompts {
        send_prompt(&app, &sid, prompt).await;

        // After each turn, developer checks messages
        app.clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/api/sessions/{sid}/messages"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        // And refreshes the graph
        app.clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/api/graph/{sid}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
    }

    // Final: check session status
    app.clone()
        .oneshot(
            Request::builder()
                .uri(format!("/api/sessions/{sid}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    drop(app);
    tokio::time::sleep(Duration::from_millis(200)).await;

    let lines = count_log_lines(&log_path);
    // 1 create + 4*(prompt + messages + graph) + 1 status = 14
    assert!(lines >= 14, "Expected at least 14 log entries, got {lines}");

    eprintln!(
        "scenario_multi_turn_conversation: {lines} API calls logged to {}",
        log_path.display()
    );
}

/// Scenario C: Developer manages multiple sessions — create, list, switch, delete.
///
/// Expected pattern: create × 3 → list → get(each) → delete(1) → list
#[tokio::test]
async fn scenario_session_management() {
    let (app, log_path) = build_app("session_mgmt");

    // Create 3 sessions
    let s1 = create_session(&app).await;
    let s2 = create_session(&app).await;
    let s3 = create_session(&app).await;

    // List all sessions
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
    assert_eq!(sessions.len(), 3);

    // Get each session individually (simulates "switching" between them)
    for sid in [&s1, &s2, &s3] {
        app.clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/api/sessions/{sid}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        // When switching, also load messages and graph
        app.clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/api/sessions/{sid}/messages"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        app.clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/api/graph/{sid}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
    }

    // Delete one session
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/api/sessions/{s2}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);

    // List again
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
    assert_eq!(sessions.len(), 2);

    drop(app);
    tokio::time::sleep(Duration::from_millis(200)).await;

    let lines = count_log_lines(&log_path);
    // 3 create + 1 list + 3*(get + messages + graph) + 1 delete + 1 list = 15
    assert!(lines >= 15, "Expected at least 15 log entries, got {lines}");

    eprintln!(
        "scenario_session_management: {lines} API calls logged to {}",
        log_path.display()
    );
}

/// Scenario D: Developer explores the graph in detail — full graph, subgraphs, individual nodes.
///
/// Expected pattern: create → prompt → get_graph → for_each_node(get_node) →
///                   pick_node(get_subgraph) → get_tasks → get_knowledge
#[tokio::test]
async fn scenario_graph_exploration() {
    let (app, log_path) = build_app("graph_explore");

    let sid = create_session(&app).await;

    // Send a prompt to populate the graph
    send_prompt(
        &app,
        &sid,
        "Analyze the project structure and suggest improvements",
    )
    .await;

    // Get full session graph
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/api/graph/{sid}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let graph: GraphResponse = serde_json::from_slice(&body).unwrap();

    // Inspect every node individually
    for node in &graph.nodes {
        app.clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/api/graph/{sid}/node/{}", node.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
    }

    // Explore subgraph from the first node at different depths
    if let Some(node) = graph.nodes.first() {
        for depth in [1, 2, 3] {
            app.clone()
                .oneshot(
                    Request::builder()
                        .uri(format!(
                            "/api/graph/{sid}/subgraph/{}?depth={depth}",
                            node.id
                        ))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
        }
    }

    // Check tasks and knowledge
    app.clone()
        .oneshot(
            Request::builder()
                .uri(format!("/api/graph/{sid}/tasks"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    app.clone()
        .oneshot(
            Request::builder()
                .uri(format!("/api/graph/{sid}/knowledge"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    drop(app);
    tokio::time::sleep(Duration::from_millis(200)).await;

    let lines = count_log_lines(&log_path);
    // 1 create + 1 prompt + 1 graph + N nodes + 3 subgraphs + tasks + knowledge
    assert!(lines >= 8, "Expected at least 8 log entries, got {lines}");

    eprintln!(
        "scenario_graph_exploration: {lines} API calls logged ({} nodes inspected) to {}",
        graph.nodes.len(),
        log_path.display()
    );
}

/// Scenario E: Developer starts a prompt, aborts, then sends a new prompt.
///
/// Expected pattern: create → prompt → abort → get_session (check status) →
///                   prompt → wait → get_messages → get_graph
#[tokio::test]
async fn scenario_abort_and_recover() {
    let (app, log_path) = build_app("abort_recover");

    let sid = create_session(&app).await;

    // Start first prompt
    app.clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/api/sessions/{sid}/prompt"))
                .header("content-type", "application/json")
                .body(Body::from(r#"{"content": "Do something complex"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    // Immediately abort
    app.clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/api/sessions/{sid}/abort"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Wait for abort to take effect
    tokio::time::sleep(Duration::from_millis(300)).await;

    // Check session status
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/api/sessions/{sid}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Send a new prompt (recovery)
    send_prompt(&app, &sid, "Actually, just explain this file").await;

    // Check messages
    app.clone()
        .oneshot(
            Request::builder()
                .uri(format!("/api/sessions/{sid}/messages"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Check graph
    app.clone()
        .oneshot(
            Request::builder()
                .uri(format!("/api/graph/{sid}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    drop(app);
    tokio::time::sleep(Duration::from_millis(200)).await;

    let lines = count_log_lines(&log_path);
    // 1 create + 1 prompt + 1 abort + 1 get_session + 1 prompt + 1 messages + 1 graph = 7
    assert!(lines >= 7, "Expected at least 7 log entries, got {lines}");

    eprintln!(
        "scenario_abort_and_recover: {lines} API calls logged to {}",
        log_path.display()
    );
}
