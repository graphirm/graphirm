# Phase 10: Usage Discovery Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Instrument the API server with structured request logging, run realistic scenario tests against MockProvider, and analyze the resulting logs to identify which endpoints, sequences, and graph traversal patterns are most common — so the UI (Phase 11) is built around what developers actually do, not what we imagine they'll do.

**Architecture:** A custom axum middleware layer logs every HTTP request as a JSONL entry (method, path, status, duration, session_id, endpoint_group) to a file via a non-blocking mpsc channel. Integration tests simulate five realistic developer workflows (single-turn chat, multi-turn conversation, session management, graph exploration, abort-and-recover) against MockProvider. A Rust example binary parses the JSONL log and produces a frequency/sequence/timing report. The report directly informs Phase 11 UI design decisions.

**Tech Stack:** axum middleware, tokio mpsc channel, serde_json (JSONL), chrono (timestamps), existing MockProvider + test_app_state infrastructure

**Models:** Scenarios use `MockProvider` by default (free, fast, deterministic). For richer patterns with real tool calls and multi-turn reasoning, run the server with a cheap/local model:
- `GRAPHIRM_MODEL=ollama/qwen2.5:7b` — free, local, no API key (needs Ollama running)
- `GRAPHIRM_MODEL=deepseek/deepseek-chat` — ~$0.14/M tokens (~20x cheaper than Anthropic)

---

## Prerequisites (verified present)

### From `graphirm-server` (Phase 8)

```rust
// crates/server/src/routes.rs — 18 route handlers
// crates/server/src/state.rs  — AppState, SessionHandle
// crates/server/src/types.rs  — all request/response types
// crates/server/src/sse.rs    — SSE handlers
// crates/server/src/error.rs  — ServerError with IntoResponse
// crates/server/tests/integration.rs — existing lifecycle + concurrency tests
```

### Test infrastructure

```rust
// crates/server/tests/integration.rs and crates/server/src/routes.rs::test_helpers
fn test_app_state() -> AppState {
    // In-memory GraphStore + MockProvider::fixed("...") + empty ToolRegistry
}
```

---

## Task 1: Create RequestLogEntry type

- [x] Complete

**Files:**
- Create: `crates/server/src/request_log.rs`
- Modify: `crates/server/src/lib.rs`

### Step 1: Write the failing test

Add to the bottom of `crates/server/src/request_log.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_log_entry_serializes_to_json() {
        let entry = RequestLogEntry {
            timestamp: chrono::Utc::now(),
            method: "GET".to_string(),
            path: "/api/sessions".to_string(),
            status: 200,
            duration_ms: 1.23,
            session_id: Some("abc-123".to_string()),
            endpoint_group: "sessions".to_string(),
        };

        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("\"method\":\"GET\""));
        assert!(json.contains("\"status\":200"));
        assert!(json.contains("\"session_id\":\"abc-123\""));
        assert!(json.contains("\"endpoint_group\":\"sessions\""));
    }

    #[test]
    fn request_log_entry_without_session_id() {
        let entry = RequestLogEntry {
            timestamp: chrono::Utc::now(),
            method: "GET".to_string(),
            path: "/api/health".to_string(),
            status: 200,
            duration_ms: 0.5,
            session_id: None,
            endpoint_group: "health".to_string(),
        };

        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("\"session_id\":null"));
    }
}
```

### Step 2: Run test to verify it fails

Run: `cargo test -p graphirm-server request_log::tests -- --nocapture 2>&1`
Expected: FAIL — module and struct don't exist yet.

### Step 3: Write the type

Create `crates/server/src/request_log.rs`:

```rust
//! Structured request logging for API usage analysis.
//!
//! Captures method, path, status, duration, and session context for every
//! HTTP request. Logged as JSONL (one JSON object per line) for easy
//! post-hoc analysis of endpoint frequency, call sequences, and timing.

use chrono::{DateTime, Utc};
use serde::Serialize;

/// A single logged HTTP request with timing and context.
#[derive(Debug, Clone, Serialize)]
pub struct RequestLogEntry {
    /// UTC timestamp when the request was received.
    pub timestamp: DateTime<Utc>,
    /// HTTP method (GET, POST, DELETE, etc.).
    pub method: String,
    /// Request path (e.g. `/api/sessions/abc-123/prompt`).
    pub path: String,
    /// HTTP response status code.
    pub status: u16,
    /// Request-to-response duration in milliseconds.
    pub duration_ms: f64,
    /// Session ID extracted from the path, if present.
    pub session_id: Option<String>,
    /// Endpoint group for aggregation: "health", "sessions", "graph", "events".
    pub endpoint_group: String,
}
```

### Step 4: Register the module

Add to `crates/server/src/lib.rs`, after the existing `pub mod` lines:

```rust
pub mod request_log;
```

And add to the re-export block:

```rust
pub use request_log::RequestLogEntry;
```

### Step 5: Run test to verify it passes

Run: `cargo test -p graphirm-server request_log::tests -- --nocapture 2>&1`
Expected: PASS — both tests pass.

### Step 6: Commit

```bash
git add crates/server/src/request_log.rs crates/server/src/lib.rs
git commit -m "feat(server): add RequestLogEntry type for structured request logging"
```

---

## Task 2: Create RequestLogger with background writer

- [x] Complete

**Files:**
- Modify: `crates/server/src/request_log.rs`

### Step 1: Write the failing test

Add to the test module in `crates/server/src/request_log.rs`:

```rust
    #[tokio::test]
    async fn request_logger_writes_to_file() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("test_request.jsonl");

        let logger = RequestLogger::new(log_path.clone());

        logger.log(RequestLogEntry {
            timestamp: chrono::Utc::now(),
            method: "POST".to_string(),
            path: "/api/sessions".to_string(),
            status: 201,
            duration_ms: 5.0,
            session_id: None,
            endpoint_group: "sessions".to_string(),
        });

        logger.log(RequestLogEntry {
            timestamp: chrono::Utc::now(),
            method: "GET".to_string(),
            path: "/api/sessions/abc/messages".to_string(),
            status: 200,
            duration_ms: 2.1,
            session_id: Some("abc".to_string()),
            endpoint_group: "sessions".to_string(),
        });

        // Flush by dropping the logger (closes the channel)
        drop(logger);
        // Give the background task time to write
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let contents = std::fs::read_to_string(&log_path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2, "Expected 2 log lines, got {}", lines.len());

        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["method"], "POST");
        assert_eq!(first["status"], 201);

        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(second["method"], "GET");
        assert_eq!(second["session_id"], "abc");
    }
```

### Step 2: Run test to verify it fails

Run: `cargo test -p graphirm-server request_log::tests::request_logger_writes_to_file -- --nocapture 2>&1`
Expected: FAIL — `RequestLogger` struct doesn't exist.

### Step 3: Add tempfile dev-dependency

Add to `crates/server/Cargo.toml` under `[dev-dependencies]`:

```toml
tempfile = "3"
```

### Step 4: Write the RequestLogger

Add to `crates/server/src/request_log.rs`, after the `RequestLogEntry` struct:

```rust
use std::io::Write;
use std::path::PathBuf;
use tokio::sync::mpsc;

/// Non-blocking request logger that writes JSONL entries via a background task.
///
/// Entries are sent through an unbounded mpsc channel to a spawned task that
/// appends them to a file. Dropping the `RequestLogger` closes the channel,
/// which causes the background task to flush remaining entries and exit.
#[derive(Clone)]
pub struct RequestLogger {
    tx: mpsc::UnboundedSender<RequestLogEntry>,
}

impl RequestLogger {
    /// Create a new logger that writes JSONL entries to `path`.
    ///
    /// Spawns a background tokio task for non-blocking writes. The file is
    /// created if it doesn't exist, or appended to if it does.
    pub fn new(path: PathBuf) -> Self {
        let (tx, mut rx) = mpsc::unbounded_channel::<RequestLogEntry>();

        tokio::spawn(async move {
            let mut file = match std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                Ok(f) => f,
                Err(e) => {
                    tracing::error!("Failed to open request log file {}: {e}", path.display());
                    return;
                }
            };

            while let Some(entry) = rx.recv().await {
                if let Ok(json) = serde_json::to_string(&entry) {
                    let _ = writeln!(file, "{json}");
                }
            }
        });

        Self { tx }
    }

    /// Log a request entry (non-blocking, fire-and-forget).
    pub fn log(&self, entry: RequestLogEntry) {
        let _ = self.tx.send(entry);
    }
}
```

### Step 5: Run test to verify it passes

Run: `cargo test -p graphirm-server request_log::tests -- --nocapture 2>&1`
Expected: PASS — all 3 tests pass.

### Step 6: Commit

```bash
git add crates/server/src/request_log.rs crates/server/Cargo.toml
git commit -m "feat(server): add RequestLogger with background JSONL writer"
```

---

## Task 3: Add URI parsing helpers

- [x] Complete

**Files:**
- Modify: `crates/server/src/request_log.rs`

### Step 1: Write the failing tests

Add to the test module in `crates/server/src/request_log.rs`:

```rust
    #[test]
    fn extract_session_id_from_session_path() {
        assert_eq!(
            extract_session_id("/api/sessions/abc-123"),
            Some("abc-123".to_string())
        );
        assert_eq!(
            extract_session_id("/api/sessions/abc-123/prompt"),
            Some("abc-123".to_string())
        );
        assert_eq!(
            extract_session_id("/api/sessions/abc-123/messages"),
            Some("abc-123".to_string())
        );
    }

    #[test]
    fn extract_session_id_from_graph_path() {
        assert_eq!(
            extract_session_id("/api/graph/abc-123"),
            Some("abc-123".to_string())
        );
        assert_eq!(
            extract_session_id("/api/graph/abc-123/node/xyz"),
            Some("abc-123".to_string())
        );
        assert_eq!(
            extract_session_id("/api/graph/abc-123/subgraph/xyz"),
            Some("abc-123".to_string())
        );
    }

    #[test]
    fn extract_session_id_from_events_path() {
        assert_eq!(
            extract_session_id("/api/events/abc-123"),
            Some("abc-123".to_string())
        );
    }

    #[test]
    fn extract_session_id_returns_none_for_paths_without_id() {
        assert_eq!(extract_session_id("/api/health"), None);
        assert_eq!(extract_session_id("/api/sessions"), None);
        assert_eq!(extract_session_id("/api/events"), None);
    }

    #[test]
    fn classify_endpoint_group_for_known_paths() {
        assert_eq!(classify_endpoint("/api/health"), "health");
        assert_eq!(classify_endpoint("/api/sessions"), "sessions");
        assert_eq!(classify_endpoint("/api/sessions/abc/prompt"), "sessions");
        assert_eq!(classify_endpoint("/api/graph/abc"), "graph");
        assert_eq!(classify_endpoint("/api/graph/abc/node/xyz"), "graph");
        assert_eq!(classify_endpoint("/api/events"), "events");
        assert_eq!(classify_endpoint("/api/events/abc"), "events");
        assert_eq!(classify_endpoint("/unknown"), "other");
    }
```

### Step 2: Run tests to verify they fail

Run: `cargo test -p graphirm-server request_log::tests -- --nocapture 2>&1`
Expected: FAIL — functions `extract_session_id` and `classify_endpoint` don't exist.

### Step 3: Implement the helpers

Add to `crates/server/src/request_log.rs`, before the test module:

```rust
/// Extract a session ID from a request path, if present.
///
/// Recognizes these patterns:
/// - `/api/sessions/{id}[/...]`
/// - `/api/graph/{id}[/...]`
/// - `/api/events/{id}`
pub fn extract_session_id(path: &str) -> Option<String> {
    let segments: Vec<&str> = path.trim_start_matches('/').split('/').collect();

    if segments.len() < 3 || segments[0] != "api" {
        return None;
    }

    match segments[1] {
        "sessions" | "graph" | "events" if segments.len() >= 3 && !segments[2].is_empty() => {
            Some(segments[2].to_string())
        }
        _ => None,
    }
}

/// Classify a request path into an endpoint group for aggregation.
pub fn classify_endpoint(path: &str) -> &'static str {
    if path.starts_with("/api/health") {
        "health"
    } else if path.starts_with("/api/sessions") {
        "sessions"
    } else if path.starts_with("/api/graph") {
        "graph"
    } else if path.starts_with("/api/events") {
        "events"
    } else {
        "other"
    }
}
```

### Step 4: Run tests to verify they pass

Run: `cargo test -p graphirm-server request_log::tests -- --nocapture 2>&1`
Expected: PASS — all tests pass.

### Step 5: Commit

```bash
git add crates/server/src/request_log.rs
git commit -m "feat(server): add URI parsing helpers for session_id extraction and endpoint classification"
```

---

## Task 4: Create request logging middleware

- [x] Complete

**Files:**
- Create: `crates/server/src/middleware.rs`
- Modify: `crates/server/src/lib.rs`

### Step 1: Write the middleware

Create `crates/server/src/middleware.rs`:

```rust
//! Axum middleware for request logging.
//!
//! Wraps every request to capture method, path, status, and duration, then
//! logs the entry through the [`RequestLogger`] attached as an axum Extension.

use axum::body::Body;
use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::Response;
use chrono::Utc;

use crate::request_log::{RequestLogEntry, RequestLogger, classify_endpoint, extract_session_id};

/// Axum middleware function that logs every request to the [`RequestLogger`].
///
/// Must be used with `axum::middleware::from_fn` and requires a
/// `RequestLogger` in an axum `Extension`.
pub async fn request_logging(
    logger: Option<axum::Extension<RequestLogger>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let Some(axum::Extension(logger)) = logger else {
        return next.run(request).await;
    };

    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    let start = std::time::Instant::now();

    let response = next.run(request).await;

    let duration = start.elapsed();
    let status = response.status().as_u16();

    logger.log(RequestLogEntry {
        timestamp: Utc::now(),
        method,
        path: path.clone(),
        status,
        duration_ms: duration.as_secs_f64() * 1000.0,
        session_id: extract_session_id(&path),
        endpoint_group: classify_endpoint(&path).to_string(),
    });

    response
}
```

### Step 2: Register the module

Add to `crates/server/src/lib.rs`, after the existing `pub mod` lines:

```rust
pub mod middleware;
```

### Step 3: Verify compilation

Run: `cargo check -p graphirm-server 2>&1`
Expected: Compiles without errors.

### Step 4: Commit

```bash
git add crates/server/src/middleware.rs crates/server/src/lib.rs
git commit -m "feat(server): add request logging middleware"
```

---

## Task 5: Wire middleware into router

- [x] Complete

**Files:**
- Modify: `crates/server/src/routes.rs`
- Modify: `crates/server/src/lib.rs`

### Step 1: Add middleware layer to router

In `crates/server/src/routes.rs`, add this import at the top:

```rust
use crate::middleware::request_logging;
```

In the `create_router` function, add the middleware layer **before** the CORS layer (so it captures all requests including CORS preflights):

```rust
pub fn create_router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        // ... all existing routes ...
        .layer(axum::middleware::from_fn(request_logging))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
```

### Step 2: Add RequestLogger extension to server startup

In `crates/server/src/lib.rs`, modify `start_server` to accept an optional log path and attach the logger extension:

Add this parameter to `ServerConfig`:

```rust
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    /// Optional path for request log JSONL output. When set, every HTTP
    /// request is logged for usage analysis.
    pub request_log_path: Option<std::path::PathBuf>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3000,
            request_log_path: None,
        }
    }
}
```

In the `start_server` function, add the logger extension conditionally:

```rust
    let mut app = create_router(state);

    if let Some(ref log_path) = server_config.request_log_path {
        let logger = crate::request_log::RequestLogger::new(log_path.clone());
        app = app.layer(axum::Extension(logger));
        info!("Request logging enabled → {}", log_path.display());
    }
```

### Step 3: Update main.rs Serve command

In `src/main.rs`, add a `--request-log` flag to the `Serve` subcommand:

```rust
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(short, long, default_value = "3000")]
        port: u16,
        /// Path to write request log JSONL file (for usage analysis)
        #[arg(long)]
        request_log: Option<PathBuf>,
    },
```

And pass it through:

```rust
    Commands::Serve { host, port, request_log } => {
        // ... existing setup ...
        let server_config = graphirm_server::ServerConfig {
            host,
            port,
            request_log_path: request_log,
        };
        // ... existing start_server call ...
    }
```

### Step 4: Verify compilation

Run: `cargo check --workspace 2>&1`
Expected: Compiles without errors. Existing tests still pass (no logger extension injected in tests → middleware is a no-op).

### Step 5: Run existing tests

Run: `cargo test -p graphirm-server 2>&1`
Expected: All existing tests pass unchanged.

### Step 6: Commit

```bash
git add crates/server/src/routes.rs crates/server/src/lib.rs src/main.rs
git commit -m "feat(server): wire request logging middleware into router with --request-log flag"
```

---

## Task 6: Test the middleware end-to-end

- [x] Complete

**Files:**
- Modify: `crates/server/tests/integration.rs`

### Step 1: Write the integration test

Add to `crates/server/tests/integration.rs`:

```rust
#[tokio::test]
async fn test_request_logging_produces_jsonl() {
    use graphirm_server::request_log::RequestLogger;
    use graphirm_server::request_log::RequestLogEntry;

    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("requests.jsonl");

    let state = test_app_state();
    let logger = RequestLogger::new(log_path.clone());

    let app = create_router(state.clone()).layer(axum::Extension(logger.clone()));

    // Make several requests
    // 1. Health check
    app.clone()
        .oneshot(
            Request::builder()
                .uri("/api/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // 2. Create session
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
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let session: SessionResponse = serde_json::from_slice(&body).unwrap();

    // 3. Get messages
    app.clone()
        .oneshot(
            Request::builder()
                .uri(format!("/api/sessions/{}/messages", session.id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // 4. Get session graph
    app.clone()
        .oneshot(
            Request::builder()
                .uri(format!("/api/graph/{}", session.id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Drop logger to flush, then wait for background task
    drop(logger);
    drop(app);
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Parse the log file
    let contents = std::fs::read_to_string(&log_path).unwrap();
    let lines: Vec<&str> = contents.lines().collect();

    assert_eq!(lines.len(), 4, "Expected 4 log entries, got {}", lines.len());

    // Verify first entry is health check
    let entry: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(entry["method"], "GET");
    assert_eq!(entry["path"], "/api/health");
    assert_eq!(entry["endpoint_group"], "health");
    assert_eq!(entry["session_id"], serde_json::Value::Null);

    // Verify second entry has correct endpoint group
    let entry: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
    assert_eq!(entry["method"], "POST");
    assert_eq!(entry["endpoint_group"], "sessions");

    // Verify third entry has session_id extracted
    let entry: serde_json::Value = serde_json::from_str(lines[2]).unwrap();
    assert_eq!(entry["session_id"], session.id);
    assert_eq!(entry["endpoint_group"], "sessions");

    // Verify fourth entry is a graph query
    let entry: serde_json::Value = serde_json::from_str(lines[3]).unwrap();
    assert_eq!(entry["endpoint_group"], "graph");
    assert_eq!(entry["session_id"], session.id);
}
```

### Step 2: Add tempfile to integration test deps

`tempfile` should already be in `[dev-dependencies]` from Task 2. Verify.

### Step 3: Run test

Run: `cargo test -p graphirm-server test_request_logging_produces_jsonl -- --nocapture 2>&1`
Expected: PASS — log file contains 4 correctly structured entries.

### Step 4: Commit

```bash
git add crates/server/tests/integration.rs
git commit -m "test(server): add integration test verifying request logging middleware"
```

---

## Task 7: Scenario — single-turn chat

- [x] Complete

**Files:**
- Create: `crates/server/tests/scenarios.rs`

### Step 1: Create the scenario test file

Create `crates/server/tests/scenarios.rs`:

```rust
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
use graphirm_server::types::{
    GraphResponse, SessionResponse, SessionStatus, SseEvent,
};
use graphirm_server::{AppState, create_router};
use graphirm_tools::ToolRegistry;

fn scenario_state() -> AppState {
    let graph = Arc::new(GraphStore::open_memory().unwrap());
    let (event_tx, _) = broadcast::channel::<SseEvent>(256);

    AppState {
        graph,
        llm: Arc::new(MockProvider::fixed("I'll help you with that. Here's my analysis of the code.")),
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
    let logger = RequestLogger::new(log_path.clone());
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

    eprintln!("scenario_single_turn_chat: {lines} API calls logged to {}", log_path.display());
}
```

### Step 2: Run the scenario

Run: `cargo test -p graphirm-server --test scenarios scenario_single_turn_chat -- --nocapture 2>&1`
Expected: PASS — creates session, prompts, queries messages and graph, logs all calls.

### Step 3: Commit

```bash
git add crates/server/tests/scenarios.rs
git commit -m "test(server): add scenario_single_turn_chat for usage discovery"
```

---

## Task 8: Scenario — multi-turn conversation

- [x] Complete

**Files:**
- Modify: `crates/server/tests/scenarios.rs`

### Step 1: Add the scenario test

Add to `crates/server/tests/scenarios.rs`:

```rust
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

    eprintln!("scenario_multi_turn_conversation: {lines} API calls logged to {}", log_path.display());
}
```

### Step 2: Run the scenario

Run: `cargo test -p graphirm-server --test scenarios scenario_multi_turn -- --nocapture 2>&1`
Expected: PASS.

### Step 3: Commit

```bash
git add crates/server/tests/scenarios.rs
git commit -m "test(server): add scenario_multi_turn_conversation for usage discovery"
```

---

## Task 9: Scenario — session management

- [x] Complete

**Files:**
- Modify: `crates/server/tests/scenarios.rs`

### Step 1: Add the scenario test

Add to `crates/server/tests/scenarios.rs`:

```rust
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

    eprintln!("scenario_session_management: {lines} API calls logged to {}", log_path.display());
}
```

### Step 2: Run the scenario

Run: `cargo test -p graphirm-server --test scenarios scenario_session_management -- --nocapture 2>&1`
Expected: PASS.

### Step 3: Commit

```bash
git add crates/server/tests/scenarios.rs
git commit -m "test(server): add scenario_session_management for usage discovery"
```

---

## Task 10: Scenario — graph exploration

- [x] Complete

**Files:**
- Modify: `crates/server/tests/scenarios.rs`

### Step 1: Add the scenario test

Add to `crates/server/tests/scenarios.rs`:

```rust
/// Scenario D: Developer explores the graph in detail — full graph, subgraphs, individual nodes.
///
/// Expected pattern: create → prompt → get_graph → for_each_node(get_node) →
///                   pick_node(get_subgraph) → get_tasks → get_knowledge
#[tokio::test]
async fn scenario_graph_exploration() {
    let (app, log_path) = build_app("graph_explore");

    let sid = create_session(&app).await;

    // Send a prompt to populate the graph
    send_prompt(&app, &sid, "Analyze the project structure and suggest improvements").await;

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
```

### Step 2: Run the scenario

Run: `cargo test -p graphirm-server --test scenarios scenario_graph_exploration -- --nocapture 2>&1`
Expected: PASS.

### Step 3: Commit

```bash
git add crates/server/tests/scenarios.rs
git commit -m "test(server): add scenario_graph_exploration for usage discovery"
```

---

## Task 11: Scenario — abort and recover

- [x] Complete

**Files:**
- Modify: `crates/server/tests/scenarios.rs`

### Step 1: Add the scenario test

Add to `crates/server/tests/scenarios.rs`:

```rust
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

    eprintln!("scenario_abort_and_recover: {lines} API calls logged to {}", log_path.display());
}
```

### Step 2: Run the scenario

Run: `cargo test -p graphirm-server --test scenarios scenario_abort_and_recover -- --nocapture 2>&1`
Expected: PASS.

### Step 3: Commit

```bash
git add crates/server/tests/scenarios.rs
git commit -m "test(server): add scenario_abort_and_recover for usage discovery"
```

---

## Task 12: Create log analysis tool

- [x] Complete

**Files:**
- Create: `crates/server/examples/analyze_request_log.rs`

### Step 1: Write the analysis tool

Create `crates/server/examples/analyze_request_log.rs`:

```rust
//! Analyze request log JSONL files produced by usage discovery scenarios.
//!
//! Reads one or more JSONL files and produces a report showing:
//! - Endpoint frequency (which endpoints are called most)
//! - Endpoint group distribution (sessions vs graph vs events)
//! - Method distribution (GET vs POST vs DELETE)
//! - Call sequences (what follows what)
//! - Timing statistics (p50, p95, max per endpoint)
//! - Session patterns (how many calls per session)
//!
//! Usage: cargo run -p graphirm-server --example analyze_request_log -- <file1.jsonl> [file2.jsonl ...]

use std::collections::HashMap;
use std::io::BufRead;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct LogEntry {
    method: String,
    path: String,
    status: u16,
    duration_ms: f64,
    session_id: Option<String>,
    endpoint_group: String,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("Usage: analyze_request_log <file1.jsonl> [file2.jsonl ...]");
        std::process::exit(1);
    }

    let mut entries: Vec<LogEntry> = Vec::new();

    for path in &args {
        let file = std::fs::File::open(path).unwrap_or_else(|e| {
            eprintln!("Failed to open {path}: {e}");
            std::process::exit(1);
        });

        for line in std::io::BufReader::new(file).lines() {
            let line = line.unwrap();
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<LogEntry>(&line) {
                Ok(entry) => entries.push(entry),
                Err(e) => eprintln!("Skipping malformed line: {e}"),
            }
        }
    }

    if entries.is_empty() {
        println!("No entries found.");
        return;
    }

    println!("═══════════════════════════════════════════════════");
    println!("  USAGE DISCOVERY REPORT");
    println!("  {} entries from {} file(s)", entries.len(), args.len());
    println!("═══════════════════════════════════════════════════\n");

    report_endpoint_frequency(&entries);
    report_group_distribution(&entries);
    report_method_distribution(&entries);
    report_call_sequences(&entries);
    report_timing_stats(&entries);
    report_session_patterns(&entries);
}

fn report_endpoint_frequency(entries: &[LogEntry]) {
    println!("── Endpoint Frequency ──────────────────────────────\n");

    let mut counts: HashMap<String, usize> = HashMap::new();
    for entry in entries {
        let key = normalize_path(&entry.method, &entry.path);
        *counts.entry(key).or_default() += 1;
    }

    let mut sorted: Vec<_> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));

    let total = entries.len() as f64;
    println!("{:<45}  {:>5}  {:>6}", "ENDPOINT", "COUNT", "  %");
    println!("{}", "-".repeat(60));
    for (endpoint, count) in &sorted {
        let pct = (*count as f64 / total) * 100.0;
        println!("{:<45}  {:>5}  {:>5.1}%", endpoint, count, pct);
    }
    println!();
}

fn report_group_distribution(entries: &[LogEntry]) {
    println!("── Endpoint Group Distribution ─────────────────────\n");

    let mut counts: HashMap<&str, usize> = HashMap::new();
    for entry in entries {
        *counts.entry(&entry.endpoint_group).or_default() += 1;
    }

    let mut sorted: Vec<_> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));

    let total = entries.len() as f64;
    for (group, count) in &sorted {
        let pct = (*count as f64 / total) * 100.0;
        let bar_len = (pct / 2.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("  {:<12}  {:>5}  ({:>5.1}%)  {}", group, count, pct, bar);
    }
    println!();
}

fn report_method_distribution(entries: &[LogEntry]) {
    println!("── Method Distribution ─────────────────────────────\n");

    let mut counts: HashMap<&str, usize> = HashMap::new();
    for entry in entries {
        *counts.entry(&entry.method).or_default() += 1;
    }

    let mut sorted: Vec<_> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));

    for (method, count) in &sorted {
        let pct = (*count as f64 / entries.len() as f64) * 100.0;
        println!("  {:<8}  {:>5}  ({:>5.1}%)", method, count, pct);
    }
    println!();
}

fn report_call_sequences(entries: &[LogEntry]) {
    println!("── Call Sequences (bigrams) ─────────────────────────\n");

    let mut bigrams: HashMap<String, usize> = HashMap::new();
    for window in entries.windows(2) {
        let a = normalize_path(&window[0].method, &window[0].path);
        let b = normalize_path(&window[1].method, &window[1].path);
        let key = format!("{a}  →  {b}");
        *bigrams.entry(key).or_default() += 1;
    }

    let mut sorted: Vec<_> = bigrams.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));

    println!("{:<80}  {:>5}", "SEQUENCE", "COUNT");
    println!("{}", "-".repeat(87));
    for (seq, count) in sorted.iter().take(15) {
        println!("{:<80}  {:>5}", seq, count);
    }
    println!();
}

fn report_timing_stats(entries: &[LogEntry]) {
    println!("── Timing (ms) ─────────────────────────────────────\n");

    let mut by_endpoint: HashMap<String, Vec<f64>> = HashMap::new();
    for entry in entries {
        let key = normalize_path(&entry.method, &entry.path);
        by_endpoint.entry(key).or_default().push(entry.duration_ms);
    }

    println!(
        "{:<45}  {:>6}  {:>6}  {:>6}  {:>6}",
        "ENDPOINT", "P50", "P95", "MAX", "N"
    );
    println!("{}", "-".repeat(75));

    let mut sorted: Vec<_> = by_endpoint.into_iter().collect();
    sorted.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    for (endpoint, mut times) in sorted {
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = times.len();
        let p50 = times[n / 2];
        let p95 = times[(n as f64 * 0.95) as usize];
        let max = times[n - 1];
        println!(
            "{:<45}  {:>6.1}  {:>6.1}  {:>6.1}  {:>6}",
            endpoint, p50, p95, max, n
        );
    }
    println!();
}

fn report_session_patterns(entries: &[LogEntry]) {
    println!("── Session Patterns ────────────────────────────────\n");

    let mut by_session: HashMap<String, Vec<String>> = HashMap::new();
    for entry in entries {
        if let Some(ref sid) = entry.session_id {
            let key = normalize_path(&entry.method, &entry.path);
            by_session.entry(sid.clone()).or_default().push(key);
        }
    }

    if by_session.is_empty() {
        println!("  No session-scoped requests found.\n");
        return;
    }

    println!("  Sessions observed: {}", by_session.len());
    println!("  Avg calls/session: {:.1}", {
        let total: usize = by_session.values().map(|v| v.len()).sum();
        total as f64 / by_session.len() as f64
    });

    // Show call sequence for each session
    println!();
    for (sid, calls) in &by_session {
        println!("  Session {}:", &sid[..8.min(sid.len())]);
        for (i, call) in calls.iter().enumerate() {
            let prefix = if i == calls.len() - 1 { "└─" } else { "├─" };
            println!("    {prefix} {call}");
        }
        println!();
    }
}

/// Normalize a path by replacing UUIDs/IDs with `:id` placeholders.
fn normalize_path(method: &str, path: &str) -> String {
    let segments: Vec<&str> = path.split('/').collect();
    let normalized: Vec<&str> = segments
        .iter()
        .enumerate()
        .map(|(i, seg)| {
            if i >= 3 && (seg.len() > 8 || seg.contains('-')) && !["prompt", "messages", "abort", "children", "tasks", "knowledge", "node", "subgraph"].contains(seg) {
                ":id"
            } else {
                seg
            }
        })
        .collect();

    format!("{} {}", method, normalized.join("/"))
}
```

### Step 2: Verify compilation

Run: `cargo build -p graphirm-server --example analyze_request_log 2>&1`
Expected: Compiles.

### Step 3: Commit

```bash
git add crates/server/examples/analyze_request_log.rs
git commit -m "feat(server): add analyze_request_log example for usage discovery reporting"
```

---

## Task 13: Run all scenarios and generate report

- [x] Complete

**Files:**
- No new files

### Step 1: Run all scenarios

Run: `cargo test -p graphirm-server --test scenarios -- --nocapture 2>&1`
Expected: All 5 scenarios pass, each printing their log file path.

### Step 2: Analyze the logs

Run: `cargo run -p graphirm-server --example analyze_request_log -- $(find target -name 'scenario_*.jsonl' -type f) 2>&1`
Expected: A formatted report showing endpoint frequencies, group distribution, call sequences, timing, and session patterns.

### Step 3: Record findings

Review the report output and note:

1. **Top endpoints by frequency** — these are what the UI must make instantly accessible
2. **Most common sequences** — these should be a single click or keystroke in the UI
3. **Group distribution** — how much of usage is "sessions" vs "graph" vs "events"
4. **Timing hotspots** — which operations feel slow and might need loading indicators
5. **Session call patterns** — the typical workflow shape

Write observations to `docs/usage-discovery-findings.md` (manual, not automated — the human interprets the data).

### Step 4: Commit

```bash
git add docs/usage-discovery-findings.md
git commit -m "docs: record Phase 10 usage discovery findings"
```

---

## Final: Run full test suite and clippy

- [x] Complete

### Step 1: Run all tests

Run: `cargo test --workspace 2>&1`
Expected: All tests pass (existing + new scenario tests).

### Step 2: Run clippy

Run: `cargo clippy --workspace --all-targets 2>&1`
Expected: No errors.

### Step 3: Format check

Run: `cargo fmt --all -- --check 2>&1`
Expected: No formatting issues.

### Step 4: Final commit

```bash
git add -A
git commit -m "chore: Phase 10 (Usage Discovery) complete — middleware, scenarios, analysis tool"
```
