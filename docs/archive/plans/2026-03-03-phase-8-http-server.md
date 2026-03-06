# Phase 8: HTTP Server + SDK Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose Graphirm's graph store and agent capabilities over a REST + SSE HTTP API, enabling web UIs, SDKs, and third-party integrations to interact with sessions, send prompts, query the graph, and stream real-time agent events.

**Architecture:** An axum-based HTTP server wraps the existing `GraphStore`, `Session`, `EventBus`, and `run_agent_loop` behind REST endpoints. Shared state (`AppState`) is injected via axum's `State` extractor, using `Arc` wrappers for thread safety. Prompt submission spawns the agent loop as a background tokio task and returns 202 immediately. Agent lifecycle events flow through a `tokio::sync::broadcast` channel and are multiplexed to connected SSE clients. A `CancellationToken` per session enables abort. The server is started via a `graphirm serve` CLI subcommand.

**Tech Stack:** axum 0.8 (HTTP framework), tower + tower-http (middleware: CORS, tracing), tokio (async runtime, broadcast, RwLock), serde + serde_json (serialization), tokio-stream (BroadcastStream for SSE), chrono (timestamps), thiserror (errors)

---

## Prerequisites (expected APIs from Phases 1–4)

Phase 8 depends on types and traits defined in earlier phases. The code below references these APIs. If the actual signatures differ slightly after Phases 1–4 land, adapt accordingly — the logic and test structure remain the same.

### From `graphirm-graph` (Phase 1)

```rust
// crates/graph/src/lib.rs — re-exports
pub use petgraph::Direction;

// Newtypes
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EdgeId(pub String);

// NodeType enum with 5 variants, each carrying data
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum NodeType {
    Interaction(InteractionData),
    Agent(AgentData),
    Content(ContentData),
    Task(TaskData),
    Knowledge(KnowledgeData),
}

// Full node/edge structs (both derive Serialize)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode { pub id: NodeId, pub node_type: NodeType, /* ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge { pub id: EdgeId, pub edge_type: EdgeType, pub source: NodeId, pub target: NodeId, /* ... */ }

// GraphStore is Send + Sync (r2d2 pool + Arc<RwLock<petgraph>>)
impl GraphStore {
    pub fn open_memory() -> Result<Self, GraphError>;
    pub fn add_node(&self, node: GraphNode) -> Result<NodeId, GraphError>;
    pub fn get_node(&self, id: &NodeId) -> Result<GraphNode, GraphError>;
    pub fn neighbors(&self, id: &NodeId, edge_type: Option<EdgeType>, direction: Direction) -> Result<Vec<GraphNode>, GraphError>;
    pub fn subgraph(&self, root: &NodeId, depth: usize) -> Result<(Vec<GraphNode>, Vec<GraphEdge>), GraphError>;
}
```

### From `graphirm-llm` (Phase 2)

```rust
#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn complete(&self, messages: Vec<LlmMessage>, tools: Vec<ToolDefinition>) -> Result<LlmResponse, LlmError>;
}

pub struct LlmResponse { pub content: String, pub tool_calls: Vec<ToolCall>, pub usage: Usage }
pub struct Usage { pub input_tokens: u32, pub output_tokens: u32 }
```

### From `graphirm-tools` (Phase 3)

```rust
pub struct ToolRegistry { /* HashMap<String, Arc<dyn Tool>> */ }
impl ToolRegistry {
    pub fn new() -> Self;
    pub fn definitions(&self) -> Vec<ToolDefinition>;
}
```

### From `graphirm-agent` (Phase 4)

```rust
pub struct Session {
    pub id: NodeId,
    pub agent_config: AgentConfig,
    pub graph: Arc<GraphStore>,
    pub created_at: DateTime<Utc>,
}
impl Session {
    pub fn new(graph: Arc<GraphStore>, config: AgentConfig) -> Result<Self, AgentError>;
    pub fn add_user_message(&self, content: &str) -> Result<NodeId, AgentError>;
}

pub struct AgentConfig {
    pub name: String,
    pub model: String,
    pub system_prompt: String,
    pub max_turns: u32,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub tools: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum AgentEvent {
    AgentStart { agent_id: NodeId },
    AgentEnd { agent_id: NodeId, node_ids: Vec<NodeId> },
    TurnStart { turn_index: u32 },
    TurnEnd { response_id: NodeId, tool_result_ids: Vec<NodeId> },
    MessageEnd { node_id: NodeId },
    ToolStart { node_id: NodeId, tool_name: String },
    ToolEnd { node_id: NodeId, is_error: bool },
    // ...
}

pub struct EventBus { /* mpsc-based pub/sub */ }
impl EventBus {
    pub fn new() -> Self;
    pub fn subscribe(&mut self) -> mpsc::Receiver<AgentEvent>;
    pub async fn emit(&self, event: AgentEvent);
}

pub async fn run_agent_loop(
    session: &Session,
    llm: &dyn LlmProvider,
    tools: &ToolRegistry,
    events: &EventBus,
    cancel: &CancellationToken,
) -> Result<(), AgentError>;
```

---

## Task 1: Update crates/server/Cargo.toml with all dependencies

- [ ] Complete

**Files:**
- Modify: `crates/server/Cargo.toml`

### Step 1: Write the updated Cargo.toml

```toml
[package]
name = "graphirm-server"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
authors.workspace = true
description = "HTTP API and SDK server for Graphirm"

[dependencies]
graphirm-graph = { path = "../graph" }
graphirm-agent = { path = "../agent" }
graphirm-llm = { path = "../llm" }
graphirm-tools = { path = "../tools" }
axum = "0.8"
tokio = { version = "1", features = ["full"] }
tokio-util = "0.7"
tokio-stream = { version = "0.1", features = ["sync"] }
tower = { version = "0.5", features = ["util"] }
tower-http = { version = "0.6", features = ["cors", "trace"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
tracing = "0.1"
chrono = { version = "0.4", features = ["serde"] }

[dev-dependencies]
tokio = { version = "1", features = ["full", "test-util"] }
async-trait = "0.1"
```

### Step 2: Verify

Run: `cargo check -p graphirm-server 2>&1`
Expected: Compiles (lib.rs is a stub at this point; may warn about unused deps).

### Step 3: Commit

```bash
git add crates/server/Cargo.toml
git commit -m "feat(server): add Phase 8 dependencies — axum, tower, tokio-stream, serde"
```

---

## Task 2: Define ServerError with IntoResponse, AppState, SessionHandle

- [ ] Complete

**Files:**
- Create: `crates/server/src/error.rs`
- Create: `crates/server/src/state.rs`
- Modify: `crates/server/src/lib.rs`

### Step 1: Write the failing tests

Add to `crates/server/src/error.rs`:

```rust
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Graph error: {0}")]
    Graph(#[from] graphirm_graph::GraphError),

    #[error("Agent error: {0}")]
    Agent(#[from] graphirm_agent::AgentError),
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            ServerError::NotFound(msg) => (StatusCode::NOT_FOUND, msg.clone()),
            ServerError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            ServerError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            ServerError::Graph(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            ServerError::Agent(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        };
        (status, Json(serde_json::json!({ "error": message }))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::StatusCode;
    use tower::ServiceExt;

    #[test]
    fn not_found_has_404_status() {
        let err = ServerError::NotFound("Session xyz".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn bad_request_has_400_status() {
        let err = ServerError::BadRequest("Missing field".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn internal_has_500_status() {
        let err = ServerError::Internal("Something broke".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }
}
```

### Step 2: Write state.rs

```rust
use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use tokio::sync::{broadcast, RwLock};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use graphirm_agent::{AgentConfig, AgentError};
use graphirm_graph::GraphStore;
use graphirm_llm::LlmProvider;
use graphirm_tools::ToolRegistry;

use crate::types::SseEvent;

#[derive(Clone)]
pub struct AppState {
    pub graph: Arc<GraphStore>,
    pub llm: Arc<dyn LlmProvider>,
    pub tools: Arc<ToolRegistry>,
    pub event_tx: broadcast::Sender<SseEvent>,
    pub sessions: Arc<RwLock<HashMap<String, SessionHandle>>>,
    pub default_config: AgentConfig,
}

pub struct SessionHandle {
    pub session: Arc<graphirm_agent::Session>,
    pub signal: CancellationToken,
    pub join_handle: Option<JoinHandle<Result<(), AgentError>>>,
    pub status: SessionStatus,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionStatus {
    Idle,
    Running,
    Completed,
    Failed,
}

impl SessionStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            SessionStatus::Idle => "idle",
            SessionStatus::Running => "running",
            SessionStatus::Completed => "completed",
            SessionStatus::Failed => "failed",
        }
    }
}

impl std::fmt::Display for SessionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_status_display() {
        assert_eq!(SessionStatus::Idle.to_string(), "idle");
        assert_eq!(SessionStatus::Running.to_string(), "running");
        assert_eq!(SessionStatus::Completed.to_string(), "completed");
        assert_eq!(SessionStatus::Failed.to_string(), "failed");
    }

    #[test]
    fn app_state_is_send_and_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<AppState>();
        assert_sync::<AppState>();
    }
}
```

### Step 3: Update lib.rs

```rust
pub mod error;
pub mod routes;
pub mod sdk;
pub mod sse;
pub mod state;
pub mod types;
```

### Step 4: Verify

Run: `cargo check -p graphirm-server 2>&1`
Expected: May fail because `types.rs`, `routes.rs`, `sse.rs`, `sdk.rs` are stubs or empty. Create minimal stubs for them now:

`crates/server/src/types.rs`:
```rust
// API request/response types — implemented in Task 3
```

`crates/server/src/routes.rs`:
```rust
// HTTP route handlers — implemented in Tasks 4–9
```

`crates/server/src/sse.rs`:
```rust
// SSE streaming handlers — implemented in Tasks 10–11
```

`crates/server/src/sdk.rs`:
```rust
// Client SDK — post-MVP
```

Run: `cargo check -p graphirm-server 2>&1`
Expected: Compiles.

Run: `cargo test -p graphirm-server 2>&1`
Expected: All tests pass (error + state tests).

### Step 5: Commit

```bash
git add crates/server/src/
git commit -m "feat(server): define AppState, SessionHandle, ServerError with IntoResponse"
```

---

## Task 3: Define API request/response types with serde roundtrip tests

- [ ] Complete

**Files:**
- Modify: `crates/server/src/types.rs`

### Step 1: Write types and tests

Replace `crates/server/src/types.rs` with:

```rust
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
    pub created_at: String,
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
        let session = SessionResponse {
            id: "abc-123".to_string(),
            agent: "graphirm".to_string(),
            model: "claude-sonnet-4-20250514".to_string(),
            created_at: "2026-03-03T12:00:00Z".to_string(),
            status: "idle".to_string(),
        };
        let json = serde_json::to_string(&session).unwrap();
        let back: SessionResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "abc-123");
        assert_eq!(back.agent, "graphirm");
        assert_eq!(back.model, "claude-sonnet-4-20250514");
        assert_eq!(back.status, "idle");
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
```

### Step 2: Verify

Run: `cargo test -p graphirm-server types::tests -- --nocapture 2>&1`
Expected: `test result: ok. 8 passed; 0 failed`

### Step 3: Commit

```bash
git add crates/server/src/types.rs
git commit -m "feat(server): define API request/response types with serde roundtrip tests"
```

---

## Task 4: Implement health endpoint with router factory and test helpers

- [ ] Complete

**Files:**
- Modify: `crates/server/src/routes.rs`
- Modify: `crates/server/src/lib.rs`

### Step 1: Write the test

Add to `crates/server/src/routes.rs`:

```rust
use axum::extract::{Json, Path, Query, State};
use axum::http::StatusCode;
use axum::routing::{delete, get, post};
use axum::Router;

use crate::error::ServerError;
use crate::state::AppState;
use crate::types::HealthResponse;

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .with_state(state)
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

#[cfg(test)]
mod test_helpers {
    use std::collections::HashMap;
    use std::sync::Arc;

    use tokio::sync::{broadcast, RwLock};

    use graphirm_agent::AgentConfig;
    use graphirm_graph::GraphStore;
    use graphirm_llm::{LlmError, LlmMessage, LlmProvider, LlmResponse, ToolDefinition, Usage};
    use graphirm_tools::ToolRegistry;

    use crate::state::AppState;
    use crate::types::SseEvent;

    pub struct NoopProvider;

    #[async_trait::async_trait]
    impl LlmProvider for NoopProvider {
        async fn complete(
            &self,
            _messages: Vec<LlmMessage>,
            _tools: Vec<ToolDefinition>,
        ) -> Result<LlmResponse, LlmError> {
            Ok(LlmResponse {
                content: "noop response".to_string(),
                tool_calls: vec![],
                usage: Usage::default(),
            })
        }
    }

    pub fn test_app_state() -> AppState {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let (event_tx, _) = broadcast::channel::<SseEvent>(256);

        AppState {
            graph,
            llm: Arc::new(NoopProvider),
            tools: Arc::new(ToolRegistry::new()),
            event_tx,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            default_config: AgentConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use http::Request;
    use tower::ServiceExt;

    use test_helpers::test_app_state;

    #[tokio::test]
    async fn test_health_returns_ok() {
        let app = create_router(test_app_state());

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let health: HealthResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(health.status, "ok");
        assert!(!health.version.is_empty());
    }

    #[tokio::test]
    async fn test_unknown_route_returns_404() {
        let app = create_router(test_app_state());

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/nonexistent")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
```

### Step 2: Update lib.rs to add public `create_router` and `start_server`

```rust
pub mod error;
pub mod routes;
pub mod sdk;
pub mod sse;
pub mod state;
pub mod types;

pub use routes::create_router;
pub use state::AppState;
```

### Step 3: Verify

Run: `cargo test -p graphirm-server routes::tests -- --nocapture 2>&1`
Expected: `test result: ok. 2 passed; 0 failed`

### Step 4: Commit

```bash
git add crates/server/src/routes.rs crates/server/src/lib.rs
git commit -m "feat(server): implement health endpoint, router factory, and test helpers"
```

---

## Task 5: Implement session CRUD — create, get, list, delete

- [ ] Complete

**Files:**
- Modify: `crates/server/src/routes.rs`

### Step 1: Write the failing tests

Add to the `tests` module in `crates/server/src/routes.rs`:

```rust
    #[tokio::test]
    async fn test_create_session() {
        let state = test_app_state();
        let app = create_router(state.clone());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/sessions")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"agent": "coder", "model": "gpt-4o"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::CREATED);

        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let session: crate::types::SessionResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(session.agent, "coder");
        assert_eq!(session.model, "gpt-4o");
        assert_eq!(session.status, "idle");
        assert!(!session.id.is_empty());
    }

    #[tokio::test]
    async fn test_create_session_defaults() {
        let state = test_app_state();
        let app = create_router(state.clone());

        let response = app
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

        assert_eq!(response.status(), StatusCode::CREATED);

        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let session: crate::types::SessionResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(session.agent, "graphirm");
    }

    #[tokio::test]
    async fn test_get_session() {
        let state = test_app_state();
        let app = create_router(state.clone());

        // Create a session first
        let create_resp = app
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
        let body = axum::body::to_bytes(create_resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let created: crate::types::SessionResponse = serde_json::from_slice(&body).unwrap();

        // Get the session
        let get_resp = app
            .oneshot(
                Request::builder()
                    .uri(&format!("/api/sessions/{}", created.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(get_resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(get_resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let fetched: crate::types::SessionResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(fetched.id, created.id);
    }

    #[tokio::test]
    async fn test_get_nonexistent_session_returns_404() {
        let app = create_router(test_app_state());

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/sessions/nonexistent-id")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_list_sessions() {
        let state = test_app_state();
        let app = create_router(state.clone());

        // Create two sessions
        for _ in 0..2 {
            app.clone()
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
        }

        // List
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/sessions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let sessions: Vec<crate::types::SessionResponse> =
            serde_json::from_slice(&body).unwrap();
        assert_eq!(sessions.len(), 2);
    }

    #[tokio::test]
    async fn test_delete_session() {
        let state = test_app_state();
        let app = create_router(state.clone());

        // Create
        let create_resp = app
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
        let body = axum::body::to_bytes(create_resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let created: crate::types::SessionResponse = serde_json::from_slice(&body).unwrap();

        // Delete
        let del_resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri(&format!("/api/sessions/{}", created.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(del_resp.status(), StatusCode::NO_CONTENT);

        // Verify gone
        let get_resp = app
            .oneshot(
                Request::builder()
                    .uri(&format!("/api/sessions/{}", created.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(get_resp.status(), StatusCode::NOT_FOUND);
    }
```

### Step 2: Implement the session handlers

Add the handler functions above the test modules in `routes.rs`, and update `create_router`:

```rust
use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use graphirm_agent::{AgentConfig, Session};

use crate::state::{SessionHandle, SessionStatus};
use crate::types::{CreateSessionRequest, SessionResponse};

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/sessions", get(list_sessions).post(create_session))
        .route(
            "/api/sessions/:id",
            get(get_session).delete(delete_session),
        )
        .with_state(state)
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

async fn create_session(
    State(state): State<AppState>,
    Json(body): Json<CreateSessionRequest>,
) -> Result<(StatusCode, Json<SessionResponse>), ServerError> {
    let config = AgentConfig {
        name: body
            .agent
            .unwrap_or_else(|| state.default_config.name.clone()),
        model: body
            .model
            .unwrap_or_else(|| state.default_config.model.clone()),
        ..state.default_config.clone()
    };

    let session = Session::new(state.graph.clone(), config).map_err(ServerError::Agent)?;
    let session_id = session.id.to_string();
    let now = chrono::Utc::now();

    let handle = SessionHandle {
        session: Arc::new(session),
        signal: CancellationToken::new(),
        join_handle: None,
        status: SessionStatus::Idle,
        created_at: now,
    };

    let response = session_handle_to_response(&session_id, &handle);

    state.sessions.write().await.insert(session_id, handle);

    Ok((StatusCode::CREATED, Json(response)))
}

async fn get_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<SessionResponse>, ServerError> {
    let sessions = state.sessions.read().await;
    let handle = sessions
        .get(&id)
        .ok_or_else(|| ServerError::NotFound(format!("Session not found: {id}")))?;

    Ok(Json(session_handle_to_response(&id, handle)))
}

async fn list_sessions(
    State(state): State<AppState>,
) -> Result<Json<Vec<SessionResponse>>, ServerError> {
    let sessions = state.sessions.read().await;
    let responses: Vec<SessionResponse> = sessions
        .iter()
        .map(|(id, handle)| session_handle_to_response(id, handle))
        .collect();

    Ok(Json(responses))
}

async fn delete_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ServerError> {
    let mut sessions = state.sessions.write().await;
    let handle = sessions
        .remove(&id)
        .ok_or_else(|| ServerError::NotFound(format!("Session not found: {id}")))?;

    // Cancel any running agent
    handle.signal.cancel();

    Ok(StatusCode::NO_CONTENT)
}

fn session_handle_to_response(id: &str, handle: &SessionHandle) -> SessionResponse {
    SessionResponse {
        id: id.to_string(),
        agent: handle.session.agent_config.name.clone(),
        model: handle.session.agent_config.model.clone(),
        created_at: handle.created_at.to_rfc3339(),
        status: handle.status.as_str().to_string(),
    }
}
```

### Step 3: Run tests

Run: `cargo test -p graphirm-server routes::tests -- --nocapture 2>&1`
Expected: All session CRUD tests pass.

### Step 4: Commit

```bash
git add crates/server/src/routes.rs
git commit -m "feat(server): implement session CRUD — create, get, list, delete"
```

---

## Task 6: Implement prompt endpoint — start agent loop in background

- [ ] Complete

**Files:**
- Modify: `crates/server/src/routes.rs`

### Step 1: Write the failing test

Add to the `tests` module in `routes.rs`:

```rust
    #[tokio::test]
    async fn test_prompt_returns_202() {
        let state = test_app_state();
        let app = create_router(state.clone());

        // Create a session
        let create_resp = app
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
        let body = axum::body::to_bytes(create_resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let created: crate::types::SessionResponse = serde_json::from_slice(&body).unwrap();

        // Send a prompt
        let prompt_resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(&format!("/api/sessions/{}/prompt", created.id))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"content": "Hello!"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(prompt_resp.status(), StatusCode::ACCEPTED);

        // Give the background task a moment to run
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Session status should have changed
        let sessions = state.sessions.read().await;
        let handle = sessions.get(&created.id).unwrap();
        assert!(
            handle.status == SessionStatus::Running
                || handle.status == SessionStatus::Completed,
            "Expected Running or Completed, got {:?}",
            handle.status
        );
    }

    #[tokio::test]
    async fn test_prompt_nonexistent_session_returns_404() {
        let app = create_router(test_app_state());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/sessions/nonexistent/prompt")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"content": "Hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
```

### Step 2: Implement the prompt handler

Add these imports and the handler in `routes.rs`:

```rust
use graphirm_agent::{run_agent_loop, EventBus};
use crate::types::{PromptRequest, SseEvent};

async fn prompt_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<PromptRequest>,
) -> Result<StatusCode, ServerError> {
    // Scope the write lock — release before spawning tasks
    let (session, cancel, event_rx) = {
        let mut sessions = state.sessions.write().await;
        let handle = sessions
            .get_mut(&id)
            .ok_or_else(|| ServerError::NotFound(format!("Session not found: {id}")))?;

        if handle.status == SessionStatus::Running {
            return Err(ServerError::BadRequest(
                "Session is already running".to_string(),
            ));
        }

        // Add user message to graph
        handle
            .session
            .add_user_message(&body.content)
            .map_err(ServerError::Agent)?;

        // Create fresh cancellation token
        let cancel = CancellationToken::new();
        handle.signal = cancel.clone();
        handle.status = SessionStatus::Running;

        // Create event bus and subscribe before moving
        let mut bus = EventBus::new();
        let rx = bus.subscribe();

        let session = handle.session.clone();

        // Store the bus in an Arc for the agent loop
        (session, cancel, (Arc::new(bus), rx))
    }; // write lock released here

    let (bus, mut rx) = event_rx;
    let event_tx = state.event_tx.clone();
    let session_id = id.clone();

    // Relay agent events to broadcast channel for SSE clients
    tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            let sse_event = agent_event_to_sse(&session_id, &event);
            let _ = event_tx.send(sse_event);
        }
    });

    // Spawn the agent loop as a background task
    let llm = state.llm.clone();
    let tools = state.tools.clone();
    let sessions = state.sessions.clone();
    let session_id = id.clone();

    let join_handle = tokio::spawn(async move {
        let result = run_agent_loop(&session, llm.as_ref(), &tools, &bus, &cancel).await;

        // Update session status when loop finishes
        let mut sessions = sessions.write().await;
        if let Some(handle) = sessions.get_mut(&session_id) {
            handle.status = match &result {
                Ok(()) => SessionStatus::Completed,
                Err(_) => SessionStatus::Failed,
            };
            handle.join_handle = None;
        }

        result
    });

    // Store the join handle
    {
        let mut sessions = state.sessions.write().await;
        if let Some(handle) = sessions.get_mut(&id) {
            handle.join_handle = Some(join_handle);
        }
    }

    Ok(StatusCode::ACCEPTED)
}

fn agent_event_to_sse(
    session_id: &str,
    event: &graphirm_agent::AgentEvent,
) -> SseEvent {
    use graphirm_agent::AgentEvent;

    let (event_type, data) = match event {
        AgentEvent::AgentStart { agent_id } => (
            "agent_start",
            serde_json::json!({ "agent_id": agent_id.to_string() }),
        ),
        AgentEvent::AgentEnd {
            agent_id,
            node_ids,
        } => (
            "agent_end",
            serde_json::json!({
                "agent_id": agent_id.to_string(),
                "node_count": node_ids.len(),
            }),
        ),
        AgentEvent::TurnStart { turn_index } => (
            "turn_start",
            serde_json::json!({ "turn_index": turn_index }),
        ),
        AgentEvent::TurnEnd {
            response_id,
            tool_result_ids,
        } => (
            "turn_end",
            serde_json::json!({
                "response_id": response_id.to_string(),
                "tool_result_count": tool_result_ids.len(),
            }),
        ),
        AgentEvent::MessageEnd { node_id } => (
            "message_end",
            serde_json::json!({ "node_id": node_id.to_string() }),
        ),
        AgentEvent::ToolStart {
            node_id,
            tool_name,
        } => (
            "tool_start",
            serde_json::json!({
                "node_id": node_id.to_string(),
                "tool_name": tool_name,
            }),
        ),
        AgentEvent::ToolEnd { node_id, is_error } => (
            "tool_end",
            serde_json::json!({
                "node_id": node_id.to_string(),
                "is_error": is_error,
            }),
        ),
        _ => (
            "unknown",
            serde_json::json!({ "debug": format!("{:?}", event) }),
        ),
    };

    SseEvent {
        session_id: session_id.to_string(),
        event_type: event_type.to_string(),
        data,
    }
}
```

### Step 3: Update create_router to include the prompt route

Update `create_router` to:

```rust
pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/sessions", get(list_sessions).post(create_session))
        .route(
            "/api/sessions/:id",
            get(get_session).delete(delete_session),
        )
        .route("/api/sessions/:id/prompt", post(prompt_session))
        .with_state(state)
}
```

### Step 4: Verify

Run: `cargo test -p graphirm-server routes::tests::test_prompt -- --nocapture 2>&1`
Expected: Both prompt tests pass.

### Step 5: Commit

```bash
git add crates/server/src/routes.rs
git commit -m "feat(server): implement prompt endpoint — spawns agent loop in background"
```

---

## Task 7: Implement abort endpoint — cancel running agent via CancellationToken

- [ ] Complete

**Files:**
- Modify: `crates/server/src/routes.rs`

### Step 1: Write the failing test

Add to the `tests` module:

```rust
    #[tokio::test]
    async fn test_abort_running_session() {
        let state = test_app_state();
        let app = create_router(state.clone());

        // Create session
        let create_resp = app
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
        let body = axum::body::to_bytes(create_resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let created: crate::types::SessionResponse = serde_json::from_slice(&body).unwrap();

        // Send prompt to start agent
        app.clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(&format!("/api/sessions/{}/prompt", created.id))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"content": "Do something"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Abort
        let abort_resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(&format!("/api/sessions/{}/abort", created.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(abort_resp.status(), StatusCode::NO_CONTENT);

        // Give the background task time to process cancellation
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        // Verify the cancellation token was triggered
        let sessions = state.sessions.read().await;
        let handle = sessions.get(&created.id).unwrap();
        assert!(handle.signal.is_cancelled());
    }

    #[tokio::test]
    async fn test_abort_nonexistent_session_returns_404() {
        let app = create_router(test_app_state());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/sessions/nonexistent/abort")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
```

### Step 2: Implement the abort handler

```rust
async fn abort_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ServerError> {
    let mut sessions = state.sessions.write().await;
    let handle = sessions
        .get_mut(&id)
        .ok_or_else(|| ServerError::NotFound(format!("Session not found: {id}")))?;

    // Cancel the running agent
    handle.signal.cancel();

    // If there's a join handle, take it and drop it (the task will finish on its own)
    if let Some(jh) = handle.join_handle.take() {
        // Spawn a cleanup task that awaits completion with a timeout
        tokio::spawn(async move {
            let _ = tokio::time::timeout(std::time::Duration::from_secs(5), jh).await;
        });
    }

    Ok(StatusCode::NO_CONTENT)
}
```

### Step 3: Update create_router

```rust
pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/sessions", get(list_sessions).post(create_session))
        .route(
            "/api/sessions/:id",
            get(get_session).delete(delete_session),
        )
        .route("/api/sessions/:id/prompt", post(prompt_session))
        .route("/api/sessions/:id/abort", post(abort_session))
        .with_state(state)
}
```

### Step 4: Verify

Run: `cargo test -p graphirm-server routes::tests::test_abort -- --nocapture 2>&1`
Expected: Both abort tests pass.

### Step 5: Commit

```bash
git add crates/server/src/routes.rs
git commit -m "feat(server): implement abort endpoint — cancels running agent via CancellationToken"
```

---

## Task 8: Implement messages and children endpoints

- [ ] Complete

**Files:**
- Modify: `crates/server/src/routes.rs`

### Step 1: Write the failing tests

Add to the `tests` module:

```rust
    #[tokio::test]
    async fn test_get_messages_empty_session() {
        let state = test_app_state();
        let app = create_router(state.clone());

        // Create session
        let create_resp = app
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
        let body = axum::body::to_bytes(create_resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let created: crate::types::SessionResponse = serde_json::from_slice(&body).unwrap();

        // Get messages
        let msg_resp = app
            .oneshot(
                Request::builder()
                    .uri(&format!("/api/sessions/{}/messages", created.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(msg_resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(msg_resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let messages: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        assert!(messages.is_empty());
    }

    #[tokio::test]
    async fn test_get_messages_with_prompt() {
        let state = test_app_state();
        let app = create_router(state.clone());

        // Create session
        let create_resp = app
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
        let body = axum::body::to_bytes(create_resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let created: crate::types::SessionResponse = serde_json::from_slice(&body).unwrap();

        // Send prompt (adds a user message)
        app.clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(&format!("/api/sessions/{}/prompt", created.id))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"content": "Hello!"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Wait for agent loop to finish
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        // Get messages — should include user message + assistant response
        let msg_resp = app
            .oneshot(
                Request::builder()
                    .uri(&format!("/api/sessions/{}/messages", created.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(msg_resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(msg_resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let messages: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        // At minimum, the user message should be there
        assert!(!messages.is_empty());
    }

    #[tokio::test]
    async fn test_get_children_returns_empty_list() {
        let state = test_app_state();
        let app = create_router(state.clone());

        // Create session
        let create_resp = app
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
        let body = axum::body::to_bytes(create_resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let created: crate::types::SessionResponse = serde_json::from_slice(&body).unwrap();

        // Get children (stub — always empty until Phase 5)
        let resp = app
            .oneshot(
                Request::builder()
                    .uri(&format!("/api/sessions/{}/children", created.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let children: Vec<crate::types::SessionResponse> =
            serde_json::from_slice(&body).unwrap();
        assert!(children.is_empty());
    }
```

### Step 2: Implement the handlers

```rust
use graphirm_graph::{Direction, EdgeType, NodeId, NodeType};

async fn get_messages(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Vec<graphirm_graph::GraphNode>>, ServerError> {
    let sessions = state.sessions.read().await;
    let handle = sessions
        .get(&id)
        .ok_or_else(|| ServerError::NotFound(format!("Session not found: {id}")))?;

    let all_neighbors = state
        .graph
        .neighbors(
            &handle.session.id,
            Some(EdgeType::Produces),
            Direction::Outgoing,
        )
        .map_err(ServerError::Graph)?;

    let messages: Vec<graphirm_graph::GraphNode> = all_neighbors
        .into_iter()
        .filter(|n| matches!(n.node_type, NodeType::Interaction(_)))
        .collect();

    Ok(Json(messages))
}

async fn get_children(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Vec<SessionResponse>>, ServerError> {
    let sessions = state.sessions.read().await;
    let _ = sessions
        .get(&id)
        .ok_or_else(|| ServerError::NotFound(format!("Session not found: {id}")))?;

    // Stub: subagent sessions are a Phase 5 feature
    Ok(Json(vec![]))
}
```

### Step 3: Update create_router

```rust
pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/sessions", get(list_sessions).post(create_session))
        .route(
            "/api/sessions/:id",
            get(get_session).delete(delete_session),
        )
        .route("/api/sessions/:id/prompt", post(prompt_session))
        .route("/api/sessions/:id/abort", post(abort_session))
        .route("/api/sessions/:id/messages", get(get_messages))
        .route("/api/sessions/:id/children", get(get_children))
        .with_state(state)
}
```

### Step 4: Verify

Run: `cargo test -p graphirm-server routes::tests::test_get_messages -- --nocapture 2>&1`
Expected: All message/children tests pass.

Run: `cargo test -p graphirm-server routes::tests::test_get_children -- --nocapture 2>&1`
Expected: Pass.

### Step 5: Commit

```bash
git add crates/server/src/routes.rs
git commit -m "feat(server): implement messages and children endpoints"
```

---

## Task 9: Implement graph query endpoints — full graph, node, subgraph, tasks, knowledge

- [ ] Complete

**Files:**
- Modify: `crates/server/src/routes.rs`

### Step 1: Write the failing tests

Add to the `tests` module:

```rust
    #[tokio::test]
    async fn test_get_session_graph() {
        let state = test_app_state();
        let app = create_router(state.clone());

        // Create session
        let create_resp = app
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
        let body = axum::body::to_bytes(create_resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let created: crate::types::SessionResponse = serde_json::from_slice(&body).unwrap();

        // Get graph
        let graph_resp = app
            .oneshot(
                Request::builder()
                    .uri(&format!("/api/graph/{}", created.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(graph_resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(graph_resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let graph: crate::types::GraphResponse = serde_json::from_slice(&body).unwrap();
        // At minimum, the agent node should be present
        assert!(!graph.nodes.is_empty());
    }

    #[tokio::test]
    async fn test_get_graph_node() {
        let state = test_app_state();
        let app = create_router(state.clone());

        // Create session (creates an Agent node in the graph)
        let create_resp = app
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
        let body = axum::body::to_bytes(create_resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let created: crate::types::SessionResponse = serde_json::from_slice(&body).unwrap();

        // Get the agent node directly
        let sessions = state.sessions.read().await;
        let handle = sessions.get(&created.id).unwrap();
        let agent_node_id = handle.session.id.to_string();
        drop(sessions);

        let node_resp = app
            .oneshot(
                Request::builder()
                    .uri(&format!("/api/graph/{}/node/{}", created.id, agent_node_id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(node_resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_graph_node_not_found() {
        let state = test_app_state();
        let app = create_router(state.clone());

        // Create session
        let create_resp = app
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
        let body = axum::body::to_bytes(create_resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let created: crate::types::SessionResponse = serde_json::from_slice(&body).unwrap();

        // Try to get nonexistent node
        let resp = app
            .oneshot(
                Request::builder()
                    .uri(&format!("/api/graph/{}/node/nonexistent", created.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_get_subgraph() {
        let state = test_app_state();
        let app = create_router(state.clone());

        // Create session
        let create_resp = app
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
        let body = axum::body::to_bytes(create_resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let created: crate::types::SessionResponse = serde_json::from_slice(&body).unwrap();

        let sessions = state.sessions.read().await;
        let agent_node_id = sessions
            .get(&created.id)
            .unwrap()
            .session
            .id
            .to_string();
        drop(sessions);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri(&format!(
                        "/api/graph/{}/subgraph/{}?depth=2",
                        created.id, agent_node_id
                    ))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let graph: crate::types::GraphResponse = serde_json::from_slice(&body).unwrap();
        assert!(!graph.nodes.is_empty());
    }

    #[tokio::test]
    async fn test_get_tasks_empty() {
        let state = test_app_state();
        let app = create_router(state.clone());

        // Create session
        let create_resp = app
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
        let body = axum::body::to_bytes(create_resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let created: crate::types::SessionResponse = serde_json::from_slice(&body).unwrap();

        let resp = app
            .oneshot(
                Request::builder()
                    .uri(&format!("/api/graph/{}/tasks", created.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let tasks: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        assert!(tasks.is_empty());
    }

    #[tokio::test]
    async fn test_get_knowledge_empty() {
        let state = test_app_state();
        let app = create_router(state.clone());

        // Create session
        let create_resp = app
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
        let body = axum::body::to_bytes(create_resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let created: crate::types::SessionResponse = serde_json::from_slice(&body).unwrap();

        let resp = app
            .oneshot(
                Request::builder()
                    .uri(&format!("/api/graph/{}/knowledge", created.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let knowledge: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        assert!(knowledge.is_empty());
    }
```

### Step 2: Implement the graph query handlers

```rust
use crate::types::{GraphResponse, SubgraphQuery};

async fn get_session_graph(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Result<Json<GraphResponse>, ServerError> {
    let sessions = state.sessions.read().await;
    let handle = sessions
        .get(&session_id)
        .ok_or_else(|| ServerError::NotFound(format!("Session not found: {session_id}")))?;

    let (nodes, edges) = state
        .graph
        .subgraph(&handle.session.id, 10)
        .map_err(ServerError::Graph)?;

    Ok(Json(GraphResponse { nodes, edges }))
}

async fn get_graph_node(
    State(state): State<AppState>,
    Path((session_id, node_id)): Path<(String, String)>,
) -> Result<Json<graphirm_graph::GraphNode>, ServerError> {
    let sessions = state.sessions.read().await;
    let _ = sessions
        .get(&session_id)
        .ok_or_else(|| ServerError::NotFound(format!("Session not found: {session_id}")))?;

    let node = state
        .graph
        .get_node(&NodeId::from(node_id.as_str()))
        .map_err(|_| ServerError::NotFound(format!("Node not found: {node_id}")))?;

    Ok(Json(node))
}

async fn get_subgraph(
    State(state): State<AppState>,
    Path((session_id, node_id)): Path<(String, String)>,
    Query(query): Query<SubgraphQuery>,
) -> Result<Json<GraphResponse>, ServerError> {
    let sessions = state.sessions.read().await;
    let _ = sessions
        .get(&session_id)
        .ok_or_else(|| ServerError::NotFound(format!("Session not found: {session_id}")))?;

    let depth = query.depth.unwrap_or(3);
    let (nodes, edges) = state
        .graph
        .subgraph(&NodeId::from(node_id.as_str()), depth)
        .map_err(ServerError::Graph)?;

    Ok(Json(GraphResponse { nodes, edges }))
}

async fn get_tasks(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Result<Json<Vec<graphirm_graph::GraphNode>>, ServerError> {
    let sessions = state.sessions.read().await;
    let handle = sessions
        .get(&session_id)
        .ok_or_else(|| ServerError::NotFound(format!("Session not found: {session_id}")))?;

    let all_neighbors = state
        .graph
        .neighbors(
            &handle.session.id,
            Some(EdgeType::Produces),
            Direction::Outgoing,
        )
        .map_err(ServerError::Graph)?;

    let tasks: Vec<graphirm_graph::GraphNode> = all_neighbors
        .into_iter()
        .filter(|n| matches!(n.node_type, NodeType::Task(_)))
        .collect();

    Ok(Json(tasks))
}

async fn get_knowledge(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Result<Json<Vec<graphirm_graph::GraphNode>>, ServerError> {
    let sessions = state.sessions.read().await;
    let handle = sessions
        .get(&session_id)
        .ok_or_else(|| ServerError::NotFound(format!("Session not found: {session_id}")))?;

    let all_neighbors = state
        .graph
        .neighbors(
            &handle.session.id,
            Some(EdgeType::Produces),
            Direction::Outgoing,
        )
        .map_err(ServerError::Graph)?;

    let knowledge: Vec<graphirm_graph::GraphNode> = all_neighbors
        .into_iter()
        .filter(|n| matches!(n.node_type, NodeType::Knowledge(_)))
        .collect();

    Ok(Json(knowledge))
}
```

### Step 3: Update create_router with all graph routes

```rust
pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/api/health", get(health))
        // Session management
        .route("/api/sessions", get(list_sessions).post(create_session))
        .route(
            "/api/sessions/:id",
            get(get_session).delete(delete_session),
        )
        .route("/api/sessions/:id/prompt", post(prompt_session))
        .route("/api/sessions/:id/abort", post(abort_session))
        .route("/api/sessions/:id/messages", get(get_messages))
        .route("/api/sessions/:id/children", get(get_children))
        // Graph queries
        .route("/api/graph/:session_id", get(get_session_graph))
        .route(
            "/api/graph/:session_id/node/:node_id",
            get(get_graph_node),
        )
        .route(
            "/api/graph/:session_id/subgraph/:node_id",
            get(get_subgraph),
        )
        .route("/api/graph/:session_id/tasks", get(get_tasks))
        .route("/api/graph/:session_id/knowledge", get(get_knowledge))
        .with_state(state)
}
```

### Step 4: Verify

Run: `cargo test -p graphirm-server routes::tests -- --nocapture 2>&1`
Expected: All graph query tests pass.

### Step 5: Commit

```bash
git add crates/server/src/routes.rs
git commit -m "feat(server): implement graph query endpoints — full graph, node, subgraph, tasks, knowledge"
```

---

## Task 10: Implement SSE streaming — global event stream

- [ ] Complete

**Files:**
- Modify: `crates/server/src/sse.rs`

### Step 1: Write the SSE handler and unit tests

Replace `crates/server/src/sse.rs` with:

```rust
use std::convert::Infallible;
use std::time::Duration;

use axum::extract::{Path, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

use crate::state::AppState;
use crate::types::SseEvent;

/// Global SSE stream — emits all agent events from all sessions.
pub async fn sse_handler(
    State(state): State<AppState>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let rx = state.event_tx.subscribe();

    let stream = BroadcastStream::new(rx)
        .filter_map(|result| result.ok())
        .map(|event: SseEvent| -> Result<Event, Infallible> {
            let data = serde_json::to_string(&event).unwrap_or_default();
            Ok(Event::default()
                .event(event.event_type)
                .data(data))
        });

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("heartbeat"),
    )
}

/// Session-scoped SSE stream — emits events filtered to a single session.
pub async fn sse_session_handler(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let rx = state.event_tx.subscribe();

    let stream = BroadcastStream::new(rx)
        .filter_map(|result| result.ok())
        .filter(move |event: &SseEvent| event.session_id == session_id)
        .map(|event: SseEvent| -> Result<Event, Infallible> {
            let data = serde_json::to_string(&event).unwrap_or_default();
            Ok(Event::default()
                .event(event.event_type)
                .data(data))
        });

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("heartbeat"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::broadcast;

    #[tokio::test]
    async fn test_broadcast_channel_delivers_events() {
        let (tx, _) = broadcast::channel::<SseEvent>(256);
        let mut rx = tx.subscribe();

        let event = SseEvent {
            session_id: "s1".to_string(),
            event_type: "turn_start".to_string(),
            data: serde_json::json!({"turn_index": 0}),
        };

        tx.send(event.clone()).unwrap();

        let received = rx.recv().await.unwrap();
        assert_eq!(received.session_id, "s1");
        assert_eq!(received.event_type, "turn_start");
    }

    #[tokio::test]
    async fn test_broadcast_multiple_subscribers() {
        let (tx, _) = broadcast::channel::<SseEvent>(256);
        let mut rx1 = tx.subscribe();
        let mut rx2 = tx.subscribe();

        let event = SseEvent {
            session_id: "s1".to_string(),
            event_type: "agent_start".to_string(),
            data: serde_json::json!({}),
        };

        tx.send(event).unwrap();

        let e1 = rx1.recv().await.unwrap();
        let e2 = rx2.recv().await.unwrap();
        assert_eq!(e1.event_type, "agent_start");
        assert_eq!(e2.event_type, "agent_start");
    }

    #[tokio::test]
    async fn test_broadcast_stream_to_sse_events() {
        let (tx, _) = broadcast::channel::<SseEvent>(256);
        let rx = tx.subscribe();

        tx.send(SseEvent {
            session_id: "s1".to_string(),
            event_type: "turn_start".to_string(),
            data: serde_json::json!({"turn_index": 0}),
        })
        .unwrap();

        tx.send(SseEvent {
            session_id: "s2".to_string(),
            event_type: "tool_start".to_string(),
            data: serde_json::json!({"tool_name": "bash"}),
        })
        .unwrap();

        // Drop the sender so the stream ends
        drop(tx);

        let stream = BroadcastStream::new(rx)
            .filter_map(|r| r.ok())
            .map(|event: SseEvent| -> Result<Event, Infallible> {
                let data = serde_json::to_string(&event).unwrap_or_default();
                Ok(Event::default().event(event.event_type).data(data))
            });

        let events: Vec<_> = stream.collect().await;
        assert_eq!(events.len(), 2);
        assert!(events[0].is_ok());
        assert!(events[1].is_ok());
    }

    #[tokio::test]
    async fn test_session_filtered_stream() {
        let (tx, _) = broadcast::channel::<SseEvent>(256);
        let rx = tx.subscribe();
        let target_session = "s1".to_string();

        tx.send(SseEvent {
            session_id: "s1".to_string(),
            event_type: "turn_start".to_string(),
            data: serde_json::json!({}),
        })
        .unwrap();

        tx.send(SseEvent {
            session_id: "s2".to_string(),
            event_type: "turn_start".to_string(),
            data: serde_json::json!({}),
        })
        .unwrap();

        tx.send(SseEvent {
            session_id: "s1".to_string(),
            event_type: "agent_end".to_string(),
            data: serde_json::json!({}),
        })
        .unwrap();

        drop(tx);

        let stream = BroadcastStream::new(rx)
            .filter_map(|r| r.ok())
            .filter(move |e: &SseEvent| e.session_id == target_session);

        let events: Vec<SseEvent> = stream.collect().await;
        assert_eq!(events.len(), 2);
        assert!(events.iter().all(|e| e.session_id == "s1"));
    }
}
```

### Step 2: Verify

Run: `cargo test -p graphirm-server sse::tests -- --nocapture 2>&1`
Expected: `test result: ok. 4 passed; 0 failed`

### Step 3: Commit

```bash
git add crates/server/src/sse.rs
git commit -m "feat(server): implement SSE streaming with broadcast channel and session filtering"
```

---

## Task 11: Wire SSE routes into the router

- [ ] Complete

**Files:**
- Modify: `crates/server/src/routes.rs`

### Step 1: Write the test

Add to the `tests` module in `routes.rs`:

```rust
    #[tokio::test]
    async fn test_sse_endpoint_returns_event_stream_content_type() {
        let state = test_app_state();
        let event_tx = state.event_tx.clone();
        let app = create_router(state);

        // Send an event so the stream has data
        event_tx
            .send(crate::types::SseEvent {
                session_id: "s1".to_string(),
                event_type: "turn_start".to_string(),
                data: serde_json::json!({"turn_index": 0}),
            })
            .unwrap();

        // The SSE endpoint starts a long-lived connection.
        // We test that the response has the correct content type.
        // Use a timeout to avoid hanging.
        let result = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            app.oneshot(
                Request::builder()
                    .uri("/api/events")
                    .body(Body::empty())
                    .unwrap(),
            ),
        )
        .await;

        // The oneshot should return a response (SSE starts streaming)
        match result {
            Ok(Ok(response)) => {
                assert_eq!(response.status(), StatusCode::OK);
                let content_type = response
                    .headers()
                    .get("content-type")
                    .unwrap()
                    .to_str()
                    .unwrap();
                assert!(
                    content_type.contains("text/event-stream"),
                    "Expected text/event-stream, got: {}",
                    content_type
                );
            }
            Ok(Err(e)) => panic!("Request error: {e}"),
            Err(_) => {
                // Timeout is acceptable — SSE connections are long-lived
            }
        }
    }
```

### Step 2: Add SSE routes to create_router

Update `create_router`:

```rust
use crate::sse::{sse_handler, sse_session_handler};

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/api/health", get(health))
        // Session management
        .route("/api/sessions", get(list_sessions).post(create_session))
        .route(
            "/api/sessions/:id",
            get(get_session).delete(delete_session),
        )
        .route("/api/sessions/:id/prompt", post(prompt_session))
        .route("/api/sessions/:id/abort", post(abort_session))
        .route("/api/sessions/:id/messages", get(get_messages))
        .route("/api/sessions/:id/children", get(get_children))
        // Graph queries
        .route("/api/graph/:session_id", get(get_session_graph))
        .route(
            "/api/graph/:session_id/node/:node_id",
            get(get_graph_node),
        )
        .route(
            "/api/graph/:session_id/subgraph/:node_id",
            get(get_subgraph),
        )
        .route("/api/graph/:session_id/tasks", get(get_tasks))
        .route("/api/graph/:session_id/knowledge", get(get_knowledge))
        // SSE
        .route("/api/events", get(sse_handler))
        .route("/api/events/:session_id", get(sse_session_handler))
        .with_state(state)
}
```

### Step 3: Verify

Run: `cargo test -p graphirm-server routes::tests::test_sse -- --nocapture 2>&1`
Expected: Test passes (or times out gracefully, which is acceptable for SSE).

### Step 4: Commit

```bash
git add crates/server/src/routes.rs
git commit -m "feat(server): wire SSE routes — global and session-scoped event streams"
```

---

## Task 12: Add middleware — CORS, request tracing, error handling

- [ ] Complete

**Files:**
- Modify: `crates/server/src/routes.rs`

### Step 1: Write the test

Add to the `tests` module:

```rust
    #[tokio::test]
    async fn test_cors_headers_present() {
        let state = test_app_state();
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("OPTIONS")
                    .uri("/api/health")
                    .header("origin", "http://localhost:3001")
                    .header("access-control-request-method", "GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        // CORS preflight should return 200 with the right headers
        assert!(
            response.status() == StatusCode::OK
                || response.status() == StatusCode::NO_CONTENT,
            "Expected 200 or 204 for CORS preflight, got: {}",
            response.status()
        );

        let headers = response.headers();
        assert!(
            headers.contains_key("access-control-allow-origin"),
            "Missing access-control-allow-origin header"
        );
    }
```

### Step 2: Add middleware layers to create_router

Update the imports and `create_router`:

```rust
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

pub fn create_router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/api/health", get(health))
        // Session management
        .route("/api/sessions", get(list_sessions).post(create_session))
        .route(
            "/api/sessions/:id",
            get(get_session).delete(delete_session),
        )
        .route("/api/sessions/:id/prompt", post(prompt_session))
        .route("/api/sessions/:id/abort", post(abort_session))
        .route("/api/sessions/:id/messages", get(get_messages))
        .route("/api/sessions/:id/children", get(get_children))
        // Graph queries
        .route("/api/graph/:session_id", get(get_session_graph))
        .route(
            "/api/graph/:session_id/node/:node_id",
            get(get_graph_node),
        )
        .route(
            "/api/graph/:session_id/subgraph/:node_id",
            get(get_subgraph),
        )
        .route("/api/graph/:session_id/tasks", get(get_tasks))
        .route("/api/graph/:session_id/knowledge", get(get_knowledge))
        // SSE
        .route("/api/events", get(sse_handler))
        .route("/api/events/:session_id", get(sse_session_handler))
        // Middleware
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
```

### Step 3: Verify

Run: `cargo test -p graphirm-server routes::tests::test_cors -- --nocapture 2>&1`
Expected: CORS test passes.

Run: `cargo test -p graphirm-server -- --nocapture 2>&1`
Expected: All tests pass.

### Step 4: Commit

```bash
git add crates/server/src/routes.rs
git commit -m "feat(server): add CORS (permissive) and request tracing middleware"
```

---

## Task 13: Wire into main.rs — `graphirm serve` subcommand

- [ ] Complete

**Files:**
- Modify: `src/main.rs`
- Modify: `crates/server/src/lib.rs`

### Step 1: Add `start_server` to the server crate

Update `crates/server/src/lib.rs`:

```rust
pub mod error;
pub mod routes;
pub mod sdk;
pub mod sse;
pub mod state;
pub mod types;

pub use routes::create_router;
pub use state::AppState;

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{broadcast, RwLock};
use tracing::info;

use graphirm_agent::AgentConfig;
use graphirm_graph::GraphStore;
use graphirm_llm::LlmProvider;
use graphirm_tools::ToolRegistry;

use crate::types::SseEvent;

pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3000,
        }
    }
}

pub async fn start_server(
    graph: Arc<GraphStore>,
    llm: Arc<dyn LlmProvider>,
    tools: Arc<ToolRegistry>,
    agent_config: AgentConfig,
    server_config: ServerConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let (event_tx, _) = broadcast::channel::<SseEvent>(1024);

    let state = AppState {
        graph,
        llm,
        tools,
        event_tx,
        sessions: Arc::new(RwLock::new(HashMap::new())),
        default_config: agent_config,
    };

    let app = create_router(state);
    let addr = format!("{}:{}", server_config.host, server_config.port);

    info!("Starting Graphirm server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Server shutdown complete");
    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install Ctrl+C handler");
    info!("Shutdown signal received");
}
```

### Step 2: Add `Serve` subcommand to main.rs

Update `src/main.rs` to add the `serve` subcommand:

```rust
mod error;

use clap::{Parser, Subcommand};
use error::GraphirmError;

#[derive(Parser)]
#[command(name = "graphirm")]
#[command(version, about = "Graph-native coding agent")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start an interactive chat session
    Chat {
        /// Resume an existing session by ID
        #[arg(short, long)]
        session: Option<String>,

        /// Model to use (e.g., "claude-sonnet-4-20250514")
        #[arg(short, long)]
        model: Option<String>,
    },

    /// Start the HTTP API server
    Serve {
        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to listen on
        #[arg(short, long, default_value = "3000")]
        port: u16,

        /// Path to the graph database
        #[arg(long, default_value = "~/.graphirm/graph.db")]
        db: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), GraphirmError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Chat { session, model } => {
            tracing::info!(
                session = session.as_deref().unwrap_or("new"),
                model = model.as_deref().unwrap_or("default"),
                "Starting chat session"
            );
            println!("graphirm chat — not yet implemented");
        }
        Commands::Serve { host, port, db } => {
            tracing::info!(host = %host, port = port, db = %db, "Starting HTTP server");
            println!("graphirm serve — not yet implemented (requires LLM provider setup)");
            // Full implementation requires LLM provider initialization from config.
            // For now, print the bind info. The actual wiring:
            //
            // let graph = Arc::new(GraphStore::open(&db)?);
            // let llm: Arc<dyn LlmProvider> = /* create from config */;
            // let tools = Arc::new(ToolRegistry::new());
            // let agent_config = AgentConfig::default();
            // let server_config = graphirm_server::ServerConfig { host, port };
            // graphirm_server::start_server(graph, llm, tools, agent_config, server_config).await?;
        }
    }

    Ok(())
}
```

### Step 3: Verify

Run: `cargo build 2>&1`
Expected: Full workspace compiles.

Run: `cargo run -- serve --help 2>&1`
Expected: Shows help text with `--host`, `--port`, `--db` options.

Run: `cargo run -- serve 2>&1`
Expected: Prints "graphirm serve — not yet implemented" and exits.

### Step 4: Commit

```bash
git add src/main.rs crates/server/src/lib.rs
git commit -m "feat(server): add 'graphirm serve' CLI subcommand and start_server() entry point"
```

---

## Task 14: Integration test — full lifecycle

- [ ] Complete

**Files:**
- Create: `crates/server/tests/integration.rs`

### Step 1: Write the integration test

Create `crates/server/tests/integration.rs`:

```rust
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tokio::sync::{broadcast, RwLock};
use tower::ServiceExt;

use graphirm_agent::AgentConfig;
use graphirm_graph::GraphStore;
use graphirm_llm::{LlmError, LlmMessage, LlmProvider, LlmResponse, ToolDefinition, Usage};
use graphirm_server::types::{
    GraphResponse, HealthResponse, SessionResponse, SseEvent,
};
use graphirm_server::{create_router, AppState};
use graphirm_tools::ToolRegistry;

struct MockProvider;

#[async_trait::async_trait]
impl LlmProvider for MockProvider {
    async fn complete(
        &self,
        _messages: Vec<LlmMessage>,
        _tools: Vec<ToolDefinition>,
    ) -> Result<LlmResponse, LlmError> {
        Ok(LlmResponse {
            content: "I processed your request.".to_string(),
            tool_calls: vec![],
            usage: Usage {
                input_tokens: 50,
                output_tokens: 10,
            },
        })
    }
}

fn test_app_state() -> AppState {
    let graph = Arc::new(GraphStore::open_memory().unwrap());
    let (event_tx, _) = broadcast::channel::<SseEvent>(256);

    AppState {
        graph,
        llm: Arc::new(MockProvider),
        tools: Arc::new(ToolRegistry::new()),
        event_tx,
        sessions: Arc::new(RwLock::new(HashMap::new())),
        default_config: AgentConfig::default(),
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
    assert_eq!(session.status, "idle");

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
                .uri(&format!("/api/sessions/{session_id}"))
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
                .uri(&format!("/api/sessions/{session_id}/prompt"))
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
    // The mock provider returns a text response (no tool calls),
    // so we expect: agent_start, turn_start, message_end, turn_end, agent_end
    assert!(
        !events.is_empty(),
        "Expected SSE events to be broadcast during agent loop"
    );
    let event_types: Vec<&str> = events.iter().map(|e| e.event_type.as_str()).collect();
    assert!(
        event_types.contains(&"agent_start"),
        "Missing agent_start event. Got: {:?}",
        event_types
    );
    assert!(
        event_types.contains(&"agent_end"),
        "Missing agent_end event. Got: {:?}",
        event_types
    );

    // 8. Get messages — should have user message + assistant response
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(&format!("/api/sessions/{session_id}/messages"))
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
                .uri(&format!("/api/graph/{session_id}"))
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
    assert!(
        !graph.edges.is_empty(),
        "Expected edges between agent and messages"
    );

    // 10. Delete session
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(&format!("/api/sessions/{session_id}"))
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
                .uri(&format!("/api/sessions/{session_id}"))
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
                        .body(Body::from(format!(
                            r#"{{"agent": "agent-{i}"}}"#
                        )))
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

    // Verify all 5 exist
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

    // All IDs are unique
    let unique: std::collections::HashSet<_> = ids.iter().collect();
    assert_eq!(unique.len(), 5);
}
```

### Step 2: Verify

Run: `cargo test -p graphirm-server --test integration -- --nocapture 2>&1`
Expected: `test result: ok. 2 passed; 0 failed`

### Step 3: Run full test suite

Run: `cargo test -p graphirm-server -- --nocapture 2>&1`
Expected: All tests pass (unit + integration).

### Step 4: Run clippy and fmt

Run: `cargo clippy -p graphirm-server --all-targets 2>&1`
Expected: No errors. Fix any warnings.

Run: `cargo fmt -p graphirm-server -- --check 2>&1`
Expected: No formatting issues.

### Step 5: Run full workspace build

Run: `cargo build 2>&1`
Expected: Full workspace compiles.

### Step 6: Commit

```bash
git add crates/server/tests/integration.rs
git commit -m "test(server): add integration test — full lifecycle: create, prompt, SSE events, graph query, delete"
```

---

## Final Checklist

After all 14 tasks, verify:

```bash
# All server crate tests pass (unit + integration)
cargo test -p graphirm-server -- --nocapture

# Full workspace still compiles
cargo build

# No lint issues
cargo clippy --all-targets --all-features

# Formatting clean
cargo fmt --all -- --check

# CLI works
cargo run -- serve --help
```

Expected final test count for `graphirm-server`: ~30 tests covering:
- ServerError IntoResponse (3 tests)
- SessionStatus display (1 test)
- AppState Send + Sync (1 test)
- API type serde roundtrips (8 tests)
- Health endpoint (2 tests)
- Session CRUD (5 tests)
- Prompt endpoint (2 tests)
- Abort endpoint (2 tests)
- Messages / children (3 tests)
- Graph queries (5 tests)
- SSE broadcast/streaming (5 tests)
- Integration lifecycle (2 tests)

### API Summary

```
GET  /api/health                                → HealthResponse
GET  /api/sessions                              → [SessionResponse]
POST /api/sessions                              → 201 + SessionResponse
GET  /api/sessions/:id                          → SessionResponse
DEL  /api/sessions/:id                          → 204
POST /api/sessions/:id/prompt                   → 202
POST /api/sessions/:id/abort                    → 204
GET  /api/sessions/:id/messages                 → [GraphNode]
GET  /api/sessions/:id/children                 → [SessionResponse] (stub)
GET  /api/graph/:sessionId                      → GraphResponse { nodes, edges }
GET  /api/graph/:sessionId/node/:nodeId         → GraphNode
GET  /api/graph/:sessionId/subgraph/:nodeId     → GraphResponse { nodes, edges }
GET  /api/graph/:sessionId/tasks                → [GraphNode]
GET  /api/graph/:sessionId/knowledge            → [GraphNode]
GET  /api/events                                → SSE stream (all sessions)
GET  /api/events/:sessionId                     → SSE stream (one session)
```
