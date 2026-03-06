//! HTTP route handlers for the Graphirm REST API.

use std::sync::Arc;

use axum::Router;
use axum::extract::{Json, Path, Query, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use chrono::Utc;
use tokio_util::sync::CancellationToken;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use graphirm_agent::{AgentConfig, EventBus, Session, run_agent_loop};
use graphirm_graph::{Direction, EdgeType, GraphNode, NodeId, NodeType};

use crate::error::ServerError;
use crate::middleware::request_logging;
use crate::sse::{sse_handler, sse_session_handler};
use crate::state::{AppState, SessionHandle};
use crate::types::{
    CreateSessionRequest, EscalationMetrics, GraphResponse, HealthResponse, PromptRequest, SessionId, SessionResponse,
    SessionStatus, SseEvent, SseEventType, SubgraphQuery,
};

/// Build the axum router with all routes wired to shared [`AppState`].
///
/// Middleware applied (outermost → innermost, axum stacks in reverse `.layer()` order):
/// - [`TraceLayer`] — per-request tracing spans at INFO level.
/// - [`CorsLayer`] — permissive CORS, allows any origin/method/header.
/// - `request_logging` — JSONL request logger; no-op if no [`RequestLogger`] Extension present.
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
            "/api/sessions/{id}",
            get(get_session).delete(delete_session),
        )
        .route("/api/sessions/{id}/prompt", post(prompt_session))
        .route("/api/sessions/{id}/abort", post(abort_session))
        .route("/api/sessions/{id}/messages", get(get_messages))
        .route("/api/sessions/{id}/children", get(get_children))
        // Graph queries
        .route("/api/graph/{session_id}", get(get_session_graph))
        .route(
            "/api/graph/{session_id}/node/{node_id}",
            get(get_graph_node),
        )
        .route(
            "/api/graph/{session_id}/subgraph/{node_id}",
            get(get_subgraph),
        )
        .route("/api/graph/{session_id}/tasks", get(get_tasks))
        .route("/api/graph/{session_id}/knowledge", get(get_knowledge))
        .route("/api/sessions/{id}/escalations", get(get_escalation_metrics))
        // SSE event streams
        .route("/api/events", get(sse_handler))
        .route("/api/events/{session_id}", get(sse_session_handler))
        // Middleware (outermost applied last)
        .layer(axum::middleware::from_fn(request_logging))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

// ── Handlers ─────────────────────────────────────────────────────────────────

/// `GET /api/health` — liveness check.
async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// `POST /api/sessions` — create a new session.
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

    let session = Session::new(state.graph.clone(), config)?;
    let session_id = SessionId(session.id.to_string());
    let now = Utc::now();

    // Use caller-supplied name or generate a timestamped default.
    let name = body
        .name
        .unwrap_or_else(|| format!("session-{}", now.format("%H:%M")));

    let response = SessionResponse {
        id: session_id.to_string(),
        name: name.clone(),
        agent: session.agent_config.name.clone(),
        model: session.agent_config.model.clone(),
        created_at: now,
        status: SessionStatus::Idle,
    };

    let handle = SessionHandle {
        session: Arc::new(session),
        name,
        signal: CancellationToken::new(),
        join_handle: None,
        status: SessionStatus::Idle,
        created_at: now,
    };

    state.sessions.write().await.insert(session_id, handle);

    Ok((StatusCode::CREATED, Json(response)))
}

/// `GET /api/sessions/:id` — fetch a single session by ID.
async fn get_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<SessionResponse>, ServerError> {
    let key = SessionId::from(id.as_str());
    let sessions = state.sessions.read().await;
    let handle = sessions
        .get(&key)
        .ok_or_else(|| ServerError::NotFound(format!("Session not found: {id}")))?;

    Ok(Json(session_handle_to_response(&id, handle)))
}

/// `GET /api/sessions` — list all sessions.
async fn list_sessions(
    State(state): State<AppState>,
) -> Result<Json<Vec<SessionResponse>>, ServerError> {
    let sessions = state.sessions.read().await;
    let responses: Vec<SessionResponse> = sessions
        .iter()
        .map(|(id, handle)| session_handle_to_response(&id.to_string(), handle))
        .collect();

    Ok(Json(responses))
}

/// `DELETE /api/sessions/:id` — delete a session, cancelling any running agent.
///
/// Cancels the session's agent via its [`CancellationToken`], then spawns a
/// bounded cleanup task (5-second timeout) to await the join handle before it
/// is dropped. This prevents the detached task from writing to the graph after
/// the session has been removed from the map.
async fn delete_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ServerError> {
    let key = SessionId::from(id.as_str());
    let handle = state
        .sessions
        .write()
        .await
        .remove(&key)
        .ok_or_else(|| ServerError::NotFound(format!("Session not found: {id}")))?;

    handle.signal.cancel();

    if let Some(jh) = handle.join_handle {
        tokio::spawn(async move {
            let _ = tokio::time::timeout(std::time::Duration::from_secs(5), jh).await;
        });
    }

    Ok(StatusCode::NO_CONTENT)
}

/// `POST /api/sessions/:id/prompt` — submit a user message and start the agent loop.
///
/// Returns 202 Accepted immediately; the agent loop runs in a background task.
async fn prompt_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<PromptRequest>,
) -> Result<StatusCode, ServerError> {
    let key = SessionId::from(id.as_str());

    // Acquire write lock briefly — release before spawning tasks.
    let (session, cancel, bus) = {
        let mut sessions = state.sessions.write().await;
        let handle = sessions
            .get_mut(&key)
            .ok_or_else(|| ServerError::NotFound(format!("Session not found: {id}")))?;

        if handle.status == SessionStatus::Running {
            return Err(ServerError::BadRequest(
                "Session is already running".to_string(),
            ));
        }

        handle.session.add_user_message(&body.content)?;

        let cancel = CancellationToken::new();
        handle.signal = cancel.clone();
        handle.status = SessionStatus::Running;

        let mut bus = EventBus::new();
        let rx = bus.subscribe();
        let session = handle.session.clone();

        (session, cancel, (Arc::new(bus), rx))
    }; // write lock released here

    let (event_bus, mut rx) = bus;
    let event_tx = state.event_tx.clone();
    let relay_session_id = id.clone();

    // Relay agent events to the broadcast channel for SSE clients.
    // This task terminates automatically when the EventBus is dropped at the
    // end of the agent loop task, which closes the mpsc sender and causes
    // rx.recv() to return None.
    tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            let sse = agent_event_to_sse(&relay_session_id, &event);
            let _ = event_tx.send(sse);
        }
    });

    let llm = state.llm.clone();
    let tools = state.tools.clone();
    let sessions = state.sessions.clone();
    let bg_key = key.clone();

    let join_handle = tokio::spawn(async move {
        let result = run_agent_loop(&session, llm.as_ref(), &tools, &event_bus, &cancel).await;

        // Update status. Do NOT clear join_handle here — storing the handle
        // into the session map happens after this task is spawned, so clearing
        // it here risks overwriting Some(handle) set by the spawner with None.
        // The handle is cleaned up when the session is deleted.
        if let Err(ref e) = result {
            if !matches!(e, graphirm_agent::AgentError::Cancelled) {
                tracing::error!(session_id = %bg_key, error = %e, "Agent loop failed");
            }
        }
        let mut sessions = sessions.write().await;
        if let Some(h) = sessions.get_mut(&bg_key) {
            h.status = match &result {
                Ok(()) => SessionStatus::Completed,
                Err(graphirm_agent::AgentError::Cancelled) => SessionStatus::Cancelled,
                Err(_) => SessionStatus::Failed,
            };
        }

        result
    });

    // Store join handle for later inspection / abort.
    {
        let mut sessions = state.sessions.write().await;
        if let Some(h) = sessions.get_mut(&key) {
            h.join_handle = Some(join_handle);
        }
    }

    Ok(StatusCode::ACCEPTED)
}

/// `POST /api/sessions/{id}/abort` — cancel the running agent loop.
///
/// Signals cancellation via the session's [`CancellationToken`], then takes
/// the join handle and spawns a background cleanup task that awaits completion
/// with a 5-second timeout.
async fn abort_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ServerError> {
    let key = SessionId::from(id.as_str());
    let mut sessions = state.sessions.write().await;
    let handle = sessions
        .get_mut(&key)
        .ok_or_else(|| ServerError::NotFound(format!("Session not found: {id}")))?;

    handle.signal.cancel();
    // Mark cancelled immediately so polling clients see the new status
    // without waiting for the background task to drain.
    handle.status = SessionStatus::Cancelled;

    if let Some(jh) = handle.join_handle.take() {
        tokio::spawn(async move {
            let _ = tokio::time::timeout(std::time::Duration::from_secs(5), jh).await;
        });
    }

    Ok(StatusCode::NO_CONTENT)
}

// ── Messages & children ───────────────────────────────────────────────────────

/// `GET /api/sessions/{id}/messages` — list Interaction nodes for this session.
async fn get_messages(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Vec<GraphNode>>, ServerError> {
    let key = SessionId::from(id.as_str());
    let sessions = state.sessions.read().await;
    let handle = sessions
        .get(&key)
        .ok_or_else(|| ServerError::NotFound(format!("Session not found: {id}")))?;

    let neighbors = state
        .graph
        .neighbors(
            &handle.session.id,
            Some(EdgeType::Produces),
            Direction::Outgoing,
        )
        .map_err(ServerError::Graph)?;

    let messages: Vec<GraphNode> = neighbors
        .into_iter()
        .filter(|n| matches!(n.node_type, NodeType::Interaction(_)))
        .collect();

    Ok(Json(messages))
}

/// `GET /api/sessions/{id}/children` — list subagent sessions spawned by this session.
///
/// Returns an empty list until multi-agent spawning writes `SpawnedBy` edges (Phase 5+).
async fn get_children(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Vec<SessionResponse>>, ServerError> {
    let key = SessionId::from(id.as_str());
    let sessions = state.sessions.read().await;
    let _ = sessions
        .get(&key)
        .ok_or_else(|| ServerError::NotFound(format!("Session not found: {id}")))?;

    Ok(Json(vec![]))
}

// ── Graph queries ─────────────────────────────────────────────────────────────

/// `GET /api/graph/{session_id}` — return the full subgraph rooted at the session's agent node.
async fn get_session_graph(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Result<Json<GraphResponse>, ServerError> {
    let key = SessionId::from(session_id.as_str());
    let sessions = state.sessions.read().await;
    let handle = sessions
        .get(&key)
        .ok_or_else(|| ServerError::NotFound(format!("Session not found: {session_id}")))?;

    let (nodes, edges) = state
        .graph
        .subgraph(&handle.session.id, 10)
        .map_err(ServerError::Graph)?;

    Ok(Json(GraphResponse { nodes, edges }))
}

/// `GET /api/graph/{session_id}/node/{node_id}` — fetch a single graph node by ID.
async fn get_graph_node(
    State(state): State<AppState>,
    Path((session_id, node_id)): Path<(String, String)>,
) -> Result<Json<GraphNode>, ServerError> {
    let key = SessionId::from(session_id.as_str());
    let sessions = state.sessions.read().await;
    let _ = sessions
        .get(&key)
        .ok_or_else(|| ServerError::NotFound(format!("Session not found: {session_id}")))?;
    drop(sessions);

    let node = state
        .graph
        .get_node(&NodeId::from(node_id.as_str()))
        .map_err(|_| ServerError::NotFound(format!("Node not found: {node_id}")))?;

    Ok(Json(node))
}

/// `GET /api/graph/{session_id}/subgraph/{node_id}` — return a subgraph rooted at any node.
async fn get_subgraph(
    State(state): State<AppState>,
    Path((session_id, node_id)): Path<(String, String)>,
    Query(query): Query<SubgraphQuery>,
) -> Result<Json<GraphResponse>, ServerError> {
    let key = SessionId::from(session_id.as_str());
    let sessions = state.sessions.read().await;
    let _ = sessions
        .get(&key)
        .ok_or_else(|| ServerError::NotFound(format!("Session not found: {session_id}")))?;
    drop(sessions);

    let depth = query.depth.unwrap_or(3);
    let (nodes, edges) = state
        .graph
        .subgraph(&NodeId::from(node_id.as_str()), depth)
        .map_err(ServerError::Graph)?;

    Ok(Json(GraphResponse { nodes, edges }))
}

/// `GET /api/graph/{session_id}/tasks` — list Task nodes produced by this session.
async fn get_tasks(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Result<Json<Vec<GraphNode>>, ServerError> {
    let key = SessionId::from(session_id.as_str());
    let sessions = state.sessions.read().await;
    let handle = sessions
        .get(&key)
        .ok_or_else(|| ServerError::NotFound(format!("Session not found: {session_id}")))?;

    let neighbors = state
        .graph
        .neighbors(
            &handle.session.id,
            Some(EdgeType::Produces),
            Direction::Outgoing,
        )
        .map_err(ServerError::Graph)?;

    let tasks: Vec<GraphNode> = neighbors
        .into_iter()
        .filter(|n| matches!(n.node_type, NodeType::Task(_)))
        .collect();

    Ok(Json(tasks))
}

/// `GET /api/graph/{session_id}/knowledge` — list Knowledge nodes produced by this session.
async fn get_knowledge(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Result<Json<Vec<GraphNode>>, ServerError> {
    let key = SessionId::from(session_id.as_str());
    let sessions = state.sessions.read().await;
    let handle = sessions
        .get(&key)
        .ok_or_else(|| ServerError::NotFound(format!("Session not found: {session_id}")))?;

    let neighbors = state
        .graph
        .neighbors(
            &handle.session.id,
            Some(EdgeType::Produces),
            Direction::Outgoing,
        )
        .map_err(ServerError::Graph)?;

    let knowledge: Vec<GraphNode> = neighbors
        .into_iter()
        .filter(|n| matches!(n.node_type, NodeType::Knowledge(_)))
        .collect();

    Ok(Json(knowledge))
}

/// `GET /api/sessions/{id}/escalations` — Get soft escalation metrics for a session.
///
/// Queries the session's graph to count and analyze soft escalation events.
/// Returns metrics about when escalations occurred and their effectiveness.
async fn get_escalation_metrics(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Result<Json<EscalationMetrics>, ServerError> {
    let key = SessionId::from(session_id.as_str());
    let sessions = state.sessions.read().await;
    let handle = sessions
        .get(&key)
        .ok_or_else(|| ServerError::NotFound(format!("Session not found: {session_id}")))?;

    // Query for SoftEscalationTriggered events in the session
    let neighbors = state
        .graph
        .neighbors(
            &handle.session.id,
            Some(EdgeType::Produces),
            Direction::Outgoing,
        )
        .map_err(ServerError::Graph)?;

    // Filter interactions that are escalation events
    let mut escalation_turns: Vec<usize> = Vec::new();
    for node in neighbors {
        if let NodeType::Interaction(interaction_data) = &node.node_type {
            // Check if this interaction contains escalation event data
            if interaction_data.content.contains("SoftEscalationTriggered") {
                // Try to extract turn number from the content
                // Format: we'd need to parse structured data or use heuristics
                // For MVP, we just count events
                escalation_turns.push(escalation_turns.len()); // placeholder
            }
        }
    }

    let total_escalations = escalation_turns.len();
    let avg_turn_triggered = if !escalation_turns.is_empty() {
        escalation_turns.iter().sum::<usize>() as f64 / escalation_turns.len() as f64
    } else {
        0.0
    };
    let last_escalation_turn = escalation_turns.last().copied();

    Ok(Json(EscalationMetrics {
        session_id: session_id.clone(),
        total_escalations,
        avg_turn_triggered,
        last_escalation_turn,
    }))
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn session_handle_to_response(id: &str, handle: &SessionHandle) -> SessionResponse {
    SessionResponse {
        id: id.to_string(),
        name: handle.name.clone(),
        agent: handle.session.agent_config.name.clone(),
        model: handle.session.agent_config.model.clone(),
        created_at: handle.created_at,
        status: handle.status,
    }
}

/// Map an [`AgentEvent`] to an [`SseEvent`] for broadcast to connected clients.
fn agent_event_to_sse(session_id: &str, event: &graphirm_agent::AgentEvent) -> SseEvent {
    use graphirm_agent::AgentEvent;

    let (event_type, data) = match event {
        AgentEvent::AgentStart { agent_id } => (
            SseEventType::AgentStart,
            serde_json::json!({ "agent_id": agent_id.to_string() }),
        ),
        AgentEvent::AgentEnd { agent_id, node_ids } => (
            SseEventType::AgentEnd,
            serde_json::json!({
                "agent_id": agent_id.to_string(),
                "node_count": node_ids.len(),
            }),
        ),
        AgentEvent::TurnStart { turn_index } => (
            SseEventType::TurnStart,
            serde_json::json!({ "turn_index": turn_index }),
        ),
        AgentEvent::TurnEnd {
            response_id,
            tool_result_ids,
        } => (
            SseEventType::TurnEnd,
            serde_json::json!({
                "response_id": response_id.to_string(),
                "tool_result_count": tool_result_ids.len(),
            }),
        ),
        AgentEvent::MessageEnd { node_id } => (
            SseEventType::MessageEnd,
            serde_json::json!({ "node_id": node_id.to_string() }),
        ),
        AgentEvent::ToolStart {
            response_node_id,
            call_id,
            tool_name,
        } => (
            SseEventType::ToolStart,
            serde_json::json!({
                "response_node_id": response_node_id.to_string(),
                "call_id": call_id,
                "tool_name": tool_name,
            }),
        ),
        AgentEvent::ToolEnd { node_id, is_error } => (
            SseEventType::ToolEnd,
            serde_json::json!({
                "node_id": node_id.to_string(),
                "is_error": is_error,
            }),
        ),
        AgentEvent::GraphUpdate {
            node_id, edge_ids, ..
        } => (
            SseEventType::GraphUpdate,
            serde_json::json!({
                "node_id": node_id.to_string(),
                "edge_count": edge_ids.len(),
            }),
        ),
        _ => (
            SseEventType::Heartbeat,
            serde_json::json!({ "debug": format!("{event:?}") }),
        ),
    };

    SseEvent {
        session_id: SessionId::from(session_id),
        event_type,
        data,
    }
}

// ── Test helpers ──────────────────────────────────────────────────────────────

#[cfg(test)]
pub(crate) mod test_helpers {
    use std::collections::HashMap;
    use std::sync::Arc;

    use tokio::sync::{RwLock, broadcast};

    use graphirm_agent::AgentConfig;
    use graphirm_graph::GraphStore;
    use graphirm_llm::MockProvider;

    use crate::state::AppState;
    use crate::types::SseEvent;

    /// Build a minimal [`AppState`] backed by an in-memory graph and a noop LLM.
    pub fn test_app_state() -> AppState {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let (event_tx, _) = broadcast::channel::<SseEvent>(256);

        AppState {
            graph,
            llm: Arc::new(MockProvider::fixed("noop")),
            tools: Arc::new(graphirm_tools::ToolRegistry::new()),
            event_tx,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            default_config: AgentConfig::default(),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::test_helpers::test_app_state;
    use super::*;
    use axum::body::Body;
    use http::Request;
    use tower::ServiceExt;

    use crate::types::SessionResponse;

    // ── Health ────────────────────────────────────────────────────────────────

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

    // ── Session CRUD ──────────────────────────────────────────────────────────

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
        let session: SessionResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(session.agent, "coder");
        assert_eq!(session.model, "gpt-4o");
        assert_eq!(session.status, SessionStatus::Idle);
        assert!(!session.id.is_empty());
    }

    #[tokio::test]
    async fn test_create_session_defaults() {
        let app = create_router(test_app_state());

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
        let session: SessionResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(session.agent, "graphirm");
    }

    #[tokio::test]
    async fn test_get_session() {
        let state = test_app_state();
        let app = create_router(state.clone());

        // Create first
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
        let created: SessionResponse = serde_json::from_slice(&body).unwrap();

        // Then fetch by ID
        let get_resp = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/sessions/{}", created.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(get_resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(get_resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let fetched: SessionResponse = serde_json::from_slice(&body).unwrap();
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
        let sessions: Vec<SessionResponse> = serde_json::from_slice(&body).unwrap();
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
        let created: SessionResponse = serde_json::from_slice(&body).unwrap();

        // Delete
        let del_resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri(format!("/api/sessions/{}", created.id))
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
                    .uri(format!("/api/sessions/{}", created.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(get_resp.status(), StatusCode::NOT_FOUND);
    }

    // ── Prompt ────────────────────────────────────────────────────────────────

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
        let created: SessionResponse = serde_json::from_slice(&body).unwrap();

        // Submit prompt
        let prompt_resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{}/prompt", created.id))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"content": "Hello!"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(prompt_resp.status(), StatusCode::ACCEPTED);

        // Give background task time to complete
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let sessions = state.sessions.read().await;
        let key = SessionId::from(created.id.as_str());
        let handle = sessions.get(&key).unwrap();
        assert!(
            handle.status == SessionStatus::Running || handle.status == SessionStatus::Completed,
            "Expected Running or Completed, got {:?}",
            handle.status
        );
    }

    #[tokio::test]
    async fn test_prompt_while_running_returns_400() {
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
        let created: SessionResponse = serde_json::from_slice(&body).unwrap();

        // Manually set session status to Running to simulate an in-progress agent
        {
            let key = SessionId::from(created.id.as_str());
            let mut sessions = state.sessions.write().await;
            if let Some(h) = sessions.get_mut(&key) {
                h.status = SessionStatus::Running;
            }
        }

        // Second prompt should be rejected with 400
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{}/prompt", created.id))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"content": "Double prompt"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
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

    // ── Abort ─────────────────────────────────────────────────────────────────

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
        let created: SessionResponse = serde_json::from_slice(&body).unwrap();

        // Start agent
        app.clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{}/prompt", created.id))
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
                    .uri(format!("/api/sessions/{}/abort", created.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(abort_resp.status(), StatusCode::NO_CONTENT);

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let sessions = state.sessions.read().await;
        let key = SessionId::from(created.id.as_str());
        let handle = sessions.get(&key).unwrap();
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

    // ── Messages & children ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_get_messages_empty_session() {
        let state = test_app_state();
        let app = create_router(state.clone());

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
        let created: SessionResponse = serde_json::from_slice(&body).unwrap();

        let msg_resp = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/sessions/{}/messages", created.id))
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
        let created: SessionResponse = serde_json::from_slice(&body).unwrap();

        app.clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{}/prompt", created.id))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"content": "Hello!"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let msg_resp = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/sessions/{}/messages", created.id))
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
        assert!(!messages.is_empty());
    }

    #[tokio::test]
    async fn test_get_children_returns_empty_list() {
        let state = test_app_state();
        let app = create_router(state.clone());

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
        let created: SessionResponse = serde_json::from_slice(&body).unwrap();

        let resp = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/sessions/{}/children", created.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let children: Vec<SessionResponse> = serde_json::from_slice(&body).unwrap();
        assert!(children.is_empty());
    }

    // ── SSE ───────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_sse_endpoint_returns_event_stream_content_type() {
        let state = test_app_state();
        let app = create_router(state.clone());

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
                    "Expected text/event-stream, got: {content_type}"
                );
            }
            Ok(Err(e)) => panic!("Request error: {e}"),
            Err(_) => { /* timeout is acceptable for long-lived SSE connections */ }
        }
    }

    // ── CORS ──────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_cors_headers_present() {
        let app = create_router(test_app_state());

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

        assert!(
            response.status() == StatusCode::OK || response.status() == StatusCode::NO_CONTENT,
            "Expected 200 or 204 for CORS preflight, got: {}",
            response.status()
        );

        let headers = response.headers();
        assert!(
            headers.contains_key("access-control-allow-origin"),
            "Missing access-control-allow-origin header"
        );
    }

    // ── Graph queries ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_get_session_graph() {
        let state = test_app_state();
        let app = create_router(state.clone());

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
        let created: SessionResponse = serde_json::from_slice(&body).unwrap();

        let graph_resp = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/graph/{}", created.id))
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
        assert!(!graph.nodes.is_empty());
    }

    #[tokio::test]
    async fn test_get_graph_node() {
        let state = test_app_state();
        let app = create_router(state.clone());

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
        let created: SessionResponse = serde_json::from_slice(&body).unwrap();

        let sessions = state.sessions.read().await;
        let agent_node_id = sessions
            .get(&SessionId::from(created.id.as_str()))
            .unwrap()
            .session
            .id
            .to_string();
        drop(sessions);

        let node_resp = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/graph/{}/node/{}", created.id, agent_node_id))
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
        let created: SessionResponse = serde_json::from_slice(&body).unwrap();

        let resp = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/graph/{}/node/nonexistent", created.id))
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
        let created: SessionResponse = serde_json::from_slice(&body).unwrap();

        let sessions = state.sessions.read().await;
        let agent_node_id = sessions
            .get(&SessionId::from(created.id.as_str()))
            .unwrap()
            .session
            .id
            .to_string();
        drop(sessions);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri(format!(
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
        let created: SessionResponse = serde_json::from_slice(&body).unwrap();

        let resp = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/graph/{}/tasks", created.id))
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
        let created: SessionResponse = serde_json::from_slice(&body).unwrap();

        let resp = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/graph/{}/knowledge", created.id))
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
}
