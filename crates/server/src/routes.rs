//! HTTP route handlers for the Graphirm REST API.

use std::sync::Arc;

use axum::Router;
use axum::extract::{Json, Path, Query, State};
use axum::http::StatusCode;
use axum::routing::{get, patch, post};
use chrono::Utc;
use tokio_util::sync::CancellationToken;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;

use graphirm_agent::workspace::sanitize_workspace_name;
use graphirm_agent::{AgentConfig, EventBus, HitlDecision, HitlGate, Session, run_agent_loop};
use graphirm_graph::{Direction, EdgeType, GraphNode, NodeId, NodeType};

use crate::error::ServerError;
use crate::middleware::request_logging;
use crate::sse::{sse_handler, sse_session_handler};
use crate::state::{AppState, SessionHandle};
use crate::types::{
    AnnotationRequest, AutoApproveRequest, CreateKnowledgeRequest, CreateSessionRequest,
    ExportQuery, GraphResponse, HealthResponse, NodeAction, NodeActionRequest,
    PinnedKnowledgeQuery, PromptRequest, RateTurnRequest, RenameSessionRequest, SessionId,
    SessionResponse, SessionStatus, SseEvent, SseEventType, StrategyReport, SubgraphQuery,
};

/// Build a brief workspace context block to inject into the system prompt.
/// Lists up to 20 entries (non-recursive) so the agent knows where it's working.
async fn build_workspace_context(path: &std::path::Path) -> String {
    let mut lines = vec![
        String::from("\n\n## Active Workspace"),
        format!("Path: {}", path.display()),
    ];

    match tokio::fs::read_dir(path).await {
        Ok(mut dir) => {
            let mut entries: Vec<String> = Vec::new();
            let mut truncated = false;
            while let Ok(Some(entry)) = dir.next_entry().await {
                if entries.len() == 20 {
                    truncated = true;
                    break;
                }
                let name = entry.file_name().to_string_lossy().to_string();
                let is_dir = entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false);
                entries.push(if is_dir {
                    format!("  {name}/")
                } else {
                    format!("  {name}")
                });
            }
            entries.sort();
            if truncated {
                entries.push("  ...".to_string());
            }
            if entries.is_empty() {
                lines.push("(empty)".to_string());
            } else {
                lines.push("Contents:".to_string());
                lines.extend(entries);
            }
        }
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "could not read workspace dir for context injection");
            lines.push("(could not read directory)".to_string());
        }
    }

    lines.join("\n")
}

/// Build the axum router with all routes wired to shared [`AppState`].
///
/// Middleware applied (outermost → innermost):
/// - [`CorsLayer`] — permissive CORS, allows any origin/method/header.
/// - [`TraceLayer`] — per-request tracing spans at INFO level.
pub fn create_router(state: AppState) -> Router {
    let web_dir = state.web_dir.clone();

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let mut router = Router::new()
        .route("/api/health", get(health))
        // Session management
        .route("/api/sessions", get(list_sessions).post(create_session))
        .route(
            "/api/sessions/{id}",
            get(get_session)
                .delete(delete_session)
                .patch(rename_session),
        )
        .route("/api/sessions/{id}/prompt", post(prompt_session))
        .route("/api/sessions/{id}/abort", post(abort_session))
        .route("/api/sessions/{id}/messages", get(get_messages))
        .route("/api/sessions/{id}/export", get(export_session))
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
        .route(
            "/api/graph/{session_id}/node/{node_id}/action",
            post(node_action),
        )
        .route("/api/graph/{session_id}/annotate", post(create_annotation))
        .route("/api/knowledge", post(create_knowledge))
        .route("/api/knowledge/pinned", get(list_pinned_knowledge))
        .route(
            "/api/sessions/{id}/turns/{turn_id}/rating",
            patch(rate_turn),
        )
        .route("/api/routing/report", get(routing_report))
        // HITL pause / resume / auto-approve
        .route("/api/sessions/{id}/pause", post(pause_session))
        .route("/api/sessions/{id}/resume", post(resume_session))
        .route("/api/sessions/{id}/auto-approve", post(toggle_auto_approve))
        // SSE event streams
        .route("/api/events", get(sse_handler))
        .route("/api/events/{session_id}", get(sse_session_handler))
        // Middleware
        .layer(axum::middleware::from_fn(request_logging))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    if let Some(dir) = web_dir {
        router = router.fallback_service(ServeDir::new(dir).append_index_html_on_directories(true));
    }

    router
}

// ── Handlers ─────────────────────────────────────────────────────────────────

/// `GET /api/health` — liveness check.
async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let session_count = state.sessions.read().await.len();
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        session_count,
    })
}

/// `POST /api/sessions` — create a new session.
async fn create_session(
    State(state): State<AppState>,
    Json(body): Json<CreateSessionRequest>,
) -> Result<(StatusCode, Json<SessionResponse>), ServerError> {
    let agent_name = body
        .agent
        .clone()
        .unwrap_or_else(|| state.default_config.name.clone());
    let mut config = AgentConfig {
        name: agent_name,
        model: body
            .model
            .unwrap_or_else(|| state.default_config.model.clone()),
        ..state.default_config.clone()
    };
    if body.enable_segments == Some(true) {
        config.segments = Some(graphirm_agent::config::SegmentConfig {
            enabled: true,
            ..graphirm_agent::config::SegmentConfig::default()
        });
    }
    config.segment_filter = body.segment_filter;

    // Resolve per-session workspace directory
    if let Some(ref root) = config.workspaces_root {
        let raw_name = body
            .workspace
            .as_deref()
            .or(body.agent.as_deref())
            .unwrap_or("session");
        let ws_name = sanitize_workspace_name(raw_name).unwrap_or_else(|| "session".to_string());
        let ws_path = root.join(&ws_name);
        tokio::fs::create_dir_all(&ws_path).await.map_err(|e| {
            ServerError::Internal(format!(
                "failed to create workspace '{}': {e}",
                ws_path.display()
            ))
        })?;
        config.working_dir = ws_path;
        config.workspace_dir = Some(config.working_dir.clone());
        config.workspace_name = Some(ws_name);
        // Inject workspace path + directory listing into system prompt
        let ws_context = build_workspace_context(&config.working_dir).await;
        config.system_prompt.push_str(&ws_context);
    }

    // Inject compact repo briefing when enabled
    if config.repo_briefing
        && let Some(briefing) =
            graphirm_agent::briefing::build_repo_briefing(&config.working_dir, state.graph.as_ref())
                .await
    {
        config.system_prompt.push_str(&briefing);
    }

    // Validate model routing: both tiers must use the same provider backend.
    if let Some(ref routing) = config.model_routing
        && !routing.same_provider()
    {
        tracing::warn!(
            cheap = %routing.cheap,
            smart = %routing.smart,
            "model routing tiers use different providers — routing disabled, using single model"
        );
        config.model_routing = None;
    }

    let hitl = Arc::new(HitlGate::new());
    let graph_for_session = state.graph.clone();
    let config_clone = config.clone();
    let mut session =
        tokio::task::spawn_blocking(move || Session::new(graph_for_session, config_clone))
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))?
            .map_err(ServerError::Agent)?;

    // Only wire up the HITL gate when the caller hasn't opted into auto-approve.
    // Programmatic clients (eval harnesses, tests) pass `auto_approve: true` to
    // bypass human confirmation for destructive tools (bash, write, edit).
    if !body.auto_approve.unwrap_or(false) {
        session = session.with_hitl(hitl.clone());
    }

    if let Some(ref retriever) = state.memory_retriever {
        session = session.with_memory_retriever(retriever.clone());
    }
    let session_id = SessionId(session.id.to_string());
    let now = Utc::now();

    let response = SessionResponse {
        id: session_id.to_string(),
        name: session.agent_config.name.clone(),
        agent: session.agent_config.name.clone(),
        model: session.agent_config.model.clone(),
        created_at: now,
        status: SessionStatus::Idle,
        workspace: session.agent_config.workspace_name.clone(),
        workspace_path: session_workspace_path(&session.agent_config),
    };

    let handle = SessionHandle {
        display_name: Arc::new(std::sync::RwLock::new(session.agent_config.name.clone())),
        session: Arc::new(session),
        signal: CancellationToken::new(),
        join_handle: None,
        status: SessionStatus::Idle,
        created_at: now,
        hitl,
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

/// `PATCH /api/sessions/:id` — rename a session.
async fn rename_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<RenameSessionRequest>,
) -> Result<Json<SessionResponse>, ServerError> {
    let name = body.name.trim().to_string();
    if name.is_empty() {
        return Err(ServerError::BadRequest(
            "name must not be empty".to_string(),
        ));
    }

    let key = SessionId::from(id.as_str());

    // Update display_name in-memory and build response while holding the read lock.
    // We clone the Arc<Session> to get the NodeId for the subsequent graph write,
    // which must happen outside the sessions lock (no await inside RwLock guard).
    let (session_arc, response) = {
        let sessions = state.sessions.read().await;
        let handle = sessions
            .get(&key)
            .ok_or_else(|| ServerError::NotFound(format!("Session not found: {id}")))?;

        *handle
            .display_name
            .write()
            .unwrap_or_else(|e| e.into_inner()) = name.clone();

        let response = session_handle_to_response(&id, handle);
        (handle.session.clone(), response)
    }; // sessions read lock released here

    // Persist the new name to the Agent node in the graph store.
    let graph = state.graph.clone();
    let session_node_id = session_arc.id.clone();
    tokio::task::spawn_blocking(move || -> Result<(), graphirm_graph::GraphError> {
        let mut node = graph.get_node(&session_node_id)?;
        if let NodeType::Agent(ref mut data) = node.node_type {
            data.name = name;
        }
        graph.update_node(&session_node_id, node)
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))?
    .map_err(ServerError::Graph)?;

    Ok(Json(response))
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
    // `add_user_message` is async (uses spawn_blocking), so we clone the Arc
    // and call it after the lock is released to avoid holding a write guard
    // across an await point.
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

        let cancel = CancellationToken::new();
        handle.signal = cancel.clone();
        handle.status = SessionStatus::Running;

        let mut bus = EventBus::new();
        let rx = bus.subscribe();
        let session = handle.session.clone();

        (session, cancel, (Arc::new(bus), rx))
    }; // write lock released here

    // Record the user message outside the lock so we don't hold a write guard
    // across the async spawn_blocking call inside add_user_message.
    session
        .add_user_message(&body.content)
        .await
        .map_err(ServerError::Agent)?;

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
        let result = run_agent_loop(&session, llm.clone(), &tools, &event_bus, &cancel).await;

        // Update status. Do NOT clear join_handle here — storing the handle
        // into the session map happens after this task is spawned, so clearing
        // it here risks overwriting Some(handle) set by the spawner with None.
        // The handle is cleaned up when the session is deleted.
        if let Err(ref e) = result
            && !matches!(e, graphirm_agent::AgentError::Cancelled)
        {
            tracing::error!(session_id = %bg_key, error = %e, "Agent loop failed");
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
    let session_id = {
        let sessions = state.sessions.read().await;
        let handle = sessions
            .get(&key)
            .ok_or_else(|| ServerError::NotFound(format!("Session not found: {id}")))?;
        handle.session.id.clone()
    };

    let graph = state.graph.clone();
    let neighbors = tokio::task::spawn_blocking(move || {
        graph.neighbors(&session_id, Some(EdgeType::Produces), Direction::Outgoing)
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))?
    .map_err(ServerError::Graph)?;

    let messages: Vec<GraphNode> = neighbors
        .into_iter()
        .filter(|n| matches!(n.node_type, NodeType::Interaction(_)))
        .collect();

    Ok(Json(messages))
}

/// `GET /api/sessions/{id}/export?format=markdown` — export session as a Markdown document.
///
/// Returns a `text/markdown` file attachment with the session's conversation and extracted
/// knowledge. Currently only `format=markdown` is supported (the default).
async fn export_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<ExportQuery>,
) -> Result<axum::response::Response, ServerError> {
    use crate::export::render_session_markdown;
    use axum::http::header;

    if query.format != "markdown" {
        return Err(ServerError::BadRequest(format!(
            "Unsupported export format: '{}'. Only 'markdown' is supported.",
            query.format
        )));
    }

    let key = SessionId::from(id.as_str());
    let (session_node_id, session_name, model, created_at) = {
        let sessions = state.sessions.read().await;
        let handle = sessions
            .get(&key)
            .ok_or_else(|| ServerError::NotFound(format!("Session not found: {id}")))?;
        let name = handle
            .display_name
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let model = handle.session.agent_config.model.clone();
        let created_at = handle.session.created_at;
        (handle.session.id.clone(), name, model, created_at)
    };

    let graph = state.graph.clone();
    let session_node_id_clone = session_node_id.clone();
    let (nodes, _edges) =
        tokio::task::spawn_blocking(move || graph.subgraph(&session_node_id_clone, 10))
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))?
            .map_err(ServerError::Graph)?;

    let markdown = render_session_markdown(&session_name, &model, created_at, &nodes);

    let filename = format!(
        "session-{}.md",
        session_name
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '-' {
                c
            } else {
                '-'
            })
            .collect::<String>()
    );

    let response = axum::response::Response::builder()
        .status(200)
        .header(header::CONTENT_TYPE, "text/markdown; charset=utf-8")
        .header(
            header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{filename}\""),
        )
        .body(axum::body::Body::from(markdown))
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    Ok(response)
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
    let session_node_id = {
        let sessions = state.sessions.read().await;
        let handle = sessions
            .get(&key)
            .ok_or_else(|| ServerError::NotFound(format!("Session not found: {session_id}")))?;
        handle.session.id.clone()
    };

    let graph = state.graph.clone();
    let (nodes, edges) = tokio::task::spawn_blocking(move || graph.subgraph(&session_node_id, 10))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .map_err(ServerError::Graph)?;

    Ok(Json(GraphResponse { nodes, edges }))
}

/// `GET /api/graph/{session_id}/node/{node_id}` — fetch a single graph node by ID.
async fn get_graph_node(
    State(state): State<AppState>,
    Path((session_id, node_id)): Path<(String, String)>,
) -> Result<Json<GraphNode>, ServerError> {
    let key = SessionId::from(session_id.as_str());
    {
        let sessions = state.sessions.read().await;
        let _ = sessions
            .get(&key)
            .ok_or_else(|| ServerError::NotFound(format!("Session not found: {session_id}")))?;
    }

    let graph = state.graph.clone();
    let target_node_id = NodeId::from(node_id.as_str());
    let node = tokio::task::spawn_blocking(move || graph.get_node(&target_node_id))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
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
    {
        let sessions = state.sessions.read().await;
        let _ = sessions
            .get(&key)
            .ok_or_else(|| ServerError::NotFound(format!("Session not found: {session_id}")))?;
    }

    let depth = query.depth.unwrap_or(3);
    let graph = state.graph.clone();
    let target_node_id = NodeId::from(node_id.as_str());
    let (nodes, edges) =
        tokio::task::spawn_blocking(move || graph.subgraph(&target_node_id, depth))
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))?
            .map_err(ServerError::Graph)?;

    Ok(Json(GraphResponse { nodes, edges }))
}

/// `GET /api/graph/{session_id}/tasks` — list Task nodes produced by this session.
async fn get_tasks(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Result<Json<Vec<GraphNode>>, ServerError> {
    let key = SessionId::from(session_id.as_str());
    let session_node_id = {
        let sessions = state.sessions.read().await;
        let handle = sessions
            .get(&key)
            .ok_or_else(|| ServerError::NotFound(format!("Session not found: {session_id}")))?;
        handle.session.id.clone()
    };

    let graph = state.graph.clone();
    let neighbors = tokio::task::spawn_blocking(move || {
        graph.neighbors(
            &session_node_id,
            Some(EdgeType::Produces),
            Direction::Outgoing,
        )
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))?
    .map_err(ServerError::Graph)?;

    let tasks: Vec<GraphNode> = neighbors
        .into_iter()
        .filter(|n| matches!(n.node_type, NodeType::Task(_)))
        .collect();

    Ok(Json(tasks))
}

/// `GET /api/graph/{session_id}/knowledge` — list Knowledge nodes produced by this session.
///
/// Knowledge nodes are not linked directly to the agent node. They are linked via
/// `DerivedFrom` edges from the knowledge node to the interaction node that triggered
/// extraction. So we do a 2-hop traversal:
///   agent → (Produces, Outgoing) → interaction nodes
///   interaction node → (DerivedFrom, Incoming) → knowledge nodes
async fn get_knowledge(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Result<Json<Vec<GraphNode>>, ServerError> {
    let key = SessionId::from(session_id.as_str());
    let session_node_id = {
        let sessions = state.sessions.read().await;
        let handle = sessions
            .get(&key)
            .ok_or_else(|| ServerError::NotFound(format!("Session not found: {session_id}")))?;
        handle.session.id.clone()
    };

    let graph = state.graph.clone();
    let knowledge = tokio::task::spawn_blocking(move || {
        // Hop 1: get all interaction nodes for this session.
        let interaction_nodes = graph.neighbors(
            &session_node_id,
            Some(EdgeType::Produces),
            Direction::Outgoing,
        )?;

        // Hop 2: for each interaction node, find knowledge nodes that derived from it.
        let mut knowledge_nodes: Vec<GraphNode> = Vec::new();
        for node in &interaction_nodes {
            if !matches!(node.node_type, NodeType::Interaction(_)) {
                continue;
            }
            let derived =
                graph.neighbors(&node.id, Some(EdgeType::DerivedFrom), Direction::Incoming)?;
            for k in derived {
                if matches!(k.node_type, NodeType::Knowledge(_)) {
                    knowledge_nodes.push(k);
                }
            }
        }

        Ok::<_, graphirm_graph::GraphError>(knowledge_nodes)
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))?
    .map_err(ServerError::Graph)?;

    Ok(Json(knowledge))
}

// ── HITL endpoints ────────────────────────────────────────────────────────────

/// `POST /api/graph/:session_id/node/:node_id/action`
///
/// Resolves a pending HITL gate for the given node. The agent loop is
/// unblocked and proceeds according to the decision.
async fn node_action(
    State(state): State<AppState>,
    Path((session_id, node_id)): Path<(SessionId, String)>,
    Json(body): Json<NodeActionRequest>,
) -> Result<StatusCode, ServerError> {
    let sessions = state.sessions.read().await;
    let handle = sessions
        .get(&session_id)
        .ok_or_else(|| ServerError::NotFound(format!("session {session_id}")))?;

    let nid = NodeId::from(node_id.as_str());
    let decision = match body.action {
        NodeAction::Approve => HitlDecision::Approve,
        NodeAction::Reject => {
            let reason = body
                .reason
                .unwrap_or_else(|| "No reason provided".to_string());
            HitlDecision::Reject(reason)
        }
        NodeAction::Modify => {
            let args = body.modified_args.ok_or_else(|| {
                ServerError::BadRequest("modified_args required for modify".to_string())
            })?;
            HitlDecision::Modify(args)
        }
    };

    let resolved = handle.hitl.resolve(&nid, decision).await;
    if resolved {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(ServerError::NotFound(format!(
            "No pending gate for node {node_id}"
        )))
    }
}

/// `POST /api/graph/{session_id}/annotate` — create a user annotation Knowledge node.
///
/// Stores the node in the graph associated with the session's agent node via a `RelatesTo` edge.
/// The optional `position` field is persisted in node metadata for canvas layout.
async fn create_annotation(
    State(state): State<AppState>,
    Path(session_id): Path<SessionId>,
    Json(body): Json<AnnotationRequest>,
) -> Result<Json<GraphNode>, ServerError> {
    use graphirm_graph::{GraphEdge, KnowledgeData};

    // Verify session exists and get its agent node ID.
    let agent_node_id = {
        let sessions = state.sessions.read().await;
        let handle = sessions
            .get(&session_id)
            .ok_or_else(|| ServerError::NotFound(format!("session {session_id}")))?;
        handle.session.id.clone()
    };

    let mut metadata = serde_json::Map::new();
    if let Some(pos) = &body.position {
        metadata.insert("position_x".to_string(), serde_json::json!(pos.x));
        metadata.insert("position_y".to_string(), serde_json::json!(pos.y));
    }
    metadata.insert("source".to_string(), serde_json::json!("user-annotation"));

    let annotation_node = GraphNode::new(NodeType::Knowledge(KnowledgeData {
        entity: body.entity,
        entity_type: body.entity_type,
        summary: body.summary,
        confidence: 1.0,
    }));
    let annotation_node = GraphNode {
        metadata: serde_json::Value::Object(metadata),
        ..annotation_node
    };

    let graph = state.graph.clone();
    let node_to_store = annotation_node.clone();
    let annotation_id = tokio::task::spawn_blocking(move || graph.add_node(node_to_store))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .map_err(ServerError::Graph)?;

    // Link annotation to the session's agent node via RelatesTo.
    let edge = GraphEdge::new(EdgeType::RelatesTo, agent_node_id, annotation_id.clone());
    let graph = state.graph.clone();
    tokio::task::spawn_blocking(move || graph.add_edge(edge))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .map_err(ServerError::Graph)?;

    // Return the created node.
    let graph = state.graph.clone();
    let node = tokio::task::spawn_blocking(move || graph.get_node(&annotation_id))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .map_err(|_| ServerError::Internal("Node vanished after insert".to_string()))?;

    Ok(Json(node))
}

/// `POST /api/knowledge` — create a global Knowledge node directly via the API.
///
/// Unlike `create_annotation`, this endpoint is NOT session-scoped. It creates
/// knowledge nodes that exist independently of any session.
async fn create_knowledge(
    State(state): State<AppState>,
    Json(body): Json<CreateKnowledgeRequest>,
) -> Result<Json<GraphNode>, ServerError> {
    use graphirm_graph::KnowledgeData;

    let mut metadata = serde_json::Map::new();
    if body.pinned {
        metadata.insert("pinned".to_string(), serde_json::json!(true));
    }
    if let Some(sid) = &body.session_id {
        metadata.insert("session_id".to_string(), serde_json::json!(sid));
    }

    let knowledge_node = GraphNode::new(NodeType::Knowledge(KnowledgeData {
        entity: body.entity,
        entity_type: body.entity_type,
        summary: body.summary,
        confidence: body.confidence.unwrap_or(1.0),
    }));
    let knowledge_node = GraphNode {
        metadata: serde_json::Value::Object(metadata),
        ..knowledge_node
    };

    let graph = state.graph.clone();
    let node_to_store = knowledge_node.clone();
    let knowledge_id = tokio::task::spawn_blocking(move || graph.add_node(node_to_store))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .map_err(ServerError::Graph)?;

    // Return the created node by fetching it back.
    let graph = state.graph.clone();
    let node = tokio::task::spawn_blocking(move || graph.get_node(&knowledge_id))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .map_err(|_| ServerError::Internal("Node vanished after insert".to_string()))?;

    Ok(Json(node))
}

/// `GET /api/knowledge/pinned` — return all pinned Knowledge nodes.
///
/// These are global rules/conventions that should always surface regardless of recency.
/// Uses `GraphStore::list_pinned_knowledge(limit)` to fetch from the database.
async fn list_pinned_knowledge(
    State(state): State<AppState>,
    Query(query): Query<PinnedKnowledgeQuery>,
) -> Result<Json<Vec<GraphNode>>, ServerError> {
    let graph = state.graph.clone();
    let nodes = tokio::task::spawn_blocking(move || graph.list_pinned_knowledge(query.limit))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .map_err(ServerError::Graph)?;

    Ok(Json(nodes))
}

/// `PATCH /api/sessions/:id/turns/:turn_id/rating` — store a 1–5 user rating on an Interaction node.
async fn rate_turn(
    State(state): State<AppState>,
    Path((_session_id, turn_id)): Path<(String, String)>,
    Json(body): Json<RateTurnRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if body.rating == 0 || body.rating > 5 {
        return Err(StatusCode::UNPROCESSABLE_ENTITY);
    }
    let graph = state.graph.clone();
    let node_id = NodeId(turn_id.clone());
    tokio::task::spawn_blocking(move || {
        let mut node = graph.get_node(&node_id).map_err(|_| StatusCode::NOT_FOUND)?;
        if let serde_json::Value::Object(ref mut map) = node.metadata {
            map.insert("user_rating".to_string(), serde_json::json!(body.rating));
        }
        graph.update_node(&node_id, node).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        Ok::<_, StatusCode>(())
    })
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)??;

    Ok(Json(serde_json::json!({ "ok": true })))
}

/// `GET /api/routing/report` — aggregate per-strategy routing statistics across all sessions.
async fn routing_report(State(state): State<AppState>) -> Json<Vec<StrategyReport>> {
    let graph = state.graph.clone();
    let reports = tokio::task::spawn_blocking(move || build_routing_report(&graph))
        .await
        .unwrap_or_default();
    Json(reports)
}

/// Query all Interaction nodes, group by `routing_strategy` metadata, and aggregate stats.
fn build_routing_report(graph: &graphirm_graph::GraphStore) -> Vec<StrategyReport> {
    use std::collections::HashMap;

    let nodes = match graph.list_nodes_by_type("interaction", None, None, 10_000) {
        Ok(nodes) => nodes,
        Err(_) => return vec![],
    };

    /// (turn_count, input_tokens, output_tokens, latency_ms, tool_errors, ratings)
    type Bucket = (u32, u64, u64, u64, u32, Vec<f64>);
    let mut groups: HashMap<String, Bucket> = HashMap::new();

    for node in &nodes {
        let meta = &node.metadata;
        let strategy = match meta.get("routing_strategy").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        let input = meta
            .get("usage_input")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let output = meta
            .get("usage_output")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let latency = meta
            .get("routing_decision_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let has_error = meta
            .get("tool_errors")
            .and_then(|v| v.as_u64())
            .map(|e| e > 0)
            .unwrap_or(false);
        let rating = meta
            .get("user_rating")
            .and_then(|v| v.as_f64());

        let entry = groups.entry(strategy).or_insert((0, 0, 0, 0, 0, vec![]));
        entry.0 += 1;
        entry.1 += input;
        entry.2 += output;
        entry.3 += latency;
        if has_error {
            entry.4 += 1;
        }
        if let Some(r) = rating {
            entry.5.push(r);
        }
    }

    let mut reports: Vec<StrategyReport> = groups
        .into_iter()
        .map(|(strategy_name, (count, input, output, latency, errors, ratings))| {
            let n = count as f64;
            let avg_user_rating = if ratings.is_empty() {
                None
            } else {
                Some(ratings.iter().sum::<f64>() / ratings.len() as f64)
            };
            StrategyReport {
                strategy_name,
                turn_count: count,
                avg_input_tokens: input as f64 / n,
                avg_output_tokens: output as f64 / n,
                avg_latency_ms: latency as f64 / n,
                error_rate: errors as f64 / n,
                avg_user_rating,
            }
        })
        .collect();
    reports.sort_by(|a, b| b.turn_count.cmp(&a.turn_count));
    reports
}

/// `POST /api/sessions/:id/pause`
async fn pause_session(
    State(state): State<AppState>,
    Path(session_id): Path<SessionId>,
) -> Result<StatusCode, ServerError> {
    let sessions = state.sessions.read().await;
    let handle = sessions
        .get(&session_id)
        .ok_or_else(|| ServerError::NotFound(format!("session {session_id}")))?;
    handle.hitl.set_paused(true);
    Ok(StatusCode::NO_CONTENT)
}

/// `POST /api/sessions/:id/resume`
async fn resume_session(
    State(state): State<AppState>,
    Path(session_id): Path<SessionId>,
) -> Result<StatusCode, ServerError> {
    let sessions = state.sessions.read().await;
    let handle = sessions
        .get(&session_id)
        .ok_or_else(|| ServerError::NotFound(format!("session {session_id}")))?;
    // Resolve the gate first so the agent loop unblocks before it sees
    // is_paused() == false. This prevents the TOCTOU window where set_paused(false)
    // fires between the while-condition check and hitl.gate(), leaving an
    // unresolvable receiver.
    let session_node_id = NodeId::from(handle.session.id.0.as_str());
    handle
        .hitl
        .resolve(&session_node_id, HitlDecision::Approve)
        .await;
    handle.hitl.set_paused(false);
    Ok(StatusCode::NO_CONTENT)
}

/// `POST /api/sessions/{id}/auto-approve` — toggle auto-approve mode for destructive tools.
///
/// Request body: `{ "enabled": true }` or `{ "enabled": false }`.
/// When enabled, the agent loop skips HITL gating and approves all tool calls automatically.
async fn toggle_auto_approve(
    State(state): State<AppState>,
    Path(session_id): Path<SessionId>,
    Json(body): Json<AutoApproveRequest>,
) -> Result<StatusCode, ServerError> {
    let sessions = state.sessions.read().await;
    let handle = sessions
        .get(&session_id)
        .ok_or_else(|| ServerError::NotFound(format!("session {session_id}")))?;
    handle.hitl.set_auto_approve(body.enabled);
    Ok(StatusCode::NO_CONTENT)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn session_workspace_path(config: &graphirm_agent::AgentConfig) -> Option<String> {
    config
        .workspace_dir
        .as_ref()
        .map(|p| p.display().to_string())
}

fn session_handle_to_response(id: &str, handle: &SessionHandle) -> SessionResponse {
    let name = handle
        .display_name
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    SessionResponse {
        id: id.to_string(),
        name: name.clone(),
        agent: name,
        model: handle.session.agent_config.model.clone(),
        created_at: handle.created_at,
        status: handle.status,
        workspace: handle.session.agent_config.workspace_name.clone(),
        workspace_path: session_workspace_path(&handle.session.agent_config),
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
            node_id,
            patch_nodes,
            recent_edges,
            ..
        } => (
            SseEventType::GraphUpdate,
            serde_json::json!({
                "node_id": node_id.to_string(),
                "nodes": patch_nodes,
                "edges": recent_edges,
            }),
        ),
        AgentEvent::AwaitingApproval {
            node_id,
            tool_name,
            arguments,
            is_pause,
        } => (
            SseEventType::AwaitingApproval,
            serde_json::json!({
                "node_id": node_id.to_string(),
                "tool_name": tool_name,
                "arguments": arguments,
                "is_pause": is_pause,
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
            memory_retriever: None,
            web_dir: None,
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

    // ── HITL pause / resume ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_pause_endpoint_sets_paused_flag() {
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

        // POST /api/sessions/:id/pause
        let pause_resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{}/pause", created.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(pause_resp.status(), StatusCode::NO_CONTENT);

        let sessions = state.sessions.read().await;
        let key = SessionId::from(created.id.as_str());
        let handle = sessions.get(&key).unwrap();
        assert!(handle.hitl.is_paused());
    }

    #[tokio::test]
    async fn test_resume_endpoint_clears_paused_flag() {
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

        // Set paused=true directly
        {
            let sessions = state.sessions.read().await;
            let key = SessionId::from(created.id.as_str());
            sessions.get(&key).unwrap().hitl.set_paused(true);
        }

        // POST /api/sessions/:id/resume
        let resume_resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/sessions/{}/resume", created.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resume_resp.status(), StatusCode::NO_CONTENT);

        let sessions = state.sessions.read().await;
        let key = SessionId::from(created.id.as_str());
        let handle = sessions.get(&key).unwrap();
        assert!(!handle.hitl.is_paused());
    }

    #[tokio::test]
    async fn test_node_action_approve_returns_204() {
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

        // Register a pending gate for node "n1" before the handler calls resolve().
        let node_id = NodeId::from("n1");
        let rx = {
            let sessions = state.sessions.read().await;
            let key = SessionId::from(created.id.as_str());
            sessions.get(&key).unwrap().hitl.gate(&node_id).await
        };

        // POST approve action
        let action_resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/graph/{}/node/n1/action", created.id))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"action":"approve"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(action_resp.status(), StatusCode::NO_CONTENT);
        // Gate receiver should have been resolved with Approve.
        assert!(matches!(rx.await.unwrap(), HitlDecision::Approve));
    }

    #[tokio::test]
    async fn test_node_action_modify_without_args_returns_400() {
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

        // POST modify without the required modified_args field
        let action_resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/graph/{}/node/n1/action", created.id))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"action":"modify"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(action_resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_node_action_nonexistent_session_returns_404() {
        let app = create_router(test_app_state());

        let action_resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/graph/nonexistent-session/node/n1/action")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"action":"approve"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(action_resp.status(), StatusCode::NOT_FOUND);
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

    // ── agent_event_to_sse ────────────────────────────────────────────────────

    #[test]
    fn agent_event_awaiting_approval_maps_to_sse_awaiting_approval() {
        use graphirm_agent::AgentEvent;
        use graphirm_graph::NodeId;

        let event = AgentEvent::AwaitingApproval {
            node_id: NodeId::from("n1"),
            tool_name: "write".to_string(),
            arguments: serde_json::json!({"path": "/tmp/x.rs"}),
            is_pause: false,
        };
        let sse = agent_event_to_sse("session-1", &event);
        assert!(
            matches!(sse.event_type, crate::types::SseEventType::AwaitingApproval),
            "expected AwaitingApproval event type"
        );
        assert_eq!(sse.session_id, crate::types::SessionId::from("session-1"));
        assert_eq!(sse.data["tool_name"], "write");
        assert_eq!(sse.data["node_id"], "n1");
        assert_eq!(sse.data["is_pause"], false);
        assert_eq!(sse.data["arguments"]["path"], "/tmp/x.rs");
    }
}

#[cfg(test)]
mod workspace_tests {
    use super::sanitize_workspace_name;

    #[test]
    fn sanitize_trims_and_lowercases() {
        assert_eq!(sanitize_workspace_name("  MyApp  "), Some("myapp".into()));
    }

    #[test]
    fn sanitize_replaces_bad_chars() {
        assert_eq!(
            sanitize_workspace_name("My App/v2"),
            Some("my-app-v2".into())
        );
    }

    #[test]
    fn sanitize_collapses_dashes() {
        assert_eq!(sanitize_workspace_name("foo--bar"), Some("foo-bar".into()));
    }

    #[test]
    fn sanitize_strips_leading_trailing_dash() {
        assert_eq!(sanitize_workspace_name("--foo--"), Some("foo".into()));
    }

    #[test]
    fn sanitize_empty_returns_none() {
        assert_eq!(sanitize_workspace_name("   "), None);
        assert_eq!(sanitize_workspace_name("---"), None);
    }

    #[test]
    fn sanitize_preserves_valid_name() {
        assert_eq!(
            sanitize_workspace_name("my-project_2"),
            Some("my-project_2".into())
        );
    }
}
