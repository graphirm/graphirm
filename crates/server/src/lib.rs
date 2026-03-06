//! Graphirm HTTP server — axum-based REST + SSE API.
//!
//! Exposes the graph store and agent loop over HTTP so web UIs, SDKs, and
//! third-party integrations can manage sessions, submit prompts, query the
//! graph, and stream real-time agent events.

pub mod error;
pub mod middleware;
pub mod request_log;
pub mod routes;
pub mod sdk;
pub mod session;
pub mod sse;
pub mod state;
pub mod types;

// Re-export the most commonly used types at the crate root.
pub use error::ServerError;
pub use request_log::RequestLogEntry;
pub use routes::create_router;
pub use state::{AppState, SessionHandle};
pub use types::{
    CreateSessionRequest, ErrorResponse, GraphResponse, HealthResponse, PromptRequest, SessionId,
    SessionResponse, SessionStatus, SseEvent, SseEventType, SubgraphQuery,
};

// Session restoration
pub use crate::session::restore_sessions_from_graph;

// ── Server entry point ────────────────────────────────────────────────────────

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{RwLock, broadcast};
use tracing::info;

use graphirm_agent::AgentConfig;
use graphirm_graph::GraphStore;
use graphirm_llm::LlmProvider;
use graphirm_tools::ToolRegistry;

/// Configuration for the HTTP server bind address.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Interface to bind to (e.g. `"127.0.0.1"` or `"0.0.0.0"`).
    pub host: String,
    /// TCP port to listen on.
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

/// Start the Graphirm HTTP server and block until a Ctrl-C signal is received.
///
/// Builds the [`AppState`], constructs the axum router, binds a TCP listener,
/// and calls [`axum::serve`] with graceful shutdown wired to `SIGINT`.
pub async fn start_server(
    graph: Arc<GraphStore>,
    llm: Arc<dyn LlmProvider>,
    tools: Arc<ToolRegistry>,
    agent_config: AgentConfig,
    server_config: ServerConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let (event_tx, _) = broadcast::channel::<SseEvent>(1024);

    // Attempt to restore previous sessions from the graph store.
    // Sessions remain in the graph and can be queried; this is primarily for
    // logging and future features (e.g., populating an active session cache).
    match restore_sessions_from_graph(&graph).await {
        Ok(sessions) => {
            if !sessions.is_empty() {
                info!("Restored {} sessions from graph", sessions.len());
            }
        }
        Err(e) => {
            tracing::warn!("Failed to restore sessions from graph: {e}");
        }
    }

    let state = AppState {
        graph,
        llm,
        tools,
        event_tx,
        sessions: Arc::new(RwLock::new(HashMap::new())),
        default_config: agent_config,
    };

    let mut app = create_router(state);
    let mut logger_task: Option<tokio::task::JoinHandle<()>> = None;

    if let Some(ref log_path) = server_config.request_log_path {
        let (logger, task) = crate::request_log::RequestLogger::new(log_path.clone());
        app = app.layer(axum::Extension(logger));
        logger_task = Some(task);
        info!("Request logging enabled → {}", log_path.display());
    }

    let addr = format!("{}:{}", server_config.host, server_config.port);

    info!("Starting Graphirm server on {addr}");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    // Drain the request log writer — give it up to 5 s to flush buffered entries.
    if let Some(task) = logger_task {
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), task).await;
    }

    info!("Server shutdown complete");
    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install Ctrl+C handler");
    info!("Shutdown signal received");
}
