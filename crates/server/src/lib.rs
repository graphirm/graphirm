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

pub use session::restore_sessions_from_graph;

// Re-export the most commonly used types at the crate root.
pub use error::ServerError;
pub use routes::create_router;
pub use state::{AppState, SessionHandle};
pub use types::{
    CreateSessionRequest, ErrorResponse, GraphResponse, HealthResponse, PromptRequest, SessionId,
    SessionResponse, SessionStatus, SseEvent, SseEventType, SubgraphQuery,
};

// ── Server entry point ────────────────────────────────────────────────────────

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{RwLock, broadcast};
use tracing::info;

use graphirm_agent::knowledge::memory::MemoryRetriever;
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
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3000,
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
    memory_retriever: Option<Arc<MemoryRetriever>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (event_tx, _) = broadcast::channel::<SseEvent>(1024);

    let state = AppState {
        graph,
        llm,
        tools,
        event_tx,
        sessions: Arc::new(RwLock::new(HashMap::new())),
        default_config: agent_config,
        memory_retriever,
    };

    let app = create_router(state);
    let addr = format!("{}:{}", server_config.host, server_config.port);

    info!("Starting Graphirm server on {addr}");

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
