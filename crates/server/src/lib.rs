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
use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::{RwLock, broadcast};
use tokio_util::sync::CancellationToken;
use tracing::info;

use graphirm_agent::knowledge::memory::MemoryRetriever;
use graphirm_agent::{AgentConfig, HitlGate, Session};
use graphirm_graph::{GraphStore, nodes::NodeId};
use graphirm_llm::LlmProvider;
use graphirm_tools::ToolRegistry;

/// Configuration for the HTTP server bind address.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Interface to bind to (e.g. `"127.0.0.1"` or `"0.0.0.0"`).
    pub host: String,
    /// TCP port to listen on (must be between 1 and 65535).
    pub port: u16,
}

impl ServerConfig {
    /// Create a new ServerConfig with validation.
    ///
    /// # Errors
    /// Returns an error if the port is not between 1 and 65535.
    pub fn new(host: String, port: u16) -> Result<Self, String> {
        if port == 0 {
            return Err(format!(
                "Invalid port: {}. Port must be between 1 and 65535",
                port
            ));
        }
        Ok(Self { host, port })
    }
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
    web_dir: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (event_tx, _) = broadcast::channel::<SseEvent>(1024);

    // Restore sessions from the graph before accepting connections.
    let restored = restore_sessions_from_graph(&graph, agent_config.workspaces_root.as_deref())
        .await
        .unwrap_or_else(|e| {
            tracing::warn!("Session restoration failed, starting with empty sessions: {e}");
            HashMap::new()
        });

    let mut initial_sessions: HashMap<SessionId, SessionHandle> = HashMap::new();
    for (id_str, meta) in restored {
        let mut config = agent_config.clone();
        config.working_dir = meta
            .workspace_path
            .clone()
            .unwrap_or_else(|| agent_config.working_dir.clone());
        config.workspace_dir = meta.workspace_path.clone();
        config.workspace_name = meta.workspace.clone();

        let node_id = NodeId(id_str.clone());
        let session = Session::restore(graph.clone(), node_id, config, meta.created_at);
        let session = if let Some(ref retriever) = memory_retriever {
            session.with_memory_retriever(retriever.clone())
        } else {
            session
        };
        let hitl = Arc::new(HitlGate::new());
        let handle = SessionHandle {
            session: Arc::new(session),
            signal: CancellationToken::new(),
            join_handle: None,
            status: crate::types::SessionStatus::Idle,
            created_at: meta.created_at,
            hitl,
        };
        initial_sessions.insert(SessionId(id_str), handle);
    }

    info!(
        restored_count = initial_sessions.len(),
        "Sessions restored from graph"
    );

    let state = AppState {
        graph,
        llm,
        tools,
        event_tx,
        sessions: Arc::new(RwLock::new(initial_sessions)),
        default_config: agent_config,
        memory_retriever,
        web_dir,
    };

    if let Some(ref dir) = state.web_dir {
        info!("Web UI serving from {}", dir.display());
    }

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
