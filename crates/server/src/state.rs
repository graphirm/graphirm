//! Shared application state and per-session bookkeeping.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use tokio::sync::{RwLock, broadcast};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use graphirm_agent::knowledge::memory::MemoryRetriever;
use graphirm_agent::{AgentConfig, AgentError, HitlGate, Session};
use graphirm_graph::GraphStore;
use graphirm_llm::LlmProvider;
use graphirm_tools::ToolRegistry;

use crate::types::{SessionId, SessionStatus, SseEvent};

/// Shared state cloned into every axum handler via `State<AppState>`.
///
/// All fields behind `Arc` so cloning is cheap. The `sessions` map is
/// protected by a `RwLock` for concurrent reads and exclusive writes.
/// `AppState: Clone + Send + Sync` — verified by the compile-time test below.
#[derive(Clone)]
pub struct AppState {
    /// Persistent graph store shared with the agent loop.
    pub graph: Arc<GraphStore>,
    /// LLM provider used by spawned agent loops.
    pub llm: Arc<dyn LlmProvider>,
    /// Tool registry passed to spawned agent loops.
    pub tools: Arc<ToolRegistry>,
    /// Broadcast channel for fan-out SSE delivery to all connected clients.
    pub event_tx: broadcast::Sender<SseEvent>,
    /// Live sessions keyed by their [`SessionId`].
    pub sessions: Arc<RwLock<HashMap<SessionId, SessionHandle>>>,
    /// Default agent config used when a `POST /sessions` body omits fields.
    pub default_config: AgentConfig,
    /// Optional embedding-based memory retriever shared across all sessions.
    /// When `Some`, each new session has cross-session memory wired in.
    pub memory_retriever: Option<Arc<MemoryRetriever>>,
}

/// Bookkeeping for a single active or completed session.
pub struct SessionHandle {
    /// The session object shared with the agent loop task.
    pub session: Arc<Session>,
    /// Cancellation token — drop or cancel to abort the running agent loop.
    pub signal: CancellationToken,
    /// Handle to the spawned agent loop task.
    ///
    /// `None` until the first prompt is submitted; `Some` while running or
    /// after completion (allows joining to collect errors).
    pub join_handle: Option<JoinHandle<Result<(), AgentError>>>,
    /// Current lifecycle status of the session.
    pub status: SessionStatus,
    /// UTC timestamp when the session was created.
    pub created_at: DateTime<Utc>,
    /// Shared HITL gate — the agent loop awaits on this; route handlers resolve it.
    pub hitl: Arc<HitlGate>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_handle_has_hitl_gate() {
        // Compile-time check: if this compiles, the field exists.
        fn assert_has_hitl(h: &SessionHandle) -> bool {
            !h.hitl.is_paused()
        }
        let _ = assert_has_hitl;
    }

    #[test]
    fn session_status_display() {
        assert_eq!(SessionStatus::Idle.to_string(), "idle");
        assert_eq!(SessionStatus::Running.to_string(), "running");
        assert_eq!(SessionStatus::Completed.to_string(), "completed");
        assert_eq!(SessionStatus::Failed.to_string(), "failed");
        assert_eq!(SessionStatus::Cancelled.to_string(), "cancelled");
    }

    #[test]
    fn app_state_is_send_and_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<AppState>();
        assert_sync::<AppState>();
    }
}
