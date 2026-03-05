use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use tokio::sync::{broadcast, RwLock};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use graphirm_agent::{AgentConfig, AgentError, Session};
use graphirm_graph::GraphStore;
use graphirm_llm::LlmProvider;
use graphirm_tools::registry::ToolRegistry;

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
    pub session: Arc<Session>,
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
