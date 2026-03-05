use serde::{Deserialize, Serialize};

// Full types implemented in Task 3; SseEvent defined here early so state.rs can reference it.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SseEvent {
    pub session_id: String,
    pub event_type: String,
    pub data: serde_json::Value,
}
