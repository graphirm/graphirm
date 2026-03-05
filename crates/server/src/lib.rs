//! Graphirm HTTP server — axum-based REST + SSE API.
//!
//! Exposes the graph store and agent loop over HTTP so web UIs, SDKs, and
//! third-party integrations can manage sessions, submit prompts, query the
//! graph, and stream real-time agent events.

pub mod error;
pub mod routes;
pub mod sdk;
pub mod sse;
pub mod state;
pub mod types;

// Re-export the most commonly used types at the crate root.
pub use error::ServerError;
pub use state::{AppState, SessionHandle};
pub use types::{
    CreateSessionRequest, ErrorResponse, GraphResponse, HealthResponse, PromptRequest, SessionId,
    SessionResponse, SessionStatus, SseEvent, SseEventType, SubgraphQuery,
};
