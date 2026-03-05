//! Server error type with axum `IntoResponse` integration.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::types::ErrorResponse;

/// All errors that can be returned by HTTP route handlers.
///
/// Implements [`IntoResponse`] so handlers can use `Result<T, ServerError>` directly.
/// New variants can be added without breaking external `match` arms thanks to `#[non_exhaustive]`.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    /// The requested resource was not found (404).
    #[error("Not found: {0}")]
    NotFound(String),

    /// The request was malformed or missing required fields (400).
    #[error("Bad request: {0}")]
    BadRequest(String),

    /// An unexpected internal error occurred (500).
    #[error("Internal error: {0}")]
    Internal(String),

    /// A graph store operation failed (500).
    #[error("Graph error: {0}")]
    Graph(#[from] graphirm_graph::GraphError),

    /// An agent loop operation failed (500).
    #[error("Agent error: {0}")]
    Agent(#[from] graphirm_agent::AgentError),
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ServerError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            ServerError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            ServerError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            ServerError::Graph(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            ServerError::Agent(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        };
        (status, Json(ErrorResponse { error: message })).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

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
