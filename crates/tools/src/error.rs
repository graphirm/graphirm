use thiserror::Error;

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("Tool not found: {0}")]
    NotFound(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Invalid arguments: {0}")]
    InvalidArguments(String),

    #[error("Timeout after {0}s")]
    Timeout(u64),

    #[error("Cancelled")]
    Cancelled,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Graph error: {0}")]
    Graph(#[from] graphirm_graph::GraphError),
}

pub type ToolResult<T> = std::result::Result<T, ToolError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_not_found() {
        let err = ToolError::NotFound("bash".into());
        assert_eq!(err.to_string(), "Tool not found: bash");
    }

    #[test]
    fn display_permission_denied() {
        let err = ToolError::PermissionDenied("write requires approval".into());
        assert_eq!(
            err.to_string(),
            "Permission denied: write requires approval"
        );
    }

    #[test]
    fn display_execution_failed() {
        let err = ToolError::ExecutionFailed("exit code 1".into());
        assert_eq!(err.to_string(), "Execution failed: exit code 1");
    }

    #[test]
    fn display_invalid_arguments() {
        let err = ToolError::InvalidArguments("missing 'path' field".into());
        assert_eq!(err.to_string(), "Invalid arguments: missing 'path' field");
    }

    #[test]
    fn display_timeout() {
        let err = ToolError::Timeout(30);
        assert_eq!(err.to_string(), "Timeout after 30s");
    }

    #[test]
    fn display_cancelled() {
        let err = ToolError::Cancelled;
        assert_eq!(err.to_string(), "Cancelled");
    }

    #[test]
    fn from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let tool_err = ToolError::from(io_err);
        assert!(tool_err.to_string().contains("file missing"));
    }

    #[test]
    fn from_json_error() {
        let json_err = serde_json::from_str::<serde_json::Value>("not json").unwrap_err();
        let tool_err = ToolError::from(json_err);
        assert!(matches!(tool_err, ToolError::Json(_)));
    }
}
