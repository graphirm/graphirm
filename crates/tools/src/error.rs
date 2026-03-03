use thiserror::Error;

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Tool execution failed: {0}")]
    Execution(String),

    #[error("Tool timeout after {0}s")]
    Timeout(u64),

    #[error("Invalid arguments: {0}")]
    InvalidArgs(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),
}
