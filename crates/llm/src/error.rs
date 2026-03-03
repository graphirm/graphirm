use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("Provider error: {0}")]
    Provider(String),

    #[error("Streaming error: {0}")]
    Stream(String),

    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("Tool call error: {0}")]
    ToolCall(String),

    #[error("Request timeout")]
    Timeout,
}
