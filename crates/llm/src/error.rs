use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("Provider error: {0}")]
    Provider(String),

    #[error("Rate limited, retry after {retry_after_ms}ms")]
    RateLimited { retry_after_ms: u64 },

    #[error("Invalid model: {0}")]
    InvalidModel(String),

    #[error("Stream error: {0}")]
    Stream(String),

    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("Request error: {0}")]
    Request(String),

    #[error("Configuration error: {0}")]
    Config(String),
}

impl LlmError {
    pub fn provider(msg: impl Into<String>) -> Self {
        Self::Provider(msg.into())
    }
    pub fn stream(msg: impl Into<String>) -> Self {
        Self::Stream(msg.into())
    }
    pub fn invalid_model(model: impl Into<String>) -> Self {
        Self::InvalidModel(model.into())
    }
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_error_display() {
        let err = LlmError::provider("connection failed");
        assert_eq!(err.to_string(), "Provider error: connection failed");
    }

    #[test]
    fn test_rate_limited_display() {
        let err = LlmError::RateLimited {
            retry_after_ms: 5000,
        };
        assert_eq!(err.to_string(), "Rate limited, retry after 5000ms");
    }

    #[test]
    fn test_invalid_model_display() {
        let err = LlmError::invalid_model("gpt-99");
        assert_eq!(err.to_string(), "Invalid model: gpt-99");
    }

    #[test]
    fn test_stream_error_display() {
        let err = LlmError::stream("unexpected EOF");
        assert_eq!(err.to_string(), "Stream error: unexpected EOF");
    }

    #[test]
    fn test_serde_error_from() {
        let json_err = serde_json::from_str::<serde_json::Value>("invalid json").unwrap_err();
        let err: LlmError = json_err.into();
        assert!(err.to_string().starts_with("Serialization error:"));
    }

    #[test]
    fn test_config_error_display() {
        let err = LlmError::config("missing API key");
        assert_eq!(err.to_string(), "Configuration error: missing API key");
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LlmError>();
    }
}
