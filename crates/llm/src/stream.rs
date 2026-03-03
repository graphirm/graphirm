use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u32>,
}

impl TokenUsage {
    pub fn new(input_tokens: u32, output_tokens: u32) -> Self {
        Self {
            input_tokens,
            output_tokens,
            cache_read_tokens: None,
            cache_write_tokens: None,
        }
    }

    pub fn total(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }
}

impl std::ops::Add for TokenUsage {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            input_tokens: self.input_tokens + rhs.input_tokens,
            output_tokens: self.output_tokens + rhs.output_tokens,
            cache_read_tokens: match (self.cache_read_tokens, rhs.cache_read_tokens) {
                (Some(a), Some(b)) => Some(a + b),
                (Some(a), None) | (None, Some(a)) => Some(a),
                (None, None) => None,
            },
            cache_write_tokens: match (self.cache_write_tokens, rhs.cache_write_tokens) {
                (Some(a), Some(b)) => Some(a + b),
                (Some(a), None) | (None, Some(a)) => Some(a),
                (None, None) => None,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    TextDelta(String),
    ThinkingDelta(String),
    ToolCallStart { id: String, name: String },
    ToolCallDelta { id: String, arguments_delta: String },
    ToolCallEnd { id: String },
    Done(TokenUsage),
    Error(String),
}

impl StreamEvent {
    pub fn text_delta(text: impl Into<String>) -> Self {
        Self::TextDelta(text.into())
    }

    pub fn thinking_delta(text: impl Into<String>) -> Self {
        Self::ThinkingDelta(text.into())
    }

    pub fn tool_call_start(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self::ToolCallStart {
            id: id.into(),
            name: name.into(),
        }
    }

    pub fn tool_call_delta(id: impl Into<String>, arguments_delta: impl Into<String>) -> Self {
        Self::ToolCallDelta {
            id: id.into(),
            arguments_delta: arguments_delta.into(),
        }
    }

    pub fn tool_call_end(id: impl Into<String>) -> Self {
        Self::ToolCallEnd { id: id.into() }
    }

    pub fn done(usage: TokenUsage) -> Self {
        Self::Done(usage)
    }

    pub fn error(msg: impl Into<String>) -> Self {
        Self::Error(msg.into())
    }

    pub fn is_done(&self) -> bool {
        matches!(self, Self::Done(_))
    }

    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_usage_new() {
        let usage = TokenUsage::new(100, 200);
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 200);
        assert_eq!(usage.total(), 300);
        assert!(usage.cache_read_tokens.is_none());
        assert!(usage.cache_write_tokens.is_none());
    }

    #[test]
    fn test_token_usage_add() {
        let a = TokenUsage::new(100, 50);
        let b = TokenUsage::new(200, 75);
        let combined = a + b;
        assert_eq!(combined.input_tokens, 300);
        assert_eq!(combined.output_tokens, 125);
        assert_eq!(combined.total(), 425);
    }

    #[test]
    fn test_token_usage_add_with_cache() {
        let a = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            cache_read_tokens: Some(10),
            cache_write_tokens: None,
        };
        let b = TokenUsage {
            input_tokens: 200,
            output_tokens: 75,
            cache_read_tokens: Some(20),
            cache_write_tokens: Some(5),
        };
        let combined = a + b;
        assert_eq!(combined.cache_read_tokens, Some(30));
        assert_eq!(combined.cache_write_tokens, Some(5));
    }

    #[test]
    fn test_token_usage_default() {
        let usage = TokenUsage::default();
        assert_eq!(usage.total(), 0);
    }

    #[test]
    fn test_token_usage_serde_roundtrip() {
        let usage = TokenUsage {
            input_tokens: 10,
            output_tokens: 20,
            cache_read_tokens: Some(5),
            cache_write_tokens: None,
        };
        let json = serde_json::to_string(&usage).unwrap();
        let restored: TokenUsage = serde_json::from_str(&json).unwrap();
        assert_eq!(usage, restored);
        assert!(!json.contains("cache_write_tokens"));
    }

    #[test]
    fn test_stream_event_text_delta() {
        let event = StreamEvent::text_delta("hello");
        assert!(!event.is_done());
        assert!(!event.is_error());
        assert!(matches!(event, StreamEvent::TextDelta(s) if s == "hello"));
    }

    #[test]
    fn test_stream_event_done() {
        let usage = TokenUsage::new(10, 20);
        let event = StreamEvent::done(usage.clone());
        assert!(event.is_done());
        assert!(!event.is_error());
    }

    #[test]
    fn test_stream_event_error() {
        let event = StreamEvent::error("something went wrong");
        assert!(!event.is_done());
        assert!(event.is_error());
    }
}
