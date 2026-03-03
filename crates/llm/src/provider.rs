use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::LlmError;
use crate::stream::{StreamEvent, TokenUsage};
use crate::tool::ToolDefinition;

// ================================================================
// Message model
// ================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    Human,
    Assistant,
    ToolResult,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: Value,
    },
    ToolResult {
        id: String,
        content: String,
        is_error: bool,
    },
}

impl ContentPart {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    pub fn tool_call(id: impl Into<String>, name: impl Into<String>, arguments: Value) -> Self {
        Self::ToolCall {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }

    pub fn tool_result(id: impl Into<String>, content: impl Into<String>, is_error: bool) -> Self {
        Self::ToolResult {
            id: id.into(),
            content: content.into(),
            is_error,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmMessage {
    pub role: Role,
    pub content: Vec<ContentPart>,
}

impl LlmMessage {
    pub fn new(role: Role, content: Vec<ContentPart>) -> Self {
        Self { role, content }
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: vec![ContentPart::text(text)],
        }
    }

    pub fn human(text: impl Into<String>) -> Self {
        Self {
            role: Role::Human,
            content: vec![ContentPart::text(text)],
        }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentPart::text(text)],
        }
    }

    pub fn tool_result(id: impl Into<String>, content: impl Into<String>, is_error: bool) -> Self {
        Self {
            role: Role::ToolResult,
            content: vec![ContentPart::tool_result(id, content, is_error)],
        }
    }
}

// ================================================================
// Completion config, response, and provider trait
// ================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionConfig {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
}

impl CompletionConfig {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            max_tokens: None,
            temperature: None,
            stop_sequences: Vec::new(),
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = sequences;
        self
    }
}

#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub content: Vec<ContentPart>,
    pub usage: TokenUsage,
    pub stop_reason: StopReason,
}

impl LlmResponse {
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    pub fn tool_calls(&self) -> Vec<&ContentPart> {
        self.content
            .iter()
            .filter(|part| matches!(part, ContentPart::ToolCall { .. }))
            .collect()
    }

    pub fn has_tool_calls(&self) -> bool {
        self.content
            .iter()
            .any(|part| matches!(part, ContentPart::ToolCall { .. }))
    }
}

#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn complete(
        &self,
        messages: Vec<LlmMessage>,
        tools: &[ToolDefinition],
        config: &CompletionConfig,
    ) -> Result<LlmResponse, LlmError>;

    async fn stream(
        &self,
        messages: Vec<LlmMessage>,
        tools: &[ToolDefinition],
        config: &CompletionConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = StreamEvent> + Send>>, LlmError>;

    fn provider_name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_serde_snake_case() {
        let role = Role::Human;
        let json = serde_json::to_string(&role).unwrap();
        assert_eq!(json, "\"human\"");

        let role = Role::ToolResult;
        let json = serde_json::to_string(&role).unwrap();
        assert_eq!(json, "\"tool_result\"");
    }

    #[test]
    fn test_role_roundtrip() {
        for role in [Role::System, Role::Human, Role::Assistant, Role::ToolResult] {
            let json = serde_json::to_string(&role).unwrap();
            let restored: Role = serde_json::from_str(&json).unwrap();
            assert_eq!(role, restored);
        }
    }

    #[test]
    fn test_content_part_text_roundtrip() {
        let part = ContentPart::text("hello world");
        let json = serde_json::to_string(&part).unwrap();
        let restored: ContentPart = serde_json::from_str(&json).unwrap();
        assert_eq!(part, restored);
    }

    #[test]
    fn test_content_part_tool_call_roundtrip() {
        let part = ContentPart::tool_call("id-1", "bash", serde_json::json!({"cmd": "ls"}));
        let json = serde_json::to_string(&part).unwrap();
        let restored: ContentPart = serde_json::from_str(&json).unwrap();
        assert_eq!(part, restored);
    }

    #[test]
    fn test_content_part_tool_result_roundtrip() {
        let part = ContentPart::tool_result("id-1", "output", false);
        let json = serde_json::to_string(&part).unwrap();
        let restored: ContentPart = serde_json::from_str(&json).unwrap();
        assert_eq!(part, restored);
    }

    #[test]
    fn test_llm_message_constructors() {
        let sys = LlmMessage::system("You are helpful");
        assert_eq!(sys.role, Role::System);
        assert_eq!(sys.content.len(), 1);

        let human = LlmMessage::human("Hello");
        assert_eq!(human.role, Role::Human);

        let assistant = LlmMessage::assistant("Hi there");
        assert_eq!(assistant.role, Role::Assistant);
    }

    #[test]
    fn test_llm_message_serde_roundtrip() {
        let msg = LlmMessage::human("Hello");
        let json = serde_json::to_string(&msg).unwrap();
        let restored: LlmMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(msg, restored);
    }

    #[test]
    fn test_completion_config_builder() {
        let config = CompletionConfig::new("claude-3-5-sonnet-20241022")
            .with_max_tokens(1024)
            .with_temperature(0.7)
            .with_stop_sequences(vec!["STOP".to_string()]);
        assert_eq!(config.model, "claude-3-5-sonnet-20241022");
        assert_eq!(config.max_tokens, Some(1024));
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.stop_sequences, vec!["STOP"]);
    }

    #[test]
    fn test_llm_response_text_content() {
        let response = LlmResponse {
            content: vec![ContentPart::text("Hello"), ContentPart::text(" world")],
            usage: TokenUsage::new(10, 20),
            stop_reason: StopReason::EndTurn,
        };
        assert_eq!(response.text_content(), "Hello world");
    }

    #[test]
    fn test_llm_response_tool_calls() {
        let response = LlmResponse {
            content: vec![
                ContentPart::text("Let me run that"),
                ContentPart::tool_call("tc-1", "bash", serde_json::json!({})),
            ],
            usage: TokenUsage::new(10, 20),
            stop_reason: StopReason::ToolUse,
        };
        assert!(response.has_tool_calls());
        assert_eq!(response.tool_calls().len(), 1);
    }

    #[test]
    fn test_types_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LlmMessage>();
        assert_send_sync::<CompletionConfig>();
        assert_send_sync::<LlmResponse>();
    }
}
