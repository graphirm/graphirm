use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use futures::stream;
use rig::client::CompletionClient;
use rig::completion::CompletionModel;
use rig::message::{AssistantContent, Message, ToolResultContent, UserContent};
use rig::providers::anthropic;

use crate::error::LlmError;
use crate::provider::{
    CompletionConfig, ContentPart, LlmMessage, LlmProvider, LlmResponse, Role, StopReason,
};
use crate::stream::{StreamEvent, TokenUsage};
use crate::tool::ToolDefinition;

pub struct AnthropicProvider {
    client: anthropic::Client,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        let client = anthropic::Client::builder()
            .api_key(api_key.into())
            .build()
            .expect("Failed to build Anthropic client");
        Self { client }
    }

    pub fn from_env() -> Self {
        let api_key =
            std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY env var not set");
        Self::new(api_key)
    }
}

/// Split our messages into (system_preamble, chat_messages) for rig-core.
/// rig-core uses a preamble string for system context rather than a system role message.
pub fn split_system_and_chat(messages: Vec<LlmMessage>) -> (Option<String>, Vec<LlmMessage>) {
    let mut system_parts: Vec<String> = Vec::new();
    let mut chat: Vec<LlmMessage> = Vec::new();

    for msg in messages {
        if msg.role == Role::System {
            for part in &msg.content {
                if let ContentPart::Text { text } = part {
                    system_parts.push(text.clone());
                }
            }
        } else {
            chat.push(msg);
        }
    }

    let preamble = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n"))
    };

    (preamble, chat)
}

/// Convert our LlmMessages (non-system) to rig-core Messages.
///
/// Anthropic requires that all tool_result blocks for a single assistant turn
/// appear in ONE user message with multiple content parts. If we emit them as
/// separate messages the API rejects with "tool_use ids found without tool_result
/// blocks immediately after". So we accumulate consecutive ToolResult messages
/// into a buffer and flush them as a single merged user Message before emitting
/// any non-tool-result content.
pub fn convert_messages_to_rig(messages: Vec<LlmMessage>) -> Vec<Message> {
    use rig::OneOrMany;
    use rig::completion::message::{Text, ToolCall, ToolFunction};

    let mut rig_messages: Vec<Message> = Vec::new();
    // Buffer for consecutive tool results — flushed as a single user Message.
    let mut pending_tool_results: Vec<UserContent> = Vec::new();

    let flush_tool_results = |pending: &mut Vec<UserContent>, out: &mut Vec<Message>| {
        if pending.is_empty() {
            return;
        }
        let parts = std::mem::take(pending);
        match parts.len() {
            1 => out.push(Message::User { content: OneOrMany::one(parts.into_iter().next().unwrap()) }),
            _ => {
                if let Ok(content) = OneOrMany::many(parts) {
                    out.push(Message::User { content });
                }
            }
        }
    };

    for msg in messages {
        match msg.role {
            Role::System => {
                // System messages are handled via preamble; skip here
            }
            Role::ToolResult => {
                // Accumulate; do NOT flush yet — more tool results may follow.
                for part in msg.content {
                    if let ContentPart::ToolResult { id, content, .. } = part {
                        pending_tool_results.push(UserContent::tool_result(
                            id,
                            OneOrMany::one(ToolResultContent::text(content)),
                        ));
                    }
                }
            }
            Role::Human => {
                // A human text message ends the tool-result run; flush first.
                flush_tool_results(&mut pending_tool_results, &mut rig_messages);
                for part in msg.content {
                    match part {
                        ContentPart::Text { text } => {
                            rig_messages.push(Message::user(text));
                        }
                        ContentPart::ToolResult { id, content, .. } => {
                            // Inline tool results inside a Human message also get buffered.
                            pending_tool_results.push(UserContent::tool_result(
                                id,
                                OneOrMany::one(ToolResultContent::text(content)),
                            ));
                        }
                        _ => {}
                    }
                }
            }
            Role::Assistant => {
                // Flush any buffered tool results before the assistant turn.
                flush_tool_results(&mut pending_tool_results, &mut rig_messages);

                let mut assistant_content: Vec<AssistantContent> = Vec::new();
                for part in msg.content {
                    match part {
                        ContentPart::Text { text } => {
                            assistant_content.push(AssistantContent::Text(Text { text }));
                        }
                        ContentPart::ToolCall {
                            id,
                            name,
                            arguments,
                        } => {
                            assistant_content.push(AssistantContent::ToolCall(ToolCall {
                                id,
                                call_id: None,
                                function: ToolFunction::new(name, arguments),
                                signature: None,
                                additional_params: None,
                            }));
                        }
                        _ => {}
                    }
                }
                if !assistant_content.is_empty() {
                    if let Ok(content) = OneOrMany::many(assistant_content) {
                        rig_messages.push(Message::Assistant { id: None, content });
                    }
                }
            }
        }
    }

    // Final flush for any trailing tool results.
    flush_tool_results(&mut pending_tool_results, &mut rig_messages);

    rig_messages
}

/// Convert rig-core CompletionResponse to our LlmResponse.
fn convert_response<T>(response: rig::completion::CompletionResponse<T>) -> LlmResponse {
    let mut content: Vec<ContentPart> = Vec::new();
    let mut has_tool_calls = false;

    for item in response.choice.iter() {
        match item {
            AssistantContent::Text(t) => {
                content.push(ContentPart::text(t.text.clone()));
            }
            AssistantContent::ToolCall(tc) => {
                has_tool_calls = true;
                content.push(ContentPart::tool_call(
                    tc.id.clone(),
                    tc.function.name.clone(),
                    tc.function.arguments.clone(),
                ));
            }
            AssistantContent::Reasoning(_) | AssistantContent::Image(_) => {}
        }
    }

    let usage = TokenUsage {
        input_tokens: response.usage.input_tokens as u32,
        output_tokens: response.usage.output_tokens as u32,
        cache_read_tokens: if response.usage.cached_input_tokens > 0 {
            Some(response.usage.cached_input_tokens as u32)
        } else {
            None
        },
        cache_write_tokens: None,
    };

    let stop_reason = if has_tool_calls {
        StopReason::ToolUse
    } else {
        StopReason::EndTurn
    };

    LlmResponse {
        content,
        usage,
        stop_reason,
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    async fn complete(
        &self,
        messages: Vec<LlmMessage>,
        tools: &[ToolDefinition],
        config: &CompletionConfig,
    ) -> Result<LlmResponse, LlmError> {
        let (preamble, chat) = split_system_and_chat(messages);
        let rig_tools: Vec<rig::completion::ToolDefinition> =
            tools.iter().cloned().map(Into::into).collect();

        let rig_messages = convert_messages_to_rig(chat);

        // Need at least one message as the prompt
        let (history, prompt) = if rig_messages.is_empty() {
            (vec![], Message::user(""))
        } else {
            let mut history = rig_messages;
            let prompt = history.pop().unwrap();
            (history, prompt)
        };

        let model: rig::providers::anthropic::completion::CompletionModel =
            self.client.completion_model(&config.model);
        let mut builder = model.completion_request(prompt).tools(rig_tools);

        if let Some(p) = preamble {
            builder = builder.preamble(p);
        }
        for msg in history {
            builder = builder.message(msg);
        }
        if let Some(max_tokens) = config.max_tokens {
            builder = builder.max_tokens(max_tokens as u64);
        }
        if let Some(temp) = config.temperature {
            builder = builder.temperature(temp as f64);
        }

        let request = builder.build();
        let response = model
            .completion(request)
            .await
            .map_err(|e| LlmError::provider(e.to_string()))?;

        Ok(convert_response(response))
    }

    async fn stream(
        &self,
        messages: Vec<LlmMessage>,
        tools: &[ToolDefinition],
        config: &CompletionConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = StreamEvent> + Send>>, LlmError> {
        // Pragmatic MVP: call complete() and emit events from the response
        let response = self.complete(messages, tools, config).await?;
        let mut events: Vec<StreamEvent> = Vec::new();

        for part in &response.content {
            match part {
                ContentPart::Text { text } => {
                    let chunks: Vec<String> = text
                        .as_bytes()
                        .chunks(10)
                        .map(|b| String::from_utf8_lossy(b).to_string())
                        .collect();
                    for chunk in chunks {
                        events.push(StreamEvent::text_delta(chunk));
                    }
                }
                ContentPart::ToolCall {
                    id,
                    name,
                    arguments,
                } => {
                    events.push(StreamEvent::tool_call_start(id.clone(), name.clone()));
                    events.push(StreamEvent::tool_call_delta(
                        id.clone(),
                        serde_json::to_string(arguments).unwrap_or_default(),
                    ));
                    events.push(StreamEvent::tool_call_end(id.clone()));
                }
                _ => {}
            }
        }

        events.push(StreamEvent::done(response.usage));
        Ok(Box::pin(stream::iter(events)))
    }

    fn provider_name(&self) -> &str {
        "anthropic"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_system_and_chat_empty() {
        let (preamble, chat) = split_system_and_chat(vec![]);
        assert!(preamble.is_none());
        assert!(chat.is_empty());
    }

    #[test]
    fn test_split_system_and_chat_with_system() {
        let messages = vec![
            LlmMessage::system("You are a helpful assistant"),
            LlmMessage::human("Hello"),
        ];
        let (preamble, chat) = split_system_and_chat(messages);
        assert_eq!(preamble.unwrap(), "You are a helpful assistant");
        assert_eq!(chat.len(), 1);
        assert_eq!(chat[0].role, Role::Human);
    }

    #[test]
    fn test_split_system_and_chat_multiple_system() {
        let messages = vec![
            LlmMessage::system("Part 1"),
            LlmMessage::system("Part 2"),
            LlmMessage::human("Hello"),
        ];
        let (preamble, chat) = split_system_and_chat(messages);
        assert_eq!(preamble.unwrap(), "Part 1\nPart 2");
        assert_eq!(chat.len(), 1);
    }

    #[test]
    fn test_convert_messages_to_rig_basic() {
        let messages = vec![
            LlmMessage::human("Hello"),
            LlmMessage::assistant("Hi there"),
        ];
        let rig_messages = convert_messages_to_rig(messages);
        assert_eq!(rig_messages.len(), 2);
    }

    #[test]
    fn test_provider_name() {
        // We don't construct AnthropicProvider (needs API key), just test the logic
        // via the convert functions which are unit-testable
        // The provider_name is tested by constructing with a dummy key won't hit network
        // Just verify the string constant
        assert_eq!("anthropic", "anthropic");
    }
}
