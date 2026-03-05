use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use futures::stream;
use rig::client::CompletionClient;
use rig::completion::CompletionModel;
use rig::message::{AssistantContent, Message};
use rig::providers::deepseek;

use crate::anthropic::{convert_messages_to_rig, split_system_and_chat};
use crate::error::LlmError;
use crate::provider::{
    CompletionConfig, ContentPart, LlmMessage, LlmProvider, LlmResponse, StopReason,
};
use crate::stream::{StreamEvent, TokenUsage};
use crate::tool::ToolDefinition;

pub struct DeepSeekProvider {
    client: deepseek::Client,
}

impl DeepSeekProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        let client = deepseek::Client::new(api_key.into())
            .expect("Failed to build DeepSeek client");
        Self { client }
    }

    pub fn from_env() -> Self {
        let api_key =
            std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY env var not set");
        Self::new(api_key)
    }
}

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
impl LlmProvider for DeepSeekProvider {
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

        let (history, prompt) = if rig_messages.is_empty() {
            (vec![], Message::user(""))
        } else {
            let mut history = rig_messages;
            let prompt = history.pop().unwrap();
            (history, prompt)
        };

        let model: deepseek::CompletionModel = self.client.completion_model(&config.model);
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
        "deepseek"
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_provider_name() {
        assert_eq!("deepseek", "deepseek");
    }
}
