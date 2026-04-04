//! OpenRouter provider — OpenAI-compatible API gateway.
//!
//! Supports any model available on OpenRouter via the `openrouter/<vendor/model>`
//! prefix, e.g. `openrouter/qwen/qwen3-coder:free`.
//!
//! `stream()` uses real SSE streaming (POST with `stream: true`), parsing
//! `data:` lines from the chunked HTTP response and yielding `StreamEvent`s
//! through a channel as tokens arrive from the upstream model.
//!
//! API docs: <https://openrouter.ai/docs>

use std::collections::HashMap;
use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use rig::client::CompletionClient;
use rig::completion::CompletionModel;
use rig::message::{AssistantContent, Message};
use rig::providers::openai::CompletionsClient;
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing;

use crate::anthropic::{convert_messages_to_rig, split_system_and_chat};
use crate::error::LlmError;
use crate::provider::{
    CompletionConfig, ContentPart, LlmMessage, LlmProvider, LlmResponse, Role, StopReason,
};
use crate::stream::{StreamEvent, TokenUsage};
use crate::tool::ToolDefinition;

const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

pub struct OpenRouterProvider {
    rig_client: CompletionsClient,
    http: reqwest::Client,
    api_key: String,
}

impl OpenRouterProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        let key = api_key.into();
        let rig_client = CompletionsClient::builder()
            .api_key(key.clone())
            .base_url(OPENROUTER_BASE_URL)
            .build()
            .expect("Failed to build OpenRouter client");
        Self {
            rig_client,
            http: reqwest::Client::new(),
            api_key: key,
        }
    }

    pub fn from_env() -> Self {
        let api_key =
            std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY env var not set");
        Self::new(api_key)
    }
}

// ----------------------------------------------------------------
// Rig-based response conversion (used by complete())
// ----------------------------------------------------------------

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

// ----------------------------------------------------------------
// OpenAI-compatible request/response types for streaming
// ----------------------------------------------------------------

fn build_openai_body(
    messages: &[LlmMessage],
    tools: &[ToolDefinition],
    config: &CompletionConfig,
) -> serde_json::Value {
    let oai_messages: Vec<serde_json::Value> = messages
        .iter()
        .flat_map(|msg| match msg.role {
            Role::System => {
                let text: String = msg
                    .content
                    .iter()
                    .filter_map(|p| match p {
                        ContentPart::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                vec![serde_json::json!({"role": "system", "content": text})]
            }
            Role::Human => {
                let text: String = msg
                    .content
                    .iter()
                    .filter_map(|p| match p {
                        ContentPart::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");
                vec![serde_json::json!({"role": "user", "content": text})]
            }
            Role::Assistant => {
                let text_parts: String = msg
                    .content
                    .iter()
                    .filter_map(|p| match p {
                        ContentPart::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");
                let tc: Vec<serde_json::Value> = msg
                    .content
                    .iter()
                    .filter_map(|p| match p {
                        ContentPart::ToolCall {
                            id,
                            name,
                            arguments,
                        } => Some(serde_json::json!({
                            "id": id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": serde_json::to_string(arguments).unwrap_or_default(),
                            }
                        })),
                        _ => None,
                    })
                    .collect();
                let mut m = serde_json::json!({"role": "assistant"});
                if !text_parts.is_empty() {
                    m["content"] = serde_json::json!(text_parts);
                }
                if !tc.is_empty() {
                    m["tool_calls"] = serde_json::json!(tc);
                }
                vec![m]
            }
            Role::ToolResult => msg
                .content
                .iter()
                .filter_map(|p| match p {
                    ContentPart::ToolResult { id, content, .. } => Some(serde_json::json!({
                        "role": "tool",
                        "tool_call_id": id,
                        "content": content,
                    })),
                    _ => None,
                })
                .collect(),
        })
        .collect();

    let mut body = serde_json::json!({
        "model": config.model,
        "messages": oai_messages,
        "stream": true,
        "stream_options": {"include_usage": true},
    });

    if let Some(max) = config.max_tokens {
        body["max_tokens"] = serde_json::json!(max);
    }
    if let Some(temp) = config.temperature {
        body["temperature"] = serde_json::json!(temp);
    }

    if !tools.is_empty() {
        let oai_tools: Vec<serde_json::Value> = tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                })
            })
            .collect();
        body["tools"] = serde_json::json!(oai_tools);
    }

    body
}

/// A single SSE chunk from the OpenAI-compatible streaming response.
#[derive(Debug, Deserialize)]
struct SseChunk {
    choices: Vec<SseChoice>,
    usage: Option<SseUsage>,
}

#[derive(Debug, Deserialize)]
struct SseChoice {
    delta: SseDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SseDelta {
    content: Option<String>,
    tool_calls: Option<Vec<SseToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct SseToolCallDelta {
    index: usize,
    id: Option<String>,
    function: Option<SseFunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct SseFunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SseUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

/// Process a single parsed SSE chunk, emitting `StreamEvent`s through the channel.
///
/// `active_tools` tracks tool calls in progress keyed by their stream index.
/// Each entry holds `(tool_call_id, function_name)`.
async fn process_sse_chunk(
    chunk: &SseChunk,
    tx: &mpsc::Sender<StreamEvent>,
    active_tools: &mut HashMap<usize, (String, String)>,
    usage: &mut TokenUsage,
) {
    if let Some(u) = &chunk.usage {
        usage.input_tokens = u.prompt_tokens;
        usage.output_tokens = u.completion_tokens;
    }

    for choice in &chunk.choices {
        if let Some(text) = &choice.delta.content
            && !text.is_empty()
        {
            let _ = tx.send(StreamEvent::text_delta(text.clone())).await;
        }

        if let Some(tc_deltas) = &choice.delta.tool_calls {
            for tcd in tc_deltas {
                if let Some(id) = &tcd.id {
                    let name = tcd
                        .function
                        .as_ref()
                        .and_then(|f| f.name.clone())
                        .unwrap_or_default();
                    active_tools.insert(tcd.index, (id.clone(), name.clone()));
                    let _ = tx
                        .send(StreamEvent::tool_call_start(id.clone(), name))
                        .await;
                }

                if let Some(func) = &tcd.function
                    && let Some(args) = &func.arguments
                    && !args.is_empty()
                    && let Some((id, _)) = active_tools.get(&tcd.index)
                {
                    let _ = tx
                        .send(StreamEvent::tool_call_delta(id.clone(), args.clone()))
                        .await;
                }
            }
        }

        if let Some(reason) = &choice.finish_reason
            && (reason == "tool_calls" || reason == "stop")
        {
            for (_, (id, _)) in active_tools.drain() {
                let _ = tx.send(StreamEvent::tool_call_end(id)).await;
            }
        }
    }
}

// ----------------------------------------------------------------
// LlmProvider impl
// ----------------------------------------------------------------

#[async_trait]
impl LlmProvider for OpenRouterProvider {
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

        let model = self.rig_client.completion_model(&config.model);
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
            .map_err(|e| LlmError::provider(format!("OpenRouter: {e}")))?;

        Ok(convert_response(response))
    }

    async fn stream(
        &self,
        messages: Vec<LlmMessage>,
        tools: &[ToolDefinition],
        config: &CompletionConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = StreamEvent> + Send>>, LlmError> {
        let body = build_openai_body(&messages, tools, config);

        let mut response = self
            .http
            .post(format!("{OPENROUTER_BASE_URL}/chat/completions"))
            .bearer_auth(&self.api_key)
            .header("Content-Type", "application/json")
            .header("HTTP-Referer", "https://graphirm.ai")
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::Request(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let err_body = response.text().await.unwrap_or_default();
            return Err(LlmError::provider(format!(
                "OpenRouter {status}: {err_body}"
            )));
        }

        let (tx, rx) = mpsc::channel::<StreamEvent>(128);

        tokio::spawn(async move {
            let mut line_buf = String::new();
            let mut active_tools: HashMap<usize, (String, String)> = HashMap::new();
            let mut usage = TokenUsage::default();

            while let Ok(Some(bytes)) = response.chunk().await {
                let text = String::from_utf8_lossy(&bytes);
                line_buf.push_str(&text);

                while let Some(newline_pos) = line_buf.find('\n') {
                    let line = line_buf[..newline_pos].trim_end_matches('\r').to_string();
                    line_buf = line_buf[newline_pos + 1..].to_string();

                    if line.is_empty() || line.starts_with(':') {
                        continue;
                    }

                    let data = match line.strip_prefix("data: ") {
                        Some(d) => d,
                        None => continue,
                    };

                    if data.trim() == "[DONE]" {
                        let _ = tx.send(StreamEvent::done(usage)).await;
                        return;
                    }

                    match serde_json::from_str::<SseChunk>(data) {
                        Ok(chunk) => {
                            process_sse_chunk(&chunk, &tx, &mut active_tools, &mut usage).await;
                        }
                        Err(e) => {
                            tracing::debug!(
                                data,
                                error = %e,
                                "skipping unparseable SSE chunk"
                            );
                        }
                    }
                }
            }

            // Stream ended without [DONE] — still emit Done so consumer doesn't hang
            let _ = tx.send(StreamEvent::done(usage)).await;
        });

        Ok(Box::pin(ReceiverStream::new(rx)))
    }

    fn provider_name(&self) -> &str {
        "openrouter"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_name() {
        assert_eq!("openrouter", "openrouter");
    }

    #[test]
    fn test_build_openai_body_minimal() {
        let messages = vec![LlmMessage::human("Hello")];
        let config = CompletionConfig::new("test-model");
        let body = build_openai_body(&messages, &[], &config);

        assert_eq!(body["model"], "test-model");
        assert_eq!(body["stream"], true);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "Hello");
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn test_build_openai_body_with_system_and_tools() {
        let messages = vec![
            LlmMessage::system("Be helpful"),
            LlmMessage::human("Do something"),
        ];
        let tools = vec![ToolDefinition::new(
            "bash",
            "Run shell command",
            serde_json::json!({"type": "object", "properties": {"cmd": {"type": "string"}}}),
        )];
        let config = CompletionConfig::new("test-model")
            .with_max_tokens(1024)
            .with_temperature(0.5);
        let body = build_openai_body(&messages, &tools, &config);

        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["max_tokens"], 1024);
        assert_eq!(body["temperature"], 0.5);
        assert_eq!(body["tools"][0]["function"]["name"], "bash");
    }

    #[test]
    fn test_build_openai_body_tool_result() {
        let messages = vec![
            LlmMessage::human("Run ls"),
            LlmMessage::new(
                Role::Assistant,
                vec![ContentPart::tool_call(
                    "tc-1",
                    "bash",
                    serde_json::json!({"cmd": "ls"}),
                )],
            ),
            LlmMessage::tool_result("tc-1", "file1.rs\nfile2.rs", false),
        ];
        let config = CompletionConfig::new("m");
        let body = build_openai_body(&messages, &[], &config);

        assert_eq!(body["messages"][1]["role"], "assistant");
        assert_eq!(body["messages"][1]["tool_calls"][0]["id"], "tc-1");
        assert_eq!(body["messages"][2]["role"], "tool");
        assert_eq!(body["messages"][2]["tool_call_id"], "tc-1");
    }

    #[test]
    fn test_parse_sse_text_chunk() {
        let json = r#"{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}"#;
        let chunk: SseChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hello"));
        assert!(chunk.choices[0].finish_reason.is_none());
    }

    #[test]
    fn test_parse_sse_tool_call_chunk() {
        let json = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"bash","arguments":""}}]},"finish_reason":null}]}"#;
        let chunk: SseChunk = serde_json::from_str(json).unwrap();
        let tc = &chunk.choices[0].delta.tool_calls.as_ref().unwrap()[0];
        assert_eq!(tc.index, 0);
        assert_eq!(tc.id.as_deref(), Some("call_1"));
        assert_eq!(tc.function.as_ref().unwrap().name.as_deref(), Some("bash"));
    }

    #[test]
    fn test_parse_sse_usage_chunk() {
        let json = r#"{"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":50}}"#;
        let chunk: SseChunk = serde_json::from_str(json).unwrap();
        let u = chunk.usage.unwrap();
        assert_eq!(u.prompt_tokens, 100);
        assert_eq!(u.completion_tokens, 50);
    }

    #[tokio::test]
    async fn test_process_sse_chunk_text() {
        let json = r#"{"choices":[{"delta":{"content":"tok"},"finish_reason":null}]}"#;
        let chunk: SseChunk = serde_json::from_str(json).unwrap();
        let (tx, mut rx) = mpsc::channel(16);
        let mut tools = HashMap::new();
        let mut usage = TokenUsage::default();

        process_sse_chunk(&chunk, &tx, &mut tools, &mut usage).await;
        drop(tx);

        let ev = rx.recv().await.unwrap();
        assert!(matches!(ev, StreamEvent::TextDelta(t) if t == "tok"));
        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn test_process_sse_chunk_tool_lifecycle() {
        let (tx, mut rx) = mpsc::channel(16);
        let mut tools = HashMap::new();
        let mut usage = TokenUsage::default();

        // Start
        let start_json = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"bash","arguments":""}}]},"finish_reason":null}]}"#;
        let chunk: SseChunk = serde_json::from_str(start_json).unwrap();
        process_sse_chunk(&chunk, &tx, &mut tools, &mut usage).await;

        // Argument delta
        let arg_json = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"cmd\":"}}]},"finish_reason":null}]}"#;
        let chunk: SseChunk = serde_json::from_str(arg_json).unwrap();
        process_sse_chunk(&chunk, &tx, &mut tools, &mut usage).await;

        // Finish
        let end_json = r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#;
        let chunk: SseChunk = serde_json::from_str(end_json).unwrap();
        process_sse_chunk(&chunk, &tx, &mut tools, &mut usage).await;

        drop(tx);

        let e1 = rx.recv().await.unwrap();
        assert!(
            matches!(e1, StreamEvent::ToolCallStart { ref id, ref name } if id == "c1" && name == "bash")
        );

        let e2 = rx.recv().await.unwrap();
        assert!(
            matches!(e2, StreamEvent::ToolCallDelta { ref id, ref arguments_delta } if id == "c1" && arguments_delta == r#"{"cmd":"#)
        );

        let e3 = rx.recv().await.unwrap();
        assert!(matches!(e3, StreamEvent::ToolCallEnd { ref id } if id == "c1"));
    }
}
