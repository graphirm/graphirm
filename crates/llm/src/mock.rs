use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use futures::Stream;
use futures::stream;

use crate::error::LlmError;
use crate::provider::{
    CompletionConfig, ContentPart, LlmMessage, LlmProvider, LlmResponse, StopReason,
};
use crate::stream::{StreamEvent, TokenUsage};
use crate::tool::ToolDefinition;

#[derive(Debug, Clone)]
pub struct MockResponse {
    pub text: Option<String>,
    pub tool_calls: Vec<(String, String, serde_json::Value)>,
    pub usage: TokenUsage,
    pub stop_reason: StopReason,
}

impl MockResponse {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            tool_calls: Vec::new(),
            usage: TokenUsage::new(10, 20),
            stop_reason: StopReason::EndTurn,
        }
    }

    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self {
            text: None,
            tool_calls: vec![(id.into(), name.into(), arguments)],
            usage: TokenUsage::new(15, 10),
            stop_reason: StopReason::ToolUse,
        }
    }

    pub fn with_usage(mut self, usage: TokenUsage) -> Self {
        self.usage = usage;
        self
    }

    fn into_llm_response(self) -> LlmResponse {
        let mut content = Vec::new();
        if let Some(text) = self.text {
            content.push(ContentPart::text(text));
        }
        for (id, name, args) in self.tool_calls {
            content.push(ContentPart::tool_call(id, name, args));
        }
        LlmResponse {
            content,
            usage: self.usage,
            stop_reason: self.stop_reason,
        }
    }
}

pub struct MockProvider {
    responses: Arc<Vec<MockResponse>>,
    call_count: Arc<AtomicUsize>,
}

impl MockProvider {
    pub fn new(responses: Vec<MockResponse>) -> Self {
        Self {
            responses: Arc::new(responses),
            call_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn fixed(text: impl Into<String>) -> Self {
        Self::new(vec![MockResponse::text(text)])
    }

    pub fn call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }

    fn next_response(&self) -> MockResponse {
        let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
        let responses = &self.responses;
        if responses.is_empty() {
            return MockResponse::text("[empty mock]");
        }
        let clamped = idx.min(responses.len() - 1);
        responses[clamped].clone()
    }
}

#[async_trait]
impl LlmProvider for MockProvider {
    async fn complete(
        &self,
        _messages: Vec<LlmMessage>,
        _tools: &[ToolDefinition],
        _config: &CompletionConfig,
    ) -> Result<LlmResponse, LlmError> {
        Ok(self.next_response().into_llm_response())
    }

    async fn stream(
        &self,
        _messages: Vec<LlmMessage>,
        _tools: &[ToolDefinition],
        _config: &CompletionConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = StreamEvent> + Send>>, LlmError> {
        let response = self.next_response();
        let mut events: Vec<StreamEvent> = Vec::new();

        if let Some(ref text) = response.text {
            let chunks: Vec<String> = text
                .as_bytes()
                .chunks(10)
                .map(|b| String::from_utf8_lossy(b).to_string())
                .collect();
            for chunk in chunks {
                events.push(StreamEvent::text_delta(chunk));
            }
        }

        for (id, name, args) in &response.tool_calls {
            events.push(StreamEvent::tool_call_start(id.clone(), name.clone()));
            events.push(StreamEvent::tool_call_delta(
                id.clone(),
                serde_json::to_string(args).unwrap_or_default(),
            ));
            events.push(StreamEvent::tool_call_end(id.clone()));
        }

        events.push(StreamEvent::done(response.usage.clone()));

        Ok(Box::pin(stream::iter(events)))
    }

    fn provider_name(&self) -> &str {
        "mock"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[tokio::test]
    async fn test_mock_provider_fixed_text() {
        let provider = MockProvider::fixed("Hello from mock");
        let config = CompletionConfig::new("mock-model");
        let response = provider.complete(vec![], &[], &config).await.unwrap();
        assert_eq!(response.text_content(), "Hello from mock");
        assert!(matches!(response.stop_reason, StopReason::EndTurn));
    }

    #[tokio::test]
    async fn test_mock_provider_sequences() {
        let provider = MockProvider::new(vec![
            MockResponse::text("first"),
            MockResponse::text("second"),
        ]);
        let config = CompletionConfig::new("mock-model");
        let r1 = provider.complete(vec![], &[], &config).await.unwrap();
        let r2 = provider.complete(vec![], &[], &config).await.unwrap();
        // After exhausting, repeats last
        let r3 = provider.complete(vec![], &[], &config).await.unwrap();
        assert_eq!(r1.text_content(), "first");
        assert_eq!(r2.text_content(), "second");
        assert_eq!(r3.text_content(), "second");
    }

    #[tokio::test]
    async fn test_mock_provider_tool_call() {
        let provider = MockProvider::new(vec![MockResponse::tool_call(
            "tc-1",
            "bash",
            serde_json::json!({"cmd": "ls"}),
        )]);
        let config = CompletionConfig::new("mock-model");
        let response = provider.complete(vec![], &[], &config).await.unwrap();
        assert!(response.has_tool_calls());
        assert!(matches!(response.stop_reason, StopReason::ToolUse));
    }

    #[tokio::test]
    async fn test_mock_provider_stream_text() {
        let provider = MockProvider::fixed("Hello world from streaming mock");
        let config = CompletionConfig::new("mock-model");
        let stream = provider.stream(vec![], &[], &config).await.unwrap();
        let events: Vec<StreamEvent> = stream.collect().await;

        let has_done = events.iter().any(|e| e.is_done());
        assert!(has_done, "Stream should end with Done event");

        let text: String = events
            .iter()
            .filter_map(|e| {
                if let StreamEvent::TextDelta(t) = e {
                    Some(t.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(text, "Hello world from streaming mock");
    }

    #[tokio::test]
    async fn test_mock_provider_stream_tool_calls() {
        let provider = MockProvider::new(vec![MockResponse::tool_call(
            "tc-1",
            "bash",
            serde_json::json!({"cmd": "ls"}),
        )]);
        let config = CompletionConfig::new("mock-model");
        let stream = provider.stream(vec![], &[], &config).await.unwrap();
        let events: Vec<StreamEvent> = stream.collect().await;

        let starts: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::ToolCallStart { .. }))
            .collect();
        assert_eq!(starts.len(), 1);
    }

    #[tokio::test]
    async fn test_mock_call_count() {
        let provider = MockProvider::fixed("hi");
        let config = CompletionConfig::new("mock-model");
        assert_eq!(provider.call_count(), 0);
        provider.complete(vec![], &[], &config).await.unwrap();
        provider.complete(vec![], &[], &config).await.unwrap();
        assert_eq!(provider.call_count(), 2);
    }

    #[test]
    fn test_mock_provider_name() {
        let provider = MockProvider::fixed("test");
        assert_eq!(provider.provider_name(), "mock");
    }
}
