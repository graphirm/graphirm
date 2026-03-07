use futures::StreamExt;
use graphirm_llm::{
    CompletionConfig, LlmMessage, LlmProvider, MockProvider, MockResponse, StopReason, StreamEvent,
    TokenUsage, ToolDefinition,
};

#[tokio::test]
async fn test_full_roundtrip_text_only() {
    let provider = MockProvider::fixed("Hello, I am your AI assistant!");
    let config = CompletionConfig::new("mock-model");
    let messages = vec![
        LlmMessage::system("You are a helpful assistant"),
        LlmMessage::human("Who are you?"),
    ];

    let response = provider.complete(messages, &[], &config).await.unwrap();
    assert_eq!(response.text_content(), "Hello, I am your AI assistant!");
    assert!(!response.has_tool_calls());
    assert!(matches!(response.stop_reason, StopReason::EndTurn));
}

#[tokio::test]
async fn test_full_roundtrip_with_tool_calls() {
    let provider = MockProvider::new(vec![
        MockResponse::tool_call("tc-1", "bash", serde_json::json!({"cmd": "ls -la"})),
        MockResponse::text("The directory contains: README.md, src/"),
    ]);
    let config = CompletionConfig::new("mock-model");

    let tool = ToolDefinition::with_properties(
        "bash",
        "Run a bash command",
        vec![("cmd", "string", "Command to run")],
        vec!["cmd"],
    );

    // Turn 1: ask a question
    let messages = vec![LlmMessage::human("What's in this directory?")];
    let turn1 = provider
        .complete(messages.clone(), &[tool.clone()], &config)
        .await
        .unwrap();
    assert!(turn1.has_tool_calls());
    assert!(matches!(turn1.stop_reason, StopReason::ToolUse));

    // Turn 2: provide tool result
    let mut messages2 = messages;
    messages2.push(LlmMessage::new(
        graphirm_llm::Role::Assistant,
        turn1.content.clone(),
    ));
    messages2.push(LlmMessage::tool_result("tc-1", "README.md  src/", false));

    let turn2 = provider
        .complete(messages2, &[tool], &config)
        .await
        .unwrap();
    assert_eq!(
        turn2.text_content(),
        "The directory contains: README.md, src/"
    );
    assert!(!turn2.has_tool_calls());
}

#[tokio::test]
async fn test_streaming_roundtrip() {
    let provider = MockProvider::new(vec![MockResponse::tool_call(
        "tc-1",
        "read_file",
        serde_json::json!({"path": "/etc/hosts"}),
    )]);
    let config = CompletionConfig::new("mock-model");
    let tool = ToolDefinition::new("read_file", "Read a file", serde_json::json!({}));

    let stream = provider
        .stream(vec![LlmMessage::human("Read /etc/hosts")], &[tool], &config)
        .await
        .unwrap();

    let events: Vec<StreamEvent> = stream.collect().await;
    let done_count = events.iter().filter(|e| e.is_done()).count();
    assert_eq!(done_count, 1, "Exactly one Done event");

    let start_count = events
        .iter()
        .filter(|e| matches!(e, StreamEvent::ToolCallStart { .. }))
        .count();
    assert_eq!(start_count, 1, "One tool call start");
}

#[tokio::test]
async fn test_provider_as_trait_object() {
    let provider: Box<dyn LlmProvider> = Box::new(MockProvider::fixed("I am a trait object"));
    let config = CompletionConfig::new("mock-model");
    let response = provider
        .complete(vec![LlmMessage::human("hello")], &[], &config)
        .await
        .unwrap();
    assert_eq!(response.text_content(), "I am a trait object");
    assert_eq!(provider.provider_name(), "mock");
}

#[tokio::test]
async fn test_token_usage_tracking() {
    let custom_usage = TokenUsage {
        input_tokens: 42,
        output_tokens: 100,
        cache_read_tokens: Some(10),
        cache_write_tokens: None,
    };
    let provider = MockProvider::new(vec![
        MockResponse::text("hi").with_usage(custom_usage.clone()),
    ]);
    let config = CompletionConfig::new("mock-model");

    let response = provider
        .complete(vec![LlmMessage::human("hello")], &[], &config)
        .await
        .unwrap();

    assert_eq!(response.usage.input_tokens, 42);
    assert_eq!(response.usage.output_tokens, 100);
    assert_eq!(response.usage.cache_read_tokens, Some(10));
    assert_eq!(response.usage.total(), 142);
}

#[tokio::test]
async fn test_multi_turn_conversation_flow() {
    let provider = MockProvider::new(vec![
        MockResponse::text("Hello! How can I help?"),
        MockResponse::text("The capital of France is Paris."),
    ]);
    let config = CompletionConfig::new("mock-model");

    // Turn 1
    let mut history = vec![LlmMessage::human("Hello!")];
    let r1 = provider
        .complete(history.clone(), &[], &config)
        .await
        .unwrap();
    assert_eq!(r1.text_content(), "Hello! How can I help?");

    // Turn 2 — append assistant response and new user message
    history.push(LlmMessage::new(
        graphirm_llm::Role::Assistant,
        r1.content.clone(),
    ));
    history.push(LlmMessage::human("What is the capital of France?"));

    let r2 = provider.complete(history, &[], &config).await.unwrap();
    assert_eq!(r2.text_content(), "The capital of France is Paris.");
}
