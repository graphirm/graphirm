// Integration tests for soft escalation feature
//
// These tests verify that:
// 1. Repeated tool calls trigger soft escalation with synthesis directive
// 2. Model respects synthesis directive and produces final output (success)
// 3. Model ignoring synthesis directive after soft escalation hits hard recursion limit

use std::sync::Arc;

use async_trait::async_trait;
use graphirm_agent::{AgentConfig, EventBus, Session};
use graphirm_graph::GraphStore;
use graphirm_llm::MockProvider;
use graphirm_tools::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry};
use serde_json::json;
use tokio_util::sync::CancellationToken;

fn test_config() -> AgentConfig {
    AgentConfig {
        name: "test_agent".to_string(),
        model: "test-model".to_string(),
        system_prompt: "You are a test agent".to_string(),
        max_turns: 15,
        soft_escalation_turn: 2,  // Check for escalation after turn 2
        soft_escalation_threshold: 2, // Trigger on 2+ repeated calls
        ..AgentConfig::default()
    }
}

/// Mock read tool that always succeeds with fake file content
struct MockReadTool;

#[async_trait]
impl Tool for MockReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn description(&self) -> &str {
        "Mock read tool for testing"
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            },
            "required": ["path"]
        })
    }

    async fn execute(
        &self,
        _arguments: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        // Always return success with some content
        Ok(ToolOutput::success("// Mock file content\nfn main() {}\n"))
    }
}

fn test_registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(MockReadTool));
    registry
}

/// Test that soft escalation prevents repeated tool calls and allows synthesis
///
/// Scenario:
/// - Turns 0-2: Mock provider returns repeated `read` tool calls on the same file
/// - Turn 2: Soft escalation triggers after executing turn 2's read
/// - Turn 2 synthesis: Model returns text (synthesis succeeds)
/// - Expected: run_agent_loop returns Ok(), no RecursionLimit error
///
/// Mock responses needed:
/// - Response 0: read (turn 0)
/// - Response 1: read (turn 1)
/// - Response 2: read (turn 2) → triggers escalation
/// - Response 3: text (turn 2 synthesis)
#[tokio::test]
async fn test_agent_loop_soft_escalation_prevents_repeated_reads() {
    // Set up in-memory graph and test configuration
    let graph = Arc::new(GraphStore::open_memory().unwrap());
    let config = test_config();
    let session = Session::new(graph, config).unwrap();

    // Start with a user message to initialize the conversation
    session
        .add_user_message("Analyze the file /same/file.rs")
        .await
        .unwrap();

    // Mock provider: turns 0-2 return repeated read calls, response 3 is synthesis text
    let responses = vec![
        graphirm_llm::MockResponse::tool_call("call-0", "read", serde_json::json!({"path": "/same/file.rs"})),
        graphirm_llm::MockResponse::tool_call("call-1", "read", serde_json::json!({"path": "/same/file.rs"})),
        graphirm_llm::MockResponse::tool_call("call-2", "read", serde_json::json!({"path": "/same/file.rs"})),
        // This is called when escalation is triggered and synthesis is requested
        graphirm_llm::MockResponse::text("Based on my analysis of /same/file.rs, I found the following..."),
    ];

    let provider = MockProvider::new(responses);
    let tools = test_registry();
    let events = EventBus::new();
    let cancel = CancellationToken::new();

    // Run the agent loop
    let result = graphirm_agent::workflow::run_agent_loop(
        &session,
        &provider,
        &tools,
        &events,
        &cancel,
    )
    .await;

    // Assert: should succeed (escalation triggered, model synthesized)
    assert!(
        result.is_ok(),
        "Expected soft escalation to succeed with synthesis, got: {:?}",
        result
    );
}

/// Test that hard recursion limit is enforced if model ignores synthesis directive
///
/// Scenario:
/// - Turns 0-2: Mock provider returns repeated `read` tool calls
/// - Turn 2: Soft escalation triggers after executing turn 2's read
/// - Turn 2 synthesis: Model IGNORES synthesis directive, returns another tool call
/// - Expected: run_agent_loop returns Err(RecursionLimit)
///
/// Mock responses needed:
/// - Response 0: read (turn 0)
/// - Response 1: read (turn 1)
/// - Response 2: read (turn 2) → triggers escalation
/// - Response 3: read (turn 2 synthesis - model ignores directive)
#[tokio::test]
async fn test_agent_loop_hard_limit_if_model_ignores_synthesis() {
    // Set up in-memory graph and test configuration
    let graph = Arc::new(GraphStore::open_memory().unwrap());
    let config = test_config();
    let session = Session::new(graph, config).unwrap();

    // Start with a user message
    session
        .add_user_message("Analyze the file /same/file.rs")
        .await
        .unwrap();

    // Mock provider: turns 0-3 all return read (ignoring synthesis directive at turn 2)
    let responses = vec![
        graphirm_llm::MockResponse::tool_call("call-0", "read", serde_json::json!({"path": "/same/file.rs"})),
        graphirm_llm::MockResponse::tool_call("call-1", "read", serde_json::json!({"path": "/same/file.rs"})),
        graphirm_llm::MockResponse::tool_call("call-2", "read", serde_json::json!({"path": "/same/file.rs"})),
        // Model ignores synthesis directive and returns another read
        graphirm_llm::MockResponse::tool_call("call-3-ignore", "read", serde_json::json!({"path": "/same/file.rs"})),
    ];

    let provider = MockProvider::new(responses);
    let tools = test_registry();
    let events = EventBus::new();
    let cancel = CancellationToken::new();

    // Run the agent loop
    let result = graphirm_agent::workflow::run_agent_loop(
        &session,
        &provider,
        &tools,
        &events,
        &cancel,
    )
    .await;

    // Assert: should fail with RecursionLimit
    assert!(
        matches!(result, Err(graphirm_agent::error::AgentError::RecursionLimit(_))),
        "Expected RecursionLimit error when model ignores synthesis, got: {:?}",
        result
    );
}

/// Test that escalation is only triggered after the specified turn threshold
///
/// Scenario:
/// - Turns 0-3: Model returns tool calls but before soft_escalation_turn (10)
/// - No escalation should be triggered
/// - Turn 4: Model returns text to finish
/// - Expected: run_agent_loop returns Ok()
#[tokio::test]
async fn test_soft_escalation_respects_turn_threshold() {
    let graph = Arc::new(GraphStore::open_memory().unwrap());
    // Use higher soft_escalation_turn so escalation doesn't trigger
    let config = AgentConfig {
        name: "test_agent".to_string(),
        model: "test-model".to_string(),
        system_prompt: "You are a test agent".to_string(),
        max_turns: 10,
        soft_escalation_turn: 10,  // Set very high so escalation doesn't trigger
        soft_escalation_threshold: 2,
        ..AgentConfig::default()
    };
    let session = Session::new(graph, config).unwrap();

    session
        .add_user_message("Read a file multiple times")
        .await
        .unwrap();

    // Mock provider: repeated calls for turns 0-3, then text on turn 4
    // Escalation turn is set to 10, so we never reach that threshold
    let responses = vec![
        graphirm_llm::MockResponse::tool_call("c1", "read", serde_json::json!({"path": "/file.rs"})),
        graphirm_llm::MockResponse::tool_call("c2", "read", serde_json::json!({"path": "/file.rs"})),
        graphirm_llm::MockResponse::tool_call("c3", "read", serde_json::json!({"path": "/file.rs"})),
        graphirm_llm::MockResponse::tool_call("c4", "read", serde_json::json!({"path": "/file.rs"})),
        graphirm_llm::MockResponse::text("Here's what I found..."),
    ];

    let provider = MockProvider::new(responses);
    let tools = test_registry();
    let events = EventBus::new();
    let cancel = CancellationToken::new();

    let result = graphirm_agent::workflow::run_agent_loop(
        &session,
        &provider,
        &tools,
        &events,
        &cancel,
    )
    .await;

    // Should succeed — escalation checks don't apply before turn 10
    assert!(result.is_ok());
}
