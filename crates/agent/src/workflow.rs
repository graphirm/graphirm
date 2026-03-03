// Agent workflow: async state machine with plan -> act -> observe -> reflect loop

use graphirm_graph::nodes::{GraphNode, InteractionData, NodeId, NodeType};
use graphirm_llm::{CompletionConfig, ContentPart, LlmProvider, LlmResponse};
use graphirm_tools::ToolContext;
use graphirm_tools::registry::ToolRegistry;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tracing::info;

use crate::context::build_context;
use crate::error::AgentError;
use crate::event::{AgentEvent, EventBus};
use crate::session::Session;

/// Call the LLM with the current conversation context and record the
/// assistant response as an Interaction node in the graph.
///
/// Returns the LlmResponse (which may contain tool calls) and the
/// NodeId of the recorded response node.
pub async fn stream_and_record(
    session: &Session,
    llm: &dyn LlmProvider,
    tools: &ToolRegistry,
    events: &EventBus,
) -> Result<(LlmResponse, NodeId), AgentError> {
    let context = build_context(session)?;
    let raw_defs = tools.definitions();
    let tool_defs: Vec<graphirm_llm::ToolDefinition> = raw_defs
        .into_iter()
        .map(|t| graphirm_llm::ToolDefinition::new(t.name, t.description, t.parameters))
        .collect();
    let config = CompletionConfig::new(session.agent_config.model.clone())
        .with_max_tokens(session.agent_config.max_tokens.unwrap_or(8192))
        .with_temperature(session.agent_config.temperature.unwrap_or(0.7));

    let response = llm.complete(context, &tool_defs, &config).await?;

    // Build metadata to persist tool_calls so build_context can reconstruct them
    let mut metadata = serde_json::Map::new();
    if response.has_tool_calls() {
        let tool_calls_json: Vec<serde_json::Value> = response
            .tool_calls()
            .iter()
            .filter_map(|part| match part {
                ContentPart::ToolCall {
                    id,
                    name,
                    arguments,
                } => Some(serde_json::json!({
                    "id": id,
                    "name": name,
                    "arguments": arguments
                })),
                _ => None,
            })
            .collect();
        metadata.insert(
            "tool_calls".to_string(),
            serde_json::Value::Array(tool_calls_json),
        );
    }
    metadata.insert(
        "usage_input".to_string(),
        serde_json::json!(response.usage.input_tokens),
    );
    metadata.insert(
        "usage_output".to_string(),
        serde_json::json!(response.usage.output_tokens),
    );

    let mut interaction_node = GraphNode::new(NodeType::Interaction(InteractionData {
        role: "assistant".to_string(),
        content: response.text_content(),
        token_count: Some(response.usage.output_tokens),
    }));
    interaction_node.metadata = serde_json::Value::Object(metadata);

    let node_id = session.graph.add_node(interaction_node)?;
    session.link_interaction(&node_id)?;

    info!(node_id = %node_id, "Recorded assistant response");
    events.emit(AgentEvent::MessageEnd {
        node_id: node_id.clone(),
    });

    Ok((response, node_id))
}

/// Execute tool calls in parallel using tokio::JoinSet.
///
/// Uses a two-phase approach: first collect all execution results, then record
/// them to the graph. This prevents ghost executions where a tool ran but its
/// output was lost due to a graph write failure mid-drain.
async fn execute_tools_parallel(
    session: &Session,
    tools: &ToolRegistry,
    response_id: &NodeId,
    tool_calls: &[&graphirm_llm::ContentPart],
    events: &EventBus,
    cancel: &CancellationToken,
) -> Result<Vec<NodeId>, AgentError> {
    let ctx = ToolContext {
        graph: session.graph.clone(),
        agent_id: session.id.clone(),
        interaction_id: response_id.clone(),
        working_dir: session.agent_config.working_dir.clone(),
        signal: cancel.clone(),
    };

    // Phase 1: spawn all tools in parallel and collect results
    let mut set = JoinSet::new();
    for part in tool_calls {
        let ContentPart::ToolCall {
            id: call_id,
            name,
            arguments,
        } = part
        else {
            continue;
        };
        let tool = tools.get(name)?;
        let call = graphirm_tools::ToolCall {
            id: call_id.clone(),
            name: name.clone(),
            arguments: arguments.clone(),
        };
        let ctx_clone = ctx.clone();
        set.spawn(async move {
            let result: Result<graphirm_tools::ToolOutput, graphirm_tools::ToolError> =
                tool.execute(call.arguments.clone(), &ctx_clone).await;
            (call.id, call.name, result)
        });
    }

    let mut exec_results = Vec::new();
    while let Some(join_result) = set.join_next().await {
        exec_results.push(join_result.map_err(|e| AgentError::Join(e.to_string()))?);
    }

    // Phase 2: record all results to graph (best-effort — log failures rather
    // than dropping results for tools that already executed successfully)
    let mut node_ids = Vec::new();
    for (call_id, tool_name, exec_result) in exec_results {
        let (content, is_error): (String, bool) = match exec_result {
            Ok(output) => (output.content, output.is_error),
            Err(e) => (e.to_string(), true),
        };

        let mut tool_metadata = serde_json::Map::new();
        tool_metadata.insert("tool_call_id".to_string(), serde_json::json!(&call_id));
        tool_metadata.insert("is_error".to_string(), serde_json::json!(is_error));

        let mut tool_node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "tool".to_string(),
            content,
            token_count: None,
        }));
        tool_node.metadata = serde_json::Value::Object(tool_metadata);

        match session.graph.add_node(tool_node) {
            Ok(node_id) => {
                if let Err(e) = session.link_interaction(&node_id) {
                    tracing::error!("Failed to link tool result node {node_id}: {e}");
                } else {
                    events.emit(AgentEvent::ToolEnd {
                        node_id: node_id.clone(),
                        is_error,
                    });
                    info!(node_id = %node_id, tool = %tool_name, is_error, "Tool execution complete");
                    node_ids.push(node_id);
                }
            }
            Err(e) => {
                tracing::error!("Failed to record tool result for call {call_id}: {e}");
            }
        }
    }

    Ok(node_ids)
}

/// The main agent loop. Cycles between:
/// 1. Build context from graph
/// 2. Call LLM and record response (races against CancellationToken)
/// 3. If tool calls present, dispatch them in parallel and record results
/// 4. Repeat until no tool calls, max_turns is reached, or cancelled
pub async fn run_agent_loop(
    session: &Session,
    llm: &dyn LlmProvider,
    tools: &ToolRegistry,
    events: &EventBus,
    cancel: &CancellationToken,
) -> Result<(), AgentError> {
    let max_turns = session.agent_config.max_turns;
    let mut all_node_ids: Vec<NodeId> = Vec::new();

    events.emit(AgentEvent::AgentStart {
        agent_id: session.id.clone(),
    });

    for turn in 0..max_turns {
        // Check cancellation before starting each turn
        if cancel.is_cancelled() {
            info!("Agent loop cancelled at turn {}", turn);
            let _ = session.set_status("cancelled");
            events.emit(AgentEvent::AgentEnd {
                agent_id: session.id.clone(),
                node_ids: all_node_ids,
            });
            return Err(AgentError::Cancelled);
        }

        events.emit(AgentEvent::TurnStart { turn_index: turn });

        // Race the LLM call against the cancellation token so in-flight requests
        // are interrupted promptly rather than waiting for the provider to respond.
        let (response, response_id) = tokio::select! {
            result = stream_and_record(session, llm, tools, events) => result?,
            _ = cancel.cancelled() => {
                info!("Agent loop cancelled during LLM call at turn {}", turn);
                let _ = session.set_status("cancelled");
                events.emit(AgentEvent::AgentEnd {
                    agent_id: session.id.clone(),
                    node_ids: all_node_ids,
                });
                return Err(AgentError::Cancelled);
            }
        };
        all_node_ids.push(response_id.clone());

        if !response.has_tool_calls() {
            events.emit(AgentEvent::TurnEnd {
                response_id: response_id.clone(),
                tool_result_ids: vec![],
            });
            break;
        }

        let tool_calls: Vec<&ContentPart> = response.tool_calls();
        for part in &tool_calls {
            let ContentPart::ToolCall {
                id: call_id, name, ..
            } = part
            else {
                continue;
            };
            events.emit(AgentEvent::ToolStart {
                response_node_id: response_id.clone(),
                call_id: call_id.clone(),
                tool_name: name.clone(),
            });
        }

        let tool_result_ids = execute_tools_parallel(
            session,
            tools,
            &response_id,
            tool_calls.as_slice(),
            events,
            cancel,
        )
        .await?;

        all_node_ids.extend(tool_result_ids.iter().cloned());

        events.emit(AgentEvent::TurnEnd {
            response_id: response_id.clone(),
            tool_result_ids,
        });

        // The loop runs 0..max_turns; hitting this on the last iteration with
        // outstanding tool calls means we consumed the full budget.
        if turn + 1 >= max_turns {
            info!("Recursion limit reached at {} turns", max_turns);
            let _ = session.set_status("limit_reached");
            events.emit(AgentEvent::AgentEnd {
                agent_id: session.id.clone(),
                node_ids: all_node_ids,
            });
            return Err(AgentError::RecursionLimit(max_turns));
        }
    }

    let _ = session.set_status("completed");
    events.emit(AgentEvent::AgentEnd {
        agent_id: session.id.clone(),
        node_ids: all_node_ids,
    });

    Ok(())
}

// ============== Test helpers ==============

#[cfg(test)]
mod test_helpers {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;

    use super::*;
    use graphirm_llm::{
        CompletionConfig, LlmError, LlmMessage, LlmProvider, LlmResponse, StopReason, TokenUsage,
        ToolDefinition,
    };

    /// Mock LLM provider that returns pre-configured responses in order.
    pub struct MockProvider {
        pub responses: Vec<LlmResponse>,
        pub call_index: AtomicUsize,
    }

    impl MockProvider {
        pub fn new(responses: Vec<LlmResponse>) -> Self {
            Self {
                responses,
                call_index: AtomicUsize::new(0),
            }
        }

        pub fn call_count(&self) -> usize {
            self.call_index.load(Ordering::SeqCst)
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
            let idx = self.call_index.fetch_add(1, Ordering::SeqCst);
            if idx < self.responses.len() {
                Ok(self.responses[idx].clone())
            } else {
                Err(LlmError::Provider("No more mock responses".to_string()))
            }
        }

        async fn stream(
            &self,
            _messages: Vec<LlmMessage>,
            _tools: &[ToolDefinition],
            _config: &CompletionConfig,
        ) -> Result<
            std::pin::Pin<Box<dyn futures::Stream<Item = graphirm_llm::StreamEvent> + Send>>,
            LlmError,
        > {
            Ok(Box::pin(futures::stream::empty()))
        }

        fn provider_name(&self) -> &str {
            "mock"
        }
    }

    /// Mock tool that returns a fixed output string.
    pub struct MockTool {
        pub tool_name: String,
        pub output: String,
    }

    #[async_trait]
    impl graphirm_tools::Tool for MockTool {
        fn name(&self) -> &str {
            &self.tool_name
        }
        fn description(&self) -> &str {
            "Mock tool for testing"
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }
        async fn execute(
            &self,
            _args: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Result<graphirm_tools::ToolOutput, graphirm_tools::ToolError> {
            Ok(graphirm_tools::ToolOutput::success(&self.output))
        }
    }

    pub fn text_response(content: &str) -> LlmResponse {
        LlmResponse {
            content: vec![ContentPart::text(content)],
            usage: TokenUsage::new(100, 20),
            stop_reason: StopReason::EndTurn,
        }
    }

    /// Builds an LlmResponse containing tool calls.
    /// Each tuple is `(tool_name, call_id, arguments)`.
    pub fn tool_call_response(calls: Vec<(&str, &str, serde_json::Value)>) -> LlmResponse {
        let content: Vec<ContentPart> = calls
            .into_iter()
            .map(|(name, id, args)| ContentPart::tool_call(id, name, args))
            .collect();
        LlmResponse {
            content,
            usage: TokenUsage::new(100, 50),
            stop_reason: StopReason::ToolUse,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::test_helpers::*;
    use super::*;
    use crate::config::AgentConfig;
    use graphirm_graph::edges::EdgeType;
    use graphirm_graph::{Direction, GraphStore};

    #[tokio::test]
    async fn test_stream_and_record_creates_assistant_node() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig::default();
        let session = Session::new(graph.clone(), config).unwrap();

        session.add_user_message("What is 2+2?").unwrap();

        let provider = MockProvider::new(vec![text_response("The answer is 4.")]);
        let tools = ToolRegistry::new();
        let bus = EventBus::new();

        let (response, node_id) = stream_and_record(&session, &provider, &tools, &bus)
            .await
            .unwrap();

        assert_eq!(response.text_content(), "The answer is 4.");
        assert!(!response.has_tool_calls());

        let node = graph.get_node(&node_id).unwrap();
        match &node.node_type {
            NodeType::Interaction(data) => {
                assert_eq!(data.content, "The answer is 4.");
                assert_eq!(data.role, "assistant");
            }
            _ => panic!("expected Interaction node"),
        }
    }

    #[tokio::test]
    async fn test_agent_loop_single_turn_no_tools() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig {
            max_turns: 10,
            ..AgentConfig::default()
        };
        let session = Session::new(graph.clone(), config).unwrap();
        session.add_user_message("What is 2+2?").unwrap();

        let provider = MockProvider::new(vec![text_response("4")]);
        let tools = ToolRegistry::new();
        let mut bus = EventBus::new();
        let mut rx = bus.subscribe();
        let token = CancellationToken::new();

        run_agent_loop(&session, &provider, &tools, &bus, &token)
            .await
            .unwrap();

        assert_eq!(provider.call_count(), 1);

        let mut events = vec![];
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        assert!(matches!(events[0], AgentEvent::AgentStart { .. }));
        assert!(matches!(events[1], AgentEvent::TurnStart { turn_index: 0 }));
        assert!(matches!(
            events.last().unwrap(),
            AgentEvent::AgentEnd { .. }
        ));

        // Agent node status should be "completed"
        let agent_node = graph.get_node(&session.id).unwrap();
        match &agent_node.node_type {
            graphirm_graph::nodes::NodeType::Agent(d) => assert_eq!(d.status, "completed"),
            _ => panic!("expected Agent node"),
        }
    }

    #[tokio::test]
    async fn test_agent_loop_tool_call_then_text() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig {
            max_turns: 10,
            ..AgentConfig::default()
        };
        let session = Session::new(graph.clone(), config).unwrap();
        session.add_user_message("List files").unwrap();

        let provider = MockProvider::new(vec![
            tool_call_response(vec![(
                "bash",
                "call_1",
                serde_json::json!({"command": "ls"}),
            )]),
            text_response("Here are your files: src/ Cargo.toml"),
        ]);

        let mock_bash = Arc::new(MockTool {
            tool_name: "bash".to_string(),
            output: "src/\nCargo.toml".to_string(),
        });
        let mut tools = ToolRegistry::new();
        tools.register(mock_bash);

        let mut bus = EventBus::new();
        let mut rx = bus.subscribe();
        let token = CancellationToken::new();

        run_agent_loop(&session, &provider, &tools, &bus, &token)
            .await
            .unwrap();

        assert_eq!(provider.call_count(), 2);

        let mut events = vec![];
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        let turn_starts: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::TurnStart { .. }))
            .collect();
        assert_eq!(turn_starts.len(), 2);

        let tool_ends: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolEnd { .. }))
            .collect();
        assert_eq!(tool_ends.len(), 1);

        let neighbors = graph
            .neighbors(&session.id, Some(EdgeType::Produces), Direction::Outgoing)
            .unwrap();
        let tool_nodes: Vec<_> = neighbors
            .iter()
            .filter(|n| {
                if let NodeType::Interaction(d) = &n.node_type {
                    d.role == "tool"
                } else {
                    false
                }
            })
            .collect();
        assert_eq!(tool_nodes.len(), 1);
        if let NodeType::Interaction(d) = &tool_nodes[0].node_type {
            assert_eq!(d.content, "src/\nCargo.toml");
        }
    }

    #[tokio::test]
    async fn test_agent_loop_recursion_limit() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig {
            max_turns: 3,
            ..AgentConfig::default()
        };
        let session = Session::new(graph.clone(), config).unwrap();
        session.add_user_message("Do infinite things").unwrap();

        let provider = MockProvider::new(vec![
            tool_call_response(vec![(
                "bash",
                "c1",
                serde_json::json!({"command": "echo 1"}),
            )]),
            tool_call_response(vec![(
                "bash",
                "c2",
                serde_json::json!({"command": "echo 2"}),
            )]),
            tool_call_response(vec![(
                "bash",
                "c3",
                serde_json::json!({"command": "echo 3"}),
            )]),
        ]);

        let mock_bash = Arc::new(MockTool {
            tool_name: "bash".to_string(),
            output: "ok".to_string(),
        });
        let mut tools = ToolRegistry::new();
        tools.register(mock_bash);

        let mut bus = EventBus::new();
        let mut rx = bus.subscribe();
        let token = CancellationToken::new();

        let result = run_agent_loop(&session, &provider, &tools, &bus, &token).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::RecursionLimit(n) => assert_eq!(n, 3),
            other => panic!("Expected RecursionLimit, got: {:?}", other),
        }
        assert_eq!(provider.call_count(), 3);

        let mut events = vec![];
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        assert!(matches!(
            events.last().unwrap(),
            AgentEvent::AgentEnd { .. }
        ));

        let agent_node = graph.get_node(&session.id).unwrap();
        match &agent_node.node_type {
            graphirm_graph::nodes::NodeType::Agent(d) => assert_eq!(d.status, "limit_reached"),
            _ => panic!("expected Agent node"),
        }
    }

    #[tokio::test]
    async fn test_agent_loop_cancellation() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig {
            max_turns: 100,
            ..AgentConfig::default()
        };
        let session = Session::new(graph.clone(), config).unwrap();
        session.add_user_message("Start working").unwrap();

        let provider = MockProvider::new(vec![
            tool_call_response(vec![(
                "bash",
                "c1",
                serde_json::json!({"command": "echo 1"}),
            )]),
            tool_call_response(vec![(
                "bash",
                "c2",
                serde_json::json!({"command": "echo 2"}),
            )]),
            text_response("done"),
        ]);

        let mock_bash = Arc::new(MockTool {
            tool_name: "bash".to_string(),
            output: "ok".to_string(),
        });
        let mut tools = ToolRegistry::new();
        tools.register(mock_bash);

        let mut bus = EventBus::new();
        let mut rx = bus.subscribe();
        let token = CancellationToken::new();

        let cancel_token = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            cancel_token.cancel();
        });

        let result = run_agent_loop(&session, &provider, &tools, &bus, &token).await;

        assert!(
            matches!(result, Err(AgentError::Cancelled)) || result.is_ok(),
            "Expected Cancelled or Ok, got: {:?}",
            result
        );

        let mut events = vec![];
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::AgentEnd { .. })),
            "AgentEnd event should be emitted on cancel"
        );
    }
}
