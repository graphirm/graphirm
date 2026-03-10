// Agent workflow: async state machine with plan -> act -> observe -> reflect loop

use graphirm_graph::edges::{EdgeType, GraphEdge};
use graphirm_graph::nodes::{ContentData, GraphNode, InteractionData, NodeId, NodeType};
use graphirm_llm::{CompletionConfig, ContentPart, LlmProvider, LlmResponse};
use graphirm_tools::ToolContext;
use graphirm_tools::registry::ToolRegistry;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tracing::info;

use crate::error::AgentError;
use crate::event::{AgentEvent, EventBus};
use crate::hitl::HitlDecision;
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
    // Append cross-session memory context to the system prompt if available.
    let suffix = session.memory_suffix().await;
    let system_prompt = if suffix.is_empty() {
        session.agent_config.system_prompt.clone()
    } else {
        format!("{}\n\n{}", session.agent_config.system_prompt, suffix)
    };
    let context_config = crate::context::ContextConfig {
        system_prompt,
        max_tokens: session
            .agent_config
            .max_tokens
            .map(|t| t as usize)
            .unwrap_or(100_000),
        ..crate::context::ContextConfig::default()
    };
    let graph_ref = session.graph.clone();
    let session_id_ref = session.id.clone();
    let window = tokio::task::spawn_blocking(move || {
        crate::context::build_context(&graph_ref, &session_id_ref, &context_config)
    })
    .await
    .map_err(|e| AgentError::Join(e.to_string()))??;
    let mut context = Vec::with_capacity(1 + window.messages.len());
    context.push(window.system);
    context.extend(window.messages);
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

    let node_id = session.record_interaction(interaction_node).await?;

    info!(node_id = %node_id, "Recorded assistant response");

    // Emit the full response as a stream of events so the TUI can render it.
    // We use complete() rather than true streaming, so we synthesise the
    // MessageStart → MessageDelta(s) → MessageEnd sequence after the fact.
    events.emit(AgentEvent::MessageStart {
        node_id: node_id.clone(),
    });
    let text = response.text_content();
    if !text.is_empty() {
        events.emit(AgentEvent::MessageDelta {
            node_id: node_id.clone(),
            delta: graphirm_llm::StreamEvent::TextDelta(text),
        });
    }
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
///
/// When `session.hitl` is `Some`, destructive tools (`write`, `edit`, `bash`)
/// are pulled out of the parallel set and processed sequentially, each awaiting
/// a human approval decision before executing.
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
        turn: session.current_turn(),
        turn_pos_counter: session.turn_position_counter(),
    };

    // Partition tool calls: destructive ones go through sequential HITL approval,
    // safe ones run in parallel without gating.
    // `.copied()` turns `&&ContentPart` (from iterating `&[&ContentPart]`) into
    // `&ContentPart` so the partition buckets are `Vec<&ContentPart>`.
    let (safe_calls, destructive_calls): (Vec<_>, Vec<_>) = tool_calls
        .iter()
        .copied()
        .partition(|part| {
            let ContentPart::ToolCall { name, .. } = part else {
                return true;
            };
            !crate::hitl::is_destructive_tool(name.as_str()) || session.hitl.is_none()
        });

    // Phase 1: spawn SAFE tools in parallel and collect results
    let mut set = JoinSet::new();
    for part in safe_calls {
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

    // Phase 2: record safe tool results to graph (best-effort — log failures
    // rather than dropping results for tools that already executed successfully)
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

        match session.record_interaction(tool_node).await {
            Ok(node_id) => {
                events.emit(AgentEvent::ToolEnd {
                    node_id: node_id.clone(),
                    is_error,
                });
                info!(node_id = %node_id, tool = %tool_name, is_error, "Tool execution complete");
                node_ids.push(node_id);
            }
            Err(e) => {
                tracing::error!("Failed to record tool result for call {call_id}: {e}");
            }
        }
    }

    // Phase 3: process destructive calls sequentially, each awaiting HITL approval.
    // `destructive_calls` is empty when `session.hitl.is_none()` (see partition above),
    // so this loop is a no-op in the non-HITL code path.
    for part in destructive_calls {
        let ContentPart::ToolCall {
            id: call_id,
            name,
            arguments,
        } = part
        else {
            continue;
        };

        // SAFETY: partition guarantees destructive_calls is non-empty only when hitl is Some.
        let hitl = session.hitl.as_ref().expect("hitl must be Some for destructive calls");

        let gate_key = NodeId::from(call_id.as_str());

        events.emit(AgentEvent::AwaitingApproval {
            node_id: gate_key.clone(),
            tool_name: name.clone(),
            arguments: arguments.clone(),
            is_pause: false,
        });

        let rx = hitl.gate(&gate_key).await;

        let decision = tokio::select! {
            result = rx => match result {
                Ok(d) => d,
                // Sender dropped — treat as rejection to avoid silently executing a destructive tool.
                Err(_) => HitlDecision::Reject("Gate sender dropped unexpectedly".to_string()),
            },
            _ = cancel.cancelled() => {
                let _ = session.set_status("cancelled").await;
                return Err(AgentError::Cancelled);
            }
        };

        match decision {
            HitlDecision::Approve | HitlDecision::Modify(_) => {
                let exec_args = match &decision {
                    HitlDecision::Modify(new_args) => new_args.clone(),
                    _ => arguments.clone(),
                };

                let tool = tools.get(name)?;
                let exec_result = tool.execute(exec_args, &ctx).await;

                let (content, is_error) = match exec_result {
                    Ok(output) => (output.content, output.is_error),
                    Err(e) => (e.to_string(), true),
                };

                let mut tool_metadata = serde_json::Map::new();
                tool_metadata.insert("tool_call_id".to_string(), serde_json::json!(call_id));
                tool_metadata.insert("is_error".to_string(), serde_json::json!(is_error));

                let mut tool_node = GraphNode::new(NodeType::Interaction(InteractionData {
                    role: "tool".to_string(),
                    content,
                    token_count: None,
                }));
                tool_node.metadata = serde_json::Value::Object(tool_metadata);

                match session.record_interaction(tool_node).await {
                    Ok(result_node_id) => {
                        let edge = GraphEdge::new(
                            EdgeType::ApprovedBy,
                            result_node_id.clone(),
                            session.id.clone(),
                        );
                        let _ = session.graph.add_edge(edge);

                        events.emit(AgentEvent::ToolEnd {
                            node_id: result_node_id.clone(),
                            is_error,
                        });
                        info!(
                            node_id = %result_node_id,
                            tool = %name,
                            is_error,
                            "Tool execution complete (HITL approved)"
                        );
                        node_ids.push(result_node_id);
                    }
                    Err(e) => {
                        tracing::error!(
                            "Failed to record HITL tool result for call {call_id}: {e}"
                        );
                    }
                }
            }
            HitlDecision::Reject(reason) => {
                let mut rejection_node = GraphNode::new(NodeType::Content(ContentData {
                    content_type: "tool_rejection".to_string(),
                    path: None,
                    body: format!("Tool call '{name}' rejected: {reason}"),
                    language: None,
                }));
                rejection_node.metadata["session_id"] = serde_json::json!(session.id.to_string());
                rejection_node.set_label(format!(
                    "content_{}_{}_1",
                    session.current_turn(),
                    session.next_turn_pos()
                ));

                match session.graph.add_node(rejection_node) {
                    Ok(rejection_id) => {
                        let _ = session.graph.add_edge(GraphEdge::new(
                            EdgeType::Produces,
                            response_id.clone(),
                            rejection_id.clone(),
                        ));
                        let _ = session.graph.add_edge(GraphEdge::new(
                            EdgeType::RejectedBy,
                            rejection_id.clone(),
                            session.id.clone(),
                        ));

                        events.emit(AgentEvent::ToolEnd {
                            node_id: rejection_id.clone(),
                            is_error: true,
                        });
                        info!(
                            node_id = %rejection_id,
                            tool = %name,
                            "Tool call rejected by human"
                        );
                        node_ids.push(rejection_id);
                    }
                    Err(e) => {
                        tracing::error!(
                            "Failed to record tool rejection for call {call_id}: {e}"
                        );
                    }
                }
            }
        }
    }

    Ok(node_ids)
}

/// Detect repeated tool calls and trigger soft escalation if detected.
/// Returns true if escalation was triggered (caller should handle synthesis directive).
fn check_soft_escalation(
    turn: u32,
    config: &crate::config::AgentConfig,
    response: &graphirm_llm::LlmResponse,
    events: &EventBus,
) -> bool {
    if turn < config.soft_escalation_turn {
        return false;
    }

    // Extract tool names from current response
    let current_tools: Vec<&str> = response
        .tool_calls()
        .iter()
        .filter_map(|part| {
            if let graphirm_llm::ContentPart::ToolCall { name, .. } = part {
                Some(name.as_str())
            } else {
                None
            }
        })
        .collect();

    if current_tools.is_empty() {
        return false;
    }

    // Simple heuristic: if calling the same tool multiple times in a row,
    // that's a sign of repetition. In a real implementation, this would
    // traverse the graph to count recent identical tool calls.
    let all_same = current_tools.iter().all(|&t| t == current_tools[0]);
    let threshold = config.soft_escalation_threshold;

    if all_same && current_tools.len() >= threshold {
        let tool_name = current_tools[0];
        let synthesis_directive = format!(
            "You've called '{}' {} times. Please synthesize what you've learned so far \
             instead of making more identical calls.",
            tool_name, current_tools.len()
        );

        events.emit(AgentEvent::SoftEscalationTriggered {
            turn,
            repeated_tool_calls: current_tools.len(),
            synthesis_directive: synthesis_directive.clone(),
        });

        return true;
    }

    false
}

/// Emit a GraphUpdate event with a fresh snapshot of the 50 most recent nodes.
/// Errors from querying the store are logged and silently swallowed so a
/// display refresh failure never crashes the agent loop.
async fn emit_graph_update(
    session: &Session,
    node_id: &NodeId,
    edge_ids: Vec<NodeId>,
    events: &EventBus,
) {
    let graph = session.graph.clone();
    let recent_nodes = match tokio::task::spawn_blocking(move || graph.list_recent_nodes(50))
        .await
    {
        Ok(Ok(nodes)) => nodes,
        Ok(Err(e)) => {
            tracing::warn!("GraphUpdate: failed to fetch recent nodes: {e}");
            return;
        }
        Err(e) => {
            tracing::warn!("GraphUpdate: spawn_blocking panicked: {e}");
            return;
        }
    };
    events.emit(AgentEvent::GraphUpdate {
        node_id: node_id.clone(),
        edge_ids: edge_ids
            .into_iter()
            .map(|id| graphirm_graph::edges::EdgeId(id.0))
            .collect(),
        recent_nodes,
    });
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

    // Pre-loop: inject relevant knowledge from past sessions into system prompt.
    if let Some(retriever) = session.memory_retriever() {
        let query = session.recent_user_message().await.unwrap_or_default();
        match retriever.retrieve_relevant(&query, 5).await {
            Ok(nodes) => {
                let context =
                    crate::knowledge::injection::format_memory_context(&nodes);
                if !context.is_empty() {
                    session.set_memory_suffix(context).await;
                    tracing::info!(
                        count = nodes.len(),
                        "Injected memory nodes into session context"
                    );
                }
            }
            Err(e) => tracing::warn!(error = %e, "Memory retrieval failed (non-fatal)"),
        }
    }

    for turn in 0..max_turns {
        // Check cancellation before starting each turn
        if cancel.is_cancelled() {
            info!("Agent loop cancelled at turn {}", turn);
            let _ = session.set_status("cancelled").await;
            events.emit(AgentEvent::AgentEnd {
                agent_id: session.id.clone(),
                node_ids: all_node_ids,
            });
            return Err(AgentError::Cancelled);
        }

        // Check manual pause flag before starting each turn.
        if let Some(ref hitl) = session.hitl {
            while hitl.is_paused() {
                events.emit(AgentEvent::AwaitingApproval {
                    node_id: session.id.clone(),
                    tool_name: "pause".to_string(),
                    arguments: serde_json::json!({}),
                    is_pause: true,
                });
                let rx = hitl.gate(&session.id).await;
                tokio::select! {
                    _ = rx => { /* unblocked by resume */ }
                    _ = cancel.cancelled() => {
                        let _ = session.set_status("cancelled").await;
                        return Err(AgentError::Cancelled);
                    }
                }
            }
        }

        events.emit(AgentEvent::TurnStart { turn_index: turn });

        // Race the LLM call against the cancellation token so in-flight requests
        // are interrupted promptly rather than waiting for the provider to respond.
        let (response, response_id) = tokio::select! {
            result = stream_and_record(session, llm, tools, events) => result?,
            _ = cancel.cancelled() => {
                info!("Agent loop cancelled during LLM call at turn {}", turn);
                let _ = session.set_status("cancelled").await;
                events.emit(AgentEvent::AgentEnd {
                    agent_id: session.id.clone(),
                    node_ids: all_node_ids,
                });
                return Err(AgentError::Cancelled);
            }
        };
        all_node_ids.push(response_id.clone());

        // Post-turn knowledge extraction + embedding — non-fatal; log and continue on error.
        if let Some(ref extraction_config) = session.agent_config.extraction {
            match crate::knowledge::extraction::post_turn_extract(
                session.graph.clone(),
                llm,
                extraction_config,
                &response_id,
            )
            .await
            {
                Ok(node_ids) => {
                    if let Some(retriever) = session.memory_retriever() {
                        for node_id in &node_ids {
                            if let Err(e) = retriever.embed_knowledge_node(node_id).await {
                                tracing::warn!(
                                    node_id = %node_id,
                                    error = %e,
                                    "Failed to embed knowledge node (non-fatal)"
                                );
                            }
                        }
                        tracing::debug!(count = node_ids.len(), "Embedded knowledge nodes");
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Knowledge extraction failed (non-fatal)");
                }
            }
        }

        if !response.has_tool_calls() {
            events.emit(AgentEvent::TurnEnd {
                response_id: response_id.clone(),
                tool_result_ids: vec![],
            });
            emit_graph_update(session, &response_id, vec![], events).await;
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

        // Check for soft escalation after tools execute
        if check_soft_escalation(turn as u32, &session.agent_config, &response, events) {
            // Agent should respond to the escalation by synthesizing findings.
            // The synthesis directive is in the SoftEscalationTriggered event.
            // For now, we continue the loop so the agent can respond with synthesis.
        }

        events.emit(AgentEvent::TurnEnd {
            response_id: response_id.clone(),
            tool_result_ids: tool_result_ids.clone(),
        });
        emit_graph_update(session, &response_id, tool_result_ids, events).await;

        // The loop runs 0..max_turns; hitting this on the last iteration with
        // outstanding tool calls means we consumed the full budget.
        if turn + 1 >= max_turns {
            info!("Recursion limit reached at {} turns", max_turns);
            let _ = session.set_status("limit_reached").await;
            events.emit(AgentEvent::AgentEnd {
                agent_id: session.id.clone(),
                node_ids: all_node_ids,
            });
            return Err(AgentError::RecursionLimit(max_turns));
        }
    }

    let _ = session.set_status("completed").await;
    events.emit(AgentEvent::AgentEnd {
        agent_id: session.id.clone(),
        node_ids: all_node_ids,
    });

    Ok(())
}

// ============== Test helpers ==============

#[cfg(test)]
mod test_helpers {
    use std::sync::Arc;
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

    /// Mock tool that tracks how many times `execute` was called.
    pub struct TrackingMockTool {
        pub tool_name: String,
        pub output: String,
        pub call_count: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl graphirm_tools::Tool for TrackingMockTool {
        fn name(&self) -> &str {
            &self.tool_name
        }
        fn description(&self) -> &str {
            "Tracking mock tool for testing"
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }
        async fn execute(
            &self,
            _args: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Result<graphirm_tools::ToolOutput, graphirm_tools::ToolError> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::test_helpers::*;
    use super::*;
    use crate::config::AgentConfig;
    use crate::hitl::{HitlDecision, HitlGate};
    use graphirm_graph::edges::EdgeType;
    use graphirm_graph::nodes::NodeType;
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
        assert_eq!(tool_nodes[0].label(), Some("interaction_1_3_1"));

        let assistant_nodes: Vec<_> = neighbors
            .iter()
            .filter(|n| {
                if let NodeType::Interaction(d) = &n.node_type {
                    d.role == "assistant"
                } else {
                    false
                }
            })
            .collect();
        assert_eq!(assistant_nodes.len(), 2);
        assert!(
            assistant_nodes
                .iter()
                .any(|node| node.label() == Some("interaction_1_2_1"))
        );
        assert!(
            assistant_nodes
                .iter()
                .any(|node| node.label() == Some("interaction_1_4_1"))
        );
    }

    #[tokio::test]
    async fn test_agent_loop_real_tool_propagates_turn_to_content_labels() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig {
            max_turns: 10,
            working_dir: temp_dir.path().to_path_buf(),
            ..AgentConfig::default()
        };
        let session = Session::new(graph.clone(), config).unwrap();
        session.add_user_message("Echo a message").unwrap();

        let provider = MockProvider::new(vec![
            tool_call_response(vec![(
                "bash",
                "call_1",
                serde_json::json!({"command": "printf tracked"}),
            )]),
            text_response("Done."),
        ]);

        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(graphirm_tools::bash::BashTool::new()));

        let bus = EventBus::new();
        let token = CancellationToken::new();

        run_agent_loop(&session, &provider, &tools, &bus, &token)
            .await
            .unwrap();

        let neighbors = graph
            .neighbors(&session.id, Some(EdgeType::Produces), Direction::Outgoing)
            .unwrap();
        let tool_nodes: Vec<_> = neighbors
            .iter()
            .filter(|n| matches!(&n.node_type, NodeType::Interaction(d) if d.role == "tool"))
            .collect();
        assert_eq!(tool_nodes.len(), 1);
        assert_eq!(tool_nodes[0].label(), Some("interaction_1_4_1"));

        let assistant_nodes: Vec<_> = neighbors
            .iter()
            .filter(|n| matches!(&n.node_type, NodeType::Interaction(d) if d.role == "assistant"))
            .collect();
        let first_assistant = assistant_nodes
            .iter()
            .find(|node| node.label() == Some("interaction_1_2_1"))
            .unwrap();

        let content_nodes = graph
            .neighbors(&first_assistant.id, Some(EdgeType::Produces), Direction::Outgoing)
            .unwrap();
        assert_eq!(content_nodes.len(), 1);
        assert_eq!(content_nodes[0].label(), Some("content_1_3_1"));
        assert_eq!(
            content_nodes[0].metadata.get("session_id"),
            Some(&serde_json::json!(session.id.to_string()))
        );
    }

    #[tokio::test]
    async fn test_agent_loop_parallel_safe_tools_keep_dense_labels() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        std::fs::write(temp_dir.path().join("a.txt"), "a").unwrap();
        std::fs::write(temp_dir.path().join("b.txt"), "b").unwrap();

        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig {
            max_turns: 10,
            working_dir: temp_dir.path().to_path_buf(),
            ..AgentConfig::default()
        };
        let session = Session::new(graph.clone(), config).unwrap();
        session.add_user_message("List and find files").unwrap();

        let provider = MockProvider::new(vec![
            tool_call_response(vec![
                ("ls", "call_ls", serde_json::json!({"path": "."})),
                ("find", "call_find", serde_json::json!({"pattern": "*.txt"})),
            ]),
            text_response("Done."),
        ]);

        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(graphirm_tools::ls::LsTool::new()));
        tools.register(Arc::new(graphirm_tools::find::FindTool::new()));

        let bus = EventBus::new();
        let token = CancellationToken::new();

        run_agent_loop(&session, &provider, &tools, &bus, &token)
            .await
            .unwrap();

        let produced = graph
            .neighbors(&session.id, Some(EdgeType::Produces), Direction::Outgoing)
            .unwrap();
        let assistant_nodes: Vec<_> = produced
            .iter()
            .filter(|n| matches!(&n.node_type, NodeType::Interaction(d) if d.role == "assistant"))
            .collect();
        let tool_nodes: Vec<_> = produced
            .iter()
            .filter(|n| matches!(&n.node_type, NodeType::Interaction(d) if d.role == "tool"))
            .collect();

        let first_assistant = assistant_nodes
            .iter()
            .find(|node| node.label() == Some("interaction_1_2_1"))
            .unwrap();
        let content_nodes = graph
            .neighbors(&first_assistant.id, Some(EdgeType::Reads), Direction::Outgoing)
            .unwrap();

        let content_labels: std::collections::HashSet<_> =
            content_nodes.iter().filter_map(|node| node.label()).collect();
        assert_eq!(content_nodes.len(), 2);
        assert_eq!(content_labels.len(), 2);
        assert!(content_labels.contains("content_1_3_1"));
        assert!(content_labels.contains("content_1_4_1"));

        let tool_labels: std::collections::HashSet<_> =
            tool_nodes.iter().filter_map(|node| node.label()).collect();
        assert_eq!(tool_nodes.len(), 2);
        assert_eq!(tool_labels.len(), 2);
        assert!(tool_labels.contains("interaction_1_5_1"));
        assert!(tool_labels.contains("interaction_1_6_1"));
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

    #[test]
    fn test_destructive_partition_with_hitl_active() {
        use crate::hitl::is_destructive_tool;
        assert!(is_destructive_tool("write"));
        assert!(is_destructive_tool("edit"));
        assert!(is_destructive_tool("bash"));
        assert!(!is_destructive_tool("read"));
        assert!(!is_destructive_tool("grep"));
        assert!(!is_destructive_tool("ls"));
    }

    // ── HITL positive-path tests ────────────────────────────────────────────

    #[tokio::test]
    async fn test_hitl_approve_allows_tool_to_run() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig {
            max_turns: 10,
            ..AgentConfig::default()
        };
        let hitl = Arc::new(HitlGate::new());
        let session = Session::new(graph.clone(), config)
            .unwrap()
            .with_hitl(hitl.clone());
        session.add_user_message("Write a file").unwrap();

        let provider = MockProvider::new(vec![
            tool_call_response(vec![(
                "write",
                "call_w1",
                serde_json::json!({"path": "/tmp/test.txt", "content": "hello"}),
            )]),
            text_response("Done!"),
        ]);

        let call_counter = Arc::new(AtomicUsize::new(0));
        let mock_write = Arc::new(TrackingMockTool {
            tool_name: "write".to_string(),
            output: "Wrote /tmp/test.txt".to_string(),
            call_count: call_counter.clone(),
        });
        let mut tools = ToolRegistry::new();
        tools.register(mock_write);

        let bus = EventBus::new();
        let token = CancellationToken::new();

        // Poll until execute_tools_parallel registers the gate, then resolve.
        // A fixed sleep is racy: under load the resolve can fire before the gate
        // is registered, leaving rx permanently pending. Retry until resolve()
        // returns true (gate found and sent to), which is guaranteed to happen
        // only after hitl.gate() has been called by the agent loop.
        let hitl_clone = hitl.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                if hitl_clone
                    .resolve(&NodeId::from("call_w1"), HitlDecision::Approve)
                    .await
                {
                    break;
                }
            }
        });

        let result = run_agent_loop(&session, &provider, &tools, &bus, &token).await;
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);
        assert_eq!(provider.call_count(), 2, "LLM should be called twice (tool turn + final)");

        // Tool execute() was invoked exactly once.
        assert_eq!(call_counter.load(Ordering::SeqCst), 1, "Tool should have been called once");

        // An ApprovedBy edge exists: tool-result node → session.id.
        let approved_sources = graph
            .neighbors(&session.id, Some(EdgeType::ApprovedBy), Direction::Incoming)
            .unwrap();
        assert_eq!(approved_sources.len(), 1, "Expected exactly one ApprovedBy edge into session");
    }

    #[tokio::test]
    async fn test_hitl_reject_skips_tool_and_continues() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig {
            max_turns: 10,
            ..AgentConfig::default()
        };
        let hitl = Arc::new(HitlGate::new());
        let session = Session::new(graph.clone(), config)
            .unwrap()
            .with_hitl(hitl.clone());
        session.add_user_message("Write a file").unwrap();

        let provider = MockProvider::new(vec![
            tool_call_response(vec![(
                "write",
                "call_w1",
                serde_json::json!({"path": "/tmp/test.txt", "content": "hello"}),
            )]),
            // Loop continues after rejection and calls LLM again.
            text_response("I was rejected, moving on."),
        ]);

        let call_counter = Arc::new(AtomicUsize::new(0));
        let mock_write = Arc::new(TrackingMockTool {
            tool_name: "write".to_string(),
            output: "Wrote /tmp/test.txt".to_string(),
            call_count: call_counter.clone(),
        });
        let mut tools = ToolRegistry::new();
        tools.register(mock_write);

        let bus = EventBus::new();
        let token = CancellationToken::new();

        let hitl_clone = hitl.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                if hitl_clone
                    .resolve(
                        &NodeId::from("call_w1"),
                        HitlDecision::Reject("no bash".to_string()),
                    )
                    .await
                {
                    break;
                }
            }
        });

        let result = run_agent_loop(&session, &provider, &tools, &bus, &token).await;
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);
        assert_eq!(provider.call_count(), 2, "LLM should be called twice");

        // Tool execute() must never have been called.
        assert_eq!(call_counter.load(Ordering::SeqCst), 0, "Tool should NOT have been called");

        // The rejection path adds a RejectedBy edge: rejection_id → session.id.
        // Query incoming RejectedBy neighbours of session.id to find the rejection Content node.
        let rejection_sources = graph
            .neighbors(&session.id, Some(EdgeType::RejectedBy), Direction::Incoming)
            .unwrap();
        assert!(
            !rejection_sources.is_empty(),
            "Expected at least one RejectedBy edge pointing to session"
        );
        assert!(
            rejection_sources.iter().any(|n| {
                matches!(&n.node_type, NodeType::Content(d) if d.content_type == "tool_rejection")
            }),
            "Expected a tool_rejection Content node connected via RejectedBy edge"
        );
        let rejection_node = rejection_sources
            .iter()
            .find(|n| matches!(&n.node_type, NodeType::Content(d) if d.content_type == "tool_rejection"))
            .unwrap();
        assert_eq!(rejection_node.label(), Some("content_1_3_1"));
        assert_eq!(
            rejection_node.metadata.get("session_id"),
            Some(&serde_json::json!(session.id.to_string()))
        );
    }

    #[tokio::test]
    async fn test_hitl_pause_blocks_then_resumes() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig {
            max_turns: 10,
            ..AgentConfig::default()
        };
        let hitl = Arc::new(HitlGate::new());
        hitl.set_paused(true);

        let session = Session::new(graph.clone(), config)
            .unwrap()
            .with_hitl(hitl.clone());
        session.add_user_message("Hello").unwrap();

        // No tool calls — just a simple text response after the pause clears.
        let provider = MockProvider::new(vec![text_response("All good.")]);
        let tools = ToolRegistry::new();
        let mut bus = EventBus::new();
        let mut rx = bus.subscribe();
        let token = CancellationToken::new();

        // Poll until the pause gate is registered (run_agent_loop entered the while
        // loop and called hitl.gate(&session.id)), then resolve it. Clear the pause
        // flag AFTER a successful resolve so the while condition is false on the
        // next iteration — if we cleared it first, the while loop might not enter
        // at all and the gate would never be registered.
        let hitl_clone = hitl.clone();
        let session_id = session.id.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                if hitl_clone
                    .resolve(&session_id, HitlDecision::Approve)
                    .await
                {
                    hitl_clone.set_paused(false);
                    break;
                }
            }
        });

        let result = run_agent_loop(&session, &provider, &tools, &bus, &token).await;
        assert!(result.is_ok(), "Expected loop to complete after resume, got: {:?}", result);
        assert_eq!(provider.call_count(), 1, "LLM should be called once after pause clears");

        // Verify that AwaitingApproval with is_pause=true was emitted.
        let mut events = vec![];
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        let pause_event = events.iter().find(|e| {
            matches!(e, AgentEvent::AwaitingApproval { is_pause, .. } if *is_pause)
        });
        assert!(
            pause_event.is_some(),
            "Expected an AwaitingApproval event with is_pause=true"
        );
    }

    #[tokio::test]
    async fn test_agent_loop_hitl_gate_not_triggered_without_session_hitl() {
        // When session.hitl is None, the agent loop runs normally even when the
        // LLM requests a destructive tool call. All calls go to the safe (parallel)
        // path because the partition predicate short-circuits on `session.hitl.is_none()`.
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig {
            max_turns: 10,
            ..AgentConfig::default()
        };
        // No .with_hitl() — hitl is None
        let session = Session::new(graph.clone(), config).unwrap();
        session.add_user_message("Write a file").unwrap();

        // LLM requests a destructive tool (write), then returns a text response.
        let provider = MockProvider::new(vec![
            tool_call_response(vec![(
                "write",
                "call_w1",
                serde_json::json!({"path": "/tmp/test.txt", "content": "hello"}),
            )]),
            text_response("Done!"),
        ]);

        let mock_write = Arc::new(MockTool {
            tool_name: "write".to_string(),
            output: "Wrote /tmp/test.txt".to_string(),
        });
        let mut tools = ToolRegistry::new();
        tools.register(mock_write);

        let bus = EventBus::new();
        let token = CancellationToken::new();

        // Without HITL the loop should complete without hanging on a gate.
        let result = run_agent_loop(&session, &provider, &tools, &bus, &token).await;
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);
        assert_eq!(provider.call_count(), 2);
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
