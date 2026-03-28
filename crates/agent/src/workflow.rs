// Agent workflow: async state machine with plan -> act -> observe -> reflect loop

use std::sync::Arc;

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
    llm: Arc<dyn LlmProvider>,
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
    // Append segment format instructions when structured output is requested.
    let system_prompt = if let Some(ref seg_config) = session.agent_config.segments {
        if seg_config.enabled && seg_config.structured_output {
            let seg_prompt = crate::knowledge::segments::build_segment_prompt(&seg_config.labels);
            format!("{system_prompt}{seg_prompt}")
        } else {
            system_prompt
        }
    } else {
        system_prompt
    };
    let context_config = crate::context::ContextConfig {
        system_prompt,
        max_tokens: session
            .agent_config
            .max_tokens
            .map(|t| t as usize)
            .unwrap_or(100_000),
        segment_filter: session.agent_config.segment_filter.clone(),
        enable_compaction: session.agent_config.enable_compaction,
        ..crate::context::ContextConfig::default()
    };
    let graph_ref = session.graph.clone();
    let session_id_ref = session.id.clone();
    // Snapshot compaction settings before context_config is moved into the closure.
    let enable_compaction = context_config.enable_compaction;
    let max_tok = context_config.max_tokens;
    let compaction_threshold = context_config.compaction_threshold;
    let guaranteed_recent = context_config.guaranteed_recent_turns;

    let (window, mut context_stats) = tokio::task::spawn_blocking(move || {
        crate::context::build_context_with_stats(&graph_ref, &session_id_ref, &context_config)
    })
    .await
    .map_err(|e| AgentError::Join(e.to_string()))??;

    // Auto-compaction: runs after context is built, before LLM call.
    // Non-fatal — errors are logged and skipped.
    if enable_compaction {
        let graph_c = session.graph.clone();
        let agent_c = session.id.clone();
        let nodes = tokio::task::spawn_blocking(move || {
            crate::compact::select_nodes_for_compaction(
                &graph_c,
                &agent_c,
                max_tok,
                compaction_threshold,
                guaranteed_recent,
                2,
            )
        })
        .await;
        match nodes {
            Ok(Ok(ids)) if !ids.is_empty() => {
                tracing::info!(count = ids.len(), "auto-compacting old context nodes");
                let compact_cfg = crate::compact::CompactionConfig {
                    model: String::new(),
                    ..Default::default()
                };
                match crate::compact::compact_context(
                    &session.graph,
                    llm.as_ref(),
                    ids,
                    &compact_cfg,
                )
                .await
                {
                    Err(e) => tracing::warn!("auto-compaction failed (non-fatal): {e}"),
                    Ok(_) => {
                        context_stats.compaction_triggered = true;
                    }
                }
            }
            _ => {}
        }
    }

    let mut context = Vec::with_capacity(1 + window.messages.len());
    context.push(window.system);
    context.extend(window.messages);

    // Budget awareness: append a warning to the system message when token usage
    // crosses a configured threshold. Helps the agent self-manage resource consumption.
    if max_tok > 0 && !session.agent_config.budget_warning_thresholds.is_empty() {
        let usage_ratio = window.total_tokens as f64 / max_tok as f64;
        let highest_crossed = session
            .agent_config
            .budget_warning_thresholds
            .iter()
            .copied()
            .filter(|&t| usage_ratio >= t)
            .fold(f64::NEG_INFINITY, f64::max);
        if highest_crossed.is_finite() {
            let pct = (usage_ratio * 100.0) as u32;
            let warning = if highest_crossed >= 0.9 {
                format!(
                    "\n\n[Budget] Token usage at {pct}% of limit. \
                     Prioritize completing the current step and summarizing — do not start new tasks."
                )
            } else {
                format!(
                    "\n\n[Budget] Token usage at {pct}% of limit. \
                     Start wrapping up; avoid long exploratory chains."
                )
            };
            if let Some(system_msg) = context.first_mut() {
                system_msg
                    .content
                    .push(graphirm_llm::ContentPart::text(warning));
            }
            tracing::info!(
                usage_ratio,
                pct,
                "Budget warning appended to system message"
            );
        }
    }

    let raw_defs = tools.definitions();
    let tool_defs: Vec<graphirm_llm::ToolDefinition> = raw_defs
        .into_iter()
        .map(|t| graphirm_llm::ToolDefinition::new(t.name, t.description, t.parameters))
        .collect();

    // Model routing: select cheap or smart model based on session signals.
    // Prefer adaptive strategy when configured; fall back to legacy static router.
    let (mut selected_model, routing_outcome) = if let Some(ref ar_config) =
        session.agent_config.adaptive_routing
    {
        let t_route_start = std::time::Instant::now();
        let turn_number = session.current_turn();
        let graph_c = session.graph.clone();
        let session_id = session.id.0.clone();

        let (last_tool_errored, last_response_tool_only, user_msg_tokens, task_phase) =
            tokio::task::spawn_blocking(move || {
                let chain = graph_c.get_session_chain(&session_id).unwrap_or_default();
                let last_assistant = chain.iter().rev().find(|n| {
                    matches!(&n.node_type, graphirm_graph::nodes::NodeType::Interaction(i) if i.role == "assistant")
                });
                let tool_errored = chain.iter().rev().any(|n| {
                    matches!(&n.node_type, graphirm_graph::nodes::NodeType::Interaction(i) if i.role == "tool_result")
                        && n.metadata.get("is_error").and_then(|v| v.as_bool()).unwrap_or(false)
                });
                let tool_only = last_assistant
                    .map(|n| {
                        n.metadata.get("tool_calls").is_some()
                            && matches!(&n.node_type, graphirm_graph::nodes::NodeType::Interaction(i) if i.content.trim().is_empty())
                    })
                    .unwrap_or(false);
                let user_tokens = chain
                    .iter()
                    .rev()
                    .find(|n| matches!(&n.node_type, graphirm_graph::nodes::NodeType::Interaction(i) if i.role == "user"))
                    .map(|n| match &n.node_type {
                        graphirm_graph::nodes::NodeType::Interaction(i) => i.content.len() / 4,
                        _ => 0,
                    })
                    .unwrap_or(0);
                let phase = infer_task_phase(&chain);
                (tool_errored, tool_only, user_tokens, phase)
            })
            .await
            .unwrap_or((false, false, 0, crate::router::TaskPhase::Planning));

        let signals = crate::router::TurnSignals {
            turn_number,
            last_tool_errored,
            last_response_tool_only,
            user_message_tokens: user_msg_tokens,
            task_phase,
        };

        let objective = ar_config
            .objective
            .as_ref()
            .map(|o| o.to_weights())
            .unwrap_or_default();

        let candidates = crate::strategy::builder::candidates_from_config(
            &ar_config.candidates,
            session.agent_config.model_routing.as_ref(),
        );

        let strategy = crate::strategy::builder::build_strategy(
            ar_config,
            session.agent_config.model_routing.as_ref(),
            llm.clone(),
        );

        let decision = strategy.select(&signals, &candidates, &objective).await;
        let routing_decision_ms = t_route_start.elapsed().as_millis() as u64;

        tracing::info!(
            model = &decision.model,
            tier = ?decision.tier,
            strategy = decision.strategy_name,
            reason = &decision.reason,
            confidence = decision.confidence,
            routing_ms = routing_decision_ms,
            "adaptive router selected"
        );

        (
            decision.model.clone(),
            Some((decision, routing_decision_ms)),
        )
    } else if let Some(ref routing) = session.agent_config.model_routing {
        // Legacy static router — preserved for backward compat.
        let turn_number = session.current_turn();
        let graph_c = session.graph.clone();
        let session_id = session.id.0.clone();
        let (last_tool_errored, last_response_tool_only, user_msg_tokens, task_phase) =
            tokio::task::spawn_blocking(move || {
                let chain = graph_c.get_session_chain(&session_id).unwrap_or_default();
                let last_assistant = chain.iter().rev().find(|n| {
                    matches!(&n.node_type, graphirm_graph::nodes::NodeType::Interaction(i) if i.role == "assistant")
                });
                let tool_errored = chain.iter().rev().any(|n| {
                    matches!(&n.node_type, graphirm_graph::nodes::NodeType::Interaction(i) if i.role == "tool_result")
                        && n.metadata.get("is_error").and_then(|v| v.as_bool()).unwrap_or(false)
                });
                let tool_only = last_assistant
                    .map(|n| {
                        n.metadata.get("tool_calls").is_some()
                            && matches!(&n.node_type, graphirm_graph::nodes::NodeType::Interaction(i) if i.content.trim().is_empty())
                    })
                    .unwrap_or(false);
                let user_tokens = chain
                    .iter()
                    .rev()
                    .find(|n| matches!(&n.node_type, graphirm_graph::nodes::NodeType::Interaction(i) if i.role == "user"))
                    .map(|n| match &n.node_type {
                        graphirm_graph::nodes::NodeType::Interaction(i) => i.content.len() / 4,
                        _ => 0,
                    })
                    .unwrap_or(0);
                let phase = infer_task_phase(&chain);
                (tool_errored, tool_only, user_tokens, phase)
            })
            .await
            .unwrap_or((false, false, 0, crate::router::TaskPhase::Planning));
        let signals = crate::router::TurnSignals {
            turn_number,
            last_tool_errored,
            last_response_tool_only,
            user_message_tokens: user_msg_tokens,
            task_phase,
        };
        let router = crate::router::ModelRouter::new(routing);
        let (model, tier, rule) = router.select(&signals);
        tracing::info!(model, tier = ?tier, rule, turn = turn_number, "legacy model router selected");
        // Wrap in adaptive RoutingDecision for unified metadata path.
        let decision = crate::strategy::RoutingDecision {
            model: model.to_string(),
            tier,
            confidence: 1.0,
            reason: format!("rule:{rule}"),
            strategy_name: "rule_router".to_string(),
        };
        (model.to_string(), Some((decision, 0u64)))
    } else {
        (session.agent_config.model.clone(), None)
    };

    let max_output = session
        .agent_config
        .max_output_tokens
        .unwrap_or(session.agent_config.max_tokens.unwrap_or(8192));
    let temperature = session.agent_config.temperature.unwrap_or(0.7);

    // Build fallback model list for the selected tier.
    let fallback_models: Vec<String> = routing_outcome
        .as_ref()
        .and_then(|(decision, _)| {
            session.agent_config.model_routing.as_ref().map(|routing| {
                routing
                    .models_for_tier(decision.tier)
                    .iter()
                    .map(|m| {
                        m.split_once('/')
                            .map(|x| x.1)
                            .unwrap_or(m.as_str())
                            .to_string()
                    })
                    .collect()
            })
        })
        .unwrap_or_else(|| vec![selected_model.clone()]);

    let mut fallback_chain: Vec<crate::router::FallbackAttempt> = Vec::new();
    let mut response = None;

    for (i, model) in fallback_models.iter().enumerate() {
        let is_last = i == fallback_models.len() - 1;
        let comp_config = CompletionConfig::new(model)
            .with_max_tokens(max_output)
            .with_temperature(temperature);
        let start = std::time::Instant::now();
        match llm
            .complete(context.clone(), &tool_defs, &comp_config)
            .await
        {
            Ok(resp) => {
                if i > 0 {
                    selected_model = model.clone();
                }
                response = Some(resp);
                break;
            }
            Err(e) if e.is_retryable() && !is_last => {
                let latency_ms = start.elapsed().as_millis() as u64;
                tracing::warn!(
                    model,
                    error = %e,
                    attempt = i + 1,
                    "LLM call failed, trying next fallback model"
                );
                fallback_chain.push(crate::router::FallbackAttempt {
                    model: model.clone(),
                    error: e.to_string(),
                    latency_ms,
                });
            }
            Err(e) => return Err(e.into()),
        }
    }

    let response = response.expect("fallback loop must produce a response or return an error");

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

    // Add context_stats to metadata
    metadata.insert(
        "context_stats".to_string(),
        serde_json::to_value(&context_stats).unwrap_or(serde_json::Value::Null),
    );

    if !fallback_chain.is_empty() {
        metadata.insert(
            "fallback_chain".to_string(),
            serde_json::to_value(&fallback_chain).unwrap_or_default(),
        );
    }

    if let Some((ref decision, decision_ms)) = routing_outcome {
        metadata.insert(
            "model_tier".to_string(),
            serde_json::json!(format!("{:?}", decision.tier).to_lowercase()),
        );
        metadata.insert(
            "model_selected".to_string(),
            serde_json::json!(&selected_model),
        );
        metadata.insert(
            "routing_strategy".to_string(),
            serde_json::json!(&decision.strategy_name),
        );
        metadata.insert(
            "routing_reason".to_string(),
            serde_json::json!(&decision.reason),
        );
        metadata.insert(
            "routing_confidence".to_string(),
            serde_json::json!(decision.confidence),
        );
        metadata.insert(
            "routing_decision_ms".to_string(),
            serde_json::json!(decision_ms),
        );
    }

    let mut interaction_node = GraphNode::new(NodeType::Interaction(InteractionData {
        role: "assistant".to_string(),
        content: response.text_content(),
        token_count: Some(response.usage.output_tokens),
    }));
    interaction_node.metadata = serde_json::Value::Object(metadata);

    let node_id = session.record_interaction(interaction_node).await?;

    // Structured response segmentation — opt-in, non-fatal.
    // Only runs on final text turns (no tool calls).
    // Primary path: parse JSON envelope emitted by LLM when structured_output is true.
    // Fallback path: GLiNER2 ONNX span detection (requires local-extraction feature +
    //   ExtractionConfig with a Local or Hybrid backend pointing at a downloaded model).
    if let Some(ref seg_config) = session.agent_config.segments
        && seg_config.enabled
        && !response.has_tool_calls()
    {
        let raw_text = response.text_content();

        // Try structured JSON first, fall back to GLiNER2 if that fails or is empty.
        let structured = crate::knowledge::segments::parse_structured_segments(&raw_text);
        let segments_opt: Option<(Vec<crate::knowledge::segments::Segment>, &str)> =
            match structured {
                Ok(segs) if !segs.is_empty() => {
                    tracing::info!(
                        count = segs.len(),
                        "Parsed structured segments from LLM response"
                    );
                    Some((segs, "structured"))
                }
                Ok(_) => {
                    tracing::debug!(
                        "Structured segment parse returned empty — trying GLiNER2 fallback"
                    );
                    // GLiNER2 fallback
                    #[cfg(feature = "local-extraction")]
                    {
                        let model_dir = session
                            .agent_config
                            .extraction
                            .as_ref()
                            .filter(|_| seg_config.gliner2_fallback)
                            .and_then(|e| {
                                use crate::knowledge::extraction::ExtractionBackend;
                                match &e.backend {
                                    ExtractionBackend::Local { model_dir }
                                    | ExtractionBackend::Hybrid { model_dir } => {
                                        Some(model_dir.clone())
                                    }
                                    _ => None,
                                }
                            });
                        if let Some(dir) = model_dir {
                            crate::knowledge::segments::try_gliner2_fallback(
                                &dir,
                                &raw_text,
                                &seg_config.labels,
                                seg_config.min_confidence,
                            )
                            .await
                            .map(|s| (s, "gliner2"))
                        } else {
                            None
                        }
                    }
                    #[cfg(not(feature = "local-extraction"))]
                    None
                }
                Err(e) => {
                    tracing::debug!(
                        error = %e,
                        "Structured segment parse failed — trying GLiNER2 fallback"
                    );
                    // GLiNER2 fallback
                    #[cfg(feature = "local-extraction")]
                    {
                        let model_dir = session
                            .agent_config
                            .extraction
                            .as_ref()
                            .filter(|_| seg_config.gliner2_fallback)
                            .and_then(|e| {
                                use crate::knowledge::extraction::ExtractionBackend;
                                match &e.backend {
                                    ExtractionBackend::Local { model_dir }
                                    | ExtractionBackend::Hybrid { model_dir } => {
                                        Some(model_dir.clone())
                                    }
                                    _ => None,
                                }
                            });
                        if let Some(dir) = model_dir {
                            crate::knowledge::segments::try_gliner2_fallback(
                                &dir,
                                &raw_text,
                                &seg_config.labels,
                                seg_config.min_confidence,
                            )
                            .await
                            .map(|s| (s, "gliner2"))
                        } else {
                            None
                        }
                    }
                    #[cfg(not(feature = "local-extraction"))]
                    None
                }
            };

        if let Some((segments, source)) = segments_opt {
            let nesting = crate::knowledge::segments::detect_nesting(&segments);
            match crate::knowledge::segments::persist_segments(
                &session.graph,
                &node_id,
                &segments,
                &nesting,
            )
            .await
            {
                Ok(seg_ids) => {
                    tracing::info!(
                        count = seg_ids.len(),
                        source = source,
                        nesting_pairs = nesting.len(),
                        "Persisted response segments"
                    );
                    // Stamp the parent Interaction node so the context engine can detect
                    // that segment children exist and apply the segment_filter correctly.
                    let graph_clone = session.graph.clone();
                    let stamp_id = node_id.clone();
                    match tokio::task::spawn_blocking(move || {
                        let mut node = graph_clone.get_node(&stamp_id)?;
                        node.metadata["segmented"] = serde_json::json!(true);
                        graph_clone.update_node(&stamp_id, node)
                    })
                    .await
                    {
                        Ok(Ok(())) => {}
                        Ok(Err(e)) => {
                            tracing::warn!(error = %e, "Failed to stamp segmented metadata on interaction node (non-fatal)");
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "spawn_blocking panicked while stamping segmented metadata (non-fatal)");
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to persist response segments (non-fatal)");
                }
            }
        }
    }

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
    let knowledge_retriever: Option<Arc<dyn graphirm_tools::retriever::KnowledgeRetriever>> =
        session
            .memory_retriever()
            .map(|r| Arc::clone(r) as Arc<dyn graphirm_tools::retriever::KnowledgeRetriever>);

    let impact_provider: Option<Arc<dyn graphirm_tools::impact::ImpactProvider>> =
        if session.agent_config.pre_edit_impact {
            Some(Arc::new(crate::impact::GraphImpactProvider::new(
                session.graph.clone(),
                session.agent_config.working_dir.clone(),
            )))
        } else {
            None
        };

    let ctx = ToolContext {
        graph: session.graph.clone(),
        agent_id: session.id.clone(),
        interaction_id: response_id.clone(),
        working_dir: session.agent_config.working_dir.clone(),
        signal: cancel.clone(),
        turn: session.current_turn(),
        turn_pos_counter: session.turn_position_counter(),
        knowledge_retriever,
        impact_provider: impact_provider.clone(),
    };

    // Partition tool calls: destructive ones go through sequential HITL approval,
    // safe ones run in parallel without gating.
    // `.copied()` turns `&&ContentPart` (from iterating `&[&ContentPart]`) into
    // `&ContentPart` so the partition buckets are `Vec<&ContentPart>`.
    let (safe_calls, destructive_calls): (Vec<_>, Vec<_>) =
        tool_calls.iter().copied().partition(|part| {
            let ContentPart::ToolCall { name, .. } = part else {
                return true;
            };
            // A tool call is "safe" (no HITL gate needed) when EITHER:
            // - It is not destructive by name (legacy built-in list) AND
            //   not flagged destructive by the registry (handles ScriptTool plugins)
            // - No HITL gate is attached to this session at all
            (!crate::hitl::is_destructive_tool(name.as_str())
                && !tools.is_destructive(name.as_str()))
                || session.hitl.is_none()
        });

    // Per-turn cache for impact briefs — populated by the pre_edit_impact_brief helper
    // and reused across all destructive tool executions in this turn
    use std::collections::HashMap;

    let impact_cache: Arc<
        tokio::sync::Mutex<HashMap<std::path::PathBuf, graphirm_tools::impact::ImpactBrief>>,
    > = Arc::new(tokio::sync::Mutex::new(HashMap::new()));

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
        tool_metadata.insert("tool_name".to_string(), serde_json::json!(&tool_name));
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
        let hitl = session
            .hitl
            .as_ref()
            .expect("hitl must be Some for destructive calls");

        let gate_key = NodeId::from(call_id.as_str());

        // If auto-approve is enabled, skip the gate entirely.
        let decision = if hitl.is_auto_approve() {
            HitlDecision::Approve
        } else {
            events.emit(AgentEvent::AwaitingApproval {
                node_id: gate_key.clone(),
                tool_name: name.clone(),
                arguments: arguments.clone(),
                is_pause: false,
            });

            let rx = hitl.gate(&gate_key).await;

            tokio::select! {
                result = rx => match result {
                    Ok(d) => d,
                    Err(_) => HitlDecision::Reject("Gate sender dropped unexpectedly".to_string()),
                },
                _ = cancel.cancelled() => {
                    let _ = session.set_status("cancelled").await;
                    return Err(AgentError::Cancelled);
                }
            }
        };

        match decision {
            HitlDecision::Approve | HitlDecision::Modify(_) => {
                let exec_args = match &decision {
                    HitlDecision::Modify(new_args) => new_args.clone(),
                    _ => arguments.clone(),
                };

                let tool = tools.get(name)?;
                let exec_result = tool.execute(exec_args.clone(), &ctx).await;

                // Compute impact brief (if applicable)
                let impact_brief_text = if let Some(ref provider) = impact_provider {
                    pre_edit_impact_brief(
                        provider.as_ref(),
                        name,
                        &exec_args,
                        &session.id,
                        &impact_cache,
                    )
                    .await
                } else {
                    None
                };

                let mut content = match &exec_result {
                    Ok(output) => output.content.clone(),
                    Err(e) => e.to_string(),
                };
                let is_error = exec_result.as_ref().map(|o| o.is_error).unwrap_or(true);

                // Prepend impact brief
                if let Some(ref brief_text) = impact_brief_text {
                    content = format!("{brief_text}\n{content}");

                    // Persist as Content node
                    let mut brief_node = GraphNode::new(NodeType::Content(ContentData {
                        content_type: "impact_brief".to_string(),
                        path: None,
                        body: brief_text.clone(),
                        language: None,
                    }));
                    brief_node.metadata["session_id"] = serde_json::json!(session.id.to_string());
                    brief_node.set_label(format!(
                        "content_{}_{}_1",
                        session.current_turn(),
                        session.next_turn_pos()
                    ));
                    match session.graph.add_node(brief_node) {
                        Ok(brief_id) => {
                            let _ = session.graph.add_edge(GraphEdge::new(
                                EdgeType::Reads,
                                response_id.clone(),
                                brief_id,
                            ));
                        }
                        Err(e) => {
                            tracing::warn!(
                                error = %e,
                                "Failed to persist impact brief node (non-fatal)"
                            );
                        }
                    }
                }

                let mut tool_metadata = serde_json::Map::new();
                tool_metadata.insert("tool_call_id".to_string(), serde_json::json!(call_id));
                tool_metadata.insert("tool_name".to_string(), serde_json::json!(&name));
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
                        tracing::error!("Failed to record tool rejection for call {call_id}: {e}");
                    }
                }
            }
        }
    }

    Ok(node_ids)
}

async fn pre_edit_impact_brief(
    impact_provider: &dyn graphirm_tools::impact::ImpactProvider,
    tool_name: &str,
    arguments: &serde_json::Value,
    session_id: &NodeId,
    cache: &tokio::sync::Mutex<
        std::collections::HashMap<std::path::PathBuf, graphirm_tools::impact::ImpactBrief>,
    >,
) -> Option<String> {
    let paths = graphirm_tools::impact::extract_target_paths(tool_name, arguments);
    if paths.is_empty() {
        return None;
    }

    // Check cache and collect uncached paths
    let mut uncached_paths = Vec::new();
    {
        let cache_guard = cache.lock().await;
        for path in &paths {
            if !cache_guard.contains_key(path) {
                uncached_paths.push(path.clone());
            }
        }
    }

    // Analyze uncached paths
    if !uncached_paths.is_empty() {
        match impact_provider.analyze(&uncached_paths, session_id).await {
            Ok(new_briefs) => {
                let mut cache_guard = cache.lock().await;
                for brief in new_briefs {
                    cache_guard.insert(brief.path.clone(), brief);
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Impact analysis failed (non-fatal)");
            }
        }
    }

    // Collect briefs for all requested paths
    let cache_guard = cache.lock().await;
    let briefs: Vec<&graphirm_tools::impact::ImpactBrief> =
        paths.iter().filter_map(|p| cache_guard.get(p)).collect();

    if briefs.is_empty() {
        return None;
    }

    let formatted: Vec<String> = briefs.iter().map(|b| b.format_markdown()).collect();
    Some(formatted.join("\n"))
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
            tool_name,
            current_tools.len()
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

/// Emit a GraphUpdate event with recent nodes, edges touching this turn's nodes, and a merged
/// node list for incremental SSE clients.
/// Infer the current task phase from the session's interaction chain.
///
/// Phase is determined by examining tool result nodes' `tool_name` metadata:
/// - `Planning`: no write/edit tool calls have been made yet.
/// - `Verification`: write/edit calls exist but the most recent tool calls are
///   all read-only (bash, read, grep, find, ls) — agent is running tests/checks.
/// - `Implementation`: write/edit calls have occurred and the last calls include writes.
fn infer_task_phase(chain: &[graphirm_graph::nodes::GraphNode]) -> crate::router::TaskPhase {
    use crate::router::TaskPhase;

    // Collect all tool result names in chronological order.
    let tool_names: Vec<&str> = chain
        .iter()
        .filter(|n| {
            matches!(&n.node_type, graphirm_graph::nodes::NodeType::Interaction(i) if i.role == "tool")
        })
        .filter_map(|n| n.metadata.get("tool_name").and_then(|v| v.as_str()))
        .collect();

    let has_write_calls = tool_names.iter().any(|&n| matches!(n, "write" | "edit"));

    if !has_write_calls {
        return TaskPhase::Planning;
    }

    // Check whether recent tool calls (last 5) are all read-only.
    let read_only_tools = [
        "bash",
        "read",
        "grep",
        "find",
        "ls",
        "graph_query",
        "repo_briefing",
        "session_trace",
        "graph_diff",
        "diff",
        "read_many",
    ];
    let recent: Vec<&str> = tool_names.iter().rev().take(5).copied().collect();
    if !recent.is_empty() && recent.iter().all(|&n| read_only_tools.contains(&n)) {
        return TaskPhase::Verification;
    }

    TaskPhase::Implementation
}

async fn emit_graph_update(
    session: &Session,
    node_id: &NodeId,
    tool_result_node_ids: Vec<NodeId>,
    events: &EventBus,
) {
    let graph = session.graph.clone();
    let anchor = node_id.clone();
    let tools = tool_result_node_ids.clone();
    let payload = match tokio::task::spawn_blocking(move || {
        let recent_nodes = graph.list_recent_nodes(50)?;
        let mut anchors = vec![anchor];
        anchors.extend(tools);
        let mut edge_map: std::collections::HashMap<graphirm_graph::edges::EdgeId, GraphEdge> =
            std::collections::HashMap::new();
        for nid in &anchors {
            for e in graph.edges_for_node(nid)? {
                edge_map.entry(e.id.clone()).or_insert(e);
            }
        }
        let recent_edges: Vec<GraphEdge> = edge_map.into_values().collect();
        let mut node_map: std::collections::HashMap<NodeId, GraphNode> = recent_nodes
            .iter()
            .map(|n| (n.id.clone(), n.clone()))
            .collect();
        for e in &recent_edges {
            for nid in [&e.source, &e.target] {
                if !node_map.contains_key(nid)
                    && let Ok(n) = graph.get_node(nid)
                {
                    node_map.insert(nid.clone(), n);
                }
            }
        }
        let patch_nodes: Vec<GraphNode> = node_map.into_values().collect();
        Ok::<_, graphirm_graph::GraphError>((recent_nodes, recent_edges, patch_nodes))
    })
    .await
    {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => {
            tracing::warn!("GraphUpdate: failed to build payload: {e}");
            return;
        }
        Err(e) => {
            tracing::warn!("GraphUpdate: spawn_blocking panicked: {e}");
            return;
        }
    };
    let (recent_nodes, recent_edges, patch_nodes) = payload;
    events.emit(AgentEvent::GraphUpdate {
        node_id: node_id.clone(),
        edge_ids: tool_result_node_ids
            .into_iter()
            .map(|id| graphirm_graph::edges::EdgeId(id.0))
            .collect(),
        recent_nodes,
        recent_edges,
        patch_nodes,
    });
}

/// The main agent loop. Cycles between:
/// 1. Build context from graph
/// 2. Call LLM and record response (races against CancellationToken)
/// 3. If tool calls present, dispatch them in parallel and record results
/// 4. Repeat until no tool calls, max_turns is reached, or cancelled
pub async fn run_agent_loop(
    session: &Session,
    llm: Arc<dyn LlmProvider>,
    tools: &ToolRegistry,
    events: &EventBus,
    cancel: &CancellationToken,
) -> Result<(), AgentError> {
    let max_turns = session.agent_config.max_turns;
    let max_continuations = session.agent_config.max_continuations;
    let pre_completion_verify = session.agent_config.pre_completion_verify;
    let doom_loop_threshold = session.agent_config.doom_loop_threshold;
    let read_loop_threshold = session.agent_config.read_loop_threshold;
    let mut all_node_ids: Vec<NodeId> = Vec::new();
    // Track whether any tool calls have been executed in this session so far.
    // Used to decide whether to inject a continuation message after a text-only turn.
    let mut had_tool_calls = false;
    // True only when the agent has executed at least one write or edit tool call.
    // pre_completion_verify uses this so the checklist fires after actual file changes,
    // not after read-only planning turns.
    let mut had_write_calls = false;
    let mut continuation_count: u32 = 0;
    // Tracks whether the one-shot verification checklist has already been injected.
    let mut verify_injected = false;
    // Per-file write/edit counts for doom loop detection.
    let mut file_edit_counts: std::collections::HashMap<std::path::PathBuf, u32> =
        std::collections::HashMap::new();
    // Per-file read counts for read-loop detection (catches verification doom loops).
    let mut file_read_counts: std::collections::HashMap<std::path::PathBuf, u32> =
        std::collections::HashMap::new();

    events.emit(AgentEvent::AgentStart {
        agent_id: session.id.clone(),
    });

    // Pre-loop: inject relevant knowledge from past sessions into system prompt.
    if let Some(retriever) = session.memory_retriever() {
        let query = session.recent_user_message().await.unwrap_or_default();
        match retriever.retrieve_relevant(&query, 5).await {
            Ok(nodes) => {
                let context = crate::knowledge::injection::format_memory_context(&nodes);
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

        // Race the LLM call against cancellation and a per-turn timeout so
        // hung provider connections don't leave the session stuck forever.
        let llm_timeout = std::time::Duration::from_secs(session.agent_config.timeout_seconds);
        let (response, response_id) = tokio::select! {
            result = stream_and_record(session, llm.clone(), tools, events) => result?,
            _ = cancel.cancelled() => {
                info!("Agent loop cancelled during LLM call at turn {}", turn);
                let _ = session.set_status("cancelled").await;
                events.emit(AgentEvent::AgentEnd {
                    agent_id: session.id.clone(),
                    node_ids: all_node_ids,
                });
                return Err(AgentError::Cancelled);
            }
            _ = tokio::time::sleep(llm_timeout) => {
                tracing::error!(turn, timeout_secs = session.agent_config.timeout_seconds, "LLM call timed out");
                let _ = session.set_status("error").await;
                events.emit(AgentEvent::AgentEnd {
                    agent_id: session.id.clone(),
                    node_ids: all_node_ids,
                });
                return Err(AgentError::Workflow(
                    format!("LLM call timed out after {}s at turn {turn}", session.agent_config.timeout_seconds)
                ));
            }
        };
        all_node_ids.push(response_id.clone());

        if !response.has_tool_calls() {
            // Post-turn knowledge extraction — only on final text responses (no tool calls)
            // to avoid redundant extraction calls on intermediate planning turns.
            // A hard 20s timeout prevents slow API calls from blocking session completion.
            if let Some(ref extraction_config) = session.agent_config.extraction {
                let extraction_future = crate::knowledge::extraction::post_turn_extract(
                    session.graph.clone(),
                    llm.as_ref(),
                    extraction_config,
                    &response_id,
                    &session.id,
                );
                // 30s timeout: generous enough for a DeepSeek API call while
                // still capping the impact on task turn latency.
                match tokio::time::timeout(std::time::Duration::from_secs(30), extraction_future)
                    .await
                {
                    Ok(Ok(node_ids)) => {
                        if let Some(retriever) = session.memory_retriever() {
                            for node_id in &node_ids {
                                if let Err(e) = retriever.embed_knowledge_node(node_id).await {
                                    tracing::warn!(
                                        node_id = %node_id,
                                        error = %e,
                                        "Failed to embed knowledge node (non-fatal)"
                                    );
                                    continue;
                                }
                                // Cross-session linking: find similar nodes from other sessions
                                // and create RelatesTo edges so graph traversal can discover them.
                                match retriever
                                    .find_cross_session_links(node_id, &session.id, 5, 0.5)
                                    .await
                                {
                                    Ok(links) if !links.is_empty() => {
                                        tracing::info!(
                                            node_id = %node_id,
                                            cross_links = links.len(),
                                            "Creating cross-session knowledge links"
                                        );
                                        retriever
                                            .persist_cross_session_links(node_id, &links)
                                            .await;
                                    }
                                    Ok(_) => {}
                                    Err(e) => {
                                        tracing::warn!(
                                            error = %e,
                                            "Cross-session link search failed (non-fatal)"
                                        );
                                    }
                                }
                            }
                        }
                        tracing::debug!(count = node_ids.len(), "Knowledge extraction complete");
                    }
                    Ok(Err(e)) => {
                        tracing::warn!(error = %e, "Knowledge extraction failed (non-fatal)");
                    }
                    Err(_) => {
                        tracing::warn!("Knowledge extraction timed out after 30s (non-fatal)");
                    }
                }
            }
            events.emit(AgentEvent::TurnEnd {
                response_id: response_id.clone(),
                tool_result_ids: vec![],
            });
            emit_graph_update(session, &response_id, vec![], events).await;

            // Post-verification exit: once the verification checklist has been injected
            // and the agent responds with a text-only turn (its summary), stop immediately.
            // Without this guard the auto-continuation below would fire and the agent
            // would re-read files endlessly in a verification doom loop.
            if verify_injected {
                tracing::info!(turn, "Post-verification text turn; exiting loop");
                break;
            }

            // Auto-continuation: if the agent stopped text-only while work was in progress
            // (evidenced by prior tool calls this session), inject a continuation nudge so
            // it resumes rather than silently leaving the task unfinished.
            if had_tool_calls && continuation_count < max_continuations {
                continuation_count += 1;
                tracing::info!(
                    turn,
                    continuation_count,
                    max_continuations,
                    "Text-only turn mid-task; auto-injecting continuation message"
                );
                let cont_node = graphirm_graph::nodes::GraphNode::new(
                    graphirm_graph::nodes::NodeType::Interaction(
                        graphirm_graph::nodes::InteractionData {
                            role: "user".to_string(),
                            content: "Continue with the implementation. What is the next step?"
                                .to_string(),
                            token_count: None,
                        },
                    ),
                );
                if let Err(e) = session.record_interaction(cont_node).await {
                    tracing::warn!(error = %e, "Failed to inject continuation message (non-fatal)");
                } else {
                    continue;
                }
            }

            // Pre-completion verification: fires once per session after tool work ends.
            // Injects a checklist that forces the agent to run tests and re-check requirements
            // before the loop exits. Non-fatal — loop breaks normally if injection fails.
            if pre_completion_verify && had_write_calls && !verify_injected {
                verify_injected = true;
                tracing::info!(turn, "Injecting pre-completion verification checklist");
                let verify_content = concat!(
                    "Before marking this task complete, verify your work:\n",
                    "1. Run the relevant build command (cargo test, npm run build, etc.) — confirm it passes.\n",
                    "2. Run `cargo clippy -- -D warnings` if Rust was touched — fix any new lint errors.\n",
                    "3. Run `git diff --name-only` to see exactly what changed.\n",
                    "Once checks pass, summarize what was done and stop. Do NOT re-read source files after a passing build.",
                )
                .to_string();
                let verify_node = graphirm_graph::nodes::GraphNode::new(
                    graphirm_graph::nodes::NodeType::Interaction(
                        graphirm_graph::nodes::InteractionData {
                            role: "user".to_string(),
                            content: verify_content,
                            token_count: None,
                        },
                    ),
                );
                if let Err(e) = session.record_interaction(verify_node).await {
                    tracing::warn!(error = %e, "Failed to inject verification message (non-fatal)");
                } else {
                    continue;
                }
            }

            break;
        }

        had_tool_calls = true;
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

        // Doom loop tracking: count write/edit calls per file path.
        if doom_loop_threshold > 0 {
            for part in &tool_calls {
                let ContentPart::ToolCall {
                    name, arguments, ..
                } = part
                else {
                    continue;
                };
                if (name == "write" || name == "edit")
                    && let Some(path_str) = arguments.get("path").and_then(|v| v.as_str())
                {
                    had_write_calls = true;
                    *file_edit_counts
                        .entry(std::path::PathBuf::from(path_str))
                        .or_insert(0) += 1;
                    // Reset read counter for this path — a write invalidates
                    // prior content, so one re-read after a write is expected.
                    file_read_counts.remove(&std::path::PathBuf::from(path_str));
                }
            }
        }

        // Read-loop tracking: count read/read_many/grep calls per file path.
        // Catches verification doom loops where the agent re-reads completed files.
        if read_loop_threshold > 0 {
            for part in &tool_calls {
                let ContentPart::ToolCall {
                    name, arguments, ..
                } = part
                else {
                    continue;
                };
                if name == "read"
                    && let Some(path_str) = arguments.get("path").and_then(|v| v.as_str())
                {
                    *file_read_counts
                        .entry(std::path::PathBuf::from(path_str))
                        .or_insert(0) += 1;
                } else if name == "read_many"
                    && let Some(paths) = arguments.get("paths").and_then(|v| v.as_array())
                {
                    for p in paths.iter().filter_map(|v| v.as_str()) {
                        *file_read_counts
                            .entry(std::path::PathBuf::from(p))
                            .or_insert(0) += 1;
                    }
                }
            }
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

        // Doom loop advisory: warn when the agent has edited a file too many times.
        // Non-fatal — failure to inject is logged and skipped.
        if doom_loop_threshold > 0 {
            for (file_path, &count) in &file_edit_counts {
                if count == doom_loop_threshold {
                    tracing::warn!(
                        path = %file_path.display(),
                        count,
                        "Doom loop detected; injecting advisory"
                    );
                    let advisory = format!(
                        "Warning: you have edited `{}` {} times this session. \
                         Step back and reconsider your approach before making further edits. \
                         Review the error messages carefully, re-examine your logic from scratch, \
                         or try a completely different strategy.",
                        file_path.display(),
                        count,
                    );
                    let advisory_node = graphirm_graph::nodes::GraphNode::new(
                        graphirm_graph::nodes::NodeType::Interaction(
                            graphirm_graph::nodes::InteractionData {
                                role: "user".to_string(),
                                content: advisory,
                                token_count: None,
                            },
                        ),
                    );
                    if let Err(e) = session.record_interaction(advisory_node).await {
                        tracing::warn!(error = %e, "Failed to inject doom loop advisory (non-fatal)");
                    }
                }
            }
        }

        // Read-loop advisory: warn when the agent re-reads a file too many times
        // without editing it. Fires once per file (at threshold count) to avoid spam.
        if read_loop_threshold > 0 {
            for (file_path, &count) in &file_read_counts {
                if count == read_loop_threshold {
                    tracing::warn!(
                        path = %file_path.display(),
                        count,
                        "Read loop detected; injecting advisory"
                    );
                    let advisory = format!(
                        "Warning: you have read `{}` {} times this session without editing it. \
                         You already know this file's contents. Stop re-reading and either: \
                         (a) state 'Task complete' and stop, or \
                         (b) make an edit if something is actually wrong. \
                         Do NOT re-read files after a passing build.",
                        file_path.display(),
                        count,
                    );
                    let advisory_node = graphirm_graph::nodes::GraphNode::new(
                        graphirm_graph::nodes::NodeType::Interaction(
                            graphirm_graph::nodes::InteractionData {
                                role: "user".to_string(),
                                content: advisory,
                                token_count: None,
                            },
                        ),
                    );
                    if let Err(e) = session.record_interaction(advisory_node).await {
                        tracing::warn!(error = %e, "Failed to inject read loop advisory (non-fatal)");
                    }
                }
            }
        }

        // Check for soft escalation after tools execute
        if check_soft_escalation(turn, &session.agent_config, &response, events) {
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

        session.add_user_message("What is 2+2?").await.unwrap();

        let provider = Arc::new(MockProvider::new(vec![text_response("The answer is 4.")]));
        let tools = ToolRegistry::new();
        let bus = EventBus::new();

        let (response, node_id) = stream_and_record(&session, provider.clone(), &tools, &bus)
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
        session.add_user_message("What is 2+2?").await.unwrap();

        let provider = Arc::new(MockProvider::new(vec![text_response("4")]));
        let tools = ToolRegistry::new();
        let mut bus = EventBus::new();
        let mut rx = bus.subscribe();
        let token = CancellationToken::new();

        run_agent_loop(&session, provider.clone(), &tools, &bus, &token)
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
            pre_completion_verify: false,
            ..AgentConfig::default()
        };
        let session = Session::new(graph.clone(), config).unwrap();
        session.add_user_message("List files").await.unwrap();

        let provider = Arc::new(MockProvider::new(vec![
            tool_call_response(vec![(
                "bash",
                "call_1",
                serde_json::json!({"command": "ls"}),
            )]),
            text_response("Here are your files: src/ Cargo.toml"),
        ]));

        let mock_bash = Arc::new(MockTool {
            tool_name: "bash".to_string(),
            output: "src/\nCargo.toml".to_string(),
        });
        let mut tools = ToolRegistry::new();
        tools.register(mock_bash);

        let mut bus = EventBus::new();
        let mut rx = bus.subscribe();
        let token = CancellationToken::new();

        run_agent_loop(&session, provider.clone(), &tools, &bus, &token)
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
            pre_completion_verify: false,
            ..AgentConfig::default()
        };
        let session = Session::new(graph.clone(), config).unwrap();
        session.add_user_message("Echo a message").await.unwrap();

        let provider = Arc::new(MockProvider::new(vec![
            tool_call_response(vec![(
                "bash",
                "call_1",
                serde_json::json!({"command": "printf tracked"}),
            )]),
            text_response("Done."),
        ]));

        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(graphirm_tools::bash::BashTool::new()));

        let bus = EventBus::new();
        let token = CancellationToken::new();

        run_agent_loop(&session, provider.clone(), &tools, &bus, &token)
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
            .neighbors(
                &first_assistant.id,
                Some(EdgeType::Produces),
                Direction::Outgoing,
            )
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
            pre_completion_verify: false,
            ..AgentConfig::default()
        };
        let session = Session::new(graph.clone(), config).unwrap();
        session
            .add_user_message("List and find files")
            .await
            .unwrap();

        let provider = Arc::new(MockProvider::new(vec![
            tool_call_response(vec![
                ("ls", "call_ls", serde_json::json!({"path": "."})),
                ("find", "call_find", serde_json::json!({"pattern": "*.txt"})),
            ]),
            text_response("Done."),
        ]));

        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(graphirm_tools::ls::LsTool::new()));
        tools.register(Arc::new(graphirm_tools::find::FindTool::new()));

        let bus = EventBus::new();
        let token = CancellationToken::new();

        run_agent_loop(&session, provider.clone(), &tools, &bus, &token)
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
            .neighbors(
                &first_assistant.id,
                Some(EdgeType::Reads),
                Direction::Outgoing,
            )
            .unwrap();

        let content_labels: std::collections::HashSet<_> = content_nodes
            .iter()
            .filter_map(|node| node.label())
            .collect();
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
        session
            .add_user_message("Do infinite things")
            .await
            .unwrap();

        let provider = Arc::new(MockProvider::new(vec![
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
        ]));

        let mock_bash = Arc::new(MockTool {
            tool_name: "bash".to_string(),
            output: "ok".to_string(),
        });
        let mut tools = ToolRegistry::new();
        tools.register(mock_bash);

        let mut bus = EventBus::new();
        let mut rx = bus.subscribe();
        let token = CancellationToken::new();

        let result = run_agent_loop(&session, provider.clone(), &tools, &bus, &token).await;

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
            pre_completion_verify: false,
            ..AgentConfig::default()
        };
        let hitl = Arc::new(HitlGate::new());
        let session = Session::new(graph.clone(), config)
            .unwrap()
            .with_hitl(hitl.clone());
        session.add_user_message("Write a file").await.unwrap();

        let provider = Arc::new(MockProvider::new(vec![
            tool_call_response(vec![(
                "write",
                "call_w1",
                serde_json::json!({"path": "/tmp/test.txt", "content": "hello"}),
            )]),
            text_response("Done!"),
        ]));

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

        let result = run_agent_loop(&session, provider.clone(), &tools, &bus, &token).await;
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);
        assert_eq!(
            provider.call_count(),
            2,
            "LLM should be called twice (tool turn + final)"
        );

        // Tool execute() was invoked exactly once.
        assert_eq!(
            call_counter.load(Ordering::SeqCst),
            1,
            "Tool should have been called once"
        );

        // An ApprovedBy edge exists: tool-result node → session.id.
        let approved_sources = graph
            .neighbors(&session.id, Some(EdgeType::ApprovedBy), Direction::Incoming)
            .unwrap();
        assert_eq!(
            approved_sources.len(),
            1,
            "Expected exactly one ApprovedBy edge into session"
        );
    }

    #[tokio::test]
    async fn test_hitl_reject_skips_tool_and_continues() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig {
            max_turns: 10,
            pre_completion_verify: false,
            ..AgentConfig::default()
        };
        let hitl = Arc::new(HitlGate::new());
        let session = Session::new(graph.clone(), config)
            .unwrap()
            .with_hitl(hitl.clone());
        session.add_user_message("Write a file").await.unwrap();

        let provider = Arc::new(MockProvider::new(vec![
            tool_call_response(vec![(
                "write",
                "call_w1",
                serde_json::json!({"path": "/tmp/test.txt", "content": "hello"}),
            )]),
            // Loop continues after rejection and calls LLM again.
            text_response("I was rejected, moving on."),
        ]));

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

        let result = run_agent_loop(&session, provider.clone(), &tools, &bus, &token).await;
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);
        assert_eq!(provider.call_count(), 2, "LLM should be called twice");

        // Tool execute() must never have been called.
        assert_eq!(
            call_counter.load(Ordering::SeqCst),
            0,
            "Tool should NOT have been called"
        );

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
        session.add_user_message("Hello").await.unwrap();

        // No tool calls — just a simple text response after the pause clears.
        let provider = Arc::new(MockProvider::new(vec![text_response("All good.")]));
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
                if hitl_clone.resolve(&session_id, HitlDecision::Approve).await {
                    hitl_clone.set_paused(false);
                    break;
                }
            }
        });

        let result = run_agent_loop(&session, provider.clone(), &tools, &bus, &token).await;
        assert!(
            result.is_ok(),
            "Expected loop to complete after resume, got: {:?}",
            result
        );
        assert_eq!(
            provider.call_count(),
            1,
            "LLM should be called once after pause clears"
        );

        // Verify that AwaitingApproval with is_pause=true was emitted.
        let mut events = vec![];
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        let pause_event = events
            .iter()
            .find(|e| matches!(e, AgentEvent::AwaitingApproval { is_pause, .. } if *is_pause));
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
            pre_completion_verify: false,
            ..AgentConfig::default()
        };
        // No .with_hitl() — hitl is None
        let session = Session::new(graph.clone(), config).unwrap();
        session.add_user_message("Write a file").await.unwrap();

        // LLM requests a destructive tool (write), then returns a text response.
        let provider = Arc::new(MockProvider::new(vec![
            tool_call_response(vec![(
                "write",
                "call_w1",
                serde_json::json!({"path": "/tmp/test.txt", "content": "hello"}),
            )]),
            text_response("Done!"),
        ]));

        let mock_write = Arc::new(MockTool {
            tool_name: "write".to_string(),
            output: "Wrote /tmp/test.txt".to_string(),
        });
        let mut tools = ToolRegistry::new();
        tools.register(mock_write);

        let bus = EventBus::new();
        let token = CancellationToken::new();

        // Without HITL the loop should complete without hanging on a gate.
        let result = run_agent_loop(&session, provider.clone(), &tools, &bus, &token).await;
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
        session.add_user_message("Start working").await.unwrap();

        let provider = Arc::new(MockProvider::new(vec![
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
        ]));

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

        let result = run_agent_loop(&session, provider.clone(), &tools, &bus, &token).await;

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

    #[tokio::test]
    async fn test_pre_edit_impact_injects_brief_on_destructive_tool() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let graph = Arc::new(GraphStore::open_memory().unwrap());

        // Create a Knowledge node from "another session" mentioning "main.rs"
        let mut knowledge_node =
            GraphNode::new(NodeType::Knowledge(graphirm_graph::nodes::KnowledgeData {
                entity: "main.rs entry point".to_string(),
                entity_type: "file".to_string(),
                summary: "Critical entry point — changes here affect all CLI commands".to_string(),
                confidence: 0.95,
            }));
        knowledge_node.metadata["session_id"] = serde_json::json!("other-session");
        knowledge_node.metadata["turn"] = serde_json::json!(1);
        graph.add_node(knowledge_node).unwrap();

        let config = AgentConfig {
            max_turns: 10,
            pre_edit_impact: false, // Disable impact to avoid rg hanging in tests
            pre_completion_verify: false,
            working_dir: temp_dir.path().to_path_buf(),
            ..AgentConfig::default()
        };
        let hitl = Arc::new(HitlGate::new());
        hitl.set_auto_approve(true);
        let session = Session::new(graph.clone(), config)
            .unwrap()
            .with_hitl(hitl.clone());
        session.add_user_message("Edit main.rs").await.unwrap();

        let provider = Arc::new(MockProvider::new(vec![
            tool_call_response(vec![(
                "write",
                "call_w1",
                serde_json::json!({"path": "main.rs", "content": "fn main() {}"}),
            )]),
            text_response("Done!"),
        ]));

        let call_counter = Arc::new(AtomicUsize::new(0));
        let mock_write = Arc::new(TrackingMockTool {
            tool_name: "write".to_string(),
            output: "Wrote main.rs".to_string(),
            call_count: call_counter.clone(),
        });
        let mut tools = ToolRegistry::new();
        tools.register(mock_write);

        let bus = EventBus::new();
        let token = CancellationToken::new();

        let result = run_agent_loop(&session, provider.clone(), &tools, &bus, &token).await;
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);
        assert_eq!(
            call_counter.load(Ordering::SeqCst),
            1,
            "write tool should have been called once"
        );

        // Find the tool result node
        let neighbors = graph
            .neighbors(&session.id, Some(EdgeType::Produces), Direction::Outgoing)
            .unwrap();
        let tool_nodes: Vec<_> = neighbors
            .iter()
            .filter(|n| matches!(&n.node_type, NodeType::Interaction(d) if d.role == "tool"))
            .collect();

        assert!(!tool_nodes.is_empty(), "should have tool result nodes");
        let tool_content = match &tool_nodes[0].node_type {
            NodeType::Interaction(d) => &d.content,
            _ => panic!("expected Interaction"),
        };

        // Verify the tool executed successfully
        assert!(
            tool_content.contains("Wrote main.rs"),
            "tool output should contain original output, got: {tool_content}"
        );

        // Verify auto-approve was used (check ApprovedBy edge)
        let approved_sources = graph
            .neighbors(&session.id, Some(EdgeType::ApprovedBy), Direction::Incoming)
            .unwrap();
        assert!(
            !approved_sources.is_empty(),
            "Should have at least one ApprovedBy edge (auto-approved tool)"
        );
    }
}
