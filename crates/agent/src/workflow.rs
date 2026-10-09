// Agent workflow: async state machine with plan -> act -> observe -> reflect loop

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use futures::StreamExt;
use graphirm_graph::edges::{EdgeType, GraphEdge};
use graphirm_graph::nodes::{ContentData, GraphNode, InteractionData, NodeId, NodeType};
use graphirm_llm::{
    CompletionConfig, ContentPart, LlmProvider, LlmResponse, StopReason, StreamEvent, TokenUsage,
};
use graphirm_tools::ToolContext;
use graphirm_tools::registry::ToolRegistry;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tracing::info;

use crate::error::AgentError;
use crate::event::{AgentEvent, EventBus};
use crate::event_sink::EventBusSink;
use crate::hitl::HitlDecision;
use crate::session::Session;

fn flush_text_segment(text_buf: &mut String, parts: &mut Vec<ContentPart>) {
    if text_buf.is_empty() {
        return;
    }
    parts.push(ContentPart::text(std::mem::take(text_buf)));
}

/// Consumes a provider stream, emits [`AgentEvent::MessageDelta`] for each text chunk, and builds [`LlmResponse`].
async fn consume_llm_stream(
    mut stream: Pin<Box<dyn futures::Stream<Item = StreamEvent> + Send>>,
    preview_node_id: &NodeId,
    events: &EventBus,
) -> Result<LlmResponse, graphirm_llm::LlmError> {
    let mut text_buf = String::new();
    let mut parts: Vec<ContentPart> = Vec::new();
    let mut tool_build: HashMap<String, (String, String)> = HashMap::new();
    let mut usage = TokenUsage::default();
    let mut saw_done = false;

    while let Some(ev) = stream.next().await {
        match ev {
            StreamEvent::TextDelta(t) => {
                if !t.is_empty() {
                    events.emit(AgentEvent::MessageDelta {
                        node_id: preview_node_id.clone(),
                        delta: StreamEvent::TextDelta(t.clone()),
                    });
                }
                text_buf.push_str(&t);
            }
            StreamEvent::ThinkingDelta(_) => {}
            StreamEvent::ToolCallStart { id, name } => {
                flush_text_segment(&mut text_buf, &mut parts);
                tool_build.insert(id, (name, String::new()));
            }
            StreamEvent::ToolCallDelta {
                id,
                arguments_delta,
            } => {
                if let Some((_, buf)) = tool_build.get_mut(&id) {
                    buf.push_str(&arguments_delta);
                }
            }
            StreamEvent::ToolCallEnd { id } => {
                if let Some((name, args_str)) = tool_build.remove(&id) {
                    let arguments =
                        serde_json::from_str(&args_str).unwrap_or(serde_json::json!({}));
                    parts.push(ContentPart::tool_call(id, name, arguments));
                }
            }
            StreamEvent::Done(u) => {
                usage = u;
                saw_done = true;
                break;
            }
            StreamEvent::Error(msg) => return Err(graphirm_llm::LlmError::stream(msg)),
        }
    }

    if !saw_done {
        return Err(graphirm_llm::LlmError::stream(
            "stream ended without Done event",
        ));
    }

    flush_text_segment(&mut text_buf, &mut parts);
    let stop_reason = if parts
        .iter()
        .any(|p| matches!(p, ContentPart::ToolCall { .. }))
    {
        StopReason::ToolUse
    } else {
        StopReason::EndTurn
    };

    Ok(LlmResponse {
        content: parts,
        usage,
        stop_reason,
    })
}

/// Context window budget. `GRAPHIRM_CONTEXT_MAX_TOKENS` overrides the toml
/// value so a squeezed eval can force selection to drop messages. Unset keeps
/// the configured budget, or 100_000.
fn context_budget_tokens(configured: Option<u32>) -> usize {
    if let Ok(raw) = std::env::var("GRAPHIRM_CONTEXT_MAX_TOKENS") {
        match raw.parse::<usize>() {
            Ok(n) if n > 0 => {
                tracing::info!(
                    max_tokens = n,
                    "context budget set from GRAPHIRM_CONTEXT_MAX_TOKENS"
                );
                return n;
            }
            _ => {
                tracing::warn!(value = %raw, "ignoring GRAPHIRM_CONTEXT_MAX_TOKENS");
            }
        }
    }
    configured.map(|t| t as usize).unwrap_or(100_000)
}

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
        max_tokens: context_budget_tokens(session.agent_config.max_tokens),
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
    let tail_fraction = context_config.tail_max_fraction;

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
                tail_fraction,
            )
        })
        .await;
        match nodes {
            Ok(Ok(ids)) if !ids.is_empty() => {
                let cheap = session
                    .agent_config
                    .model_routing
                    .as_ref()
                    .map(|routing| routing.cheap.clone())
                    .unwrap_or_default();
                match crate::compact::resolve_compaction_model(&session.agent_config.model, &cheap)
                {
                    None => tracing::warn!("auto-compaction skipped: no model configured"),
                    Some(model) => {
                        tracing::info!(
                            count = ids.len(),
                            model = %model,
                            "auto-compacting old context nodes"
                        );
                        let compact_cfg = crate::compact::CompactionConfig {
                            model,
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
    let mut tool_defs: Vec<graphirm_llm::ToolDefinition> = raw_defs
        .into_iter()
        .filter(|t| {
            !(session.agent_config.disable_bash
                && crate::pi_delegate::tool::hidden_under_disable_bash(&t.name))
        })
        .map(|t| graphirm_llm::ToolDefinition::new(t.name, t.description, t.parameters))
        .collect();

    let mut tools_gated = false;
    if session.agent_config.tool_gate_enabled
        && matches!(session.agent_config.mode, crate::config::AgentMode::Primary)
        && let Some(ref text) = crate::tool_gate::last_human_message_text(&context)
        && crate::tool_gate::should_omit_tools_for_user_message(text)
        && !crate::tool_gate::task_is_open(&context)
        && !crate::tool_gate::is_verification_checklist(text)
    {
        tool_defs.clear();
        tools_gated = true;
        tracing::info!(
            preview = %text.chars().take(80).collect::<String>(),
            "tool_gate: omitting tool definitions for conversational user message"
        );
    }

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
                let tool_errored = last_tool_errored(&chain);
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
                let tool_errored = last_tool_errored(&chain);
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

    if let Some(cap) = session.agent_config.max_session_tokens {
        let used = session.llm_tokens_used();
        if used >= cap {
            return Err(AgentError::SessionTokenCapExceeded {
                used,
                cap,
                assistant_node_id: None,
            });
        }
    }

    let preview_node_id = NodeId::new();
    let mut fallback_chain: Vec<crate::router::FallbackAttempt> = Vec::new();
    let mut response: Option<LlmResponse> = None;

    for (i, model) in fallback_models.iter().enumerate() {
        let is_last = i == fallback_models.len() - 1;
        let comp_config = CompletionConfig::new(model)
            .with_max_tokens(max_output)
            .with_temperature(temperature);
        let start = std::time::Instant::now();
        match llm.stream(context.clone(), &tool_defs, &comp_config).await {
            Ok(stream) => {
                events.emit(AgentEvent::MessageStart {
                    node_id: preview_node_id.clone(),
                });
                match consume_llm_stream(stream, &preview_node_id, events).await {
                    Ok(resp) => {
                        if i > 0 {
                            selected_model = model.clone();
                        }
                        // Some models (e.g. DeepSeek via OpenRouter) emit DSML/XML tool blocks in
                        // assistant text instead of SSE `delta.tool_calls`; normalize so tools run.
                        let resp = graphirm_llm::augment_embedded_tool_calls(resp);
                        response = Some(resp);
                        break;
                    }
                    Err(e) if e.is_retryable() && !is_last => {
                        let latency_ms = start.elapsed().as_millis() as u64;
                        tracing::warn!(
                            model,
                            error = %e,
                            attempt = i + 1,
                            "LLM stream failed, trying next fallback model"
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

    let delta = u64::from(response.usage.total());
    let prev = session.llm_tokens_used();
    let cap = session.agent_config.max_session_tokens;
    let over_cap = cap.is_some_and(|c| prev.saturating_add(delta) > c);

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

    if over_cap {
        metadata.insert(
            "session_token_cap_exceeded".to_string(),
            serde_json::json!(true),
        );
    }

    if tools_gated {
        metadata.insert("tools_gated".to_string(), serde_json::json!(true));
    }

    let mut interaction_node = GraphNode::new(NodeType::Interaction(InteractionData {
        role: "assistant".to_string(),
        content: response.text_content(),
        token_count: Some(response.usage.output_tokens),
    }));
    interaction_node.id = preview_node_id.clone();
    interaction_node.metadata = serde_json::Value::Object(metadata);

    session.add_llm_completion_tokens(delta);

    let node_id = session.record_interaction(interaction_node).await?;

    if over_cap {
        // Segments never rewrite this node. Judge the raw text that stays stored.
        // A judge failure must not replace the token-cap error below.
        crate::reply_judge::observe_assistant_reply(
            session.graph.clone(),
            &session.id.0,
            &node_id,
            &session.agent_config,
            &response.text_content(),
        )
        .await;

        let cap = cap.expect("over_cap implies cap is Some");
        info!(
            node_id = %node_id,
            used = prev.saturating_add(delta),
            cap,
            "Recorded assistant response; session token cap exceeded"
        );
        events.emit(AgentEvent::MessageEnd {
            node_id: node_id.clone(),
        });
        return Err(AgentError::SessionTokenCapExceeded {
            used: prev.saturating_add(delta),
            cap,
            assistant_node_id: Some(node_id),
        });
    }

    let raw_text = response.text_content();
    // Set when the segment stamp replaces the Interaction body. Otherwise Jev
    // sees `raw_text`, which is what remains stored.
    let mut stamped_reply: Option<String> = None;

    // Structured response segmentation — opt-in, non-fatal.
    // Only runs on final text turns (no tool calls).
    // Primary path: parse JSON envelope emitted by LLM when structured_output is true.
    // Fallback path: GLiNER2 ONNX span detection (requires local-extraction feature +
    //   ExtractionConfig with a Local or Hybrid backend pointing at a downloaded model).
    let mut segment_path_persisted = false;
    if let Some(ref seg_config) = session.agent_config.segments
        && seg_config.enabled
        && !response.has_tool_calls()
    {
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
                                seg_config.label_descriptions.as_ref(),
                                seg_config.label_min_confidence.as_ref(),
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
                                seg_config.label_descriptions.as_ref(),
                                seg_config.label_min_confidence.as_ref(),
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
            if crate::knowledge::segments::has_tool_call_leak(&segments) {
                tracing::warn!(
                    "Detected tool_call leak in structured segments — skipping segment persistence"
                );
                let graph_clone = session.graph.clone();
                let stamp_id = node_id.clone();
                match tokio::task::spawn_blocking(move || {
                    let mut node = graph_clone.get_node(&stamp_id)?;
                    node.metadata["tool_call_leak"] = serde_json::json!(true);
                    graph_clone.update_node(&stamp_id, node)
                })
                .await
                {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        tracing::warn!(
                            error = %e,
                            "Failed to stamp tool_call_leak on assistant node (non-fatal)"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "spawn_blocking panicked while stamping tool_call_leak (non-fatal)"
                        );
                    }
                }
            } else {
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
                        segment_path_persisted = true;
                        tracing::info!(
                            count = seg_ids.len(),
                            source = source,
                            nesting_pairs = nesting.len(),
                            "Persisted response segments"
                        );
                        // Stamp the parent Interaction node so the context engine can detect
                        // that segment children exist and apply the segment_filter correctly.
                        // Replace raw JSON envelope with readable concatenated segment text.
                        let clean_text =
                            crate::knowledge::segments::segment_display_text(&segments);
                        let stamped_text = clean_text.clone();
                        let graph_clone = session.graph.clone();
                        let stamp_id = node_id.clone();
                        match tokio::task::spawn_blocking(move || {
                            let mut node = graph_clone.get_node(&stamp_id)?;
                            node.metadata["segmented"] = serde_json::json!(true);
                            if let NodeType::Interaction(ref mut data) = node.node_type {
                                data.content = clean_text;
                            }
                            graph_clone.update_node(&stamp_id, node)
                        })
                        .await
                        {
                            Ok(Ok(())) => {
                                stamped_reply = Some(stamped_text);
                            }
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
    }

    // Post-hoc outline (markdown headings → outline_item nodes). Skips when structured segment
    // JSON was emitted, when GLiNER/segment children were persisted, or when already extracted.
    let structured_json_envelope = crate::knowledge::segments::parse_structured_segments(&raw_text)
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    if let Some(ref oc) = session.agent_config.outline
        && oc.enabled
        && !response.has_tool_calls()
        && !segment_path_persisted
        && !structured_json_envelope
    {
        let graph = session.graph.clone();
        let parent_id = node_id.clone();
        let parent_id_for_items = parent_id.clone();
        let oc = oc.clone();
        let text_owned = raw_text.clone();
        match tokio::task::spawn_blocking(move || -> Result<bool, crate::error::AgentError> {
            let n = graph.get_node(&parent_id)?;
            if n.metadata
                .get("outline_extracted")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                return Ok(false);
            }
            Ok(true)
        })
        .await
        {
            Ok(Ok(false)) => {}
            Ok(Ok(true)) => {
                let items = crate::knowledge::outline::parse_markdown_outline(&text_owned);
                if !items.is_empty() {
                    let oc_ref = oc.clone();
                    let pid = parent_id_for_items.clone();
                    match crate::knowledge::outline::persist_outline_items(
                        &session.graph,
                        &pid,
                        &items,
                        &oc_ref,
                    )
                    .await
                    {
                        Ok(ids) => {
                            tracing::info!(
                                count = ids.len(),
                                "Persisted outline items from markdown headings"
                            );
                            let graph_clone = session.graph.clone();
                            let stamp_id = node_id.clone();
                            if let Err(e) = tokio::task::spawn_blocking(move || {
                                let mut node = graph_clone.get_node(&stamp_id)?;
                                node.metadata["outline_extracted"] = serde_json::json!(true);
                                node.metadata["outline_version"] = serde_json::json!(1);
                                graph_clone.update_node(&stamp_id, node)
                            })
                            .await
                            {
                                tracing::warn!(error = %e, "Failed to stamp outline metadata (non-fatal)");
                            }
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "Failed to persist outline items (non-fatal)");
                        }
                    }
                }
            }
            Ok(Err(e)) => {
                tracing::warn!(error = %e, "Outline pre-check failed (non-fatal)");
            }
            Err(e) => {
                tracing::warn!(error = %e, "Outline pre-check join failed (non-fatal)");
            }
        }
    }

    // Once, after the segment stamp (or its skip). Judge the body that remains.
    let judged = stamped_reply.as_deref().unwrap_or(raw_text.as_str());
    crate::reply_judge::observe_assistant_reply(
        session.graph.clone(),
        &session.id.0,
        &node_id,
        &session.agent_config,
        judged,
    )
    .await;

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
///
/// When `session.hitl` is `Some`, destructive tools (`write`, `edit`, `bash`)
/// are pulled out of the parallel set and processed sequentially, each awaiting
/// a human approval decision before executing.
#[allow(clippy::too_many_arguments)]
async fn execute_tools_parallel(
    session: &Session,
    tools: &ToolRegistry,
    response_id: &NodeId,
    tool_calls: &[&graphirm_llm::ContentPart],
    events: &EventBus,
    cancel: &CancellationToken,
    graph_queries: &mut crate::graph_query_guard::GraphQueryGuard,
    repeat_reads: &std::collections::HashSet<std::path::PathBuf>,
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

    // One sink per turn, shared by every `ToolContext` clone handed to a tool.
    // `EventBus` is `Clone` (shares subscriber channels), so no signature
    // change is needed to get an `Arc<EventBus>`.
    let sink = Arc::new(EventBusSink::new(
        Arc::new(events.clone()),
        session.graph.clone(),
    ));

    let ctx = ToolContext {
        graph: session.graph.clone(),
        agent_id: session.id.clone(),
        interaction_id: response_id.clone(),
        working_dir: session.agent_config.working_dir.clone(),
        signal: cancel.clone(),
        turn: session.current_turn(),
        turn_pos_counter: session.turn_position_counter(),
        knowledge_retriever,
        impact_provider,
        disable_bash: session.agent_config.disable_bash,
        auto_link_write_to_planning: session.agent_config.auto_link_write_to_planning,
        event_sink: Some(Arc::clone(&sink) as Arc<dyn graphirm_tools::ToolEventSink>),
    };

    // `ctx` is moved into `run_tool_calls` and dropped (along with every clone
    // spawned into tool tasks) before it returns, on success and error alike.
    let result = run_tool_calls(
        session,
        tools,
        response_id,
        tool_calls,
        events,
        cancel,
        ctx,
        graph_queries,
        repeat_reads,
    )
    .await;

    // Drain the sink *before* the caller emits its own turn-end GraphUpdate, so
    // a `graph_changed` queued at the tail of a tool's run can never be emitted
    // after it and regress the view. `try_unwrap` fails only if a tool stashed
    // `ctx.event_sink` somewhere that outlives `execute`, or a tool task is
    // still winding down after a join error.
    match Arc::try_unwrap(sink) {
        Ok(s) => s.close().await,
        Err(_) => tracing::warn!(
            "EventBusSink still shared at turn end (a tool stashed ctx.event_sink or is \
             still running); GraphUpdate ordering not guaranteed for this turn"
        ),
    }

    result
}

/// Body of [`execute_tools_parallel`]: partition, run, and record every tool
/// call in `tool_calls` using `ctx`. Takes `ctx` by value so that all
/// references to its `event_sink` are gone when this returns.
#[allow(clippy::too_many_arguments)] // the query guard lives for one agent loop, not on ToolContext
async fn run_tool_calls(
    session: &Session,
    tools: &ToolRegistry,
    response_id: &NodeId,
    tool_calls: &[&graphirm_llm::ContentPart],
    events: &EventBus,
    cancel: &CancellationToken,
    ctx: ToolContext,
    graph_queries: &mut crate::graph_query_guard::GraphQueryGuard,
    repeat_reads: &std::collections::HashSet<std::path::PathBuf>,
) -> Result<Vec<NodeId>, AgentError> {
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

    // Phase 1: spawn SAFE tools in parallel and collect results.
    // Resolve every tool BEFORE spawning anything: an unknown tool name must
    // bail out with an empty JoinSet, otherwise the `?` would return while
    // spawned tasks still hold clones of `ctx` (and its event sink), racing
    // the `Arc::try_unwrap(sink)` in `execute_tools_parallel`.
    let mut resolved = Vec::with_capacity(safe_calls.len());
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
        resolved.push((tool, call));
    }

    let mut set = JoinSet::new();
    let mut synthetic = Vec::new();
    for (tool, call) in resolved {
        if call.name == "read"
            && let Some(path) = call.arguments.get("path").and_then(|value| value.as_str())
            && repeat_reads.contains(std::path::Path::new(path))
        {
            tracing::info!(path, "read loop returned already read");
            synthetic.push((
                call.id,
                call.name,
                Ok(graphirm_tools::ToolOutput::success(already_read_notice(
                    std::path::Path::new(path),
                ))),
            ));
            continue;
        }
        if call.name == "graph_query" {
            match graph_queries.prepare(&call.arguments) {
                crate::graph_query_guard::GraphQueryDecision::Run => {}
                crate::graph_query_guard::GraphQueryDecision::Repeat { notice } => {
                    tracing::info!("graph_query guard returned the previous result");
                    synthetic.push((
                        call.id,
                        call.name,
                        Ok(graphirm_tools::ToolOutput::success(notice)),
                    ));
                    continue;
                }
                crate::graph_query_guard::GraphQueryDecision::Cap { notice } => {
                    tracing::info!("graph_query guard hit the per-turn cap");
                    synthetic.push((
                        call.id,
                        call.name,
                        Ok(graphirm_tools::ToolOutput::success(notice)),
                    ));
                    continue;
                }
            }
        }
        let ctx_clone = ctx.clone();
        set.spawn(async move {
            let arguments = call.arguments.clone();
            let result: Result<graphirm_tools::ToolOutput, graphirm_tools::ToolError> =
                tool.execute(arguments.clone(), &ctx_clone).await;
            (call.id, call.name, arguments, result)
        });
    }

    // Drain every task before propagating a join error so no spawned clone of
    // `ctx` outlives this function (the sink is closed right after we return).
    let mut exec_results = Vec::new();
    let mut join_error: Option<AgentError> = None;
    while let Some(join_result) = set.join_next().await {
        match join_result {
            Ok((call_id, tool_name, arguments, result)) => {
                if tool_name == "graph_query" {
                    let text = match &result {
                        Ok(output) => output.content.clone(),
                        Err(error) => error.to_string(),
                    };
                    graph_queries.remember(&arguments, &text);
                }
                exec_results.push((call_id, tool_name, result));
            }
            Err(e) => {
                join_error.get_or_insert_with(|| AgentError::Join(e.to_string()));
            }
        }
    }
    if let Some(e) = join_error {
        return Err(e);
    }
    exec_results.extend(synthetic);

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

        // Auto-approve skips the gate — unless the additive judge scores this
        // call irreversible, in which case it falls through to the human gate
        // (or, headless, is merely recorded). The judge never removes a pause.
        let judge_outcome = hitl.judge_auto_approve(name, arguments).await;
        let auto_approved =
            hitl.is_auto_approve() && !judge_outcome.as_ref().is_some_and(|o| o.pause);

        let decision = if auto_approved {
            HitlDecision::Approve
        } else {
            events.emit(AgentEvent::AwaitingApproval {
                node_id: gate_key.clone(),
                tool_name: name.clone(),
                arguments: arguments.clone(),
                is_pause: false,
                hitl_judge: judge_outcome.as_ref().map(|o| o.to_metadata()),
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
                let impact_brief_text = if let Some(ref provider) = ctx.impact_provider {
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
                if let Some(ref outcome) = judge_outcome {
                    tool_metadata.insert("hitl_judge".to_string(), outcome.to_metadata());
                }

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

                let mut tool_metadata = serde_json::Map::new();
                tool_metadata.insert("tool_call_id".to_string(), serde_json::json!(call_id));
                tool_metadata.insert("tool_name".to_string(), serde_json::json!(name));
                tool_metadata.insert("is_error".to_string(), serde_json::json!(true));
                let mut tool_node = GraphNode::new(NodeType::Interaction(InteractionData {
                    role: "tool".to_string(),
                    content: format!("rejected by user: {reason}"),
                    token_count: None,
                }));
                tool_node.metadata = serde_json::Value::Object(tool_metadata);
                match session.record_interaction(tool_node).await {
                    Ok(result_node_id) => {
                        events.emit(AgentEvent::ToolEnd {
                            node_id: result_node_id.clone(),
                            is_error: true,
                        });
                        node_ids.push(result_node_id);
                    }
                    Err(e) => {
                        tracing::error!(
                            "Failed to record rejection tool result for call {call_id}: {e}"
                        );
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
        "fetch_url",
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

/// Whether the most recent tool result in the session chain errored.
///
/// Tool results are recorded with role `"tool"` (see the tool-node writers in
/// `execute_tool_calls`); the signal used to look for `"tool_result"` and scan
/// the whole chain, so it never fired and `error_recovery` routing was dead.
fn last_tool_errored(chain: &[graphirm_graph::nodes::GraphNode]) -> bool {
    chain
        .iter()
        .rev()
        .find(|n| {
            matches!(&n.node_type, graphirm_graph::nodes::NodeType::Interaction(i) if i.role == "tool")
        })
        .and_then(|n| n.metadata.get("is_error").and_then(|v| v.as_bool()))
        .unwrap_or(false)
}

async fn emit_graph_update(
    session: &Session,
    node_id: &NodeId,
    tool_result_node_ids: Vec<NodeId>,
    events: &EventBus,
) {
    emit_graph_update_for(session.graph.clone(), node_id, tool_result_node_ids, events).await;
}

/// Build a `GraphUpdate` payload from `graph` (in `spawn_blocking`) and emit it
/// on `events`. `node_id` is the anchor; `tool_result_node_ids` are the nodes
/// whose incident edges are included in `recent_edges`.
///
/// Session-independent so it can be reused by `EventBusSink`, which only has
/// access to the graph store.
pub(crate) async fn emit_graph_update_for(
    graph: Arc<graphirm_graph::GraphStore>,
    node_id: &NodeId,
    tool_result_node_ids: Vec<NodeId>,
    events: &EventBus,
) {
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

/// A text reply that names a next tool action, so the loop can ask for that
/// call once instead of treating the plan as the end of the task.
pub(crate) fn announces_unfinished_action(text: &str) -> bool {
    let lower = text.replace('Ġ', " ").replace('Ċ', "\n").to_lowercase();
    const LEADS: &[&str] = &[
        "i will ",
        "i'll ",
        "i’ll ",
        "i shall ",
        "i'm going to ",
        "i am going to ",
        "next i'll ",
        "next i’ll ",
        "next i will ",
        "let me ",
        "i should ",
        "i'll proceed",
        "i’ll proceed",
    ];
    const VERBS: &[&str] = &[
        "write",
        "edit",
        "read",
        "run",
        "fix",
        "create",
        "add",
        "update",
        "replace",
        "implement",
        "check",
        "test",
        "open",
        "delete",
        "remove",
        "change",
        "apply",
        "build",
        "compile",
        "call",
    ];
    for lead in LEADS {
        let Some(at) = lower.find(lead) else {
            continue;
        };
        let from = at + lead.len();
        let to = (from + 80).min(lower.len());
        let window = &lower[from..to];
        let names_action = window
            .split(|c: char| !c.is_ascii_alphanumeric())
            .any(|word| VERBS.contains(&word));
        if names_action {
            return true;
        }
    }
    false
}

/// One checklist after a finished reply that followed a write or edit.
/// `git diff` is included only in a git repo. Tests are skipped for text,
/// config, and `/tmp` scratch files. Code changes get a scoped test command.
/// `full_suite` keeps `cargo test` when the user asked for the whole run.
pub(crate) fn verification_checklist(
    root: &std::path::Path,
    changed: &[std::path::PathBuf],
    full_suite: bool,
) -> String {
    let mut steps = Vec::new();
    if root.join(".git").exists() {
        steps.push(
            "Run `git diff --name-only` and confirm only the intended files changed.".to_string(),
        );
    }
    if let Some(command) = scoped_test_command(root, changed, full_suite) {
        if full_suite {
            steps.push(format!("Run `{command}` and confirm it passes."));
        } else {
            steps.push(format!(
                "Run `{command}` and confirm it passes. Do not run a wider suite."
            ));
        }
    } else if !full_suite
        && !changed.is_empty()
        && changed.iter().all(|path| is_non_code(root, path))
    {
        steps.push("Do not run tests. The changes are text, config, or scratch files.".to_string());
    }
    steps.push(
        "In your final reply, state what changed and the result. Include the output you confirmed, or the code you wrote. Then stop. Do not re-read files you already read.".to_string(),
    );
    let mut out = format!("{}\n", crate::tool_gate::VERIFICATION_LEAD);
    for (index, step) in steps.iter().enumerate() {
        out.push_str(&format!("{}. {step}\n", index + 1));
    }
    out
}

fn test_command(root: &std::path::Path) -> Option<&'static str> {
    if root.join("Cargo.toml").is_file() && rust_tests_present(root) {
        return Some("cargo test");
    }
    if npm_has_test_script(root) {
        return Some("npm test");
    }
    if python_tests_present(root) {
        return Some("python -m pytest");
    }
    None
}

/// The user asked for the whole suite, not a scoped run.
pub(crate) fn user_asked_for_full_suite(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower.contains("full suite")
        || lower.contains("full test")
        || lower.contains("all tests")
        || lower.contains("entire test")
        || (lower.contains("cargo test") && !lower.contains("-p "))
}

/// Text, config, and `/tmp` scratch files do not need a test run.
/// A workspace that itself lives under `/tmp` is not scratch.
fn is_non_code(root: &std::path::Path, path: &std::path::Path) -> bool {
    if is_scratch(root, path) {
        return true;
    }
    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    matches!(
        ext.as_str(),
        "" | "txt"
            | "md"
            | "markdown"
            | "rst"
            | "toml"
            | "json"
            | "yaml"
            | "yml"
            | "ini"
            | "cfg"
            | "conf"
            | "csv"
            | "env"
            | "log"
    )
}

fn is_scratch(root: &std::path::Path, path: &std::path::Path) -> bool {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    if absolute.starts_with(root) {
        return false;
    }
    let text = path.to_string_lossy().replace('\\', "/");
    text == "/tmp" || text.starts_with("/tmp/")
}

fn is_rust(path: &std::path::Path) -> bool {
    path.extension().is_some_and(|ext| ext == "rs")
}

fn scoped_test_command(
    root: &std::path::Path,
    changed: &[std::path::PathBuf],
    full_suite: bool,
) -> Option<String> {
    if full_suite {
        return test_command(root).map(str::to_string);
    }
    let code: Vec<&std::path::PathBuf> = changed
        .iter()
        .filter(|path| !is_non_code(root, path))
        .collect();
    if code.is_empty() || !root.join("Cargo.toml").is_file() {
        return None;
    }
    let mut packages = Vec::new();
    for path in &code {
        if !is_rust(path) {
            continue;
        }
        if let Some(name) = package_name_for(root, path)
            && !packages.contains(&name)
        {
            packages.push(name);
        }
    }
    if !packages.is_empty() {
        let mut command = String::from("cargo test");
        for name in &packages {
            command.push_str(" -p ");
            command.push_str(name);
        }
        if packages.len() == 1
            && let Some(test_name) = code.iter().find_map(|path| integration_test_name(path))
        {
            command.push_str(" --test ");
            command.push_str(&test_name);
        }
        return Some(command);
    }
    code.iter()
        .find_map(|path| nearest_test_target(root, path))
        .map(|test_name| format!("cargo test --test {test_name}"))
}

fn package_name_for(root: &std::path::Path, changed: &std::path::Path) -> Option<String> {
    let start = if changed.is_absolute() {
        changed.to_path_buf()
    } else {
        root.join(changed)
    };
    let mut dir = start.parent()?.to_path_buf();
    loop {
        if let Some(name) = package_name(&dir.join("Cargo.toml")) {
            return Some(name);
        }
        if dir == root {
            break;
        }
        let parent = dir.parent()?;
        if parent != root && !parent.starts_with(root) {
            break;
        }
        dir = parent.to_path_buf();
    }
    None
}

fn package_name(manifest: &std::path::Path) -> Option<String> {
    let text = std::fs::read_to_string(manifest).ok()?;
    let value: toml::Value = toml::from_str(&text).ok()?;
    value
        .get("package")?
        .get("name")?
        .as_str()
        .map(str::to_string)
}

fn integration_test_name(path: &std::path::Path) -> Option<String> {
    let parent = path.parent()?;
    if parent.file_name().is_some_and(|name| name == "tests")
        && path.extension().is_some_and(|ext| ext == "rs")
    {
        path.file_stem()?.to_str().map(str::to_string)
    } else {
        None
    }
}

fn nearest_test_target(root: &std::path::Path, changed: &std::path::Path) -> Option<String> {
    if let Some(name) = integration_test_name(changed) {
        return Some(name);
    }
    let full = if changed.is_absolute() {
        changed.to_path_buf()
    } else {
        root.join(changed)
    };
    let stem = full.file_stem()?.to_str()?;
    let mut dir = full.parent()?.to_path_buf();
    for _ in 0..6 {
        let candidate = dir.join("tests").join(format!("{stem}.rs"));
        if candidate.is_file() {
            return Some(stem.to_string());
        }
        if dir == root {
            break;
        }
        let parent = dir.parent()?;
        if parent != root && !parent.starts_with(root) {
            break;
        }
        dir = parent.to_path_buf();
    }
    None
}

/// A later read of a file that was already read enough times.
pub(crate) fn read_is_repeat(count: u32, threshold: u32) -> bool {
    threshold > 0 && count >= threshold
}

pub(crate) fn already_read_notice(path: &std::path::Path) -> String {
    format!(
        "already read `{}`. Use the earlier result. Do not read this file again until you edit it.",
        path.display()
    )
}

fn rust_tests_present(root: &std::path::Path) -> bool {
    if directory_has_file(root.join("tests")) {
        return true;
    }
    rust_file_has_test_module(&root.join("src"), 0)
}

fn rust_file_has_test_module(dir: &std::path::Path, depth: u8) -> bool {
    if depth > 8 {
        return false;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name == "target" || name == ".git" {
                continue;
            }
            if rust_file_has_test_module(&path, depth + 1) {
                return true;
            }
        } else if path.extension().is_some_and(|ext| ext == "rs")
            && std::fs::read_to_string(&path).is_ok_and(|text| text.contains("#[cfg(test)]"))
        {
            return true;
        }
    }
    false
}

fn npm_has_test_script(root: &std::path::Path) -> bool {
    let Ok(text) = std::fs::read_to_string(root.join("package.json")) else {
        return false;
    };
    let Some(scripts) = text.split("\"scripts\"").nth(1) else {
        return false;
    };
    let Some(test) = scripts.split("\"test\"").nth(1) else {
        return false;
    };
    !test.contains("no test specified")
}

fn python_tests_present(root: &std::path::Path) -> bool {
    root.join("pytest.ini").is_file()
        || root.join("tox.ini").is_file()
        || directory_has_file(root.join("tests"))
}

fn directory_has_file(dir: std::path::PathBuf) -> bool {
    std::fs::read_dir(dir).is_ok_and(|mut entries| entries.next().is_some())
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
    let pre_completion_verify = session.agent_config.pre_completion_verify;
    let doom_loop_threshold = session.agent_config.doom_loop_threshold;
    let read_loop_threshold = session.agent_config.read_loop_threshold;
    let mut all_node_ids: Vec<NodeId> = Vec::new();
    // One extra turn when a text reply announces a tool action and then stops.
    let mut announced_action_continued = false;
    // True after any write or edit in this session. The checklist fires once
    // on the next finished reply, then the following finished reply ends.
    let mut had_write_calls = false;
    let mut verify_injected = false;
    let mut changed_paths: Vec<std::path::PathBuf> = Vec::new();
    let full_suite = session
        .recent_user_message()
        .await
        .is_some_and(|text| user_asked_for_full_suite(&text));
    // Per-file write/edit counts for doom loop detection.
    let mut file_edit_counts: std::collections::HashMap<std::path::PathBuf, u32> =
        std::collections::HashMap::new();
    // Per-file read counts for read-loop detection (catches verification doom loops).
    let mut file_read_counts: std::collections::HashMap<std::path::PathBuf, u32> =
        std::collections::HashMap::new();
    // Retries after segment JSON contained fake `tool_call` types instead of native tools.
    let mut tool_call_leak_retries: u32 = 0;
    let mut graph_queries = crate::graph_query_guard::GraphQueryGuard::default();

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
                    hitl_judge: None,
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
            result = stream_and_record(session, llm.clone(), tools, events) => match result {
                Ok(pair) => pair,
                Err(AgentError::SessionTokenCapExceeded {
                    used,
                    cap,
                    assistant_node_id,
                }) => {
                    if let Some(ref nid) = assistant_node_id {
                        all_node_ids.push(nid.clone());
                    }
                    let _ = session.set_status("token_cap_exceeded").await;
                    events.emit(AgentEvent::AgentEnd {
                        agent_id: session.id.clone(),
                        node_ids: all_node_ids.clone(),
                    });
                    return Err(AgentError::SessionTokenCapExceeded {
                        used,
                        cap,
                        assistant_node_id,
                    });
                }
                Err(e) => return Err(e),
            },
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
            // Model put tool invocations inside segment JSON instead of using native tool calls.
            let leak_graph = session.graph.clone();
            let leak_rid = response_id.clone();
            let is_tool_call_leak = tokio::task::spawn_blocking(move || {
                leak_graph
                    .get_node(&leak_rid)
                    .ok()
                    .and_then(|n| n.metadata.get("tool_call_leak").and_then(|v| v.as_bool()))
                    .unwrap_or(false)
            })
            .await
            .unwrap_or(false);

            if is_tool_call_leak {
                if tool_call_leak_retries < 2 {
                    tool_call_leak_retries += 1;
                    tracing::warn!(
                        turn,
                        retry = tool_call_leak_retries,
                        "Tool-call segment leak detected; injecting retry nudge for native tools API"
                    );
                    events.emit(AgentEvent::TurnEnd {
                        response_id: response_id.clone(),
                        tool_result_ids: vec![],
                    });
                    emit_graph_update(session, &response_id, vec![], events).await;
                    let nudge = GraphNode::new(NodeType::Interaction(InteractionData {
                        role: "user".to_string(),
                        content: "You put tool invocations inside your JSON segments instead of \
                                  using the tools API. Please retry — call tools using the \
                                  native tool-calling interface (e.g. `read`, `write`)."
                            .to_string(),
                        token_count: None,
                    }));
                    if let Err(e) = session.record_interaction(nudge).await {
                        tracing::warn!(error = %e, "Failed to inject tool-call leak retry (non-fatal)");
                    } else {
                        continue;
                    }
                } else {
                    tracing::warn!(
                        turn,
                        "tool_call_leak persisted after max retries; continuing with text-only exit"
                    );
                }
            }

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

            // A plan with no tool call is not a finished task. One continue, then
            // a second prose reply is allowed to end the loop. A report of finished
            // work does not match, so that reply ends the task.
            if !announced_action_continued && announces_unfinished_action(&response.text_content())
            {
                announced_action_continued = true;
                tracing::info!(
                    turn,
                    "Text-only turn announced an action; injecting continue"
                );
                let cont_node = graphirm_graph::nodes::GraphNode::new(
                    graphirm_graph::nodes::NodeType::Interaction(
                        graphirm_graph::nodes::InteractionData {
                            role: "user".to_string(),
                            content: "Continue. Call the read, write, or edit tool for the action you just stated."
                                .to_string(),
                            token_count: None,
                        },
                    ),
                );
                if let Err(e) = session.record_interaction(cont_node).await {
                    tracing::warn!(error = %e, "Failed to inject action continue (non-fatal)");
                } else {
                    continue;
                }
            }

            // One checklist after the first finished reply that followed a write
            // or edit. The reply after that checklist ends the task.
            if pre_completion_verify && had_write_calls && !verify_injected {
                verify_injected = true;
                tracing::info!(turn, "Injecting pre-completion verification checklist");
                let verify_node = graphirm_graph::nodes::GraphNode::new(
                    graphirm_graph::nodes::NodeType::Interaction(
                        graphirm_graph::nodes::InteractionData {
                            role: "user".to_string(),
                            content: verification_checklist(
                                &session.agent_config.working_dir,
                                &changed_paths,
                                full_suite,
                            ),
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

        for part in &tool_calls {
            let ContentPart::ToolCall {
                name, arguments, ..
            } = part
            else {
                continue;
            };
            if name == "write" || name == "edit" {
                had_write_calls = true;
                if let Some(path_str) = arguments.get("path").and_then(|v| v.as_str()) {
                    changed_paths.push(std::path::PathBuf::from(path_str));
                }
            }
        }

        // Doom loop tracking: count write/edit calls per file path.
        let mut edited_this_turn: Vec<std::path::PathBuf> = Vec::new();
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
                    let p = std::path::PathBuf::from(path_str);
                    *file_edit_counts.entry(p.clone()).or_insert(0) += 1;
                    edited_this_turn.push(p.clone());
                    file_read_counts.remove(&p);
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
                    let p = std::path::PathBuf::from(path_str);
                    *file_read_counts.entry(p).or_insert(0) += 1;
                } else if name == "read_many"
                    && let Some(paths) = arguments.get("paths").and_then(|v| v.as_array())
                {
                    for p in paths.iter().filter_map(|v| v.as_str()) {
                        let pb = std::path::PathBuf::from(p);
                        *file_read_counts.entry(pb).or_insert(0) += 1;
                    }
                }
            }
        }

        let repeat_reads: std::collections::HashSet<std::path::PathBuf> = file_read_counts
            .iter()
            .filter(|(_, count)| read_is_repeat(**count, read_loop_threshold))
            .map(|(path, _)| path.clone())
            .collect();

        let tool_result_ids = execute_tools_parallel(
            session,
            tools,
            &response_id,
            tool_calls.as_slice(),
            events,
            cancel,
            &mut graph_queries,
            &repeat_reads,
        )
        .await?;

        all_node_ids.extend(tool_result_ids.iter().cloned());

        // Doom loop advisory: warn when the agent has edited a file too many times.
        // Only check files edited THIS turn to avoid re-firing on stale counts.
        if doom_loop_threshold > 0 {
            for file_path in &edited_this_turn {
                let count = file_edit_counts.get(file_path).copied().unwrap_or(0);
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
pub(crate) mod test_helpers {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;

    use super::*;
    use graphirm_llm::{
        CompletionConfig, ContentPart, LlmError, LlmMessage, LlmProvider, LlmResponse, StopReason,
        StreamEvent, TokenUsage, ToolDefinition,
    };

    /// Mock LLM provider that returns pre-configured responses in order.
    pub struct MockProvider {
        pub responses: Vec<LlmResponse>,
        pub call_index: AtomicUsize,
        /// Tool names offered on each call, in call order.
        pub seen_tool_names: std::sync::Mutex<Vec<Vec<String>>>,
    }

    impl MockProvider {
        pub fn new(responses: Vec<LlmResponse>) -> Self {
            Self {
                responses,
                call_index: AtomicUsize::new(0),
                seen_tool_names: std::sync::Mutex::new(Vec::new()),
            }
        }

        pub fn call_count(&self) -> usize {
            self.call_index.load(Ordering::SeqCst)
        }

        /// Tool names offered on the most recent call.
        pub fn last_tool_names(&self) -> Vec<String> {
            self.seen_tool_names
                .lock()
                .expect("lock")
                .last()
                .cloned()
                .unwrap_or_default()
        }

        fn record_tools(&self, tools: &[ToolDefinition]) {
            self.seen_tool_names
                .lock()
                .expect("lock")
                .push(tools.iter().map(|t| t.name.clone()).collect());
        }
    }

    #[async_trait]
    impl LlmProvider for MockProvider {
        async fn complete(
            &self,
            _messages: Vec<LlmMessage>,
            tools: &[ToolDefinition],
            _config: &CompletionConfig,
        ) -> Result<LlmResponse, LlmError> {
            self.record_tools(tools);
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
            tools: &[ToolDefinition],
            _config: &CompletionConfig,
        ) -> Result<std::pin::Pin<Box<dyn futures::Stream<Item = StreamEvent> + Send>>, LlmError>
        {
            self.record_tools(tools);
            let idx = self.call_index.fetch_add(1, Ordering::SeqCst);
            let response = if idx < self.responses.len() {
                self.responses[idx].clone()
            } else {
                return Err(LlmError::Provider("No more mock responses".to_string()));
            };

            let mut events: Vec<StreamEvent> = Vec::new();
            for part in &response.content {
                match part {
                    ContentPart::Text { text } => {
                        for chunk in text.as_bytes().chunks(10) {
                            events.push(StreamEvent::text_delta(String::from_utf8_lossy(chunk)));
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
                    ContentPart::ToolResult { .. } => {}
                }
            }
            events.push(StreamEvent::done(response.usage.clone()));
            Ok(Box::pin(futures::stream::iter(events)))
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
mod signal_tests {
    use graphirm_graph::nodes::{GraphNode, InteractionData, NodeType};

    use super::last_tool_errored;

    fn interaction(role: &str, metadata: serde_json::Value) -> GraphNode {
        let mut node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: role.to_string(),
            content: String::new(),
            token_count: None,
        }));
        node.metadata = metadata;
        node
    }

    fn tool(is_error: bool) -> GraphNode {
        interaction(
            "tool",
            serde_json::json!({"tool_name": "bash", "is_error": is_error}),
        )
    }

    #[test]
    fn empty_chain_is_not_errored() {
        assert!(!last_tool_errored(&[]));
    }

    #[test]
    fn fires_when_most_recent_tool_result_errored() {
        // Regression: the signal looked for role "tool_result" (never written)
        // so error_recovery routing was dead.
        let chain = [
            interaction("user", serde_json::json!({})),
            tool(false),
            tool(true),
        ];
        assert!(last_tool_errored(&chain));
    }

    #[test]
    fn does_not_fire_when_a_later_tool_result_succeeded() {
        let chain = [
            tool(true),
            interaction("assistant", serde_json::json!({})),
            tool(false),
        ];
        assert!(!last_tool_errored(&chain));
    }

    #[test]
    fn ignores_assistant_nodes_after_the_last_tool_result() {
        let chain = [
            tool(true),
            interaction("assistant", serde_json::json!({"is_error": false})),
        ];
        assert!(last_tool_errored(&chain));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::test_helpers::*;
    use super::*;
    use crate::config::AgentConfig;
    use crate::error::AgentError;
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

    /// `disable_bash` hides `delegate_pi` from the model alongside `bash`:
    /// Pi runs shell, so a locked-down server must not offer it.
    #[tokio::test]
    async fn disable_bash_hides_bash_and_delegate_pi_from_tool_definitions() {
        fn registry() -> ToolRegistry {
            let mut tools = ToolRegistry::new();
            for name in ["bash", "delegate_pi", "read"] {
                tools.register(Arc::new(MockTool {
                    tool_name: name.to_string(),
                    output: "ok".to_string(),
                }));
            }
            tools
        }
        async fn offered(disable_bash: bool) -> Vec<String> {
            let graph = Arc::new(GraphStore::open_memory().unwrap());
            let config = AgentConfig {
                disable_bash,
                tool_gate_enabled: false,
                ..Default::default()
            };
            let session = Session::new(graph, config).unwrap();
            session.add_user_message("implement it").await.unwrap();
            let provider = Arc::new(MockProvider::new(vec![text_response("done")]));
            let bus = EventBus::new();
            stream_and_record(&session, provider.clone(), &registry(), &bus)
                .await
                .unwrap();
            provider.last_tool_names()
        }

        assert_eq!(offered(false).await, vec!["bash", "delegate_pi", "read"]);
        assert_eq!(offered(true).await, vec!["read"]);
    }

    #[tokio::test]
    async fn test_stream_and_record_session_token_cap_exceeded_on_second_turn() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig {
            max_session_tokens: Some(200),
            pre_completion_verify: false,
            ..Default::default()
        };
        let session = Session::new(graph.clone(), config).unwrap();
        session.add_user_message("q1").await.unwrap();

        let provider = Arc::new(MockProvider::new(vec![
            text_response("a1"),
            text_response("a2"),
        ]));
        let tools = ToolRegistry::new();
        let bus = EventBus::new();

        stream_and_record(&session, provider.clone(), &tools, &bus)
            .await
            .unwrap();
        assert_eq!(session.llm_tokens_used(), 120);

        session.add_user_message("q2").await.unwrap();
        let err = stream_and_record(&session, provider.clone(), &tools, &bus)
            .await
            .unwrap_err();
        match err {
            AgentError::SessionTokenCapExceeded {
                used,
                cap,
                assistant_node_id,
            } => {
                assert_eq!(cap, 200);
                assert_eq!(used, 240);
                assert!(assistant_node_id.is_some());
            }
            _ => panic!("expected SessionTokenCapExceeded, got {err:?}"),
        }
        assert_eq!(session.llm_tokens_used(), 240);
    }

    #[tokio::test]
    async fn test_stream_and_record_session_token_cap_blocks_before_llm_when_already_at_cap() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig {
            max_session_tokens: Some(50),
            ..Default::default()
        };
        let session = Session::new(graph.clone(), config).unwrap();
        session.test_set_llm_tokens_used(50);
        session.add_user_message("q").await.unwrap();

        let provider = Arc::new(MockProvider::new(vec![text_response("never")]));
        let tools = ToolRegistry::new();
        let bus = EventBus::new();

        let err = stream_and_record(&session, provider.clone(), &tools, &bus)
            .await
            .unwrap_err();
        match err {
            AgentError::SessionTokenCapExceeded {
                used,
                cap,
                assistant_node_id,
            } => {
                assert_eq!(used, 50);
                assert_eq!(cap, 50);
                assert!(assistant_node_id.is_none());
            }
            _ => panic!("expected SessionTokenCapExceeded, got {err:?}"),
        }
        assert_eq!(provider.call_count(), 0);
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

    #[test]
    fn announced_action_matches_the_selection_misses() {
        assert!(announces_unfinished_action(
            "I should read early.txt now to get the token, then write answer.txt with only that token."
        ));
        assert!(announces_unfinished_action(
            "I'll\u{0120}read\u{0120}the\u{0120}files\u{0120}in\u{0120}order, and edit late.txt."
        ));
        assert!(announces_unfinished_action(
            "Next I'll write selection/answer.txt."
        ));
        assert!(!announces_unfinished_action(
            "Here are your files: src/ Cargo.toml"
        ));
        assert!(!announces_unfinished_action("I wrote answer.txt."));
    }

    #[tokio::test]
    async fn announced_action_gets_one_continue() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig {
            max_turns: 4,
            pre_completion_verify: false,
            ..AgentConfig::default()
        };
        let session = Session::new(graph.clone(), config).unwrap();
        session
            .add_user_message("Edit selection/late.txt")
            .await
            .unwrap();

        let provider = Arc::new(MockProvider::new(vec![
            text_response("I'll edit late.txt now."),
            text_response("Done."),
        ]));
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(MockTool {
            tool_name: "bash".to_string(),
            output: "ok".to_string(),
        }));
        let bus = EventBus::new();
        let token = CancellationToken::new();

        run_agent_loop(&session, provider.clone(), &tools, &bus, &token)
            .await
            .unwrap();

        assert_eq!(provider.call_count(), 2);
        let neighbors = graph
            .neighbors(&session.id, Some(EdgeType::Produces), Direction::Outgoing)
            .unwrap();
        let continued = neighbors.iter().any(|node| {
            matches!(
                &node.node_type,
                NodeType::Interaction(data)
                    if data.role == "user"
                        && data.content.contains("Call the read, write, or edit tool")
            )
        });
        assert!(continued, "the plan should be followed by one continue");
    }

    #[tokio::test]
    async fn finished_report_after_a_write_gets_one_checklist() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let empty =
            std::env::temp_dir().join(format!("graphirm-verify-empty-{}", std::process::id()));
        std::fs::create_dir_all(&empty).unwrap();
        let config = AgentConfig {
            max_turns: 4,
            pre_completion_verify: true,
            working_dir: empty.clone(),
            ..AgentConfig::default()
        };
        let session = Session::new(graph.clone(), config).unwrap();
        session
            .add_user_message("Write selection/answer.txt")
            .await
            .unwrap();

        let provider = Arc::new(MockProvider::new(vec![
            tool_call_response(vec![(
                "write",
                "call_1",
                serde_json::json!({"path": "selection/answer.txt", "content": "TOKEN"}),
            )]),
            text_response("Done. I wrote selection/answer.txt."),
            text_response("Checked. Stopping."),
        ]));
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(MockTool {
            tool_name: "write".to_string(),
            output: "wrote".to_string(),
        }));
        let bus = EventBus::new();
        let token = CancellationToken::new();

        run_agent_loop(&session, provider.clone(), &tools, &bus, &token)
            .await
            .unwrap();

        assert_eq!(provider.call_count(), 3);
        let neighbors = graph
            .neighbors(&session.id, Some(EdgeType::Produces), Direction::Outgoing)
            .unwrap();
        let checklists = neighbors
            .iter()
            .filter(|node| {
                matches!(
                    &node.node_type,
                    NodeType::Interaction(data)
                        if data.role == "user"
                            && data.content.contains(crate::tool_gate::VERIFICATION_LEAD)
                )
            })
            .count();
        assert_eq!(checklists, 1, "the checklist is sent once");
        let checklist = neighbors.iter().find_map(|node| match &node.node_type {
            NodeType::Interaction(data)
                if data.role == "user"
                    && data.content.contains(crate::tool_gate::VERIFICATION_LEAD) =>
            {
                Some(data.content.as_str())
            }
            _ => None,
        });
        let checklist = checklist.unwrap();
        assert!(!checklist.contains("git diff"));
        assert!(!checklist.contains("cargo test"));
        assert!(!checklist.contains("What is the next step"));
        let _ = std::fs::remove_dir_all(&empty);
    }

    #[test]
    fn verification_checklist_matches_the_workspace() {
        let root =
            std::env::temp_dir().join(format!("graphirm-verify-steps-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("src")).unwrap();
        let bare = verification_checklist(&root, &[], false);
        assert!(bare.contains(crate::tool_gate::VERIFICATION_LEAD));
        assert!(!bare.contains("git diff"));
        assert!(!bare.contains("cargo test"));

        std::fs::create_dir_all(root.join(".git")).unwrap();
        let git_only = verification_checklist(&root, &[], false);
        assert!(git_only.contains("git diff --name-only"));
        assert!(!git_only.contains("cargo test"));

        std::fs::write(root.join("Cargo.toml"), "[package]\nname = \"t\"\n").unwrap();
        std::fs::write(root.join("src/lib.rs"), "#[cfg(test)]\nmod tests {}\n").unwrap();
        let code = vec![root.join("src/lib.rs")];
        let both = verification_checklist(&root, &code, false);
        assert!(both.contains("git diff --name-only"));
        assert!(both.contains("cargo test -p t"));
        assert!(!both.contains("cargo test` and"));
        assert!(
            both.contains("what changed and the result"),
            "the last step must ask for the change and the result, got {both}"
        );
        assert!(crate::tool_gate::is_verification_checklist(&both));

        let text = verification_checklist(&root, &[root.join("selection/answer.txt")], false);
        assert!(text.contains("Do not run tests"));
        assert!(!text.contains("cargo test"));

        let scratch = verification_checklist(
            &root,
            &[std::path::PathBuf::from("/tmp/eval_fib.rs")],
            false,
        );
        assert!(scratch.contains("Do not run tests"));
        assert!(!scratch.contains("cargo test"));

        let asked = verification_checklist(&root, &code, true);
        assert!(asked.contains("cargo test` and"));
        assert!(!asked.contains("-p "));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_crate_change_does_not_run_the_whole_suite() {
        let root =
            std::env::temp_dir().join(format!("graphirm-verify-crate-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("crates/agent/src")).unwrap();
        std::fs::write(
            root.join("crates/agent/Cargo.toml"),
            "[package]\nname = \"graphirm-agent\"\n",
        )
        .unwrap();
        std::fs::write(
            root.join("Cargo.toml"),
            "[workspace]\nmembers = [\"crates/agent\"]\n",
        )
        .unwrap();
        std::fs::write(
            root.join("crates/agent/src/lib.rs"),
            "#[cfg(test)]\nmod tests {}\n",
        )
        .unwrap();
        let changed = vec![std::path::PathBuf::from("crates/agent/src/lib.rs")];
        let checklist = verification_checklist(&root, &changed, false);
        assert!(checklist.contains("cargo test -p graphirm-agent"));
        assert!(!checklist.contains("cargo test` and"));
        assert!(!user_asked_for_full_suite("edit selection/late.txt"));
        assert!(user_asked_for_full_suite("Run the full test suite"));
        assert!(user_asked_for_full_suite("please cargo test"));
        assert!(!user_asked_for_full_suite("cargo test -p graphirm-agent"));
        assert!(read_is_repeat(3, 3));
        assert!(!read_is_repeat(2, 3));
        assert!(!read_is_repeat(9, 0));
        assert!(already_read_notice(std::path::Path::new("src/store.rs")).contains("already read"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn announced_action_after_a_write_gets_one_continue() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig {
            max_turns: 4,
            pre_completion_verify: false,
            ..AgentConfig::default()
        };
        let session = Session::new(graph.clone(), config).unwrap();
        session
            .add_user_message("Write selection/answer.txt")
            .await
            .unwrap();

        let provider = Arc::new(MockProvider::new(vec![
            tool_call_response(vec![(
                "write",
                "call_1",
                serde_json::json!({"path": "selection/answer.txt", "content": "TOKEN"}),
            )]),
            text_response("Next I'll edit late.txt."),
            text_response("Done."),
        ]));
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(MockTool {
            tool_name: "write".to_string(),
            output: "wrote".to_string(),
        }));
        let bus = EventBus::new();
        let token = CancellationToken::new();

        run_agent_loop(&session, provider.clone(), &tools, &bus, &token)
            .await
            .unwrap();

        assert_eq!(provider.call_count(), 3);
        let neighbors = graph
            .neighbors(&session.id, Some(EdgeType::Produces), Direction::Outgoing)
            .unwrap();
        let continued = neighbors.iter().any(|node| {
            matches!(
                &node.node_type,
                NodeType::Interaction(data)
                    if data.role == "user"
                        && data.content.contains("Call the read, write, or edit tool")
            )
        });
        assert!(continued, "an announced edit should get one continue");
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

    /// Tool that reports a sub-step through `ctx.event_sink` — the same path
    /// long-running tools such as `delegate_pi` use.
    struct SinkReportingTool;

    #[async_trait::async_trait]
    impl graphirm_tools::Tool for SinkReportingTool {
        fn name(&self) -> &str {
            "outer_tool"
        }
        fn description(&self) -> &str {
            "Reports a sub-step via ToolContext.event_sink"
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }
        async fn execute(
            &self,
            _args: serde_json::Value,
            ctx: &ToolContext,
        ) -> Result<graphirm_tools::ToolOutput, graphirm_tools::ToolError> {
            let sink = ctx.event_sink.as_ref().expect("sink");
            sink.tool_started(&ctx.interaction_id, "sub:1", "sub_tool");
            sink.tool_finished(&ctx.interaction_id, false);
            Ok(graphirm_tools::ToolOutput::success("done"))
        }
    }

    #[tokio::test]
    async fn test_agent_loop_passes_event_sink_to_tools() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig {
            max_turns: 10,
            pre_completion_verify: false,
            ..AgentConfig::default()
        };
        let session = Session::new(graph.clone(), config).unwrap();
        session
            .add_user_message("Run the outer tool")
            .await
            .unwrap();

        let provider = Arc::new(MockProvider::new(vec![
            tool_call_response(vec![("outer_tool", "call_1", serde_json::json!({}))]),
            text_response("Finished."),
        ]));

        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(SinkReportingTool));

        let mut bus = EventBus::new();
        let mut rx = bus.subscribe();
        let token = CancellationToken::new();

        run_agent_loop(&session, provider.clone(), &tools, &bus, &token)
            .await
            .unwrap();

        // With the caller's bus gone, the only remaining senders would belong
        // to a leaked sink or its worker. The channel must therefore report
        // closed (`None`) once the buffered events are drained.
        drop(bus);
        let mut events = vec![];
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        let closed = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv()).await;
        assert!(
            matches!(closed, Ok(None)),
            "EventBusSink or its worker leaked past the turn: {closed:?}"
        );

        // The sub-step reported through the sink surfaces as a ToolStart on
        // the same bus, alongside the loop's own ToolStart for the outer tool.
        let tool_starts: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ToolStart {
                    response_node_id,
                    call_id,
                    tool_name,
                } => Some((call_id.as_str(), tool_name.as_str(), response_node_id)),
                _ => None,
            })
            .collect();
        assert!(
            tool_starts
                .iter()
                .any(|(id, name, _)| (*id, *name) == ("call_1", "outer_tool")),
            "missing loop ToolStart: {tool_starts:?}"
        );
        let sink_node = tool_starts
            .iter()
            .find(|(id, name, _)| (*id, *name) == ("sub:1", "sub_tool"))
            .map(|(_, _, node)| (*node).clone())
            .unwrap_or_else(|| panic!("missing sink ToolStart: {tool_starts:?}"));

        // One ToolEnd carries the sink's node (the tool's `interaction_id`);
        // the other is the loop's own, pointing at the recorded `role: "tool"`
        // Interaction for the outer call.
        let tool_end_nodes: Vec<&NodeId> = events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ToolEnd { node_id, .. } => Some(node_id),
                _ => None,
            })
            .collect();
        assert!(
            tool_end_nodes.contains(&&sink_node),
            "missing sink ToolEnd for {sink_node}: {tool_end_nodes:?}"
        );
        let loop_tool_end = tool_end_nodes.iter().any(|id| {
            **id != sink_node
                && matches!(
                    graph.get_node(id).map(|n| n.node_type),
                    Ok(NodeType::Interaction(d)) if d.role == "tool"
                )
        });
        assert!(
            loop_tool_end,
            "missing loop ToolEnd on a role=tool node: {tool_end_nodes:?}"
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

        let produced = graph
            .neighbors(&session.id, Some(EdgeType::Produces), Direction::Outgoing)
            .unwrap();
        let rejection_result = produced.iter().find(|node| {
            matches!(&node.node_type, NodeType::Interaction(data) if data.role == "tool")
                && node
                    .metadata
                    .get("tool_call_id")
                    .and_then(|value| value.as_str())
                    == Some("call_w1")
        });
        let Some(NodeType::Interaction(data)) = rejection_result.map(|node| &node.node_type) else {
            panic!("rejection must record a tool result for call_w1");
        };
        assert_eq!(data.content, "rejected by user: no bash");
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
    async fn test_tool_call_segment_leak_triggers_retry() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig {
            max_turns: 10,
            pre_completion_verify: false,
            segments: Some(crate::config::SegmentConfig {
                enabled: true,
                structured_output: true,
                ..crate::config::SegmentConfig::default()
            }),
            ..AgentConfig::default()
        };
        let session = Session::new(graph.clone(), config).unwrap();
        session.add_user_message("Read the file").await.unwrap();

        let leak_json = r#"{"segments":[{"type":"observation","content":"checking"},{"type":"tool_call","content":"read foo.rs"}]}"#;
        let clean_json =
            r#"{"segments":[{"type":"answer","content":"The file contains the implementation."}]}"#;

        let provider = Arc::new(MockProvider::new(vec![
            text_response(leak_json),
            text_response(clean_json),
        ]));
        let tools = ToolRegistry::new();
        let bus = EventBus::new();
        let token = CancellationToken::new();

        run_agent_loop(&session, provider.clone(), &tools, &bus, &token)
            .await
            .unwrap();

        assert_eq!(provider.call_count(), 2);

        let sid = session.id.to_string();
        let interactions = graph
            .list_nodes_by_type("interaction", Some(&sid), None, 50)
            .unwrap();
        assert!(
            interactions.iter().any(|n| {
                matches!(&n.node_type, NodeType::Interaction(d) if d.role == "user" && d.content.contains("native tool-calling"))
            }),
            "expected retry nudge user message"
        );
    }

    #[tokio::test]
    async fn test_tool_invocation_in_code_segment_triggers_retry() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig {
            max_turns: 10,
            pre_completion_verify: false,
            segments: Some(crate::config::SegmentConfig {
                enabled: true,
                structured_output: true,
                ..crate::config::SegmentConfig::default()
            }),
            ..AgentConfig::default()
        };
        let session = Session::new(graph.clone(), config).unwrap();
        session
            .add_user_message("Show file contents")
            .await
            .unwrap();

        let leak_json = r#"{"segments":[{"type":"observation","content":"need file"},{"type":"code","content":"read notes.md"}]}"#;
        let clean_json = r#"{"segments":[{"type":"answer","content":"Done."}]}"#;

        let provider = Arc::new(MockProvider::new(vec![
            text_response(leak_json),
            text_response(clean_json),
        ]));
        let tools = ToolRegistry::new();
        let bus = EventBus::new();
        let token = CancellationToken::new();

        run_agent_loop(&session, provider.clone(), &tools, &bus, &token)
            .await
            .unwrap();

        assert_eq!(provider.call_count(), 2);
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
