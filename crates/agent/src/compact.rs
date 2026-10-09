// Context compaction: summarize old context, prune graph branches

use graphirm_graph::{EdgeType, GraphEdge, GraphNode, GraphStore, KnowledgeData, NodeId, NodeType};
use graphirm_llm::{CompletionConfig, LlmMessage, LlmProvider};

use crate::context::{estimate_tokens, estimate_tokens_str, get_text_content};
use crate::error::AgentError;

#[cfg(test)]
use chrono::Duration;

#[derive(Debug, Clone)]
pub struct CompactionConfig {
    pub model: String,
    pub max_summary_tokens: usize,
    pub min_nodes_to_compact: usize,
}

/// Session model first, then the first non-empty cheap-tier model.
/// `None` means compaction has no model and must not call the provider.
pub fn resolve_compaction_model(session_model: &str, cheap_models: &[String]) -> Option<String> {
    let session_model = session_model.trim();
    if !session_model.is_empty() {
        return Some(session_model.to_string());
    }
    cheap_models
        .iter()
        .map(|model| model.trim())
        .find(|model| !model.is_empty())
        .map(str::to_string)
}

/// `off` when compaction is disabled, `ready` when a model resolves, `unconfigured` otherwise.
pub fn compaction_status(
    enabled: bool,
    session_model: &str,
    cheap_models: &[String],
) -> &'static str {
    if !enabled {
        "off"
    } else if resolve_compaction_model(session_model, cheap_models).is_some() {
        "ready"
    } else {
        "unconfigured"
    }
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            // Empty model means "not configured" — callers must set this
            // explicitly. Using "mock" here would cause silent mock behaviour
            // in production if the default is accidentally used.
            model: String::new(),
            max_summary_tokens: 500,
            min_nodes_to_compact: 3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompactionResult {
    pub summary_node_id: NodeId,
    pub compacted_node_ids: Vec<NodeId>,
    pub tokens_saved: usize,
}

/// System and user text for one compaction call.
/// Identifiers, paths, values, decisions, and unfinished instructions come first, copied as written.
pub fn compaction_prompt(max_summary_tokens: usize, transcript: &str) -> (String, String) {
    let system = "You are a concise summarizer. Copy identifiers, file paths, values, \
         decisions, and unfinished instructions word for word, and put those ahead of any narrative."
        .to_string();
    let human = format!(
        "Copy identifiers, file paths, values, decisions, and unfinished instructions \
         word for word, and put those ahead of any narrative. \
         Keep it under {max_summary_tokens} tokens.\n\n{transcript}"
    );
    (system, human)
}

/// Compact old context nodes by summarizing them via an LLM call.
///
/// Steps:
/// 1. Collect the text content of all nodes to compact
/// 2. Build a summarization prompt
/// 3. Call LLM with complete()
/// 4. Create a Knowledge node with the summary
/// 5. Add Summarizes edges from the Knowledge node to each compacted node
/// 6. Mark original nodes as compacted (metadata["compacted"] = true)
pub async fn compact_context(
    graph: &GraphStore,
    llm: &dyn LlmProvider,
    nodes_to_compact: Vec<NodeId>,
    config: &CompactionConfig,
) -> Result<CompactionResult, AgentError> {
    if config.model.trim().is_empty() {
        return Err(AgentError::Context(
            "compaction model is empty; set the session model or a cheap routing tier".into(),
        ));
    }

    if nodes_to_compact.len() < config.min_nodes_to_compact {
        return Err(AgentError::Context(format!(
            "Need at least {} nodes to compact, got {}",
            config.min_nodes_to_compact,
            nodes_to_compact.len()
        )));
    }

    // Collect text from nodes
    let mut texts = Vec::new();
    let mut original_tokens = 0_usize;

    for node_id in &nodes_to_compact {
        let node = graph
            .get_node(node_id)
            .map_err(|e| AgentError::Context(e.to_string()))?;
        original_tokens += estimate_tokens(&node);
        let content = get_text_content(&node);
        if !content.is_empty() {
            texts.push(content.to_string());
        }
    }

    if let Some(prior) = prior_summary_text(graph, &nodes_to_compact)? {
        texts.insert(0, format!("Previous summary:\n{prior}"));
    }

    let combined = texts.join("\n---\n");
    let (system, prompt) = compaction_prompt(config.max_summary_tokens, &combined);

    let messages = vec![LlmMessage::system(system), LlmMessage::human(prompt)];

    let completion_config =
        CompletionConfig::new(&config.model).with_max_tokens(config.max_summary_tokens as u32);

    let response = llm
        .complete(messages, &[], &completion_config)
        .await
        .map_err(|e| AgentError::Context(format!("Compaction LLM call failed: {e}")))?;

    let summary_text = response.text_content();
    let summary_tokens = estimate_tokens_str(&summary_text);
    tracing::info!(
        summary_tokens,
        summary = %summary_text,
        "compaction summary"
    );

    // Create Knowledge node with summary
    let summary_node = GraphNode::new(NodeType::Knowledge(KnowledgeData {
        entity: "session_summary".to_string(),
        entity_type: "compaction".to_string(),
        summary: summary_text,
        confidence: 1.0,
    }));
    let summary_node_id = summary_node.id.clone();
    graph
        .add_node(summary_node)
        .map_err(|e| AgentError::Context(e.to_string()))?;

    // Add Summarizes edges from summary to each compacted node
    for node_id in &nodes_to_compact {
        graph
            .add_edge(GraphEdge::new(
                EdgeType::Summarizes,
                summary_node_id.clone(),
                node_id.clone(),
            ))
            .map_err(|e| AgentError::Context(e.to_string()))?;
    }

    // Mark original nodes as compacted
    for node_id in &nodes_to_compact {
        let mut node = graph
            .get_node(node_id)
            .map_err(|e| AgentError::Context(e.to_string()))?;

        // Ensure metadata is always a JSON object before inserting the flag.
        // Nodes loaded from DB with corrupted/null metadata would silently skip
        // the flag otherwise.
        if !node.metadata.is_object() {
            node.metadata = serde_json::Value::Object(serde_json::Map::new());
        }
        node.metadata
            .as_object_mut()
            .expect("just initialized as object")
            .insert("compacted".to_string(), serde_json::Value::Bool(true));

        graph
            .update_node(node_id, node)
            .map_err(|e| AgentError::Context(e.to_string()))?;
    }

    let tokens_saved = original_tokens.saturating_sub(summary_tokens);

    Ok(CompactionResult {
        summary_node_id,
        compacted_node_ids: nodes_to_compact,
        tokens_saved,
    })
}

/// The user message that opened the task, plus the latest user message when
/// a later one exists. Both stay out of compaction. A later user message is
/// the current request; the first one is the instruction that started the work.
pub fn pinned_task_nodes<'a>(
    nodes_oldest_first: impl IntoIterator<Item = &'a GraphNode>,
) -> Vec<GraphNode> {
    let mut oldest: Option<&GraphNode> = None;
    let mut newest: Option<&GraphNode> = None;
    for node in nodes_oldest_first {
        if !is_user_message(node) {
            continue;
        }
        if oldest.is_none() {
            oldest = Some(node);
        }
        newest = Some(node);
    }
    match (oldest, newest) {
        (Some(first), Some(last)) if first.id != last.id => vec![first.clone(), last.clone()],
        (Some(first), _) => vec![first.clone()],
        _ => Vec::new(),
    }
}

fn is_user_message(node: &GraphNode) -> bool {
    matches!(&node.node_type, NodeType::Interaction(data) if data.role == "user")
}

/// Newest compaction summary already stored on this thread, so the next
/// summary can carry its identifiers forward instead of replacing them.
fn prior_summary_text(graph: &GraphStore, seeds: &[NodeId]) -> Result<Option<String>, AgentError> {
    use crate::context::latest_compaction_summary;

    let mut latest: Option<GraphNode> = None;
    for seed in seeds {
        let thread = graph
            .conversation_thread(seed)
            .map_err(|e| AgentError::Context(e.to_string()))?;
        let Some(found) = latest_compaction_summary(graph, &thread)? else {
            continue;
        };
        let newer = latest
            .as_ref()
            .is_none_or(|current| found.created_at > current.created_at);
        if newer {
            latest = Some(found);
        }
    }
    Ok(latest
        .as_ref()
        .map(get_text_content)
        .filter(|text| !text.is_empty())
        .map(str::to_string))
}

/// Check if a node has been compacted (excluded from future context builds).
pub fn is_compacted(node: &GraphNode) -> bool {
    node.metadata
        .get("compacted")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// Select nodes for compaction when context exceeds `threshold_ratio` of `max_tokens`.
///
/// `guaranteed_recent_turns` counts context units, the same way `build_context`
/// does. A tool exchange is one unit: it is compacted entirely or not at all.
/// `tail_max_fraction` is the same cap the context tail uses. Returns empty
/// when compaction is not needed.
pub fn select_nodes_for_compaction(
    graph: &GraphStore,
    agent_id: &NodeId,
    max_tokens: usize,
    threshold_ratio: f64,
    guaranteed_recent_turns: usize,
    min_nodes_to_compact: usize,
    tail_max_fraction: f64,
) -> Result<Vec<NodeId>, AgentError> {
    use crate::context::{
        estimate_tokens, find_current_turn, group_interaction_units, tail_unit_indexes,
    };
    let current_turn = match find_current_turn(graph, agent_id)? {
        Some(n) => n,
        None => return Ok(vec![]),
    };
    let thread = graph
        .conversation_thread(&current_turn.id)
        .map_err(AgentError::Graph)?;
    let mut candidates: Vec<graphirm_graph::GraphNode> = thread
        .into_iter()
        .filter(|node| !is_compacted(node))
        .collect();
    let total_tokens: usize = candidates.iter().map(estimate_tokens).sum();
    let threshold = (max_tokens as f64 * threshold_ratio) as usize;
    if total_tokens < threshold {
        return Ok(vec![]);
    }
    candidates.reverse();
    let pinned_ids: std::collections::HashSet<NodeId> = pinned_task_nodes(candidates.iter())
        .into_iter()
        .map(|node| node.id)
        .collect();
    let units = group_interaction_units(&candidates);
    let in_tail = tail_unit_indexes(
        &units,
        guaranteed_recent_turns,
        max_tokens,
        tail_max_fraction,
    );
    let eligible: Vec<NodeId> = units
        .iter()
        .enumerate()
        .filter(|(index, _)| !in_tail[*index])
        .flat_map(|(_, unit)| unit.iter().map(|node| node.id.clone()))
        .filter(|id| !pinned_ids.contains(id))
        .collect();
    if eligible.len() < min_nodes_to_compact {
        return Ok(vec![]);
    }
    Ok(eligible)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use graphirm_graph::{AgentData, GraphEdge, GraphNode, InteractionData, NodeType};
    use graphirm_llm::MockProvider;

    #[test]
    fn compaction_instruction_copies_facts_ahead_of_narrative() {
        let (system, human) = compaction_prompt(500, "early TOKEN_EARLY_184729");
        assert!(system.contains("word for word"), "{system}");
        assert!(human.contains("identifiers"), "{human}");
        assert!(human.contains("file paths"), "{human}");
        assert!(human.contains("values"), "{human}");
        assert!(human.contains("decisions"), "{human}");
        assert!(human.contains("unfinished instructions"), "{human}");
        assert!(human.contains("ahead of any narrative"), "{human}");
        assert!(human.contains("Keep it under 500 tokens"), "{human}");
        assert!(human.contains("TOKEN_EARLY_184729"), "{human}");
    }

    #[tokio::test]
    async fn next_context_keeps_the_latest_summary_and_the_token() {
        use crate::context::{ContextConfig, build_context};
        use chrono::{Duration, Utc};
        use graphirm_llm::ContentPart;

        let graph = GraphStore::open_memory().unwrap();
        let agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "test".to_string(),
            model: "mock".to_string(),
            system_prompt: None,
            status: "running".to_string(),
        }));
        let agent_id = agent.id.clone();
        graph.add_node(agent).unwrap();

        let contents = [
            "early file token TOKEN_EARLY_184729 lives in selection/early.txt",
            "assistant saw the early file",
            "read selection/late.txt LABEL=unset",
            "kept turn edit selection/late.txt",
        ];
        let mut prev: Option<NodeId> = None;
        let mut ids = Vec::new();
        for (index, content) in contents.iter().enumerate() {
            let mut node = GraphNode::new(NodeType::Interaction(InteractionData {
                role: if index % 2 == 0 { "user" } else { "assistant" }.to_string(),
                content: (*content).to_string(),
                token_count: None,
            }));
            node.created_at = Utc::now() - Duration::minutes(10 - index as i64);
            node.updated_at = node.created_at;
            let id = node.id.clone();
            graph.add_node(node).unwrap();
            graph
                .add_edge(GraphEdge::new(
                    EdgeType::Produces,
                    agent_id.clone(),
                    id.clone(),
                ))
                .unwrap();
            if let Some(parent) = &prev {
                graph
                    .add_edge(GraphEdge::new(
                        EdgeType::RespondsTo,
                        id.clone(),
                        parent.clone(),
                    ))
                    .unwrap();
            }
            prev = Some(id.clone());
            ids.push(id);
        }

        let mut stale = GraphNode::new(NodeType::Knowledge(KnowledgeData {
            entity: "session_summary".to_string(),
            entity_type: "compaction".to_string(),
            summary: "STALE_SUMMARY".to_string(),
            confidence: 1.0,
        }));
        stale.created_at = Utc::now() - Duration::hours(2);
        stale.updated_at = stale.created_at;
        let stale_id = stale.id.clone();
        graph.add_node(stale).unwrap();
        graph
            .add_edge(GraphEdge::new(
                EdgeType::Summarizes,
                stale_id,
                ids[0].clone(),
            ))
            .unwrap();

        let llm = MockProvider::fixed("PINNED_SUMMARY TOKEN_EARLY_184729 selection/early.txt");
        compact_context(
            &graph,
            &llm,
            ids[..3].to_vec(),
            &CompactionConfig {
                model: "mock".to_string(),
                max_summary_tokens: 500,
                min_nodes_to_compact: 3,
            },
        )
        .await
        .unwrap();

        let window = build_context(
            &graph,
            &agent_id,
            &ContextConfig {
                max_tokens: 8,
                system_prompt: "S".to_string(),
                guaranteed_recent_turns: 1,
                ..ContextConfig::default()
            },
        )
        .unwrap();
        let all_text: String = window
            .messages
            .iter()
            .flat_map(|message| message.content.iter())
            .filter_map(|part| match part {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ");

        assert!(
            all_text.contains("PINNED_SUMMARY TOKEN_EARLY_184729"),
            "{all_text}"
        );
        assert!(!all_text.contains("early file token"), "{all_text}");
        assert!(!all_text.contains("STALE_SUMMARY"), "{all_text}");
        assert!(
            all_text.find("PINNED_SUMMARY").unwrap() < all_text.find("kept turn").unwrap(),
            "{all_text}"
        );
    }

    /// The second summary is whatever the model returns. This provider copies
    /// the early token into that return only when the prompt still contains it,
    /// which happens only if the previous summary was fed back in.
    struct CarryForward;

    #[async_trait::async_trait]
    impl LlmProvider for CarryForward {
        async fn complete(
            &self,
            messages: Vec<LlmMessage>,
            _tools: &[graphirm_llm::ToolDefinition],
            _config: &CompletionConfig,
        ) -> Result<graphirm_llm::LlmResponse, graphirm_llm::LlmError> {
            let blob = messages
                .iter()
                .flat_map(|message| message.content.iter())
                .filter_map(|part| match part {
                    graphirm_llm::ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            let text = if blob.contains("TOKEN_FIRST_991") {
                "second summary TOKEN_FIRST_991"
            } else {
                "second summary lost the token"
            };
            Ok(graphirm_llm::LlmResponse {
                content: vec![graphirm_llm::ContentPart::text(text)],
                usage: graphirm_llm::TokenUsage::default(),
                stop_reason: graphirm_llm::StopReason::EndTurn,
            })
        }

        async fn stream(
            &self,
            _messages: Vec<LlmMessage>,
            _tools: &[graphirm_llm::ToolDefinition],
            _config: &CompletionConfig,
        ) -> Result<
            std::pin::Pin<Box<dyn futures::Stream<Item = graphirm_llm::StreamEvent> + Send>>,
            graphirm_llm::LlmError,
        > {
            Err(graphirm_llm::LlmError::stream("unused"))
        }

        fn provider_name(&self) -> &str {
            "carry-forward"
        }
    }

    #[tokio::test]
    async fn second_compaction_keeps_the_first_token() {
        let graph = GraphStore::open_memory().unwrap();
        let mut prev: Option<NodeId> = None;
        let mut ids = Vec::new();
        let contents = [
            "The early token is TOKEN_FIRST_991",
            "ack",
            "middle note",
            "ack",
            "later note",
            "ack",
        ];
        for (index, content) in contents.iter().enumerate() {
            let mut node = GraphNode::new(NodeType::Interaction(InteractionData {
                role: if index % 2 == 0 { "user" } else { "assistant" }.to_string(),
                content: (*content).to_string(),
                token_count: None,
            }));
            node.created_at = Utc::now() - Duration::minutes(20 - index as i64);
            node.updated_at = node.created_at;
            let id = node.id.clone();
            graph.add_node(node).unwrap();
            if let Some(parent) = &prev {
                graph
                    .add_edge(GraphEdge::new(
                        EdgeType::RespondsTo,
                        id.clone(),
                        parent.clone(),
                    ))
                    .unwrap();
            }
            prev = Some(id.clone());
            ids.push(id);
        }

        let config = CompactionConfig {
            model: "mock".to_string(),
            max_summary_tokens: 500,
            min_nodes_to_compact: 3,
        };
        compact_context(
            &graph,
            &MockProvider::fixed("first summary TOKEN_FIRST_991"),
            ids[..3].to_vec(),
            &config,
        )
        .await
        .unwrap();

        let second = compact_context(&graph, &CarryForward, ids[3..].to_vec(), &config)
            .await
            .unwrap();
        let summary = graph.get_node(&second.summary_node_id).unwrap();
        let NodeType::Knowledge(data) = &summary.node_type else {
            panic!("expected a knowledge summary");
        };
        assert!(data.summary.contains("TOKEN_FIRST_991"), "{}", data.summary);
    }

    #[tokio::test]
    async fn task_instruction_stays_in_the_payload() {
        use crate::context::{ContextConfig, build_context};

        let graph = GraphStore::open_memory().unwrap();
        let agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "test".to_string(),
            model: "mock".to_string(),
            system_prompt: None,
            status: "running".to_string(),
        }));
        let agent_id = agent.id.clone();
        graph.add_node(agent).unwrap();

        let task = "Write selection/answer.txt now. The file must contain only the token.";
        let filler = "pad ".repeat(80);
        let contents = [
            task,
            "ack",
            filler.as_str(),
            "ack",
            filler.as_str(),
            "newest turn stays",
        ];
        let mut prev: Option<NodeId> = None;
        let mut ids = Vec::new();
        for (index, content) in contents.iter().enumerate() {
            let mut node = GraphNode::new(NodeType::Interaction(InteractionData {
                role: if index % 2 == 0 { "user" } else { "assistant" }.to_string(),
                content: (*content).to_string(),
                token_count: None,
            }));
            node.created_at = Utc::now() - Duration::minutes(20 - index as i64);
            node.updated_at = node.created_at;
            let id = node.id.clone();
            graph.add_node(node).unwrap();
            graph
                .add_edge(GraphEdge::new(
                    EdgeType::Produces,
                    agent_id.clone(),
                    id.clone(),
                ))
                .unwrap();
            if let Some(parent) = &prev {
                graph
                    .add_edge(GraphEdge::new(
                        EdgeType::RespondsTo,
                        id.clone(),
                        parent.clone(),
                    ))
                    .unwrap();
            }
            prev = Some(id.clone());
            ids.push(id);
        }

        let selected = select_nodes_for_compaction(&graph, &agent_id, 200, 0.5, 1, 3, 0.5).unwrap();
        assert!(
            !selected.contains(&ids[0]),
            "the task message must not be compacted: {selected:?}"
        );
        compact_context(
            &graph,
            &MockProvider::fixed("TOKEN_EARLY_184729"),
            selected,
            &CompactionConfig {
                model: "mock".to_string(),
                max_summary_tokens: 500,
                min_nodes_to_compact: 3,
            },
        )
        .await
        .unwrap();

        let task_node = graph.get_node(&ids[0]).unwrap();
        assert!(!is_compacted(&task_node));

        let window = build_context(
            &graph,
            &agent_id,
            &ContextConfig {
                max_tokens: 20,
                system_prompt: "S".to_string(),
                guaranteed_recent_turns: 1,
                ..ContextConfig::default()
            },
        )
        .unwrap();
        let all_text: String = window
            .messages
            .iter()
            .flat_map(|message| message.content.iter())
            .filter_map(|part| match part {
                graphirm_llm::ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            all_text.contains("Write selection/answer.txt now"),
            "{all_text}"
        );
    }

    #[test]
    fn resolve_compaction_model_prefers_the_session_model() {
        assert_eq!(
            resolve_compaction_model("openrouter/deepseek/deepseek-v4-flash", &[]).as_deref(),
            Some("openrouter/deepseek/deepseek-v4-flash")
        );
        assert_eq!(
            resolve_compaction_model("", &["openrouter/deepseek/deepseek-v4-flash".to_string()])
                .as_deref(),
            Some("openrouter/deepseek/deepseek-v4-flash")
        );
        assert_eq!(resolve_compaction_model("  ", &[String::new()]), None);
    }

    #[test]
    fn compaction_config_defaults() {
        let config = CompactionConfig::default();
        assert_eq!(config.max_summary_tokens, 500);
        assert_eq!(config.min_nodes_to_compact, 3);
    }

    #[tokio::test]
    async fn compact_context_creates_knowledge_node() {
        let graph = GraphStore::open_memory().unwrap();

        let mut node_ids = Vec::new();
        for i in 0..5 {
            let node = GraphNode::new(NodeType::Interaction(InteractionData {
                role: if i % 2 == 0 { "user" } else { "assistant" }.to_string(),
                content: format!("Message {i} with some discussion about the project."),
                token_count: None,
            }));
            let id = node.id.clone();
            graph.add_node(node).unwrap();
            node_ids.push(id);
        }

        let llm = MockProvider::fixed(
            "Summary: 5 messages discussing project. Key points: \
             code review feedback, main.rs changes, test additions.",
        );

        let config = CompactionConfig {
            model: "mock".to_string(),
            max_summary_tokens: 100,
            min_nodes_to_compact: 3,
        };

        let result = compact_context(&graph, &llm, node_ids.clone(), &config)
            .await
            .unwrap();

        let summary_node = graph.get_node(&result.summary_node_id).unwrap();
        match &summary_node.node_type {
            NodeType::Knowledge(data) => {
                assert_eq!(data.entity_type, "compaction");
                assert!(data.summary.contains("Summary"));
            }
            other => panic!("Expected Knowledge node, got {:?}", other),
        }

        let summarized = graph
            .neighbors(
                &result.summary_node_id,
                Some(EdgeType::Summarizes),
                graphirm_graph::Direction::Outgoing,
            )
            .unwrap();
        assert_eq!(summarized.len(), 5);

        assert_eq!(result.compacted_node_ids.len(), 5);

        for id in &node_ids {
            let node = graph.get_node(id).unwrap();
            assert!(is_compacted(&node), "Node {id} should be marked compacted");
        }

        assert!(result.tokens_saved > 0, "Should save tokens");
    }

    #[tokio::test]
    async fn compact_context_rejects_too_few_nodes() {
        let graph = GraphStore::open_memory().unwrap();

        let node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "solo".to_string(),
            token_count: None,
        }));
        let id = node.id.clone();
        graph.add_node(node).unwrap();

        let llm = MockProvider::fixed("summary");
        let config = CompactionConfig {
            model: "mock".to_string(),
            ..CompactionConfig::default()
        };

        let result = compact_context(&graph, &llm, vec![id], &config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn compact_context_rejects_an_empty_model() {
        let graph = GraphStore::open_memory().unwrap();
        let mut ids = Vec::new();
        for i in 0..3 {
            let node = GraphNode::new(NodeType::Interaction(InteractionData {
                role: "user".to_string(),
                content: format!("message {i}"),
                token_count: None,
            }));
            ids.push(node.id.clone());
            graph.add_node(node).unwrap();
        }
        let llm = MockProvider::fixed("summary");
        let err = compact_context(&graph, &llm, ids, &CompactionConfig::default())
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("compaction model is empty"),
            "{err}"
        );
    }

    #[test]
    fn is_compacted_false_by_default() {
        let node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "normal".to_string(),
            token_count: None,
        }));
        assert!(!is_compacted(&node));
    }

    #[test]
    fn is_compacted_true_when_marked() {
        let mut node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "compacted".to_string(),
            token_count: None,
        }));
        node.metadata = serde_json::json!({"compacted": true});
        assert!(is_compacted(&node));
    }

    #[test]
    fn select_nodes_below_threshold_returns_empty() {
        let graph = GraphStore::open_memory().unwrap();

        let agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "test-agent".to_string(),
            model: "mock".to_string(),
            system_prompt: Some("You are helpful.".to_string()),
            status: "running".to_string(),
        }));
        let agent_id = agent.id.clone();
        graph.add_node(agent).unwrap();

        // Create 5 short messages (each ~2 words = ~3 tokens)
        let mut prev_id: Option<NodeId> = None;
        for i in 0..5 {
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            let mut node = GraphNode::new(NodeType::Interaction(InteractionData {
                role: role.to_string(),
                content: format!("Message {i}"), // 2 words
                token_count: None,
            }));
            node.created_at = Utc::now() - Duration::hours(5 - i as i64);
            node.updated_at = node.created_at;
            let node_id = node.id.clone();
            graph.add_node(node).unwrap();
            graph
                .add_edge(GraphEdge::new(
                    EdgeType::Produces,
                    agent_id.clone(),
                    node_id.clone(),
                ))
                .unwrap();
            if let Some(pid) = &prev_id {
                graph
                    .add_edge(GraphEdge::new(
                        EdgeType::RespondsTo,
                        node_id.clone(),
                        pid.clone(),
                    ))
                    .unwrap();
            }
            prev_id = Some(node_id);
        }

        // High max_tokens with 0.80 threshold = 102400 tokens, but total is only ~15 tokens
        let nodes = select_nodes_for_compaction(
            &graph, &agent_id, 128_000, // max_tokens
            0.80,    // threshold_ratio
            2,       // guaranteed_recent_turns
            2,       // min_nodes_to_compact
            0.5,
        )
        .unwrap();
        assert!(nodes.is_empty(), "Should return empty when below threshold");
    }

    #[test]
    fn select_nodes_above_threshold_returns_oldest() {
        let graph = GraphStore::open_memory().unwrap();

        let agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "test-agent".to_string(),
            model: "mock".to_string(),
            system_prompt: Some("You are helpful.".to_string()),
            status: "running".to_string(),
        }));
        let agent_id = agent.id.clone();
        graph.add_node(agent).unwrap();

        // Create 10 messages with more tokens each (10 words each = ~14 tokens)
        let mut prev_id: Option<NodeId> = None;
        for i in 0..10 {
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            let content = if i % 2 == 0 {
                format!("Message {i} about Rust programming language and development")
            } else {
                format!("Response {i} to user query about Rust programming language")
            };
            // 10 words each → ~14 tokens each
            let mut node = GraphNode::new(NodeType::Interaction(InteractionData {
                role: role.to_string(),
                content,
                token_count: None,
            }));
            node.created_at = Utc::now() - Duration::hours(10 - i as i64);
            node.updated_at = node.created_at;
            let node_id = node.id.clone();
            graph.add_node(node).unwrap();
            graph
                .add_edge(GraphEdge::new(
                    EdgeType::Produces,
                    agent_id.clone(),
                    node_id.clone(),
                ))
                .unwrap();
            if let Some(pid) = &prev_id {
                graph
                    .add_edge(GraphEdge::new(
                        EdgeType::RespondsTo,
                        node_id.clone(),
                        pid.clone(),
                    ))
                    .unwrap();
            }
            prev_id = Some(node_id);
        }

        // Low max_tokens with 0.80 threshold = 80 tokens
        // 10 nodes × ~14 tokens = ~140 tokens > threshold → should select oldest
        let nodes = select_nodes_for_compaction(
            &graph, &agent_id, 100,  // max_tokens (low to trigger compaction, threshold = 80)
            0.80, // threshold_ratio
            2,    // guaranteed_recent_turns
            2,    // min_nodes_to_compact
            0.5,
        )
        .unwrap();
        // Should select oldest 6 nodes (10 - 2 guaranteed = 8 eligible, but only need 2+)
        assert!(
            !nodes.is_empty(),
            "Should return nodes when above threshold"
        );
        assert!(
            nodes.len() >= 2,
            "Should return at least min_nodes_to_compact"
        );
    }

    #[test]
    fn select_nodes_skips_already_compacted() {
        let graph = GraphStore::open_memory().unwrap();

        let agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "test-agent".to_string(),
            model: "mock".to_string(),
            system_prompt: Some("You are helpful.".to_string()),
            status: "running".to_string(),
        }));
        let agent_id = agent.id.clone();
        graph.add_node(agent).unwrap();

        // Create 10 messages
        let mut prev_id: Option<NodeId> = None;
        for i in 0..10 {
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            let mut node = GraphNode::new(NodeType::Interaction(InteractionData {
                role: role.to_string(),
                content: format!("Message {i} about Rust programming language and development"),
                token_count: None,
            }));
            node.created_at = Utc::now() - Duration::hours(10 - i as i64);
            node.updated_at = node.created_at;
            if i < 3 {
                node.metadata = serde_json::json!({"compacted": true});
            }
            let node_id = node.id.clone();
            graph.add_node(node).unwrap();
            graph
                .add_edge(GraphEdge::new(
                    EdgeType::Produces,
                    agent_id.clone(),
                    node_id.clone(),
                ))
                .unwrap();
            if let Some(pid) = &prev_id {
                graph
                    .add_edge(GraphEdge::new(
                        EdgeType::RespondsTo,
                        node_id.clone(),
                        pid.clone(),
                    ))
                    .unwrap();
            }
            prev_id = Some(node_id);
        }

        let nodes = select_nodes_for_compaction(
            &graph, &agent_id, 200,  // max_tokens
            0.80, // threshold_ratio
            2,    // guaranteed_recent_turns
            2,    // min_nodes_to_compact
            0.5,
        )
        .unwrap();
        // Should NOT include compacted nodes (0, 1, 2)
        for node_id in &nodes {
            let node = graph.get_node(node_id).unwrap();
            assert!(
                !is_compacted(&node),
                "Compacted node {node_id} should not be selected"
            );
        }
    }

    #[test]
    fn select_nodes_respects_min_nodes() {
        let graph = GraphStore::open_memory().unwrap();

        let agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "test-agent".to_string(),
            model: "mock".to_string(),
            system_prompt: Some("You are helpful.".to_string()),
            status: "running".to_string(),
        }));
        let agent_id = agent.id.clone();
        graph.add_node(agent).unwrap();

        // Create 5 messages
        let mut prev_id: Option<NodeId> = None;
        for i in 0..5 {
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            let mut node = GraphNode::new(NodeType::Interaction(InteractionData {
                role: role.to_string(),
                content: format!("Message {i} about Rust programming language and development"),
                token_count: None,
            }));
            node.created_at = Utc::now() - Duration::hours(5 - i as i64);
            node.updated_at = node.created_at;
            let node_id = node.id.clone();
            graph.add_node(node).unwrap();
            graph
                .add_edge(GraphEdge::new(
                    EdgeType::Produces,
                    agent_id.clone(),
                    node_id.clone(),
                ))
                .unwrap();
            if let Some(pid) = &prev_id {
                graph
                    .add_edge(GraphEdge::new(
                        EdgeType::RespondsTo,
                        node_id.clone(),
                        pid.clone(),
                    ))
                    .unwrap();
            }
            prev_id = Some(node_id);
        }

        // Set min_nodes_to_compact to 4, but with guaranteed_recent_turns=3,
        // only 2 nodes are eligible (5 - 3 = 2), so should return empty
        let nodes = select_nodes_for_compaction(
            &graph, &agent_id, 200,  // max_tokens
            0.80, // threshold_ratio
            3,    // guaranteed_recent_turns (leaves only 2 eligible)
            4,    // min_nodes_to_compact (more than eligible)
            0.5,
        )
        .unwrap();
        assert!(
            nodes.is_empty(),
            "Should return empty when too few eligible nodes"
        );
    }

    /// T7. A tool exchange that straddles the old node tail is compacted whole
    /// or not at all.
    #[test]
    fn t7_compaction_keeps_a_tool_exchange_whole() {
        let graph = GraphStore::open_memory().unwrap();
        let agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "test-agent".to_string(),
            model: "mock".to_string(),
            system_prompt: Some("You are helpful.".to_string()),
            status: "running".to_string(),
        }));
        let agent_id = agent.id.clone();
        graph.add_node(agent).unwrap();

        let mut prev_id: Option<NodeId> = None;
        let mut link = |role: &str, content: String, metadata: serde_json::Value| -> NodeId {
            let mut node = GraphNode::new(NodeType::Interaction(InteractionData {
                role: role.to_string(),
                content,
                token_count: None,
            }));
            node.metadata = metadata;
            node.created_at = Utc::now() - Duration::minutes(20);
            node.updated_at = node.created_at;
            let node_id = node.id.clone();
            graph.add_node(node).unwrap();
            graph
                .add_edge(GraphEdge::new(
                    EdgeType::Produces,
                    agent_id.clone(),
                    node_id.clone(),
                ))
                .unwrap();
            if let Some(pid) = &prev_id {
                graph
                    .add_edge(GraphEdge::new(
                        EdgeType::RespondsTo,
                        node_id.clone(),
                        pid.clone(),
                    ))
                    .unwrap();
            }
            prev_id = Some(node_id.clone());
            node_id
        };

        for i in 0..4 {
            link(
                "user",
                format!("older-{i} {}", "word ".repeat(20)),
                serde_json::json!({}),
            );
        }
        let call_id = link(
            "assistant",
            "calling".to_string(),
            serde_json::json!({"tool_calls": [{"id": "call_1", "name": "read", "arguments": {}}]}),
        );
        let result_id = link(
            "tool",
            "result body".to_string(),
            serde_json::json!({"tool_call_id": "call_1"}),
        );

        let nodes = select_nodes_for_compaction(
            &graph, &agent_id, 100, // max_tokens, threshold 80
            0.80, 1, // one recent unit: the exchange, not the result alone
            2, 0.5,
        )
        .unwrap();
        assert!(
            !nodes.is_empty(),
            "older messages are over the threshold and should compact"
        );
        let call_in = nodes.iter().any(|id| id == &call_id);
        let result_in = nodes.iter().any(|id| id == &result_id);
        assert_eq!(
            call_in, result_in,
            "tool exchange must be compacted whole or not at all"
        );
    }
}
