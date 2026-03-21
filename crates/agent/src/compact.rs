// Context compaction: summarize old context, prune graph branches

use graphirm_graph::{EdgeType, GraphEdge, GraphNode, GraphStore, KnowledgeData, NodeId, NodeType};
use graphirm_llm::{CompletionConfig, LlmMessage, LlmProvider};

use crate::context::{estimate_tokens, estimate_tokens_str, get_text_content};
use crate::error::AgentError;

#[cfg(test)]
use chrono::Duration;
#[cfg(test)]
use graphirm_graph::{AgentData, InteractionData};

#[derive(Debug, Clone)]
pub struct CompactionConfig {
    pub model: String,
    pub max_summary_tokens: usize,
    pub min_nodes_to_compact: usize,
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

    // Build summarization prompt
    let combined = texts.join("\n---\n");
    let prompt = format!(
        "Summarize the following conversation context into a concise summary \
         that preserves key information, decisions, and file changes. \
         Keep it under {} tokens.\n\n{}",
        config.max_summary_tokens, combined
    );

    let messages = vec![
        LlmMessage::system(
            "You are a concise summarizer. Produce a factual summary preserving \
             key technical details, file paths, decisions, and outcomes.",
        ),
        LlmMessage::human(prompt),
    ];

    let completion_config =
        CompletionConfig::new(&config.model).with_max_tokens(config.max_summary_tokens as u32);

    let response = llm
        .complete(messages, &[], &completion_config)
        .await
        .map_err(|e| AgentError::Context(format!("Compaction LLM call failed: {e}")))?;

    let summary_text = response.text_content();
    let summary_tokens = estimate_tokens_str(&summary_text);

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

/// Check if a node has been compacted (excluded from future context builds).
pub fn is_compacted(node: &GraphNode) -> bool {
    node.metadata
        .get("compacted")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// Select nodes for compaction when context exceeds `threshold_ratio` of `max_tokens`.
///
/// Returns node IDs of older conversation messages (excluding the most recent
/// `guaranteed_recent_turns`) that should be compacted. Returns empty Vec if
/// compaction is not needed.
pub fn select_nodes_for_compaction(
    graph: &GraphStore,
    agent_id: &NodeId,
    max_tokens: usize,
    threshold_ratio: f64,
    guaranteed_recent_turns: usize,
    min_nodes_to_compact: usize,
) -> Result<Vec<NodeId>, AgentError> {
    use crate::context::{estimate_tokens, find_current_turn};
    // Get the leaf turn node
    let current_turn = match find_current_turn(graph, agent_id)? {
        Some(n) => n,
        None => return Ok(vec![]),
    };
    // Walk back through conversation thread (newest-first)
    let thread = graph
        .conversation_thread(&current_turn.id)
        .map_err(AgentError::Graph)?;
    // Filter out already-compacted
    let candidates: Vec<&graphirm_graph::GraphNode> =
        thread.iter().filter(|n| !is_compacted(n)).collect();
    // Estimate total tokens
    let total_tokens: usize = candidates.iter().map(|n| estimate_tokens(n)).sum();
    let threshold = (max_tokens as f64 * threshold_ratio) as usize;
    if total_tokens < threshold {
        return Ok(vec![]);
    }
    // Skip the `guaranteed_recent_turns` newest nodes
    let skip = guaranteed_recent_turns.min(candidates.len());
    let eligible: Vec<NodeId> = candidates[skip..].iter().map(|n| n.id.clone()).collect();
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
        let config = CompactionConfig::default();

        let result = compact_context(&graph, &llm, vec![id], &config).await;
        assert!(result.is_err());
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
        )
        .unwrap();
        assert!(
            nodes.is_empty(),
            "Should return empty when too few eligible nodes"
        );
    }
}
