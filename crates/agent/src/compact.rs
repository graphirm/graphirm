// Context compaction: summarize old context, prune graph branches

use graphirm_graph::{EdgeType, GraphEdge, GraphNode, GraphStore, KnowledgeData, NodeId, NodeType};
use graphirm_llm::{CompletionConfig, LlmMessage, LlmProvider};

use crate::context::{estimate_tokens, estimate_tokens_str, get_text_content};
use crate::error::AgentError;

#[derive(Debug, Clone)]
pub struct CompactionConfig {
    pub model: String,
    pub max_summary_tokens: usize,
    pub min_nodes_to_compact: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            model: "mock".to_string(),
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

        if let Some(obj) = node.metadata.as_object_mut() {
            obj.insert("compacted".to_string(), serde_json::Value::Bool(true));
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use graphirm_graph::{GraphNode, InteractionData, NodeType};
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
}
