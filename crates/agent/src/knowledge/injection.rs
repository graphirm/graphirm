//! Cross-session memory injection: retrieves relevant knowledge from past sessions
//! and formats it as system context for the agent's first turn.

use graphirm_graph::{GraphNode, NodeType};

use crate::error::AgentError;

use super::memory::MemoryRetriever;

/// Extract `(entity, entity_type, summary)` from a Knowledge node.
/// Returns `None` for any other node type.
fn extract_knowledge_fields(node: &GraphNode) -> Option<(String, String, String)> {
    if let NodeType::Knowledge(data) = &node.node_type {
        Some((
            data.entity.clone(),
            data.entity_type.clone(),
            data.summary.clone(),
        ))
    } else {
        None
    }
}

/// Format retrieved knowledge nodes into a system-context block
/// that can be prepended to the agent's messages.
pub fn format_memory_context(nodes: &[GraphNode]) -> String {
    if nodes.is_empty() {
        return String::new();
    }

    let mut lines = vec!["## Relevant knowledge from past sessions\n".to_string()];

    for node in nodes {
        if let Some((name, entity_type, description)) = extract_knowledge_fields(node) {
            lines.push(format!("- **{}** ({}): {}", name, entity_type, description));
        }
    }

    lines.join("\n")
}

/// Retrieve relevant knowledge from past sessions and format it
/// as context for the agent's first turn in a new session.
pub async fn build_session_context(
    retriever: &MemoryRetriever,
    initial_prompt: &str,
    max_knowledge_items: usize,
) -> Result<String, AgentError> {
    let nodes = retriever
        .retrieve_relevant(initial_prompt, max_knowledge_items)
        .await?;

    Ok(format_memory_context(&nodes))
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use async_trait::async_trait;
    use graphirm_graph::{vector::VectorIndex, GraphNode, GraphStore, KnowledgeData, NodeType};
    use graphirm_llm::{EmbeddingProvider, LlmError};
    use tokio::sync::RwLock;

    use crate::knowledge::memory::MemoryRetriever;

    struct MockEmbeddingProvider;

    #[async_trait]
    impl EmbeddingProvider for MockEmbeddingProvider {
        async fn embed(&self, text: &str) -> Result<Vec<f32>, LlmError> {
            let mut vec = vec![0.0f32; 64];
            for (i, byte) in text.bytes().enumerate() {
                vec[i % 64] += byte as f32 / 255.0;
            }
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut vec {
                    *v /= norm;
                }
            }
            Ok(vec)
        }
    }

    fn knowledge_node(entity: &str, entity_type: &str, summary: &str) -> GraphNode {
        GraphNode::new(NodeType::Knowledge(KnowledgeData {
            entity: entity.to_string(),
            entity_type: entity_type.to_string(),
            summary: summary.to_string(),
            confidence: 0.9,
        }))
    }

    #[tokio::test]
    async fn test_format_memory_context() {
        let nodes = vec![
            knowledge_node("JWT Auth", "pattern", "Token-based authentication"),
            knowledge_node("bcrypt", "library", "Password hashing for secure storage"),
        ];

        let context = format_memory_context(&nodes);

        assert!(context.contains("Relevant knowledge from past sessions"));
        assert!(context.contains("JWT Auth"));
        assert!(context.contains("pattern"));
        assert!(context.contains("Token-based authentication"));
        assert!(context.contains("bcrypt"));
        assert!(context.contains("library"));
        assert!(context.contains("Password hashing for secure storage"));
    }

    #[tokio::test]
    async fn test_format_memory_context_empty() {
        let nodes: Vec<GraphNode> = vec![];
        let context = format_memory_context(&nodes);
        assert!(context.is_empty());
    }

    #[tokio::test]
    async fn test_build_session_context_full_pipeline() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let vector_index = Arc::new(RwLock::new(VectorIndex::new(64)));
        let llm: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbeddingProvider);

        let retriever = MemoryRetriever::new(graph.clone(), vector_index.clone(), llm, 64);

        let topics = vec![
            ("JWT Auth", "pattern", "Token-based authentication using JSON Web Tokens"),
            ("bcrypt", "library", "Password hashing library for secure storage"),
            ("OAuth2 flow", "pattern", "Authorization protocol for third-party access"),
        ];

        for (name, entity_type, desc) in &topics {
            let node_id = graph
                .add_node(knowledge_node(name, entity_type, desc))
                .unwrap();
            retriever.embed_knowledge_node(&node_id).await.unwrap();
        }

        {
            let mut idx = vector_index.write().await;
            idx.rebuild();
        }

        let context = build_session_context(&retriever, "implement user login", 3)
            .await
            .unwrap();

        assert!(!context.is_empty());
        assert!(context.contains("Relevant knowledge from past sessions"));
    }
}
