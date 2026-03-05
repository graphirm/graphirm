//! Cross-session memory retrieval using HNSW vector search.

use std::sync::Arc;

use tokio::sync::RwLock;

use graphirm_graph::{vector::VectorIndex, GraphStore, NodeId, NodeType};
use graphirm_llm::EmbeddingProvider;

use crate::error::AgentError;

pub struct MemoryRetriever {
    graph: Arc<GraphStore>,
    vector_index: Arc<RwLock<VectorIndex>>,
    llm: Arc<dyn EmbeddingProvider>,
    embedding_dimension: usize,
}

impl MemoryRetriever {
    pub fn new(
        graph: Arc<GraphStore>,
        vector_index: Arc<RwLock<VectorIndex>>,
        llm: Arc<dyn EmbeddingProvider>,
        embedding_dimension: usize,
    ) -> Self {
        Self {
            graph,
            vector_index,
            llm,
            embedding_dimension,
        }
    }

    /// Embed a knowledge node's entity and summary, storing the vector
    /// in both SQLite and the in-memory HNSW index.
    pub async fn embed_knowledge_node(&self, node_id: &NodeId) -> Result<(), AgentError> {
        let node = self.graph.get_node(node_id)?;

        let text = match &node.node_type {
            NodeType::Knowledge(data) => {
                format!("[{}] {}: {}", data.entity_type, data.entity, data.summary)
            }
            _ => {
                return Err(AgentError::Workflow(format!(
                    "Node {node_id} is not a Knowledge node"
                )));
            }
        };

        let embedding = self.llm.embed(&text).await.map_err(AgentError::Llm)?;

        if embedding.len() != self.embedding_dimension {
            return Err(AgentError::Workflow(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.embedding_dimension,
                embedding.len()
            )));
        }

        self.graph.set_embedding(node_id, &embedding)?;

        {
            let mut idx = self.vector_index.write().await;
            idx.insert(node_id.clone(), embedding);
        }

        tracing::debug!(node_id = %node_id, "Embedded knowledge node");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use async_trait::async_trait;
    use graphirm_graph::{GraphNode, KnowledgeData, NodeType};
    use graphirm_llm::LlmError;

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

    fn knowledge_node(entity: &str, summary: &str) -> GraphNode {
        GraphNode::new(NodeType::Knowledge(KnowledgeData {
            entity: entity.to_string(),
            entity_type: "pattern".to_string(),
            summary: summary.to_string(),
            confidence: 0.9,
        }))
    }

    #[tokio::test]
    async fn test_embed_knowledge_node_stores_embedding() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let vector_index = Arc::new(RwLock::new(VectorIndex::new(64)));
        let llm: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbeddingProvider);

        let retriever = MemoryRetriever::new(graph.clone(), vector_index.clone(), llm, 64);

        let node_id = graph
            .add_node(knowledge_node("JWT Authentication", "Token-based auth"))
            .unwrap();

        retriever.embed_knowledge_node(&node_id).await.unwrap();

        let embedding = graph.get_embedding(&node_id).unwrap();
        assert!(embedding.is_some());
        assert_eq!(embedding.unwrap().len(), 64);
    }

    #[tokio::test]
    async fn test_embed_knowledge_node_inserts_into_hnsw() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let vector_index = Arc::new(RwLock::new(VectorIndex::new(64)));
        let llm: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbeddingProvider);

        let retriever = MemoryRetriever::new(graph.clone(), vector_index.clone(), llm, 64);

        let node_id = graph
            .add_node(knowledge_node("JWT Authentication", "Token-based auth"))
            .unwrap();

        retriever.embed_knowledge_node(&node_id).await.unwrap();

        {
            let mut idx = vector_index.write().await;
            idx.rebuild();
        }

        let idx = vector_index.read().await;
        assert_eq!(idx.len(), 1);
    }

    #[tokio::test]
    async fn test_embed_non_knowledge_node_returns_error() {
        use graphirm_graph::{InteractionData, NodeType};

        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let vector_index = Arc::new(RwLock::new(VectorIndex::new(64)));
        let llm: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbeddingProvider);

        let retriever = MemoryRetriever::new(graph.clone(), vector_index.clone(), llm, 64);

        let node_id = graph
            .add_node(GraphNode::new(NodeType::Interaction(InteractionData {
                role: "user".to_string(),
                content: "hello".to_string(),
                token_count: None,
            })))
            .unwrap();

        let result = retriever.embed_knowledge_node(&node_id).await;
        assert!(matches!(result, Err(AgentError::Workflow(_))));
    }

    #[tokio::test]
    async fn test_embed_dimension_mismatch_returns_error() {
        struct TinyProvider;

        #[async_trait]
        impl EmbeddingProvider for TinyProvider {
            async fn embed(&self, _text: &str) -> Result<Vec<f32>, LlmError> {
                Ok(vec![1.0, 0.0]) // 2-dim, but retriever expects 64
            }
        }

        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let vector_index = Arc::new(RwLock::new(VectorIndex::new(64)));
        let llm: Arc<dyn EmbeddingProvider> = Arc::new(TinyProvider);

        let retriever = MemoryRetriever::new(graph.clone(), vector_index.clone(), llm, 64);

        let node_id = graph
            .add_node(knowledge_node("Entity", "Summary"))
            .unwrap();

        let result = retriever.embed_knowledge_node(&node_id).await;
        assert!(matches!(result, Err(AgentError::Workflow(_))));
    }
}
