//! Cross-session memory retrieval using HNSW vector search.

use std::sync::Arc;

use tokio::sync::RwLock;

use graphirm_graph::{GraphNode, GraphStore, NodeId, NodeType, vector::VectorIndex};
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

    /// Convenience constructor — creates a fresh in-memory HNSW index.
    ///
    /// Use this for new sessions. For persistent cross-session memory,
    /// load embeddings from the graph store and insert them into the index
    /// before calling this (future work).
    pub fn from_store(
        graph: Arc<GraphStore>,
        llm: Arc<dyn EmbeddingProvider>,
        embedding_dimension: usize,
    ) -> Self {
        let vector_index = Arc::new(RwLock::new(VectorIndex::new(embedding_dimension)));
        Self::new(graph, vector_index, llm, embedding_dimension)
    }

    /// Embed a knowledge node's entity and summary, storing the vector
    /// in both SQLite and the in-memory HNSW index.
    pub async fn embed_knowledge_node(&self, node_id: &NodeId) -> Result<(), AgentError> {
        // Fetch the node in a blocking thread pool task.
        let graph = self.graph.clone();
        let nid = node_id.clone();
        let node = tokio::task::spawn_blocking(move || graph.get_node(&nid))
            .await
            .map_err(|e| AgentError::Join(e.to_string()))??;

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

        // Persist the embedding in a blocking thread pool task.
        let graph = self.graph.clone();
        let nid = node_id.clone();
        let emb = embedding.clone();
        tokio::task::spawn_blocking(move || graph.set_embedding(&nid, &emb))
            .await
            .map_err(|e| AgentError::Join(e.to_string()))??;

        {
            let mut idx = self.vector_index.write().await;
            idx.insert(node_id.clone(), embedding);
        }

        tracing::debug!(node_id = %node_id, "Embedded knowledge node");
        Ok(())
    }

    /// Load all embeddings persisted in the graph store into the in-memory HNSW index.
    ///
    /// Call this once at server startup after building the retriever to restore
    /// cross-session memory across restarts. Returns the number of embeddings loaded.
    ///
    /// Embeddings whose dimension doesn't match `self.embedding_dimension` are skipped
    /// with a warning — this happens when the embedding backend was changed between runs.
    pub async fn hydrate_from_graph(&self) -> Result<usize, AgentError> {
        let graph = self.graph.clone();
        let all_embeddings = tokio::task::spawn_blocking(move || graph.get_all_embeddings())
            .await
            .map_err(|e| AgentError::Join(e.to_string()))?
            .map_err(|e| AgentError::Workflow(format!("Failed to load embeddings: {e}")))?;

        let total = all_embeddings.len();
        let mut loaded = 0usize;

        {
            let mut idx = self.vector_index.write().await;
            for (node_id, embedding) in all_embeddings {
                if embedding.len() != self.embedding_dimension {
                    tracing::warn!(
                        node_id = %node_id,
                        stored_dim = embedding.len(),
                        expected_dim = self.embedding_dimension,
                        "Skipping embedding with mismatched dimension (backend changed?)"
                    );
                    continue;
                }
                idx.insert(node_id, embedding);
                loaded += 1;
            }
            if loaded > 0 {
                idx.rebuild();
            }
        }

        tracing::info!(loaded, skipped = total - loaded, "Hydrated HNSW index from graph store");
        Ok(loaded)
    }

    /// Embed a query string and search the HNSW index for the k most
    /// similar knowledge nodes. Returns full node objects from the graph.
    pub async fn retrieve_relevant(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<GraphNode>, AgentError> {
        let query_embedding = self.llm.embed(query).await.map_err(AgentError::Llm)?;

        let index = self.vector_index.read().await;
        let candidates = index.search(&query_embedding, k);
        drop(index);

        // Fetch all candidate nodes from the graph in one spawn_blocking call.
        let graph = self.graph.clone();
        let node_ids: Vec<NodeId> = candidates.into_iter().map(|(id, _)| id).collect();
        let fetched: Vec<Result<GraphNode, _>> =
            tokio::task::spawn_blocking(move || node_ids.into_iter().map(|id| graph.get_node(&id)).collect())
                .await
                .map_err(|e| AgentError::Join(e.to_string()))?;

        let mut nodes = Vec::with_capacity(fetched.len());
        for result in fetched {
            match result {
                Ok(node) => nodes.push(node),
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "Knowledge node in HNSW index but missing from graph"
                    );
                }
            }
        }

        Ok(nodes)
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
    async fn test_retrieve_relevant_finds_similar_knowledge() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let vector_index = Arc::new(RwLock::new(VectorIndex::new(64)));
        let llm: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbeddingProvider);

        let retriever = MemoryRetriever::new(graph.clone(), vector_index.clone(), llm, 64);

        let topics = vec![
            (
                "JWT Auth",
                "pattern",
                "Token-based authentication using JSON Web Tokens",
            ),
            (
                "bcrypt",
                "library",
                "Password hashing library for secure storage",
            ),
            (
                "PostgreSQL",
                "database",
                "Relational database for persistent storage",
            ),
            (
                "React hooks",
                "pattern",
                "React state management with useState and useEffect",
            ),
            (
                "OAuth2 flow",
                "pattern",
                "Authorization protocol for third-party access",
            ),
        ];

        for (name, entity_type, desc) in &topics {
            let node_id = graph
                .add_node(GraphNode::new(NodeType::Knowledge(KnowledgeData {
                    entity: name.to_string(),
                    entity_type: entity_type.to_string(),
                    summary: desc.to_string(),
                    confidence: 0.9,
                })))
                .unwrap();
            retriever.embed_knowledge_node(&node_id).await.unwrap();
        }

        {
            let mut idx = vector_index.write().await;
            idx.rebuild();
        }

        let results = retriever
            .retrieve_relevant("implement user login", 3)
            .await
            .unwrap();
        assert_eq!(results.len(), 3);

        for node in &results {
            assert!(matches!(node.node_type, NodeType::Knowledge(_)));
        }
    }

    #[tokio::test]
    async fn test_retrieve_relevant_empty_index() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let vector_index = Arc::new(RwLock::new(VectorIndex::new(64)));
        let llm: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbeddingProvider);

        let retriever = MemoryRetriever::new(graph.clone(), vector_index, llm, 64);

        let results = retriever.retrieve_relevant("anything", 5).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_retrieve_relevant_k_larger_than_index() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let vector_index = Arc::new(RwLock::new(VectorIndex::new(64)));
        let llm: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbeddingProvider);

        let retriever = MemoryRetriever::new(graph.clone(), vector_index.clone(), llm, 64);

        let node_id = graph
            .add_node(knowledge_node("Only Node", "The only knowledge node"))
            .unwrap();
        retriever.embed_knowledge_node(&node_id).await.unwrap();

        {
            let mut idx = vector_index.write().await;
            idx.rebuild();
        }

        let results = retriever.retrieve_relevant("query", 100).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_memory_retriever_from_store() {
        struct DummyEmbed;

        #[async_trait::async_trait]
        impl graphirm_llm::EmbeddingProvider for DummyEmbed {
            async fn embed(&self, _: &str) -> Result<Vec<f32>, graphirm_llm::LlmError> {
                Ok(vec![0.0f32; 768])
            }
        }

        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let retriever = MemoryRetriever::from_store(graph, Arc::new(DummyEmbed), 768);
        // Just verify construction without panic and that basic ops work
        let results = retriever.retrieve_relevant("anything", 5).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_hydrate_from_graph_loads_existing_embeddings() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let vector_index = Arc::new(RwLock::new(VectorIndex::new(64)));
        let llm: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbeddingProvider);

        // Pre-populate SQLite embeddings directly (simulating a previous run).
        let node_id = graph
            .add_node(knowledge_node("Rust ownership", "Memory safety without GC"))
            .unwrap();
        let embedding = vec![0.1f32; 64];
        graph.set_embedding(&node_id, &embedding).unwrap();

        let retriever = MemoryRetriever::new(graph.clone(), vector_index.clone(), llm, 64);
        let count = retriever.hydrate_from_graph().await.unwrap();

        assert_eq!(count, 1);

        {
            let mut idx = vector_index.write().await;
            idx.rebuild();
        }
        let idx = vector_index.read().await;
        assert_eq!(idx.len(), 1);
    }

    #[tokio::test]
    async fn test_hydrate_from_graph_empty_store() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let vector_index = Arc::new(RwLock::new(VectorIndex::new(64)));
        let llm: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbeddingProvider);

        let retriever = MemoryRetriever::new(graph.clone(), vector_index, llm, 64);
        let count = retriever.hydrate_from_graph().await.unwrap();

        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_hydrate_from_graph_skips_wrong_dimension() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let vector_index = Arc::new(RwLock::new(VectorIndex::new(64)));
        let llm: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbeddingProvider);

        let node_id = graph
            .add_node(knowledge_node("Stale node", "Embedded with old model"))
            .unwrap();
        // Store a 32-dim embedding but retriever expects 64-dim.
        graph.set_embedding(&node_id, &vec![0.1f32; 32]).unwrap();

        let retriever = MemoryRetriever::new(graph.clone(), vector_index, llm, 64);
        let count = retriever.hydrate_from_graph().await.unwrap();

        assert_eq!(count, 0);
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

        let node_id = graph.add_node(knowledge_node("Entity", "Summary")).unwrap();

        let result = retriever.embed_knowledge_node(&node_id).await;
        assert!(matches!(result, Err(AgentError::Workflow(_))));
    }
}
