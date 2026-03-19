use async_trait::async_trait;
use graphirm_graph::nodes::{GraphNode, NodeId};

use crate::ToolError;

/// A ranked knowledge search result: node + cosine similarity score (0..1).
#[derive(Debug, Clone)]
pub struct KnowledgeResult {
    pub node: GraphNode,
    pub node_id: NodeId,
    pub score: f64,
}

/// Semantic retrieval over the HNSW knowledge index.
///
/// Implemented by `MemoryRetriever` in `graphirm-agent`. Stored as
/// `Option<Arc<dyn KnowledgeRetriever>>` in `ToolContext` so `graph_query`
/// can access it without creating a circular crate dependency.
#[async_trait]
pub trait KnowledgeRetriever: Send + Sync {
    /// Embed `query`, search HNSW, and return up to `k` Knowledge nodes
    /// ordered by descending similarity (highest first).
    async fn retrieve_semantic(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<KnowledgeResult>, ToolError>;
}
