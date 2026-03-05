use std::sync::Arc;

use async_trait::async_trait;
use graphirm_agent::knowledge::{
    extraction::{ExtractionConfig, extract_knowledge},
    injection::build_session_context,
    memory::MemoryRetriever,
};
use graphirm_graph::{
    vector::VectorIndex, Direction, EdgeType, GraphEdge, GraphNode, GraphStore, InteractionData,
    KnowledgeData, NodeType,
};
use graphirm_llm::{EmbeddingProvider, LlmError, MockProvider};
use tokio::sync::RwLock;

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

const AUTH_EXTRACTION_JSON: &str = r#"{
    "entities": [
        {
            "entity_type": "pattern",
            "name": "JWT Authentication",
            "description": "Token-based auth using jsonwebtoken crate with RS256 signing",
            "confidence": 0.95,
            "relationships": [{"target_name": "argon2 password hashing", "relationship": "used_with"}]
        },
        {
            "entity_type": "library",
            "name": "argon2 password hashing",
            "description": "Secure password hashing using the argon2 crate",
            "confidence": 0.90,
            "relationships": []
        },
        {
            "entity_type": "architecture",
            "name": "Redis session store",
            "description": "Using Redis for session storage to enable horizontal scaling",
            "confidence": 0.85,
            "relationships": [{"target_name": "JWT Authentication", "relationship": "alternative_to"}]
        },
        {
            "entity_type": "pattern",
            "name": "axum auth middleware",
            "description": "Authentication guard implemented as axum middleware layer",
            "confidence": 0.88,
            "relationships": [{"target_name": "JWT Authentication", "relationship": "implements"}]
        }
    ]
}"#;

/// Full pipeline: Session A extracts knowledge → embed → rebuild index → Session B retrieves.
#[tokio::test]
async fn test_full_knowledge_pipeline() {
    let graph = Arc::new(GraphStore::open_memory().unwrap());

    // Session A: create conversation nodes
    let user_id = graph
        .add_node(GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "How should I implement user authentication in my Rust web app?".to_string(),
            token_count: None,
        })))
        .unwrap();

    let assistant_id = graph
        .add_node(GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".to_string(),
            content: "Use JWT with argon2 for password hashing. Consider Redis for session storage. Use axum middleware for auth guards.".to_string(),
            token_count: None,
        })))
        .unwrap();

    graph
        .add_edge(GraphEdge::new(
            EdgeType::RespondsTo,
            assistant_id.clone(),
            user_id.clone(),
        ))
        .unwrap();

    // Extract knowledge using a mock LLM that returns 4 auth entities
    let config = ExtractionConfig {
        enabled: true,
        min_confidence: 0.7,
        ..ExtractionConfig::default()
    };
    let llm = MockProvider::fixed(AUTH_EXTRACTION_JSON);
    let messages = vec![
        (
            "user".to_string(),
            "How should I implement user authentication in my Rust web app?".to_string(),
        ),
        (
            "assistant".to_string(),
            "Use JWT with argon2 for password hashing. Consider Redis for session storage. Use axum middleware for auth guards.".to_string(),
        ),
    ];

    let knowledge_ids = extract_knowledge(&graph, &llm, &messages, &assistant_id, &config)
        .await
        .unwrap();

    assert_eq!(
        knowledge_ids.len(),
        4,
        "All 4 entities above 0.7 confidence should be extracted"
    );

    // Embed each knowledge node into the HNSW index
    let vector_index = Arc::new(RwLock::new(VectorIndex::new(64)));
    let embedding_provider: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbeddingProvider);
    let retriever = MemoryRetriever::new(graph.clone(), vector_index.clone(), embedding_provider, 64);

    for id in &knowledge_ids {
        retriever.embed_knowledge_node(id).await.unwrap();
    }

    // Rebuild HNSW so search is available
    {
        let mut idx = vector_index.write().await;
        idx.rebuild();
    }

    // Session B: query "implement user login with password" → relevant auth knowledge surfaces
    let context = build_session_context(&retriever, "implement user login with password", 4)
        .await
        .unwrap();

    assert!(!context.is_empty(), "Context should not be empty");
    assert!(
        context.contains("Relevant knowledge from past sessions"),
        "Context must contain the section header"
    );

    // At least one auth-related term should appear in the retrieved context
    let lower = context.to_lowercase();
    assert!(
        lower.contains("jwt")
            || lower.contains("argon2")
            || lower.contains("auth")
            || lower.contains("session"),
        "Context should contain auth-related knowledge. Got:\n{context}"
    );

    // Every knowledge node must have a DerivedFrom edge pointing to the assistant message
    for id in &knowledge_ids {
        let derived = graph
            .neighbors(id, Some(EdgeType::DerivedFrom), Direction::Outgoing)
            .unwrap();
        assert!(
            derived.iter().any(|n| n.id == assistant_id),
            "Knowledge node {id} should have a DerivedFrom edge to the assistant message"
        );
    }

    // Inter-entity RelatesTo edges: JWT→argon2, Redis→JWT, axum→JWT = 3 edges total
    let relates_to_count: usize = knowledge_ids
        .iter()
        .map(|id| {
            graph
                .neighbors(id, Some(EdgeType::RelatesTo), Direction::Outgoing)
                .unwrap_or_default()
                .len()
        })
        .sum();
    assert!(
        relates_to_count >= 3,
        "Should have at least 3 RelatesTo edges (JWT→argon2, Redis→JWT, axum→JWT), got {relates_to_count}"
    );
}

/// Simulates app restart: store has embeddings, rebuild HNSW from SQLite, then query.
#[tokio::test]
async fn test_rebuild_index_from_store_then_query() {
    let graph = Arc::new(GraphStore::open_memory().unwrap());
    let embedding_provider = MockEmbeddingProvider;

    // Simulate a previous session: create knowledge nodes with embeddings stored in SQLite
    let topics = [
        (
            "REST API design",
            "architecture",
            "Designing RESTful APIs with proper resource naming and HTTP status codes",
        ),
        (
            "GraphQL API schema",
            "pattern",
            "Schema-first API design using GraphQL with typed queries and mutations",
        ),
        (
            "Password hashing",
            "library",
            "Secure password storage using bcrypt or argon2 crates",
        ),
    ];

    for (name, entity_type, summary) in &topics {
        let node_id = graph
            .add_node(GraphNode::new(NodeType::Knowledge(KnowledgeData {
                entity: name.to_string(),
                entity_type: entity_type.to_string(),
                summary: summary.to_string(),
                confidence: 0.9,
            })))
            .unwrap();

        let text = format!("[{entity_type}] {name}: {summary}");
        let embedding = embedding_provider.embed(&text).await.unwrap();
        graph.set_embedding(&node_id, &embedding).unwrap();
    }

    // Simulate app restart: rebuild HNSW entirely from SQLite
    let rebuilt_index = VectorIndex::rebuild_from_store(&graph, 64).unwrap();
    assert_eq!(
        rebuilt_index.len(),
        3,
        "All 3 stored embeddings should be loaded from the graph store"
    );

    let vector_index = Arc::new(RwLock::new(rebuilt_index));
    let emb: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbeddingProvider);
    let retriever = MemoryRetriever::new(graph.clone(), vector_index, emb, 64);

    // Query with k=2 — must return exactly 2 results regardless of which nodes win
    let results = retriever.retrieve_relevant("API design", 2).await.unwrap();
    assert_eq!(
        results.len(),
        2,
        "Should retrieve exactly 2 results for k=2 with 3 nodes in index"
    );

    for node in &results {
        assert!(
            matches!(node.node_type, NodeType::Knowledge(_)),
            "All retrieved nodes should be Knowledge nodes"
        );
    }
}
