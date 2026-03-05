//! Integration tests for the local ONNX extraction backend.
//!
//! These tests verify the `ExtractionBackend` routing logic and the
//! `OnnxExtractor` error-path behavior without requiring a real GLiNER2 model.
//! Tests that exercise actual ONNX inference are gated on the `local-extraction`
//! feature and require a model artifact at the configured path.

use graphirm_agent::knowledge::extraction::{
    ExtractionBackend, ExtractionConfig, extract_knowledge_with_backend,
};
use graphirm_graph::{GraphNode, GraphStore, InteractionData, NodeType};

fn make_source_node(graph: &GraphStore) -> graphirm_graph::NodeId {
    graph
        .add_node(GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "How should I handle authentication?".to_string(),
            token_count: None,
        })))
        .unwrap()
}

#[tokio::test]
async fn test_local_backend_without_feature_returns_error() {
    let graph = GraphStore::open_memory().unwrap();
    let config = ExtractionConfig {
        enabled: true,
        min_confidence: 0.5,
        backend: ExtractionBackend::Local {
            model_path: "/nonexistent/model.onnx".to_string(),
            tokenizer_path: "/nonexistent/tokenizer.json".to_string(),
        },
        ..ExtractionConfig::default()
    };

    let source_id = make_source_node(&graph);
    let messages = vec![("user".to_string(), "test message".to_string())];

    let result = extract_knowledge_with_backend(
        &graph,
        None,
        #[cfg(not(feature = "local-extraction"))]
        None::<()>,
        #[cfg(feature = "local-extraction")]
        None,
        &messages,
        &source_id,
        &config,
    )
    .await;

    // Without `local-extraction` feature: returns a feature-not-available error.
    // With the feature but no extractor: returns a "no OnnxExtractor provided" error.
    // Either way, the call must not silently succeed with zero nodes.
    assert!(result.is_err());
}

#[tokio::test]
async fn test_hybrid_backend_without_feature_returns_error() {
    let graph = GraphStore::open_memory().unwrap();
    let config = ExtractionConfig {
        enabled: true,
        min_confidence: 0.5,
        backend: ExtractionBackend::Hybrid {
            model_path: "/nonexistent/model.onnx".to_string(),
            tokenizer_path: "/nonexistent/tokenizer.json".to_string(),
        },
        ..ExtractionConfig::default()
    };

    let source_id = make_source_node(&graph);
    let messages = vec![("user".to_string(), "test message".to_string())];

    let result = extract_knowledge_with_backend(
        &graph,
        None,
        #[cfg(not(feature = "local-extraction"))]
        None::<()>,
        #[cfg(feature = "local-extraction")]
        None,
        &messages,
        &source_id,
        &config,
    )
    .await;

    assert!(result.is_err());
}
