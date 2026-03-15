//! Structured response segment parsing and graph persistence.

use std::sync::Arc;

use graphirm_graph::{ContentData, EdgeType, GraphEdge, GraphNode, GraphStore, NodeId, NodeType};
use serde::{Deserialize, Serialize};

use crate::error::AgentError;

/// A parsed segment from an LLM structured response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    /// The segment type label (e.g. "code", "reasoning", "answer").
    pub segment_type: String,
    /// The text content of this segment.
    pub content: String,
    /// Character offset where this segment starts in the concatenated raw text.
    #[serde(default)]
    pub start: usize,
    /// Character offset where this segment ends (exclusive) in the concatenated raw text.
    #[serde(default)]
    pub end: usize,
}

/// Wire format the LLM is asked to produce.
#[derive(Debug, Deserialize)]
struct StructuredResponse {
    segments: Vec<SegmentWire>,
}

#[derive(Debug, Deserialize)]
struct SegmentWire {
    #[serde(rename = "type")]
    segment_type: String,
    content: String,
}

/// Try to parse LLM response text as a structured segment JSON envelope.
///
/// Expected format: `{"segments": [{"type": "...", "content": "..."}, ...]}`.
/// Strips markdown code fences if the model wrapped the JSON.
/// Returns `Err` if the text is not valid JSON or missing the `segments` key.
pub fn parse_structured_segments(text: &str) -> Result<Vec<Segment>, AgentError> {
    let trimmed = text.trim();
    let clean = if trimmed.starts_with("```") {
        trimmed
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim()
    } else {
        trimmed
    };

    let parsed: StructuredResponse = serde_json::from_str(clean)
        .map_err(|e| AgentError::Workflow(format!("Segment JSON parse failed: {e}")))?;

    // Compute monotonically increasing character offsets from concatenation order.
    let mut offset = 0usize;
    let segments = parsed
        .segments
        .into_iter()
        .map(|w| {
            let start = offset;
            let end = start + w.content.len();
            offset = end;
            Segment {
                segment_type: w.segment_type,
                content: w.content,
                start,
                end,
            }
        })
        .collect();

    Ok(segments)
}

/// Find parent-child nesting pairs among segments based on character span containment.
///
/// Returns `(parent_index, child_index)` pairs where the child's span falls entirely
/// within the parent's span (but is not identical to it).
pub fn detect_nesting(segments: &[Segment]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (i, parent) in segments.iter().enumerate() {
        for (j, child) in segments.iter().enumerate() {
            if i == j {
                continue;
            }
            let child_inside_parent = child.start >= parent.start && child.end <= parent.end;
            let not_identical = child.start != parent.start || child.end != parent.end;
            if child_inside_parent && not_identical {
                pairs.push((i, j));
            }
        }
    }
    pairs
}

/// Create Content nodes for each segment and link them to the parent Interaction node
/// via Contains edges (with `order` in edge metadata). Also adds Contains edges for
/// detected nesting pairs (parent segment → child segment).
///
/// Returns the NodeId of each created segment node, in order.
pub async fn persist_segments(
    store: &Arc<GraphStore>,
    parent_id: &NodeId,
    segments: &[Segment],
    nesting: &[(usize, usize)],
) -> Result<Vec<NodeId>, AgentError> {
    let store = Arc::clone(store);
    let parent_id = parent_id.clone();
    let segments = segments.to_vec();
    let nesting = nesting.to_vec();

    tokio::task::spawn_blocking(move || {
        let mut node_ids = Vec::with_capacity(segments.len());

        for (i, segment) in segments.iter().enumerate() {
            let language = if segment.segment_type == "code" {
                Some("unknown".to_string())
            } else {
                None
            };

            let mut node = GraphNode::new(NodeType::Content(ContentData {
                content_type: segment.segment_type.clone(),
                path: None,
                body: segment.content.clone(),
                language,
            }));

            node.metadata = serde_json::json!({
                "segment_start": segment.start,
                "segment_end": segment.end,
            });

            let node_id = store.add_node(node)?;

            let edge = GraphEdge::new(EdgeType::Contains, parent_id.clone(), node_id.clone())
                .with_metadata(serde_json::json!({ "order": i }));
            store.add_edge(edge)?;

            node_ids.push(node_id);
        }

        for (parent_idx, child_idx) in &nesting {
            let edge = GraphEdge::new(
                EdgeType::Contains,
                node_ids[*parent_idx].clone(),
                node_ids[*child_idx].clone(),
            );
            store.add_edge(edge)?;
        }

        Ok(node_ids)
    })
    .await
    .map_err(|e| AgentError::Join(e.to_string()))?
}

/// Run GLiNER2 over the raw response text with segment labels and return
/// `Segment`s with character offsets from the model.
///
/// Only available with the `local-extraction` feature flag.
/// Real integration testing requires a downloaded model; set `GLINER2_MODEL_DIR`
/// and use `OnnxExtractor::new()` to construct the extractor.
#[cfg(feature = "local-extraction")]
pub async fn segment_extract_gliner2(
    extractor: &super::local_extraction::OnnxExtractor,
    text: &str,
    labels: &[String],
    min_confidence: f64,
) -> Result<Vec<Segment>, AgentError> {
    let raw = extractor.extract_raw(text, labels, min_confidence).await?;
    let segments = raw
        .into_iter()
        .map(|e| Segment {
            segment_type: e.entity_type,
            content: e.text,
            start: e.start,
            end: e.end,
        })
        .collect();
    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_structured_output_valid() {
        let json = r#"{"segments": [
            {"type": "reasoning", "content": "The bug is in line 42."},
            {"type": "code", "content": "fn fix() {}"}
        ]}"#;
        let result = parse_structured_segments(json);
        assert!(result.is_ok());
        let segments = result.unwrap();
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].segment_type, "reasoning");
        assert_eq!(segments[0].content, "The bug is in line 42.");
        assert_eq!(segments[1].segment_type, "code");
        assert_eq!(segments[1].content, "fn fix() {}");
    }

    #[test]
    fn test_parse_structured_output_offsets() {
        let json = r#"{"segments": [
            {"type": "reasoning", "content": "hello"},
            {"type": "code", "content": "world"}
        ]}"#;
        let segments = parse_structured_segments(json).unwrap();
        assert_eq!(segments[0].start, 0);
        assert_eq!(segments[0].end, 5); // len("hello")
        assert_eq!(segments[1].start, 5);
        assert_eq!(segments[1].end, 10); // len("world")
    }

    #[test]
    fn test_parse_structured_output_strips_code_fences() {
        let json = "```json\n{\"segments\": [{\"type\": \"answer\", \"content\": \"done\"}]}\n```";
        let result = parse_structured_segments(json);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_parse_structured_output_invalid_json() {
        let result = parse_structured_segments("This is plain text, not JSON.");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_structured_output_missing_segments_key() {
        let result = parse_structured_segments(r#"{"items": []}"#);
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_nesting_found() {
        let segments = vec![
            Segment {
                segment_type: "plan".into(),
                content: "long plan text here foo bar baz".into(),
                start: 0,
                end: 100,
            },
            Segment {
                segment_type: "code".into(),
                content: "fn f(){}".into(),
                start: 30,
                end: 60,
            },
            Segment {
                segment_type: "answer".into(),
                content: "done".into(),
                start: 110,
                end: 130,
            },
        ];
        let pairs = detect_nesting(&segments);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], (0, 1)); // plan contains code
    }

    #[test]
    fn test_detect_nesting_no_overlap() {
        let segments = vec![
            Segment {
                segment_type: "reasoning".into(),
                content: "a".into(),
                start: 0,
                end: 50,
            },
            Segment {
                segment_type: "code".into(),
                content: "b".into(),
                start: 55,
                end: 100,
            },
        ];
        let pairs = detect_nesting(&segments);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_detect_nesting_identical_spans_not_nested() {
        let segments = vec![
            Segment {
                segment_type: "plan".into(),
                content: "x".into(),
                start: 0,
                end: 10,
            },
            Segment {
                segment_type: "code".into(),
                content: "x".into(),
                start: 0,
                end: 10,
            },
        ];
        let pairs = detect_nesting(&segments);
        assert!(pairs.is_empty()); // identical span = not nested
    }

    #[tokio::test]
    async fn test_persist_segments_creates_nodes_and_edges() {
        use graphirm_graph::{EdgeType, GraphNode, GraphStore, InteractionData, NodeType};
        use std::sync::Arc;

        let store = Arc::new(GraphStore::open(":memory:").unwrap());
        let parent = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".into(),
            content: "raw text".into(),
            token_count: Some(10),
        }));
        let parent_id = store.add_node(parent).unwrap();

        let segments = vec![
            Segment {
                segment_type: "reasoning".into(),
                content: "because X".into(),
                start: 0,
                end: 9,
            },
            Segment {
                segment_type: "code".into(),
                content: "fn f(){}".into(),
                start: 10,
                end: 18,
            },
        ];

        let node_ids = persist_segments(&store, &parent_id, &segments, &[])
            .await
            .unwrap();
        assert_eq!(node_ids.len(), 2);

        for (i, nid) in node_ids.iter().enumerate() {
            let node = store.get_node(nid).unwrap();
            match node.node_type {
                NodeType::Content(data) => {
                    assert_eq!(data.content_type, segments[i].segment_type);
                    assert_eq!(data.body, segments[i].content);
                }
                _ => panic!("Expected Content node at index {i}"),
            }
        }

        let edges = store.edges_for_node(&parent_id).unwrap();
        let contains: Vec<_> = edges
            .iter()
            .filter(|e| e.edge_type == EdgeType::Contains && e.source == parent_id)
            .collect();
        assert_eq!(contains.len(), 2);
    }

    /// Verify `segment_extract_gliner2` exists and the module compiles with the feature enabled.
    /// Real integration testing requires a downloaded GLiNER2 model (`GLINER2_MODEL_DIR`).
    #[cfg(feature = "local-extraction")]
    #[test]
    fn test_segment_extract_gliner2_compiles() {
        // Compilation with `--features local-extraction` is the primary check.
        // We cannot instantiate OnnxExtractor without a real model directory,
        // so this test body is intentionally empty.
    }

    #[tokio::test]
    async fn test_persist_segments_with_nesting() {
        use graphirm_graph::{EdgeType, GraphNode, GraphStore, InteractionData, NodeType};
        use std::sync::Arc;

        let store = Arc::new(GraphStore::open(":memory:").unwrap());
        let parent = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".into(),
            content: "raw".into(),
            token_count: Some(5),
        }));
        let parent_id = store.add_node(parent).unwrap();

        let segments = vec![
            Segment {
                segment_type: "plan".into(),
                content: "plan with code inside".into(),
                start: 0,
                end: 100,
            },
            Segment {
                segment_type: "code".into(),
                content: "fn f(){}".into(),
                start: 30,
                end: 60,
            },
        ];
        let nesting = vec![(0usize, 1usize)];

        let node_ids = persist_segments(&store, &parent_id, &segments, &nesting)
            .await
            .unwrap();
        assert_eq!(node_ids.len(), 2);

        let edges = store.edges_for_node(&node_ids[0]).unwrap();
        let nested: Vec<_> = edges
            .iter()
            .filter(|e| {
                e.edge_type == EdgeType::Contains
                    && e.source == node_ids[0]
                    && e.target == node_ids[1]
            })
            .collect();
        assert_eq!(nested.len(), 1);
    }
}
