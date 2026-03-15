//! Integration test: full segment parse → graph persistence round-trip.

use std::sync::Arc;
use graphirm_graph::{EdgeType, GraphNode, GraphStore, InteractionData, NodeType};
use graphirm_agent::knowledge::segments::{
    detect_nesting, parse_structured_segments, persist_segments,
};

#[tokio::test]
async fn test_full_segment_round_trip() {
    let store = Arc::new(GraphStore::open(":memory:").unwrap());

    // Create a parent Interaction node (simulating a recorded assistant response)
    let parent = GraphNode::new(NodeType::Interaction(InteractionData {
        role: "assistant".into(),
        content: r#"{"segments":[{"type":"reasoning","content":"The bug is X."},{"type":"code","content":"fn fix() {}"},{"type":"answer","content":"Fixed."}]}"#.into(),
        token_count: Some(20),
    }));
    let parent_id = store.add_node(parent).unwrap();

    // Parse the structured output
    let raw = r#"{"segments":[{"type":"reasoning","content":"The bug is X."},{"type":"code","content":"fn fix() {}"},{"type":"answer","content":"Fixed."}]}"#;
    let segments = parse_structured_segments(raw).unwrap();
    assert_eq!(segments.len(), 3);
    assert_eq!(segments[0].segment_type, "reasoning");
    assert_eq!(segments[1].segment_type, "code");
    assert_eq!(segments[2].segment_type, "answer");

    // No nesting in flat sequential segments
    let nesting = detect_nesting(&segments);
    assert!(nesting.is_empty());

    // Persist to graph
    let node_ids = persist_segments(&store, &parent_id, &segments, &nesting).await.unwrap();
    assert_eq!(node_ids.len(), 3);

    // Verify segment Content nodes have correct types and bodies
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

    // Verify Contains edges from parent → each segment
    let edges = store.edges_for_node(&parent_id).unwrap();
    let contains_edges: Vec<_> = edges
        .iter()
        .filter(|e| e.edge_type == EdgeType::Contains && e.source == parent_id)
        .collect();
    assert_eq!(contains_edges.len(), 3);
}

#[tokio::test]
async fn test_segment_round_trip_with_nesting() {
    let store = Arc::new(GraphStore::open(":memory:").unwrap());

    let parent = GraphNode::new(NodeType::Interaction(InteractionData {
        role: "assistant".into(),
        content: "raw".into(),
        token_count: Some(5),
    }));
    let parent_id = store.add_node(parent).unwrap();

    // Manually construct segments with overlapping spans to trigger nesting
    use graphirm_agent::knowledge::segments::Segment;
    let segments = vec![
        Segment { segment_type: "plan".into(), content: "overall plan".into(), start: 0, end: 100 },
        Segment { segment_type: "code".into(), content: "fn f(){}".into(), start: 30, end: 60 },
        Segment { segment_type: "answer".into(), content: "done".into(), start: 110, end: 130 },
    ];

    let nesting = detect_nesting(&segments);
    assert_eq!(nesting.len(), 1);
    assert_eq!(nesting[0], (0, 1)); // plan contains code

    let node_ids = persist_segments(&store, &parent_id, &segments, &nesting).await.unwrap();
    assert_eq!(node_ids.len(), 3);

    // Check nesting edge: segment[0] (plan) → segment[1] (code)
    let edges = store.edges_for_node(&node_ids[0]).unwrap();
    let nesting_edge: Vec<_> = edges
        .iter()
        .filter(|e| e.edge_type == EdgeType::Contains && e.target == node_ids[1])
        .collect();
    assert_eq!(nesting_edge.len(), 1);
}
