//! Integration test: full segment parse → graph persistence round-trip.

use std::sync::Arc;
use graphirm_graph::{AgentData, EdgeType, GraphEdge, GraphNode, GraphStore, InteractionData, NodeType};
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

#[tokio::test]
async fn test_segment_filter_excludes_non_matching_types() {
    use graphirm_agent::context::{build_context, ContextConfig};

    let store = Arc::new(GraphStore::open(":memory:").unwrap());

    // Create an Agent node (session root required by build_context)
    let agent_node = GraphNode::new(NodeType::Agent(AgentData {
        name: "test-agent".to_string(),
        model: "mock".to_string(),
        system_prompt: Some("You are helpful.".to_string()),
        status: "running".to_string(),
    }));
    let agent_id = agent_node.id.clone();
    store.add_node(agent_node).unwrap();

    // Create a user Interaction node linked to the agent
    let user_node = GraphNode::new(NodeType::Interaction(InteractionData {
        role: "user".to_string(),
        content: "Write a Python adder.".to_string(),
        token_count: Some(5),
    }));
    let user_id = user_node.id.clone();
    store.add_node(user_node).unwrap();
    store
        .add_edge(GraphEdge::new(EdgeType::Produces, agent_id.clone(), user_id.clone()))
        .unwrap();

    // Create an assistant Interaction node (metadata["segmented"] will be stamped after persist)
    let assistant_node = GraphNode::new(NodeType::Interaction(InteractionData {
        role: "assistant".to_string(),
        content: "raw json placeholder".to_string(),
        token_count: Some(10),
    }));
    let assistant_id = assistant_node.id.clone();
    store.add_node(assistant_node).unwrap();

    // Link assistant to agent (Produces) and to user turn (RespondsTo)
    store
        .add_edge(GraphEdge::new(EdgeType::Produces, agent_id.clone(), assistant_id.clone()))
        .unwrap();
    store
        .add_edge(GraphEdge::new(EdgeType::RespondsTo, assistant_id.clone(), user_id.clone()))
        .unwrap();

    // Persist reasoning + code segments as child Content nodes
    use graphirm_agent::knowledge::segments::Segment;
    let segments = vec![
        Segment { segment_type: "reasoning".to_string(), content: "I think step by step.".to_string(), start: 0, end: 22 },
        Segment { segment_type: "code".to_string(), content: "def add(a, b): return a + b".to_string(), start: 23, end: 50 },
    ];
    let nesting = detect_nesting(&segments);
    persist_segments(&store, &assistant_id, &segments, &nesting).await.unwrap();

    // Stamp "segmented": true on the parent node, mirroring the production path in workflow.rs.
    let mut stamped = store.get_node(&assistant_id).unwrap();
    stamped.metadata["segmented"] = serde_json::json!(true);
    store.update_node(&assistant_id, stamped).unwrap();

    // Build context filtering to "code" segments only
    let config = ContextConfig {
        max_tokens: 10_000,
        system_prompt: "You are helpful.".to_string(),
        guaranteed_recent_turns: 4,
        recency_decay: 0.1,
        enable_compaction: false,
        segment_filter: Some(vec!["code".to_string()]),
        ..ContextConfig::default()
    };

    let window = build_context(&*store, &agent_id, &config).unwrap();

    // Find the assistant message in the context window
    let assistant_msg = window
        .messages
        .iter()
        .find(|m| {
            m.content.iter().any(|part| {
                matches!(part, graphirm_llm::ContentPart::Text { text } if text.contains("def add"))
            })
        })
        .expect("assistant message with code should be in context");

    // The reconstructed message must contain the code segment
    let full_text: String = assistant_msg
        .content
        .iter()
        .filter_map(|part| {
            if let graphirm_llm::ContentPart::Text { text } = part { Some(text.as_str()) } else { None }
        })
        .collect::<Vec<_>>()
        .join("\n");

    assert!(
        full_text.contains("def add"),
        "Expected code segment content in assistant message, got: {full_text}"
    );
    assert!(
        !full_text.contains("I think step by step"),
        "Reasoning segment must be excluded by segment_filter, got: {full_text}"
    );
}
