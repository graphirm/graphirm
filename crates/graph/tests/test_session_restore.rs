use graphirm_graph::nodes::AgentData;
use graphirm_graph::{GraphNode, GraphStore, NodeId, NodeType};

#[test]
fn test_get_agent_nodes_returns_all_agents() {
    let store = GraphStore::open_memory().unwrap();

    // Create 2 agent nodes
    let agent1 = GraphNode {
        id: NodeId(uuid::Uuid::new_v4().to_string()),
        node_type: NodeType::Agent(AgentData {
            name: "build".to_string(),
            model: "claude-sonnet-4".to_string(),
            system_prompt: None,
            status: "completed".to_string(),
        }),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        metadata: serde_json::json!({}),
    };

    let agent2 = GraphNode {
        id: NodeId(uuid::Uuid::new_v4().to_string()),
        node_type: NodeType::Agent(AgentData {
            name: "explore".to_string(),
            model: "claude-haiku-4".to_string(),
            system_prompt: None,
            status: "idle".to_string(),
        }),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        metadata: serde_json::json!({}),
    };

    let _id1 = store.add_node(agent1.clone()).unwrap();
    let _id2 = store.add_node(agent2.clone()).unwrap();

    let agents = store.get_agent_nodes().unwrap();

    assert_eq!(agents.len(), 2);
    assert!(agents.iter().any(|a| a.0.id == agent1.id));
    assert!(agents.iter().any(|a| a.0.id == agent2.id));
}
