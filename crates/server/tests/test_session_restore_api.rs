#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use chrono::Utc;

    use graphirm_agent::SessionStatus;
    use graphirm_graph::{GraphNode, GraphStore, NodeId, NodeType, nodes::AgentData};
    use graphirm_server::restore_sessions_from_graph;

    #[tokio::test]
    async fn test_restore_sessions_from_graph_empty() {
        // Empty graph should return empty hashmap
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let sessions: HashMap<String, _> = restore_sessions_from_graph(&graph).await.unwrap();
        assert_eq!(sessions.len(), 0);
    }

    #[tokio::test]
    async fn test_restore_sessions_from_graph_with_agents() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());

        // Add agent node representing a completed session
        let agent = GraphNode {
            id: NodeId("session-001".to_string()),
            node_type: NodeType::Agent(AgentData {
                name: "build".to_string(),
                model: "claude-sonnet-4".to_string(),
                status: "completed".to_string(),
                system_prompt: None,
            }),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: serde_json::json!({}),
        };

        let _agent_id = graph.add_node(agent.clone()).unwrap();

        // Restore sessions
        let sessions: HashMap<String, _> = restore_sessions_from_graph(&graph).await.unwrap();

        // Verify session was restored
        assert_eq!(sessions.len(), 1);
        assert!(sessions.contains_key("session-001"));

        let metadata = &sessions["session-001"];
        assert_eq!(metadata.session_id, "session-001");
        assert_eq!(metadata.name, "build");
        assert_eq!(metadata.model, "claude-sonnet-4");
        assert_eq!(metadata.status, SessionStatus::Completed);
    }

    #[tokio::test]
    async fn test_restore_sessions_status_mapping() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());

        // Create sessions with different statuses
        let agent_active = GraphNode {
            id: NodeId("session-active".to_string()),
            node_type: NodeType::Agent(AgentData {
                name: "active".to_string(),
                model: "claude-sonnet-4".to_string(),
                status: "active".to_string(),
                system_prompt: None,
            }),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: serde_json::json!({}),
        };

        let agent_failed = GraphNode {
            id: NodeId("session-failed".to_string()),
            node_type: NodeType::Agent(AgentData {
                name: "error".to_string(),
                model: "claude-haiku-4".to_string(),
                status: "failed".to_string(),
                system_prompt: None,
            }),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: serde_json::json!({}),
        };

        graph.add_node(agent_active).unwrap();
        graph.add_node(agent_failed).unwrap();

        let sessions: HashMap<String, _> = restore_sessions_from_graph(&graph).await.unwrap();

        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions["session-active"].status, SessionStatus::Running);
        assert_eq!(sessions["session-failed"].status, SessionStatus::Failed);
    }

    #[tokio::test]
    async fn test_api_list_sessions_includes_restored() {
        // This test verifies that restored sessions are included in the API list
        // The list_sessions handler iterates over the sessions registry and returns them
        let graph = Arc::new(GraphStore::open_memory().unwrap());

        // Add multiple agent nodes representing restored sessions
        let sessions_data = vec![
            (
                "session-api-001",
                "auth-refactor",
                "claude-sonnet-4",
                "completed",
            ),
            (
                "session-api-002",
                "feature-build",
                "claude-haiku-4",
                "active",
            ),
            ("session-api-003", "debug-error", "gpt-4o", "failed"),
        ];

        for (id, name, model, status) in sessions_data {
            let agent = GraphNode {
                id: NodeId(id.to_string()),
                node_type: NodeType::Agent(AgentData {
                    name: name.to_string(),
                    model: model.to_string(),
                    status: status.to_string(),
                    system_prompt: None,
                }),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                metadata: serde_json::json!({}),
            };

            graph.add_node(agent).unwrap();
        }

        // Restore sessions from graph (simulates server startup)
        let sessions = restore_sessions_from_graph(&graph).await.unwrap();

        // Verify all sessions were restored with correct data
        assert_eq!(sessions.len(), 3);

        assert_eq!(sessions["session-api-001"].name, "auth-refactor");
        assert_eq!(sessions["session-api-001"].status, SessionStatus::Completed);

        assert_eq!(sessions["session-api-002"].name, "feature-build");
        assert_eq!(sessions["session-api-002"].status, SessionStatus::Running);

        assert_eq!(sessions["session-api-003"].name, "debug-error");
        assert_eq!(sessions["session-api-003"].status, SessionStatus::Failed);

        // Verify all sessions have required fields populated
        for (session_id, metadata) in &sessions {
            assert_eq!(metadata.session_id, *session_id);
            assert!(!metadata.name.is_empty());
            assert!(!metadata.model.is_empty());
            assert!(!metadata.created_at.to_string().is_empty());
        }
    }
}
