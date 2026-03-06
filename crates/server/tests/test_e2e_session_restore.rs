#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use chrono::Utc;

    use graphirm_agent::SessionStatus;
    use graphirm_graph::{GraphNode, GraphStore, NodeId, NodeType, nodes::AgentData};
    use graphirm_server::restore_sessions_from_graph;

    #[tokio::test]
    async fn test_e2e_session_restoration_flow() {
        // 1. Create graph with multiple sessions in different states
        let graph = Arc::new(GraphStore::open_memory().unwrap());

        let sessions_data = vec![
            ("session-1", "auth-refactor", "completed", "claude-sonnet-4"),
            ("session-2", "feature-build", "active", "claude-haiku-4"),
            ("session-3", "debug-error", "failed", "gpt-4o"),
        ];

        for (id, name, status, model) in sessions_data {
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

        // 2. Call restore (simulates server startup)
        let sessions = restore_sessions_from_graph(&graph).await.unwrap();

        // 3. Verify all sessions restored with correct metadata
        assert_eq!(sessions.len(), 3);

        assert_eq!(sessions["session-1"].name, "auth-refactor");
        assert_eq!(sessions["session-1"].status, SessionStatus::Completed);
        assert_eq!(sessions["session-1"].model, "claude-sonnet-4");

        assert_eq!(sessions["session-2"].name, "feature-build");
        assert_eq!(sessions["session-2"].status, SessionStatus::Running);
        assert_eq!(sessions["session-2"].model, "claude-haiku-4");

        assert_eq!(sessions["session-3"].name, "debug-error");
        assert_eq!(sessions["session-3"].status, SessionStatus::Failed);
        assert_eq!(sessions["session-3"].model, "gpt-4o");

        // 4. Verify all sessions have required fields
        for (session_id, metadata) in &sessions {
            assert!(!session_id.is_empty());
            assert_eq!(metadata.session_id, *session_id);
            assert!(!metadata.name.is_empty());
            assert!(!metadata.model.is_empty());
            assert!(!metadata.created_at.to_string().is_empty());
        }
    }

    #[tokio::test]
    async fn test_e2e_empty_graph_restoration() {
        // Verify behavior when graph has no sessions
        let graph = Arc::new(GraphStore::open_memory().unwrap());

        let sessions = restore_sessions_from_graph(&graph).await.unwrap();

        assert_eq!(sessions.len(), 0);
    }

    #[tokio::test]
    async fn test_e2e_single_session_restoration() {
        // Test simple case with one session
        let graph = Arc::new(GraphStore::open_memory().unwrap());

        let agent = GraphNode {
            id: NodeId("single-session".to_string()),
            node_type: NodeType::Agent(AgentData {
                name: "test-session".to_string(),
                model: "claude-opus-4".to_string(),
                status: "idle".to_string(),
                system_prompt: None,
            }),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: serde_json::json!({}),
        };

        graph.add_node(agent).unwrap();

        let sessions = restore_sessions_from_graph(&graph).await.unwrap();

        assert_eq!(sessions.len(), 1);
        assert!(sessions.contains_key("single-session"));

        let metadata = &sessions["single-session"];
        assert_eq!(metadata.session_id, "single-session");
        assert_eq!(metadata.name, "test-session");
        assert_eq!(metadata.model, "claude-opus-4");
        assert_eq!(metadata.status, SessionStatus::Idle);
    }

    #[tokio::test]
    async fn test_e2e_all_status_types_restoration() {
        // Verify all status mappings work correctly
        let graph = Arc::new(GraphStore::open_memory().unwrap());

        let status_mappings = vec![
            ("session-idle", "idle", SessionStatus::Idle),
            ("session-active", "active", SessionStatus::Running),
            ("session-completed", "completed", SessionStatus::Completed),
            ("session-failed", "failed", SessionStatus::Failed),
            ("session-unknown", "unknown_status", SessionStatus::Running), // Unknown maps to Running
        ];

        for (id, status_str, _expected_status) in &status_mappings {
            let agent = GraphNode {
                id: NodeId(id.to_string()),
                node_type: NodeType::Agent(AgentData {
                    name: id.to_string(),
                    model: "test-model".to_string(),
                    status: status_str.to_string(),
                    system_prompt: None,
                }),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                metadata: serde_json::json!({}),
            };

            graph.add_node(agent).unwrap();
        }

        let sessions = restore_sessions_from_graph(&graph).await.unwrap();

        assert_eq!(sessions.len(), 5);

        for (id, _status_str, expected_status) in &status_mappings {
            let session_id_str = id.to_string();
            let metadata = &sessions[&session_id_str];
            assert_eq!(
                metadata.status, *expected_status,
                "Status mismatch for session {id}"
            );
        }
    }
}
