use async_trait::async_trait;
use serde_json::json;

use crate::{Tool, ToolContext, ToolError, ToolOutput};

pub struct GraphQueryTool;

impl GraphQueryTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GraphQueryTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for GraphQueryTool {
    fn name(&self) -> &str {
        "graph_query"
    }

    fn description(&self) -> &str {
        "not implemented"
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "mode": { "type": "string" }
            },
            "required": ["mode"]
        })
    }

    async fn execute(
        &self,
        _args: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        Err(ToolError::ExecutionFailed("not implemented".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::make_test_context;
    use graphirm_graph::nodes::{
        GraphNode, InteractionData, KnowledgeData, NodeType, TaskData, TaskStatus,
    };
    use serde_json::json;

    #[tokio::test]
    async fn bfs_returns_traversed_nodes_from_seeded_graph() {
        let ctx = make_test_context();
        let graph = &ctx.graph;

        let root = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "root message".to_string(),
            token_count: None,
        }));
        let root_id = root.id.clone();
        let child = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".to_string(),
            content: "response".to_string(),
            token_count: None,
        }));
        let child_id = child.id.clone();

        graph.add_node(root).unwrap();
        graph.add_node(child).unwrap();
        graph
            .add_edge(graphirm_graph::edges::GraphEdge::new(
                graphirm_graph::edges::EdgeType::RespondsTo,
                child_id.clone(),
                root_id.clone(),
            ))
            .unwrap();

        let tool = GraphQueryTool::new();
        let result = tool
            .execute(
                json!({
                    "mode": "bfs",
                    "node_id": root_id.to_string(),
                    "depth": 2
                }),
                &ctx,
            )
            .await;

        // skeleton returns error — test will fail on this assertion once implemented
        assert!(result.is_ok(), "bfs should return Ok, got: {result:?}");
        let out = result.unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains(&child_id.to_string()));
    }

    #[tokio::test]
    async fn bfs_nonexistent_start_node_returns_error() {
        let ctx = make_test_context();
        let tool = GraphQueryTool::new();
        let result = tool
            .execute(
                json!({
                    "mode": "bfs",
                    "node_id": "00000000-0000-0000-0000-000000000000",
                    "depth": 1
                }),
                &ctx,
            )
            .await;
        assert!(result.is_err(), "bfs with nonexistent node should be Err");
    }

    #[tokio::test]
    async fn list_type_returns_filtered_task_nodes() {
        let ctx = make_test_context();
        let graph = &ctx.graph;

        let session_id = ctx.agent_id.to_string();

        let mut task_failed = GraphNode::new(NodeType::Task(TaskData {
            title: "Fix bug".to_string(),
            description: "Critical fix".to_string(),
            status: TaskStatus::Failed,
            priority: Some(1),
        }));
        task_failed.metadata["session_id"] = json!(session_id.clone());
        graph.add_node(task_failed).unwrap();

        let mut task_pending = GraphNode::new(NodeType::Task(TaskData {
            title: "New feature".to_string(),
            description: "Pending work".to_string(),
            status: TaskStatus::Pending,
            priority: None,
        }));
        task_pending.metadata["session_id"] = json!(session_id.clone());
        graph.add_node(task_pending).unwrap();

        let tool = GraphQueryTool::new();
        let result = tool
            .execute(
                json!({
                    "mode": "list_type",
                    "node_type": "task",
                    "session_id": session_id,
                    "metadata": {"status": "failed"},
                    "limit": 5
                }),
                &ctx,
            )
            .await;

        assert!(result.is_ok(), "list_type should return Ok, got: {result:?}");
        let out = result.unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("task") || out.content.contains("Fix bug"));
    }

    #[tokio::test]
    async fn search_returns_matching_knowledge_nodes() {
        let ctx = make_test_context();
        let graph = &ctx.graph;

        let k = GraphNode::new(NodeType::Knowledge(KnowledgeData {
            entity: "authenticate_user".to_string(),
            entity_type: "function".to_string(),
            summary: "Handles JWT authentication".to_string(),
            confidence: 0.9,
        }));
        graph.add_node(k).unwrap();

        let tool = GraphQueryTool::new();
        let result = tool
            .execute(
                json!({
                    "mode": "search",
                    "query": "auth"
                }),
                &ctx,
            )
            .await;

        assert!(result.is_ok(), "search should return Ok, got: {result:?}");
        let out = result.unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("authenticate_user") || out.content.contains("auth"));
    }

    #[tokio::test]
    async fn invalid_mode_returns_invalid_arguments_error() {
        let ctx = make_test_context();
        let tool = GraphQueryTool::new();
        let result = tool
            .execute(json!({"mode": "teleport"}), &ctx)
            .await;
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[tokio::test]
    async fn empty_search_returns_success_not_error() {
        let ctx = make_test_context();
        let tool = GraphQueryTool::new();
        let result = tool
            .execute(
                json!({
                    "mode": "search",
                    "query": "xyzzy_no_match_expected"
                }),
                &ctx,
            )
            .await;
        assert!(result.is_ok(), "empty search should be Ok, got: {result:?}");
        let out = result.unwrap();
        assert!(!out.is_error);
    }
}
