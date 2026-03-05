pub mod bash;
pub mod edit;
pub mod error;
pub mod executor;
pub mod find;
pub mod grep;
pub mod ls;
pub mod permissions;
pub mod read;
pub mod registry;
pub mod write;

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

pub use error::ToolError;
use graphirm_graph::GraphStore;
use graphirm_graph::nodes::NodeId;
pub use registry::ToolRegistry;

/// Context passed to every tool execution.
#[derive(Clone)]
pub struct ToolContext {
    pub graph: Arc<GraphStore>,
    pub agent_id: NodeId,
    pub interaction_id: NodeId,
    pub working_dir: PathBuf,
    pub signal: CancellationToken,
}

/// Result of a tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
    pub node_id: Option<NodeId>,
}

impl ToolOutput {
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
            node_id: None,
        }
    }

    pub fn success_with_node(content: impl Into<String>, node_id: NodeId) -> Self {
        Self {
            content: content.into(),
            is_error: false,
            node_id: Some(node_id),
        }
    }

    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
            node_id: None,
        }
    }
}

/// JSON Schema definition sent to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// A tool call request from the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Every tool implements this trait.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> serde_json::Value;

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError>;

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: self.parameters(),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use async_trait::async_trait;
    use graphirm_graph::nodes::{AgentData, GraphNode, InteractionData, NodeType};
    use serde_json::json;

    pub fn make_test_context() -> ToolContext {
        let graph = Arc::new(GraphStore::open_memory().expect("memory graph"));
        let agent_node = GraphNode::new(NodeType::Agent(AgentData {
            name: "test-agent".to_string(),
            model: "test".to_string(),
            system_prompt: None,
            status: "active".to_string(),
        }));
        let interaction_node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "test".to_string(),
            token_count: None,
        }));
        let agent_id = graph.add_node(agent_node).expect("agent node");
        let interaction_id = graph.add_node(interaction_node).expect("interaction node");
        ToolContext {
            graph,
            agent_id,
            interaction_id,
            working_dir: PathBuf::from("/tmp"),
            signal: CancellationToken::new(),
        }
    }

    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "Echoes input back"
        }
        fn parameters(&self) -> serde_json::Value {
            json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string", "description": "Message to echo" }
                },
                "required": ["message"]
            })
        }
        async fn execute(
            &self,
            args: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, ToolError> {
            let message = args["message"]
                .as_str()
                .ok_or_else(|| ToolError::InvalidArguments("missing 'message'".into()))?;
            Ok(ToolOutput::success(message))
        }
    }

    #[test]
    fn tool_definition() {
        let tool = EchoTool;
        let def = tool.definition();
        assert_eq!(def.name, "echo");
        assert_eq!(def.description, "Echoes input back");
        assert!(def.parameters["properties"]["message"].is_object());
    }

    #[tokio::test]
    async fn tool_execute_success() {
        let tool = EchoTool;
        let ctx = make_test_context();
        let result = tool
            .execute(json!({"message": "hello"}), &ctx)
            .await
            .unwrap();
        assert_eq!(result.content, "hello");
        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn tool_execute_invalid_args() {
        let tool = EchoTool;
        let ctx = make_test_context();
        let err = tool.execute(json!({}), &ctx).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[test]
    fn tool_output_constructors() {
        let ok = ToolOutput::success("done");
        assert_eq!(ok.content, "done");
        assert!(!ok.is_error);
        assert!(ok.node_id.is_none());

        let ok_node = ToolOutput::success_with_node("done", NodeId::new());
        assert!(ok_node.node_id.is_some());

        let err = ToolOutput::error("failed");
        assert_eq!(err.content, "failed");
        assert!(err.is_error);
    }
}
