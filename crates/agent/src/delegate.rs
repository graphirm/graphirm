// SubagentTool: the `delegate` tool that allows the primary agent to spawn subagents

use std::sync::Arc;

use async_trait::async_trait;
use graphirm_graph::nodes::NodeId;
use graphirm_graph::GraphStore;
use graphirm_tools::registry::ToolRegistry;
use graphirm_tools::{Tool, ToolContext, ToolError, ToolOutput};
use tokio_util::sync::CancellationToken;

use crate::event::EventBus;
use crate::multi::{spawn_subagent, wait_for_subagents, AgentRegistry, LlmFactory};

/// Tool that allows the primary agent to delegate work to a subagent.
///
/// When the primary agent calls `delegate(agent="explore", task="...")`,
/// this tool:
/// 1. Looks up the agent config from the registry
/// 2. Creates Task + Agent nodes with DelegatesTo/SpawnedBy edges
/// 3. Runs the subagent's agent loop
/// 4. Waits for completion
/// 5. Returns a summary of what the subagent produced
///
/// Subagents receive a **child** cancellation token derived from the parent's token,
/// so cancelling the parent automatically cancels all subagents it spawned.
pub struct SubagentTool {
    graph: Arc<GraphStore>,
    agents: Arc<AgentRegistry>,
    llm_factory: LlmFactory,
    base_tools: Arc<ToolRegistry>,
    events: Arc<EventBus>,
    parent_agent_id: NodeId,
    /// Parent agent's cancel token. A child token is passed to each spawned subagent
    /// so that cancelling the parent also cancels its children.
    cancel: CancellationToken,
}

impl SubagentTool {
    pub fn new(
        graph: Arc<GraphStore>,
        agents: Arc<AgentRegistry>,
        llm_factory: LlmFactory,
        base_tools: Arc<ToolRegistry>,
        events: Arc<EventBus>,
        parent_agent_id: NodeId,
        cancel: CancellationToken,
    ) -> Self {
        Self {
            graph,
            agents,
            llm_factory,
            base_tools,
            events,
            parent_agent_id,
            cancel,
        }
    }
}

#[async_trait]
impl Tool for SubagentTool {
    fn name(&self) -> &str {
        "delegate"
    }

    fn description(&self) -> &str {
        "Delegate a task to a specialized subagent. The subagent runs independently \
         with scoped tools and context, writes results to the graph, and returns a \
         summary when complete."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "Name of the subagent to delegate to (e.g. 'explore')"
                },
                "task": {
                    "type": "string",
                    "description": "Description of the task for the subagent to perform"
                },
                "context_nodes": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional list of node IDs to include as context for the subagent"
                }
            },
            "required": ["agent", "task"]
        })
    }

    async fn execute(&self, args: serde_json::Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let agent_name = args["agent"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("Missing 'agent' field".to_string()))?;

        let task_description = args["task"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("Missing 'task' field".to_string()))?;

        let context_nodes: Vec<NodeId> = args["context_nodes"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(NodeId::from))
                    .collect()
            })
            .unwrap_or_default();

        // Child token: cancelling the parent agent also cancels this subagent.
        let cancel = self.cancel.child_token();

        let handle = spawn_subagent(
            &self.graph,
            &self.agents,
            &self.llm_factory,
            &self.base_tools,
            &self.events,
            &self.parent_agent_id,
            agent_name,
            task_description,
            context_nodes,
            cancel,
        )
        .await
        .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        let task_id = handle.task_id.clone();
        let agent_id = handle.agent_id.clone();
        let name = handle.name.clone();

        wait_for_subagents(vec![handle])
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        let output = self.collect_subagent_output(&agent_id)?;

        let summary = format!(
            "Subagent '{}' completed.\nTask: {}\nAgent ID: {}\nTask ID: {}\n\nOutput:\n{}",
            name, task_description, agent_id, task_id, output
        );

        Ok(ToolOutput::success(summary))
    }
}

impl SubagentTool {
    /// Read the subagent's final assistant responses from the graph.
    fn collect_subagent_output(&self, agent_id: &NodeId) -> Result<String, ToolError> {
        use graphirm_graph::edges::EdgeType;
        use graphirm_graph::Direction;

        let neighbors = self
            .graph
            .neighbors(agent_id, Some(EdgeType::Produces), Direction::Outgoing)
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        let mut outputs = Vec::new();
        for node in &neighbors {
            match &node.node_type {
                graphirm_graph::nodes::NodeType::Interaction(data) => {
                    if data.role == "assistant" && !data.content.is_empty() {
                        outputs.push(data.content.clone());
                    }
                }
                graphirm_graph::nodes::NodeType::Knowledge(data) => {
                    outputs.push(format!("[Knowledge] {}: {}", data.entity, data.summary));
                }
                graphirm_graph::nodes::NodeType::Content(data) => {
                    let preview_len = data.body.len().min(200);
                    outputs.push(format!(
                        "[Content] {}: {}",
                        data.path.as_deref().unwrap_or("unknown"),
                        &data.body[..preview_len]
                    ));
                }
                _ => {}
            }
        }

        if outputs.is_empty() {
            Ok("(no output produced)".to_string())
        } else {
            Ok(outputs.join("\n\n"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AgentConfig, AgentMode};
    use crate::event::EventBus;
    use crate::multi::AgentRegistry;
    use graphirm_graph::edges::EdgeType;
    use graphirm_graph::nodes::{AgentData, GraphNode, NodeType};
    use graphirm_graph::{Direction, GraphStore};
    use graphirm_llm::{MockProvider, MockResponse};
    use graphirm_tools::registry::ToolRegistry;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn make_test_context(graph: &Arc<GraphStore>) -> ToolContext {
        use graphirm_graph::nodes::{AgentData, GraphNode, InteractionData, NodeType};
        let agent_node = GraphNode::new(NodeType::Agent(AgentData {
            name: "test".to_string(),
            model: "test".to_string(),
            system_prompt: None,
            status: "active".to_string(),
        }));
        let interaction_node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "test".to_string(),
            token_count: None,
        }));
        let agent_id = graph.add_node(agent_node).unwrap();
        let interaction_id = graph.add_node(interaction_node).unwrap();
        ToolContext {
            graph: graph.clone(),
            agent_id,
            interaction_id,
            working_dir: PathBuf::from("."),
            signal: CancellationToken::new(),
        }
    }

    fn mock_factory() -> LlmFactory {
        Arc::new(|_model: &str| -> Box<dyn graphirm_llm::LlmProvider> {
            Box::new(MockProvider::new(vec![MockResponse::text(
                "Exploration complete. Found auth patterns using JWT.",
            )]))
        })
    }

    fn test_registry() -> AgentRegistry {
        let mut configs = HashMap::new();
        configs.insert(
            "explore".to_string(),
            AgentConfig {
                name: "explore".to_string(),
                mode: AgentMode::Subagent,
                model: "test-model".to_string(),
                description: "Fast code exploration".to_string(),
                system_prompt: "You explore code.".to_string(),
                max_turns: 5,
                ..AgentConfig::default()
            },
        );
        AgentRegistry::from_configs(configs).unwrap()
    }

    #[test]
    fn test_delegate_tool_name_and_schema() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let registry = Arc::new(test_registry());
        let tools = Arc::new(ToolRegistry::new());
        let events = Arc::new(EventBus::new());
        let factory = mock_factory();
        let parent_id = NodeId::from("parent-1");

        let delegate = SubagentTool::new(
            graph,
            registry,
            factory,
            tools,
            events,
            parent_id,
            CancellationToken::new(),
        );

        assert_eq!(delegate.name(), "delegate");
        assert!(delegate.description().contains("subagent"));

        let schema = delegate.parameters();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["agent"].is_object());
        assert!(schema["properties"]["task"].is_object());
        assert_eq!(schema["required"], serde_json::json!(["agent", "task"]));
    }

    #[tokio::test]
    async fn test_delegate_tool_execute() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let registry = Arc::new(test_registry());
        let tools = Arc::new(ToolRegistry::new());
        let events = Arc::new(EventBus::new());
        let factory = mock_factory();

        let parent_agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "build".to_string(),
            model: "claude".to_string(),
            system_prompt: None,
            status: "running".to_string(),
        }));
        let parent_id = parent_agent.id.clone();
        graph.add_node(parent_agent).unwrap();

        let delegate = SubagentTool::new(
            graph.clone(),
            registry,
            factory,
            tools,
            events,
            parent_id.clone(),
            CancellationToken::new(),
        );

        let args = serde_json::json!({
            "agent": "explore",
            "task": "Analyze authentication patterns in src/auth/"
        });

        let ctx = make_test_context(&graph);
        let result = delegate.execute(args, &ctx).await.unwrap();

        assert!(result.content.contains("explore"));
        assert!(result.content.contains("completed"));

        // Verify graph structure: parent --DelegatesTo--> task
        let delegated = graph
            .neighbors(&parent_id, Some(EdgeType::DelegatesTo), Direction::Outgoing)
            .unwrap();
        assert_eq!(delegated.len(), 1);
    }

    #[tokio::test]
    async fn test_delegate_tool_unknown_agent() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let registry = Arc::new(test_registry());
        let tools = Arc::new(ToolRegistry::new());
        let events = Arc::new(EventBus::new());
        let factory = mock_factory();
        let parent_id = NodeId::from("parent-1");

        let delegate = SubagentTool::new(
            graph.clone(),
            registry,
            factory,
            tools,
            events,
            parent_id,
            CancellationToken::new(),
        );

        let args = serde_json::json!({
            "agent": "nonexistent",
            "task": "Do something"
        });

        let ctx = make_test_context(&graph);
        let result = delegate.execute(args, &ctx).await;
        assert!(result.is_err());
    }
}
