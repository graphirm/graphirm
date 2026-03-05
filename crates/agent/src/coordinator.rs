// Coordinator: orchestrates the primary agent with auto-injected delegate tool.
//
// Lives in its own module to avoid a circular import: multi.rs ← delegate.rs ← multi.rs.
// coordinator.rs is the only file that imports from both.

use std::sync::Arc;

use graphirm_graph::nodes::NodeId;
use graphirm_graph::GraphStore;
use graphirm_tools::registry::ToolRegistry;
use tokio_util::sync::CancellationToken;

use crate::delegate::SubagentTool;
use crate::error::AgentError;
use crate::event::EventBus;
use crate::multi::{AgentRegistry, LlmFactory};
use crate::session::Session;
use crate::workflow::run_agent_loop;

/// Orchestrates multi-agent workflows.
///
/// Owns the agent registry, LLM factory, shared graph, base tools, and event bus.
/// `run_primary` automatically injects the `delegate` tool into the primary agent's
/// tool registry so the LLM can call it to spawn subagents. Cancellation propagates
/// from primary to all spawned subagents via child tokens.
pub struct Coordinator {
    graph: Arc<GraphStore>,
    agents: Arc<AgentRegistry>,
    llm_factory: LlmFactory,
    tools: Arc<ToolRegistry>,
    events: Arc<EventBus>,
}

impl Coordinator {
    pub fn new(
        graph: Arc<GraphStore>,
        agents: Arc<AgentRegistry>,
        llm_factory: LlmFactory,
        tools: Arc<ToolRegistry>,
        events: Arc<EventBus>,
    ) -> Self {
        Self {
            graph,
            agents,
            llm_factory,
            tools,
            events,
        }
    }

    /// Run the primary agent with the given user prompt.
    ///
    /// Automatically registers the `delegate` tool in the primary agent's tool
    /// registry. The delegate tool uses a child cancellation token so that
    /// cancelling `cancel` also cancels all subagents it spawns.
    ///
    /// Returns the Session's agent `NodeId`.
    pub async fn run_primary(
        &self,
        prompt: &str,
        cancel: CancellationToken,
    ) -> Result<NodeId, AgentError> {
        let primary_config = self
            .agents
            .primary()
            .ok_or_else(|| {
                AgentError::Workflow("No primary agent configured in registry".to_string())
            })?
            .clone();

        let session = Session::new(self.graph.clone(), primary_config.clone())?;
        session.add_user_message(prompt)?;

        // Build primary tool registry: base tools + delegate tool with child cancel token.
        // The child token propagates cancellation: cancelling `cancel` also cancels subagents.
        let delegate = SubagentTool::new(
            self.graph.clone(),
            self.agents.clone(),
            self.llm_factory.clone(),
            self.tools.clone(),
            self.events.clone(),
            session.id.clone(),
            cancel.child_token(),
        );

        let mut primary_tools = ToolRegistry::new();
        for name in self.tools.list() {
            if let Ok(tool) = self.tools.get(name) {
                primary_tools.register(tool);
            }
        }
        primary_tools.register(Arc::new(delegate));

        let llm = (self.llm_factory)(&primary_config.model);
        run_agent_loop(&session, llm.as_ref(), &primary_tools, &self.events, &cancel).await?;

        Ok(session.id)
    }

    pub fn registry(&self) -> &Arc<AgentRegistry> {
        &self.agents
    }

    pub fn graph(&self) -> &Arc<GraphStore> {
        &self.graph
    }

    pub fn llm_factory(&self) -> &LlmFactory {
        &self.llm_factory
    }

    pub fn tools(&self) -> &Arc<ToolRegistry> {
        &self.tools
    }

    pub fn events(&self) -> &Arc<EventBus> {
        &self.events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    use graphirm_graph::edges::EdgeType;
    use graphirm_graph::nodes::NodeType;
    use graphirm_graph::{Direction, GraphStore};
    use graphirm_llm::{MockProvider, MockResponse};

    fn write_toml(dir: &TempDir, name: &str, content: &str) -> std::path::PathBuf {
        let path = dir.path().join(format!("{}.toml", name));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    fn build_toml() -> &'static str {
        r#"
[agent]
name = "build"
mode = "primary"
model = "anthropic/claude-sonnet-4"
description = "Default agent with full tool access"
system_prompt = "You are a coding assistant."
max_turns = 50
tools = ["bash", "read", "write", "edit"]

[permissions]
bash = "allow"
write = "allow"
edit = "allow"
"#
    }

    fn explore_toml() -> &'static str {
        r#"
[agent]
name = "explore"
mode = "subagent"
model = "anthropic/claude-haiku-4"
description = "Fast, read-only codebase exploration"
system_prompt = "You explore code."
max_turns = 10
tools = ["read"]

[permissions]
bash = "deny"
write = "deny"
edit = "deny"
"#
    }

    fn mock_factory_text(text: &str) -> LlmFactory {
        let text = text.to_string();
        Arc::new(move |_model: &str| -> Box<dyn graphirm_llm::LlmProvider> {
            Box::new(MockProvider::new(vec![MockResponse::text(text.clone())]))
        })
    }

    #[tokio::test]
    async fn test_coordinator_run_primary_simple() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let dir = TempDir::new().unwrap();
        write_toml(&dir, "build", build_toml());
        let registry = Arc::new(AgentRegistry::load_from_dir(dir.path()).unwrap());
        let tools = Arc::new(ToolRegistry::new());
        let events = Arc::new(EventBus::new());
        let factory = mock_factory_text("Hello from the primary agent!");

        let coordinator = Coordinator::new(graph.clone(), registry, factory, tools, events);

        let cancel = CancellationToken::new();
        let session_id = coordinator
            .run_primary("What is 2+2?", cancel)
            .await
            .unwrap();

        let agent_node = graph.get_node(&session_id).unwrap();
        assert!(matches!(agent_node.node_type, NodeType::Agent(_)));

        let neighbors = graph
            .neighbors(&session_id, Some(EdgeType::Produces), Direction::Outgoing)
            .unwrap();
        assert!(neighbors.len() >= 2, "Expected user msg + assistant response");
    }

    #[tokio::test]
    async fn test_coordinator_run_primary_no_primary_agent() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let dir = TempDir::new().unwrap();
        write_toml(&dir, "explore", explore_toml());
        let registry = Arc::new(AgentRegistry::load_from_dir(dir.path()).unwrap());
        let tools = Arc::new(ToolRegistry::new());
        let events = Arc::new(EventBus::new());
        let factory = mock_factory_text("unused");

        let coordinator = Coordinator::new(graph, registry, factory, tools, events);

        let cancel = CancellationToken::new();
        let result = coordinator.run_primary("Hello", cancel).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AgentError::Workflow(_)));
    }

    #[tokio::test]
    async fn test_coordinator_injects_delegate_tool() {
        use graphirm_llm::LlmProvider;

        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let dir = TempDir::new().unwrap();
        write_toml(&dir, "build", build_toml());
        write_toml(&dir, "explore", explore_toml());
        let registry = Arc::new(AgentRegistry::load_from_dir(dir.path()).unwrap());
        let tools = Arc::new(ToolRegistry::new());
        let events = Arc::new(EventBus::new());

        // Primary calls delegate, then gives a final answer
        let factory: LlmFactory = Arc::new(|model: &str| -> Box<dyn LlmProvider> {
            if model.contains("sonnet") {
                Box::new(MockProvider::new(vec![
                    MockResponse::tool_call(
                        "tc1",
                        "delegate",
                        serde_json::json!({
                            "agent": "explore",
                            "task": "Explore the codebase"
                        }),
                    ),
                    MockResponse::text("Exploration done. Here is the summary."),
                ]))
            } else {
                Box::new(MockProvider::new(vec![MockResponse::text("done")]))
            }
        });

        let coordinator = Coordinator::new(graph.clone(), registry, factory, tools, events);
        let cancel = CancellationToken::new();
        let session_id = coordinator
            .run_primary("Review the codebase.", cancel)
            .await
            .unwrap();

        // Primary agent should have delegated at least one task
        let delegated = graph
            .neighbors(&session_id, Some(EdgeType::DelegatesTo), Direction::Outgoing)
            .unwrap();
        assert_eq!(
            delegated.len(),
            1,
            "Primary should have delegated exactly 1 task via Coordinator"
        );
    }
}
