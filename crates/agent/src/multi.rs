// Multi-agent: coordinator pattern, subagent spawning, result aggregation

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use graphirm_graph::edges::{EdgeType, GraphEdge};
use graphirm_graph::nodes::{GraphNode, NodeId, NodeType, TaskData};
use graphirm_graph::{Direction, GraphStore};
use graphirm_llm::LlmProvider;
use graphirm_tools::registry::ToolRegistry;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::info;

use crate::config::{AgentConfig, AgentMode};
use crate::context::build_subagent_context;
use crate::error::AgentError;
use crate::event::EventBus;
use crate::session::Session;
use crate::workflow::run_agent_loop;

/// Registry of agent configurations loaded from TOML files.
pub struct AgentRegistry {
    configs: HashMap<String, AgentConfig>,
}

impl AgentRegistry {
    /// Load all `*.toml` agent config files from a directory.
    pub fn load_from_dir(path: &Path) -> Result<Self, AgentError> {
        let mut configs = HashMap::new();

        if !path.exists() {
            return Ok(Self { configs });
        }

        let entries = std::fs::read_dir(path).map_err(|e| {
            AgentError::Workflow(format!("Failed to read dir {}: {}", path.display(), e))
        })?;

        for entry in entries {
            let entry = entry
                .map_err(|e| AgentError::Workflow(format!("Failed to read entry: {}", e)))?;
            let file_path = entry.path();

            if file_path.extension().and_then(|e| e.to_str()) != Some("toml") {
                continue;
            }

            let config = AgentConfig::from_file(&file_path)?;
            configs.insert(config.name.clone(), config);
        }

        Ok(Self { configs })
    }

    /// Create a registry from a pre-built map (useful for testing).
    pub fn from_configs(configs: HashMap<String, AgentConfig>) -> Self {
        Self { configs }
    }

    /// Get an agent config by name.
    pub fn get(&self, name: &str) -> Option<&AgentConfig> {
        self.configs.get(name)
    }

    /// List all registered agent names.
    pub fn list(&self) -> Vec<&str> {
        self.configs.keys().map(|s| s.as_str()).collect()
    }

    /// Find the primary agent config (mode = "primary").
    pub fn primary(&self) -> Option<&AgentConfig> {
        self.configs.values().find(|c| c.mode == AgentMode::Primary)
    }

    /// Find all subagent configs.
    pub fn subagents(&self) -> Vec<&AgentConfig> {
        self.configs
            .values()
            .filter(|c| c.mode == AgentMode::Subagent)
            .collect()
    }
}

/// Factory function type: takes a model string, returns a boxed LlmProvider.
pub type LlmFactory = Arc<dyn Fn(&str) -> Box<dyn LlmProvider> + Send + Sync>;

/// Handle to a running subagent. Returned by `spawn_subagent()`.
#[derive(Debug)]
pub struct SubagentHandle {
    pub agent_id: NodeId,
    pub task_id: NodeId,
    pub name: String,
    pub join_handle: JoinHandle<Result<(), AgentError>>,
}

/// Build a scoped ToolRegistry for a subagent based on its permission config.
/// Tools explicitly denied are excluded. Unlisted tools are allowed.
fn build_scoped_tools(base_tools: &ToolRegistry, config: &AgentConfig) -> ToolRegistry {
    let mut scoped = ToolRegistry::new();
    for name in base_tools.list() {
        if config.is_tool_allowed(name) {
            if let Ok(tool) = base_tools.get(name) {
                scoped.register(tool);
            }
        }
    }
    scoped
}

/// Spawn a subagent to work on a specific task.
///
/// Creates the graph structure:
/// ```text
/// Parent Agent --DelegatesTo--> Task --SpawnedBy--> Subagent
/// ```
///
/// The subagent runs `run_agent_loop()` inside a `tokio::spawn` with:
/// - A scoped ToolRegistry (filtered by permissions)
/// - Context built from the task description + referenced nodes
/// - Its own Session (creating an Agent node in the graph)
///
/// Returns a `SubagentHandle` with the JoinHandle to await completion.
pub async fn spawn_subagent(
    graph: &Arc<GraphStore>,
    agents: &AgentRegistry,
    llm_factory: &LlmFactory,
    base_tools: &Arc<ToolRegistry>,
    events: &Arc<EventBus>,
    parent_agent_id: &NodeId,
    agent_name: &str,
    task_description: &str,
    context_nodes: Vec<NodeId>,
    cancel: CancellationToken,
) -> Result<SubagentHandle, AgentError> {
    // Look up agent config
    let agent_config = agents
        .get(agent_name)
        .ok_or_else(|| AgentError::AgentNotFound(agent_name.to_string()))?
        .clone();

    // Create Task node
    let task_node = GraphNode::new(NodeType::Task(TaskData {
        title: format!("Delegated to {}", agent_name),
        description: task_description.to_string(),
        status: "pending".to_string(),
        priority: None,
    }));
    let task_id = task_node.id.clone();
    graph.add_node(task_node)?;

    // DelegatesTo edge: parent agent → task
    graph.add_edge(GraphEdge::new(
        EdgeType::DelegatesTo,
        parent_agent_id.clone(),
        task_id.clone(),
    ))?;

    // Create subagent Session (creates Agent node)
    let session = Session::new(graph.clone(), agent_config.clone())?;
    let agent_id = session.id.clone();

    // SpawnedBy edge: task → subagent
    graph.add_edge(GraphEdge::new(
        EdgeType::SpawnedBy,
        task_id.clone(),
        agent_id.clone(),
    ))?;

    // Build scoped context: skip system message (Session handles that),
    // add user messages from task + context nodes
    let scoped_messages =
        build_subagent_context(graph, &agent_config, &task_id, &context_nodes)?;
    for msg in &scoped_messages {
        if msg.role != graphirm_llm::Role::System {
            let content_text = msg
                .content
                .iter()
                .filter_map(|p| match p {
                    graphirm_llm::ContentPart::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            session.add_user_message(&content_text)?;
        }
    }

    // Build scoped tools (filtered by permissions)
    let scoped_tools = build_scoped_tools(base_tools, &agent_config);

    // Create LLM provider via factory
    let llm = (llm_factory)(&agent_config.model);

    let events_clone = events.clone();
    let agent_name_owned = agent_name.to_string();
    let task_id_for_status = task_id.clone();
    let graph_for_status = graph.clone();

    info!(
        agent = %agent_name_owned,
        task_id = %task_id,
        agent_id = %agent_id,
        "Spawning subagent"
    );

    let join_handle = tokio::spawn(async move {
        let result =
            run_agent_loop(&session, llm.as_ref(), &scoped_tools, &events_clone, &cancel).await;

        // Update task status in graph
        let status = if result.is_ok() {
            "completed"
        } else {
            "failed"
        };
        if let Ok(mut task_node) = graph_for_status.get_node(&task_id_for_status) {
            if let NodeType::Task(ref mut data) = task_node.node_type {
                data.status = status.to_string();
            }
            let _ = graph_for_status.update_node(&task_id_for_status, task_node);
        }

        info!(agent = %agent_name_owned, status, "Subagent finished");
        result
    });

    Ok(SubagentHandle {
        agent_id,
        task_id,
        name: agent_name.to_string(),
        join_handle,
    })
}

/// Wait for all subagent handles to complete.
///
/// Returns a list of (task_id, agent_id) for each completed subagent.
/// Collects errors but continues waiting for all handles to prevent leaks.
/// Only returns an error if all subagents failed AND results are empty.
pub async fn wait_for_subagents(
    handles: Vec<SubagentHandle>,
) -> Result<Vec<(NodeId, NodeId)>, AgentError> {
    let mut results = Vec::new();
    let mut errors = Vec::new();

    for handle in handles {
        match handle.join_handle.await {
            Ok(Ok(())) => {
                info!(agent = %handle.name, "Subagent completed successfully");
                results.push((handle.task_id, handle.agent_id));
            }
            Ok(Err(e)) => {
                info!(agent = %handle.name, error = %e, "Subagent failed");
                errors.push(AgentError::SubagentFailed {
                    name: handle.name.clone(),
                    reason: e.to_string(),
                });
                results.push((handle.task_id, handle.agent_id));
            }
            Err(join_err) => {
                errors.push(AgentError::Join(format!(
                    "Subagent '{}' panicked: {}",
                    handle.name, join_err
                )));
            }
        }
    }

    if !errors.is_empty() && results.is_empty() {
        return Err(errors.remove(0));
    }

    Ok(results)
}

/// Wait for all tasks that a given task depends on (via DependsOn edges) to
/// reach "completed" status. Polls the graph every 100ms.
///
/// Used to enforce task ordering: if Task B DependsOn Task A, call
/// `wait_for_dependencies(graph, task_b_id)` before spawning Task B.
pub async fn wait_for_dependencies(
    graph: &GraphStore,
    task_id: &NodeId,
) -> Result<(), AgentError> {
    let deps = graph
        .neighbors(task_id, Some(EdgeType::DependsOn), Direction::Outgoing)
        .map_err(|e| AgentError::Context(format!("Failed to read dependencies: {}", e)))?;

    if deps.is_empty() {
        return Ok(());
    }

    let dep_ids: Vec<NodeId> = deps.iter().map(|n| n.id.clone()).collect();

    loop {
        let mut all_done = true;
        for dep_id in &dep_ids {
            let node = graph
                .get_node(dep_id)
                .map_err(|e| AgentError::Context(format!("Failed to read dep: {}", e)))?;
            if let NodeType::Task(data) = &node.node_type {
                if data.status != "completed" {
                    all_done = false;
                    break;
                }
            }
        }
        if all_done {
            return Ok(());
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}

/// Collect output produced by a subagent via graph traversal.
///
/// Starting from a Task node, follows SpawnedBy → Agent → Produces → outputs.
/// Returns text content from assistant responses, knowledge, and content nodes.
pub fn collect_subagent_results(
    graph: &GraphStore,
    task_id: &NodeId,
) -> Result<Vec<String>, AgentError> {
    // task --SpawnedBy--> agent
    let spawned = graph
        .neighbors(task_id, Some(EdgeType::SpawnedBy), Direction::Outgoing)
        .map_err(|e| AgentError::Context(e.to_string()))?;

    let mut results = Vec::new();

    for agent_node in &spawned {
        // agent --Produces--> interactions/content/knowledge
        let outputs = graph
            .neighbors(&agent_node.id, Some(EdgeType::Produces), Direction::Outgoing)
            .map_err(|e| AgentError::Context(e.to_string()))?;

        for node in &outputs {
            match &node.node_type {
                NodeType::Interaction(data) if data.role == "assistant" => {
                    if !data.content.is_empty() {
                        results.push(data.content.clone());
                    }
                }
                NodeType::Knowledge(data) => {
                    results.push(format!(
                        "[Knowledge] {} ({}): {}",
                        data.entity, data.entity_type, data.summary
                    ));
                }
                NodeType::Content(data) => {
                    let path = data.path.as_deref().unwrap_or("unknown");
                    let preview = if data.body.len() > 500 {
                        format!("{}...", &data.body[..500])
                    } else {
                        data.body.clone()
                    };
                    results.push(format!("[File: {}]\n{}", path, preview));
                }
                _ => {}
            }
        }
    }

    Ok(results)
}

/// Orchestrates multi-agent workflows.
///
/// The Coordinator owns the agent registry, LLM factory, shared graph, tools,
/// and event bus. It runs the primary agent loop and provides the plumbing for
/// subagent spawning via the `delegate` tool.
pub struct Coordinator {
    graph: Arc<GraphStore>,
    agents: AgentRegistry,
    llm_factory: LlmFactory,
    tools: Arc<ToolRegistry>,
    events: Arc<EventBus>,
}

impl Coordinator {
    pub fn new(
        graph: Arc<GraphStore>,
        agents: AgentRegistry,
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
    /// Returns the Session's agent NodeId.
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

        let llm = (self.llm_factory)(&primary_config.model);

        run_agent_loop(&session, llm.as_ref(), &self.tools, &self.events, &cancel).await?;

        Ok(session.id)
    }

    /// Access the agent registry (for SubagentTool).
    pub fn registry(&self) -> &AgentRegistry {
        &self.agents
    }

    /// Access the graph store.
    pub fn graph(&self) -> &Arc<GraphStore> {
        &self.graph
    }

    /// Access the LLM factory.
    pub fn llm_factory(&self) -> &LlmFactory {
        &self.llm_factory
    }

    /// Access the base tool registry.
    pub fn tools(&self) -> &Arc<ToolRegistry> {
        &self.tools
    }

    /// Access the event bus.
    pub fn events(&self) -> &Arc<EventBus> {
        &self.events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    use graphirm_graph::nodes::{AgentData, GraphNode, NodeType, TaskData};
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
system_prompt = "You explore code. Read files and report findings."
max_turns = 10
tools = ["read", "grep", "find", "ls"]

[permissions]
bash = "deny"
write = "deny"
edit = "deny"
"#
    }

    fn mock_factory_text(text: &str) -> LlmFactory {
        let text = text.to_string();
        Arc::new(move |_model: &str| -> Box<dyn LlmProvider> {
            Box::new(MockProvider::new(vec![MockResponse::text(text.clone())]))
        })
    }

    // ── AgentRegistry tests ──────────────────────────────────────────────────

    #[test]
    fn test_registry_load_from_dir() {
        let dir = TempDir::new().unwrap();
        write_toml(&dir, "build", build_toml());
        write_toml(&dir, "explore", explore_toml());

        let registry = AgentRegistry::load_from_dir(dir.path()).unwrap();
        assert_eq!(registry.list().len(), 2);
        assert!(registry.get("build").is_some());
        assert!(registry.get("explore").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_primary() {
        let dir = TempDir::new().unwrap();
        write_toml(&dir, "build", build_toml());
        write_toml(&dir, "explore", explore_toml());

        let registry = AgentRegistry::load_from_dir(dir.path()).unwrap();
        let primary = registry.primary().unwrap();
        assert_eq!(primary.name, "build");
        assert_eq!(primary.mode, AgentMode::Primary);
    }

    #[test]
    fn test_registry_list_names() {
        let dir = TempDir::new().unwrap();
        write_toml(&dir, "build", build_toml());
        write_toml(&dir, "explore", explore_toml());

        let registry = AgentRegistry::load_from_dir(dir.path()).unwrap();
        let mut names = registry.list();
        names.sort();
        assert_eq!(names, vec!["build", "explore"]);
    }

    #[test]
    fn test_registry_empty_dir() {
        let dir = TempDir::new().unwrap();
        let registry = AgentRegistry::load_from_dir(dir.path()).unwrap();
        assert!(registry.list().is_empty());
        assert!(registry.primary().is_none());
    }

    #[test]
    fn test_registry_skips_non_toml_files() {
        let dir = TempDir::new().unwrap();
        write_toml(&dir, "build", build_toml());
        std::fs::write(dir.path().join("README.md"), "# Agents").unwrap();

        let registry = AgentRegistry::load_from_dir(dir.path()).unwrap();
        assert_eq!(registry.list().len(), 1);
    }

    // ── spawn_subagent tests ─────────────────────────────────────────────────

    fn explore_config() -> AgentConfig {
        AgentConfig {
            name: "explore".to_string(),
            mode: AgentMode::Subagent,
            model: "test-model".to_string(),
            description: "Explore code".to_string(),
            system_prompt: "You explore code.".to_string(),
            max_turns: 5,
            ..AgentConfig::default()
        }
    }

    fn worker_config() -> AgentConfig {
        AgentConfig {
            name: "worker".to_string(),
            mode: AgentMode::Subagent,
            model: "test".to_string(),
            system_prompt: "Work.".to_string(),
            max_turns: 3,
            ..AgentConfig::default()
        }
    }

    #[tokio::test]
    async fn test_spawn_subagent_creates_graph_nodes() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let mut agents = HashMap::new();
        agents.insert("explore".to_string(), explore_config());
        let registry = AgentRegistry::from_configs(agents);
        let tools = Arc::new(ToolRegistry::new());
        let events = Arc::new(EventBus::new());
        let factory = mock_factory_text("I found some patterns.");

        let parent_agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "build".to_string(),
            model: "claude".to_string(),
            system_prompt: None,
            status: "running".to_string(),
        }));
        let parent_id = parent_agent.id.clone();
        graph.add_node(parent_agent).unwrap();

        let cancel = CancellationToken::new();

        let handle = spawn_subagent(
            &graph,
            &registry,
            &factory,
            &tools,
            &events,
            &parent_id,
            "explore",
            "Analyze the auth module",
            vec![],
            cancel,
        )
        .await
        .unwrap();

        // Wait for subagent to complete
        handle.join_handle.await.unwrap().unwrap();

        // Verify Task node was created
        let task_node = graph.get_node(&handle.task_id).unwrap();
        assert!(matches!(task_node.node_type, NodeType::Task(_)));

        // Verify Agent node was created for subagent
        let agent_node = graph.get_node(&handle.agent_id).unwrap();
        assert!(matches!(agent_node.node_type, NodeType::Agent(_)));

        // Verify DelegatesTo edge: parent → task
        let delegated = graph
            .neighbors(&parent_id, Some(EdgeType::DelegatesTo), Direction::Outgoing)
            .unwrap();
        assert_eq!(delegated.len(), 1);
        assert_eq!(delegated[0].id, handle.task_id);

        // Verify SpawnedBy edge: task → subagent
        let spawned = graph
            .neighbors(&handle.task_id, Some(EdgeType::SpawnedBy), Direction::Outgoing)
            .unwrap();
        assert_eq!(spawned.len(), 1);
        assert_eq!(spawned[0].id, handle.agent_id);
    }

    #[tokio::test]
    async fn test_spawn_subagent_unknown_agent() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let registry = AgentRegistry::from_configs(HashMap::new());
        let tools = Arc::new(ToolRegistry::new());
        let events = Arc::new(EventBus::new());
        let factory = mock_factory_text("unused");

        let parent_agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "build".to_string(),
            model: "claude".to_string(),
            system_prompt: None,
            status: "running".to_string(),
        }));
        let parent_id = parent_agent.id.clone();
        graph.add_node(parent_agent).unwrap();

        let cancel = CancellationToken::new();
        let result = spawn_subagent(
            &graph,
            &registry,
            &factory,
            &tools,
            &events,
            &parent_id,
            "nonexistent",
            "Do something",
            vec![],
            cancel,
        )
        .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AgentError::AgentNotFound(_)));
    }

    // ── wait_for_subagents tests ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_wait_for_subagents_all_succeed() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let mut agents = HashMap::new();
        agents.insert("explore".to_string(), explore_config());
        let registry = AgentRegistry::from_configs(agents);
        let tools = Arc::new(ToolRegistry::new());
        let events = Arc::new(EventBus::new());
        let factory = mock_factory_text("Done exploring.");

        let parent_agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "build".to_string(),
            model: "claude".to_string(),
            system_prompt: None,
            status: "running".to_string(),
        }));
        let parent_id = parent_agent.id.clone();
        graph.add_node(parent_agent).unwrap();

        let cancel = CancellationToken::new();

        let h1 = spawn_subagent(
            &graph,
            &registry,
            &factory,
            &tools,
            &events,
            &parent_id,
            "explore",
            "Task A",
            vec![],
            cancel.clone(),
        )
        .await
        .unwrap();

        let h2 = spawn_subagent(
            &graph,
            &registry,
            &factory,
            &tools,
            &events,
            &parent_id,
            "explore",
            "Task B",
            vec![],
            cancel.clone(),
        )
        .await
        .unwrap();

        let results = wait_for_subagents(vec![h1, h2]).await.unwrap();
        assert_eq!(results.len(), 2);

        for (task_id, _agent_id) in &results {
            let task_node = graph.get_node(task_id).unwrap();
            if let NodeType::Task(data) = &task_node.node_type {
                assert_eq!(data.status, "completed");
            }
        }
    }

    // ── Coordinator tests ────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_coordinator_run_primary_simple() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let dir = TempDir::new().unwrap();
        write_toml(&dir, "build", build_toml());
        let registry = AgentRegistry::load_from_dir(dir.path()).unwrap();
        let tools = Arc::new(ToolRegistry::new());
        let events = Arc::new(EventBus::new());
        let factory = mock_factory_text("Hello from the primary agent!");

        let coordinator = Coordinator::new(graph.clone(), registry, factory, tools, events);

        let cancel = CancellationToken::new();
        let session_id = coordinator
            .run_primary("What is 2+2?", cancel)
            .await
            .unwrap();

        // Verify session created an agent node
        let agent_node = graph.get_node(&session_id).unwrap();
        assert!(matches!(agent_node.node_type, NodeType::Agent(_)));

        // Verify the user message was recorded
        let neighbors = graph
            .neighbors(&session_id, Some(EdgeType::Produces), Direction::Outgoing)
            .unwrap();
        assert!(neighbors.len() >= 2); // user msg + assistant response
    }

    #[tokio::test]
    async fn test_coordinator_run_primary_no_primary_agent() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let dir = TempDir::new().unwrap();
        write_toml(&dir, "explore", explore_toml()); // only subagent, no primary
        let registry = AgentRegistry::load_from_dir(dir.path()).unwrap();
        let tools = Arc::new(ToolRegistry::new());
        let events = Arc::new(EventBus::new());
        let factory = mock_factory_text("unused");

        let coordinator = Coordinator::new(graph, registry, factory, tools, events);

        let cancel = CancellationToken::new();
        let result = coordinator.run_primary("Hello", cancel).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AgentError::Workflow(_)));
    }

    // ── wait_for_dependencies tests ──────────────────────────────────────────

    #[tokio::test]
    async fn test_task_dependency_waits_for_prerequisite() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let mut agents = HashMap::new();
        agents.insert("worker".to_string(), worker_config());
        let registry = AgentRegistry::from_configs(agents);
        let tools = Arc::new(ToolRegistry::new());
        let events = Arc::new(EventBus::new());

        let call_order = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));

        let order_a = call_order.clone();
        let factory_a: LlmFactory = Arc::new(move |_| {
            order_a.lock().unwrap().push("A".to_string());
            Box::new(MockProvider::new(vec![MockResponse::text("A done")]))
        });

        let order_b = call_order.clone();
        let factory_b: LlmFactory = Arc::new(move |_| {
            order_b.lock().unwrap().push("B".to_string());
            Box::new(MockProvider::new(vec![MockResponse::text("B done")]))
        });

        let parent_agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "build".to_string(),
            model: "claude".to_string(),
            system_prompt: None,
            status: "running".to_string(),
        }));
        let parent_id = parent_agent.id.clone();
        graph.add_node(parent_agent).unwrap();

        let cancel = CancellationToken::new();

        // Spawn Task A
        let handle_a = spawn_subagent(
            &graph,
            &registry,
            &factory_a,
            &tools,
            &events,
            &parent_id,
            "worker",
            "Task A",
            vec![],
            cancel.clone(),
        )
        .await
        .unwrap();

        let task_a_id = handle_a.task_id.clone();

        // Create Task B that depends on Task A
        let task_b_node = GraphNode::new(NodeType::Task(TaskData {
            title: "Task B".to_string(),
            description: "Depends on Task A".to_string(),
            status: "pending".to_string(),
            priority: None,
        }));
        let task_b_id = task_b_node.id.clone();
        graph.add_node(task_b_node).unwrap();

        // Add DependsOn edge: Task B --DependsOn--> Task A
        graph
            .add_edge(GraphEdge::new(
                EdgeType::DependsOn,
                task_b_id.clone(),
                task_a_id.clone(),
            ))
            .unwrap();

        // Wait for A to finish before spawning B
        wait_for_subagents(vec![handle_a]).await.unwrap();
        wait_for_dependencies(&graph, &task_b_id).await.unwrap();

        // Now spawn Task B
        let handle_b = spawn_subagent(
            &graph,
            &registry,
            &factory_b,
            &tools,
            &events,
            &parent_id,
            "worker",
            "Task B",
            vec![],
            cancel.clone(),
        )
        .await
        .unwrap();

        wait_for_subagents(vec![handle_b]).await.unwrap();

        let order = call_order.lock().unwrap();
        assert!(order.len() >= 2);
        let a_idx = order.iter().position(|x| x == "A").unwrap();
        let b_idx = order.iter().position(|x| x == "B").unwrap();
        assert!(a_idx < b_idx, "Task A should run before Task B");
    }

    // ── Parallel subagent execution test ─────────────────────────────────────

    #[tokio::test]
    async fn test_parallel_subagent_execution() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let mut agents = HashMap::new();
        agents.insert("explore".to_string(), explore_config());
        let registry = AgentRegistry::from_configs(agents);
        let tools = Arc::new(ToolRegistry::new());
        let events = Arc::new(EventBus::new());
        let factory: LlmFactory = Arc::new(move |_| {
            Box::new(MockProvider::new(vec![MockResponse::text("done")]))
        });

        let parent_agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "build".to_string(),
            model: "claude".to_string(),
            system_prompt: None,
            status: "running".to_string(),
        }));
        let parent_id = parent_agent.id.clone();
        graph.add_node(parent_agent).unwrap();

        let cancel = CancellationToken::new();

        // Spawn 3 subagents concurrently
        let mut handles = Vec::new();
        for i in 0..3 {
            let h = spawn_subagent(
                &graph,
                &registry,
                &factory,
                &tools,
                &events,
                &parent_id,
                "explore",
                &format!("Task {}", i),
                vec![],
                cancel.clone(),
            )
            .await
            .unwrap();
            handles.push(h);
        }

        let results = wait_for_subagents(handles).await.unwrap();
        assert_eq!(results.len(), 3);

        // Verify all 3 tasks are in the graph
        let delegated = graph
            .neighbors(&parent_id, Some(EdgeType::DelegatesTo), Direction::Outgoing)
            .unwrap();
        assert_eq!(delegated.len(), 3);

        // Verify each task is completed
        for task_node in &delegated {
            if let NodeType::Task(data) = &task_node.node_type {
                assert_eq!(data.status, "completed");
            }
        }
    }

    // ── collect_subagent_results tests ───────────────────────────────────────

    #[tokio::test]
    async fn test_collect_subagent_results_from_graph() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let mut agents = HashMap::new();
        agents.insert("explore".to_string(), explore_config());
        let registry = AgentRegistry::from_configs(agents);
        let tools = Arc::new(ToolRegistry::new());
        let events = Arc::new(EventBus::new());
        let factory = mock_factory_text("Found: auth uses JWT tokens with 24h expiry.");

        let parent_agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "build".to_string(),
            model: "claude".to_string(),
            system_prompt: None,
            status: "running".to_string(),
        }));
        let parent_id = parent_agent.id.clone();
        graph.add_node(parent_agent).unwrap();

        let cancel = CancellationToken::new();

        let handle = spawn_subagent(
            &graph,
            &registry,
            &factory,
            &tools,
            &events,
            &parent_id,
            "explore",
            "Analyze auth tokens",
            vec![],
            cancel,
        )
        .await
        .unwrap();

        let task_id = handle.task_id.clone();
        wait_for_subagents(vec![handle]).await.unwrap();

        let results = collect_subagent_results(&graph, &task_id).unwrap();

        assert!(!results.is_empty());
        let has_jwt_content = results.iter().any(|r| r.contains("JWT"));
        assert!(has_jwt_content, "Results should contain subagent output: {:?}", results);
    }
}
