// Session management: create, resume, list, archive sessions

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use graphirm_graph::GraphStore;
use graphirm_graph::edges::{EdgeType, GraphEdge};
use graphirm_graph::nodes::{AgentData, GraphNode, InteractionData, NodeId, NodeType};

use crate::config::AgentConfig;
use crate::error::AgentError;
use crate::hitl::HitlGate;
use crate::knowledge::memory::MemoryRetriever;

/// Graph store for a new session.
///
/// When `fresh_graph_per_session` is false, this returns the shared store.
/// When it is true, this opens a new empty SQLite file and does not copy nodes.
pub fn open_session_graph(
    shared: Arc<GraphStore>,
    config: &AgentConfig,
) -> Result<Arc<GraphStore>, AgentError> {
    if !config.fresh_graph_per_session {
        return Ok(shared);
    }
    let dir = config
        .fresh_graph_dir
        .clone()
        .unwrap_or_else(|| std::env::temp_dir().join("graphirm-fresh"));
    std::fs::create_dir_all(&dir)
        .map_err(|err| AgentError::Workflow(format!("fresh graph dir: {err}")))?;
    let path = dir.join(format!("{}.db", NodeId::new()));
    let path_str = path
        .to_str()
        .ok_or_else(|| AgentError::Workflow("fresh graph path is not utf-8".to_string()))?;
    Ok(Arc::new(GraphStore::open(path_str)?))
}

/// Lifecycle status of a restored or active session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionStatus {
    Idle,
    Running,
    Completed,
    Failed,
}

/// Lightweight metadata for a session, used during restoration on startup.
#[derive(Debug, Clone)]
pub struct SessionMetadata {
    pub session_id: String,
    pub name: String,
    pub model: String,
    pub created_at: DateTime<Utc>,
    pub status: SessionStatus,
    /// Workspace name stored in the Agent node's metadata (e.g. `"myapp"`).
    pub workspace: Option<String>,
    /// Fully-resolved workspace path (`workspaces_root/workspace`).
    /// `None` when no workspaces root was provided at restore time.
    pub workspace_path: Option<std::path::PathBuf>,
}

impl SessionMetadata {
    pub fn from_agent_node_id(
        session_id: String,
        name: String,
        model: String,
        created_at: DateTime<Utc>,
        status: SessionStatus,
    ) -> Self {
        Self {
            session_id,
            name,
            model,
            created_at,
            status,
            workspace: None,
            workspace_path: None,
        }
    }
}

pub struct Session {
    pub id: NodeId,
    pub agent_config: AgentConfig,
    pub graph: Arc<GraphStore>,
    pub created_at: DateTime<Utc>,
    /// Optional human-in-the-loop gate. When `Some`, the agent pauses before
    /// destructive tool calls and at manual pause points.
    pub hitl: Option<Arc<HitlGate>>,
    /// Optional cross-session memory retriever. When `Some`, knowledge nodes
    /// are embedded after each turn and injected into new sessions.
    memory_retriever: Option<Arc<MemoryRetriever>>,
    /// Runtime suffix appended to the system prompt containing relevant
    /// knowledge retrieved from past sessions via HNSW vector search.
    runtime_system_suffix: tokio::sync::Mutex<String>,
    /// ID of the most recent Interaction node in this session's conversation
    /// chain. Used to create `RespondsTo` edges that link messages into a
    /// traversable DAG (rather than a flat star off the agent node).
    /// Arc-wrapped so the NodeId can be cloned into `spawn_blocking` closures.
    last_interaction_id: Arc<Mutex<Option<NodeId>>>,
    /// Node ids sent on the previous model call, in send order.
    packed_prefix: Mutex<Vec<NodeId>>,
    turn_counter: AtomicU32,
    turn_pos_counter: Arc<AtomicU32>,
    /// Cumulative LLM usage tokens (input + output) for this session across
    /// [`crate::workflow::stream_and_record`] completions. Used with
    /// [`AgentConfig::max_session_tokens`].
    pub(crate) llm_tokens_used: Arc<AtomicU64>,
    llm_input_tokens: Arc<AtomicU64>,
    llm_output_tokens: Arc<AtomicU64>,
    llm_cost_micro_usd: Arc<AtomicU64>,
    llm_cost_reports: Arc<AtomicU64>,
    llm_reported_cost_micro_usd: Arc<AtomicU64>,
    llm_reported_cost_reports: Arc<AtomicU64>,
    frozen_input_per_token: std::sync::Mutex<Option<f64>>,
    frozen_cache_per_token: std::sync::Mutex<Option<f64>>,
    frozen_output_per_token: std::sync::Mutex<Option<f64>>,
}

impl Session {
    pub fn new(graph: Arc<GraphStore>, mut config: AgentConfig) -> Result<Self, AgentError> {
        config.apply_disable_bash_system_notice();
        config.apply_pi_delegate_system_notice();
        let now = Utc::now();
        let mut agent_node = GraphNode::new(NodeType::Agent(AgentData {
            name: config.name.clone(),
            model: config.model.clone(),
            system_prompt: Some(config.system_prompt.clone()),
            status: "active".to_string(),
        }));
        agent_node.set_label("agent_0_1_1");
        agent_node.metadata["session_id"] = serde_json::json!(agent_node.id.to_string());
        if let Some(ws) = &config.workspace_name {
            agent_node.metadata["workspace"] = serde_json::json!(ws);
        }
        let id = graph.add_node(agent_node)?;
        Ok(Self {
            id,
            agent_config: config,
            graph,
            created_at: now,
            hitl: None,
            memory_retriever: None,
            runtime_system_suffix: tokio::sync::Mutex::new(String::new()),
            last_interaction_id: Arc::new(Mutex::new(None)),
            packed_prefix: Mutex::new(Vec::new()),
            turn_counter: AtomicU32::new(0),
            turn_pos_counter: Arc::new(AtomicU32::new(0)),
            llm_tokens_used: Arc::new(AtomicU64::new(0)),
            llm_input_tokens: Arc::new(AtomicU64::new(0)),
            llm_output_tokens: Arc::new(AtomicU64::new(0)),
            llm_cost_micro_usd: Arc::new(AtomicU64::new(0)),
            llm_cost_reports: Arc::new(AtomicU64::new(0)),
            llm_reported_cost_micro_usd: Arc::new(AtomicU64::new(0)),
            llm_reported_cost_reports: Arc::new(AtomicU64::new(0)),
            frozen_input_per_token: std::sync::Mutex::new(None),
            frozen_cache_per_token: std::sync::Mutex::new(None),
            frozen_output_per_token: std::sync::Mutex::new(None),
        })
    }

    /// Reconstruct a [`Session`] from an existing Agent node stored in the graph.
    ///
    /// Unlike [`Session::new`], this does **not** create a new Agent node — it
    /// binds to the node that is already in the graph under `node_id`.
    /// Used at server startup to restore sessions that survived a restart.
    pub fn restore(
        graph: Arc<GraphStore>,
        node_id: graphirm_graph::nodes::NodeId,
        mut config: AgentConfig,
        created_at: DateTime<Utc>,
    ) -> Self {
        config.apply_disable_bash_system_notice();
        config.apply_pi_delegate_system_notice();
        Self {
            id: node_id,
            agent_config: config,
            graph,
            created_at,
            hitl: None,
            memory_retriever: None,
            runtime_system_suffix: tokio::sync::Mutex::new(String::new()),
            last_interaction_id: Arc::new(Mutex::new(None)),
            packed_prefix: Mutex::new(Vec::new()),
            turn_counter: AtomicU32::new(0),
            turn_pos_counter: Arc::new(AtomicU32::new(0)),
            llm_tokens_used: Arc::new(AtomicU64::new(0)),
            llm_input_tokens: Arc::new(AtomicU64::new(0)),
            llm_output_tokens: Arc::new(AtomicU64::new(0)),
            llm_cost_micro_usd: Arc::new(AtomicU64::new(0)),
            llm_cost_reports: Arc::new(AtomicU64::new(0)),
            llm_reported_cost_micro_usd: Arc::new(AtomicU64::new(0)),
            llm_reported_cost_reports: Arc::new(AtomicU64::new(0)),
            frozen_input_per_token: std::sync::Mutex::new(None),
            frozen_cache_per_token: std::sync::Mutex::new(None),
            frozen_output_per_token: std::sync::Mutex::new(None),
        }
    }

    /// Total LLM tokens (input + output) accumulated for this session from
    /// completion [`graphirm_llm::TokenUsage`] totals.
    pub fn llm_tokens_used(&self) -> u64 {
        self.llm_tokens_used.load(Ordering::Relaxed)
    }

    pub fn llm_input_tokens(&self) -> u64 {
        self.llm_input_tokens.load(Ordering::Relaxed)
    }

    pub fn llm_output_tokens(&self) -> u64 {
        self.llm_output_tokens.load(Ordering::Relaxed)
    }

    /// Sum of provider-reported costs, in millionths of a US dollar.
    pub fn llm_cost_micro_usd(&self) -> u64 {
        self.llm_cost_micro_usd.load(Ordering::Relaxed)
    }

    /// How many completions included a cost figure.
    pub fn llm_cost_reports(&self) -> u64 {
        self.llm_cost_reports.load(Ordering::Relaxed)
    }

    pub(crate) fn record_llm_usage(&self, input: u32, output: u32, cost_micro: Option<u64>) {
        self.llm_input_tokens
            .fetch_add(u64::from(input), Ordering::SeqCst);
        self.llm_output_tokens
            .fetch_add(u64::from(output), Ordering::SeqCst);
        self.llm_tokens_used
            .fetch_add(u64::from(input) + u64::from(output), Ordering::SeqCst);
        if let Some(cost) = cost_micro {
            self.llm_cost_micro_usd.fetch_add(cost, Ordering::SeqCst);
            self.llm_cost_reports.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Per-token prices from `eval/swebench/prices.lock.json`. When set, the
    /// session cost and `--max-cost-usd` use uncached input, cached input, and
    /// output at these rates.
    pub fn set_frozen_prices(
        &self,
        input_per_token: f64,
        cache_read_per_token: f64,
        output_per_token: f64,
    ) {
        *self
            .frozen_input_per_token
            .lock()
            .expect("frozen input price") = Some(input_per_token);
        *self
            .frozen_cache_per_token
            .lock()
            .expect("frozen cache price") = Some(cache_read_per_token);
        *self
            .frozen_output_per_token
            .lock()
            .expect("frozen output price") = Some(output_per_token);
    }

    pub fn frozen_prices_set(&self) -> bool {
        self.frozen_input_per_token
            .lock()
            .expect("frozen input price")
            .is_some()
    }

    /// Uncached input, cached input, and output times the frozen rates, in
    /// millionths of a dollar. Cached tokens are the provider's cache-read
    /// count, capped at the prompt size.
    pub fn frozen_cost_micro(&self, input: u32, cache_read: u32, output: u32) -> Option<u64> {
        let input_rate = (*self
            .frozen_input_per_token
            .lock()
            .expect("frozen input price"))?;
        let cache_rate = (*self
            .frozen_cache_per_token
            .lock()
            .expect("frozen cache price"))?;
        let output_rate = (*self
            .frozen_output_per_token
            .lock()
            .expect("frozen output price"))?;
        Some(Self::priced_micro(
            input,
            cache_read,
            output,
            input_rate,
            cache_rate,
            output_rate,
        ))
    }

    pub fn note_reported_cost(&self, cost_micro: u64) {
        self.llm_reported_cost_micro_usd
            .fetch_add(cost_micro, Ordering::SeqCst);
        self.llm_reported_cost_reports
            .fetch_add(1, Ordering::SeqCst);
    }

    pub fn llm_reported_cost_micro_usd(&self) -> u64 {
        self.llm_reported_cost_micro_usd.load(Ordering::Relaxed)
    }

    pub fn llm_reported_cost_reports(&self) -> u64 {
        self.llm_reported_cost_reports.load(Ordering::Relaxed)
    }

    /// `input_per_token` and `output_per_token` are US dollars per token.
    /// Every prompt token is billed at the input rate. This is the list price.
    pub fn tokens_to_micro_usd(input: u32, output: u32, input_rate: f64, output_rate: f64) -> u64 {
        Self::priced_micro(input, 0, output, input_rate, input_rate, output_rate)
    }

    /// Uncached prompt tokens use `input_rate`. Cached prompt tokens use
    /// `cache_rate`. Completion tokens use `output_rate`.
    pub fn priced_micro(
        input: u32,
        cache_read: u32,
        output: u32,
        input_rate: f64,
        cache_rate: f64,
        output_rate: f64,
    ) -> u64 {
        let cached = cache_read.min(input);
        let uncached = input - cached;
        let usd = f64::from(uncached) * input_rate
            + f64::from(cached) * cache_rate
            + f64::from(output) * output_rate;
        if usd <= 0.0 {
            return 0;
        }
        usd.mul_add(1_000_000.0, 0.5) as u64
    }

    /// Seeds cumulative LLM token usage (crate tests only).
    #[cfg(test)]
    pub fn test_set_llm_tokens_used(&self, n: u64) {
        self.llm_tokens_used.store(n, Ordering::SeqCst);
    }

    /// Attach a [`HitlGate`] to this session, enabling HITL approval flow.
    pub fn with_hitl(mut self, gate: Arc<HitlGate>) -> Self {
        self.hitl = Some(gate);
        self
    }

    /// Attach a memory retriever for cross-session knowledge injection.
    pub fn with_memory_retriever(mut self, retriever: Arc<MemoryRetriever>) -> Self {
        self.memory_retriever = Some(retriever);
        self
    }

    pub fn memory_retriever(&self) -> Option<&Arc<MemoryRetriever>> {
        self.memory_retriever.as_ref()
    }

    /// Read the current runtime system suffix (cross-session memory context).
    pub async fn memory_suffix(&self) -> String {
        self.runtime_system_suffix.lock().await.clone()
    }

    /// Replace the runtime system suffix with new memory context.
    pub async fn set_memory_suffix(&self, suffix: String) {
        *self.runtime_system_suffix.lock().await = suffix;
    }

    /// Return the content of the most recent user Interaction node, if any.
    ///
    /// Used to form the retrieval query for pre-loop memory injection.
    /// Async because `get_node` is a blocking `GraphStore` call.
    pub async fn recent_user_message(&self) -> Option<String> {
        let last_id = {
            let last = self
                .last_interaction_id
                .lock()
                .expect("last_interaction_id lock poisoned");
            last.clone()?
        };
        let graph = self.graph.clone();
        let node = tokio::task::spawn_blocking(move || graph.get_node(&last_id))
            .await
            .ok()?
            .ok()?;
        if let NodeType::Interaction(ref data) = node.node_type
            && data.role == "user"
        {
            return Some(data.content.clone());
        }
        None
    }

    /// Node ids sent on the previous model call, in that order.
    pub fn packed_prefix_ids(&self) -> Vec<NodeId> {
        self.packed_prefix
            .lock()
            .expect("packed_prefix lock poisoned")
            .clone()
    }

    /// Remember the ids just sent, so the next pack keeps that order.
    pub fn note_packed_ids(&self, ids: Vec<NodeId>) {
        *self
            .packed_prefix
            .lock()
            .expect("packed_prefix lock poisoned") = ids;
    }

    /// Bind to the oldest Agent already in `graph`, or create one.
    ///
    /// A second `graphirm run` on the same file continues that agent's
    /// conversation. The new task message chains onto the previous leaf.
    /// Each process opens the store once, so this does not rely on the
    /// agent-node cache staying fresh inside one process.
    pub fn open(graph: Arc<GraphStore>, config: AgentConfig) -> Result<Self, AgentError> {
        let mut agents = graph.get_agent_nodes()?;
        if agents.is_empty() {
            return Self::new(graph, config);
        }
        agents.sort_by_key(|(node, _)| node.created_at);
        let (node, _) = agents.into_iter().next().expect("agent list is non-empty");
        let session = Self::restore(graph.clone(), node.id.clone(), config, node.created_at);
        let chain = graph.get_session_chain(&session.id.to_string())?;
        if let Some(leaf) = chain.last() {
            *session
                .last_interaction_id
                .lock()
                .expect("last_interaction_id lock poisoned") = Some(leaf.id.clone());
        }
        Ok(session)
    }

    /// Add a user message to this session's conversation.
    /// Returns the NodeId of the created Interaction node.
    pub async fn add_user_message(&self, content: &str) -> Result<NodeId, AgentError> {
        self.add_user_message_with_context(content, None).await
    }

    /// Add the user message that states the task for this run.
    ///
    /// The `task` mark is what the packer pins. A later retry nudge is an
    /// ordinary user message and does not take the pin.
    pub async fn add_task_message(&self, content: &str) -> Result<NodeId, AgentError> {
        let turn = self.turn_counter.fetch_add(1, Ordering::SeqCst) + 1;
        self.turn_pos_counter.store(1, Ordering::SeqCst);
        let mut interaction_node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: content.to_string(),
            token_count: None,
        }));
        interaction_node.metadata["session_id"] = serde_json::json!(self.id.to_string());
        interaction_node.metadata["task"] = serde_json::json!(true);
        interaction_node.set_label(format!("interaction_{turn}_1_1"));
        self.persist_interaction(interaction_node, None).await
    }

    /// Add a user message, optionally linking `RespondsTo` to a specific prior node (steer / fork).
    ///
    /// When `responds_to` is `None`, the new message chains to the session's previous interaction
    /// (same as [`add_user_message`]). When set, the new user node `RespondsTo` that node instead,
    /// enabling branch-from-here without rewriting history.
    pub async fn add_user_message_with_context(
        &self,
        content: &str,
        responds_to: Option<NodeId>,
    ) -> Result<NodeId, AgentError> {
        let turn = self.turn_counter.fetch_add(1, Ordering::SeqCst) + 1;
        self.turn_pos_counter.store(1, Ordering::SeqCst);
        let mut interaction_node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: content.to_string(),
            token_count: None,
        }));
        interaction_node.metadata["session_id"] = serde_json::json!(self.id.to_string());
        interaction_node.set_label(format!("interaction_{turn}_1_1"));
        self.persist_interaction(interaction_node, responds_to)
            .await
    }

    pub fn current_turn(&self) -> u32 {
        self.turn_counter.load(Ordering::SeqCst)
    }

    pub fn next_turn_pos(&self) -> u32 {
        self.turn_pos_counter.fetch_add(1, Ordering::SeqCst) + 1
    }

    pub fn turn_position_counter(&self) -> Arc<AtomicU32> {
        self.turn_pos_counter.clone()
    }

    pub async fn record_interaction(&self, mut node: GraphNode) -> Result<NodeId, AgentError> {
        let turn = self.current_turn();
        let pos = self.next_turn_pos();
        node.metadata["session_id"] = serde_json::json!(self.id.to_string());
        node.set_label(format!("interaction_{turn}_{pos}_1"));
        self.persist_interaction(node, None).await
    }

    async fn persist_interaction(
        &self,
        node: GraphNode,
        responds_to_override: Option<NodeId>,
    ) -> Result<NodeId, AgentError> {
        let graph = self.graph.clone();
        let node_id = tokio::task::spawn_blocking(move || graph.add_node(node))
            .await
            .map_err(|e| AgentError::Join(e.to_string()))??;
        self.link_interaction(&node_id, responds_to_override)
            .await?;
        Ok(node_id)
    }

    /// Record an Interaction node as part of this session's conversation chain.
    ///
    /// Adds:
    /// - A `Produces` edge from the Agent node to the new Interaction node.
    /// - A `RespondsTo` edge from the new node to the previous Interaction node
    ///   (if any), forming a traversable conversation chain.
    pub async fn link_interaction(
        &self,
        node_id: &NodeId,
        responds_to_override: Option<NodeId>,
    ) -> Result<(), AgentError> {
        let graph = self.graph.clone();
        let agent_id = self.id.clone();
        let node_id_clone = node_id.clone();

        // Grab the previous ID and update the slot atomically under the mutex,
        // then do the blocking graph writes outside the lock.
        let prev_id = {
            let mut last = self
                .last_interaction_id
                .lock()
                .expect("last_interaction_id lock poisoned");
            let prev = responds_to_override.or_else(|| last.clone());
            *last = Some(node_id.clone());
            prev
        };

        tokio::task::spawn_blocking(move || -> Result<(), AgentError> {
            // Agent → node (Produces)
            graph.add_edge(GraphEdge::new(
                EdgeType::Produces,
                agent_id,
                node_id_clone.clone(),
            ))?;
            // node → previous (RespondsTo) — chains the conversation for graph traversal
            if let Some(prev) = prev_id {
                graph.add_edge(GraphEdge::new(EdgeType::RespondsTo, node_id_clone, prev))?;
            }
            Ok(())
        })
        .await
        .map_err(|e| AgentError::Join(e.to_string()))?
    }

    /// Update the Agent node's status field in the graph.
    /// Typical values: `"active"`, `"completed"`, `"cancelled"`, `"limit_reached"`.
    pub async fn set_status(&self, status: &str) -> Result<(), AgentError> {
        let graph = self.graph.clone();
        let agent_id = self.id.clone();
        let status = status.to_string();
        tokio::task::spawn_blocking(move || -> Result<(), AgentError> {
            let mut node = graph.get_node(&agent_id)?;
            if let NodeType::Agent(ref mut data) = node.node_type {
                data.status = status;
            }
            graph.update_node(&agent_id, node)?;
            Ok(())
        })
        .await
        .map_err(|e| AgentError::Join(e.to_string()))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hitl::HitlGate;
    use graphirm_graph::Direction;

    #[test]
    fn frozen_price_matches_the_openrouter_table() {
        // 1_000 input at $2/1M and 100 output at $6/1M = $0.0026 = 2600 micro-dollars.
        assert_eq!(
            Session::tokens_to_micro_usd(1_000, 100, 0.000002, 0.000006),
            2_600
        );
        // 600 uncached at $2/1M, 400 cached at $0.50/1M, 100 output at $6/1M = $0.0020.
        assert_eq!(
            Session::priced_micro(1_000, 400, 100, 0.000002, 0.0000005, 0.000006),
            2_000
        );
    }

    #[test]
    fn test_session_creates_agent_node_in_graph() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig::default();
        let session = Session::new(graph.clone(), config).unwrap();

        let node = graph.get_node(&session.id).unwrap();
        match &node.node_type {
            NodeType::Agent(data) => {
                assert_eq!(data.name, "graphirm");
                assert_eq!(node.label(), Some("agent_0_1_1"));
            }
            _ => panic!("expected Agent node"),
        }
    }

    #[test]
    fn open_reuses_the_agent_and_chains_the_next_task() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("graph.db");
        let path_str = path.to_str().unwrap().to_string();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async {
            let graph = Arc::new(GraphStore::open(&path_str).unwrap());
            let session = Session::open(graph, AgentConfig::default()).unwrap();
            session.add_task_message("TASK-ONE").await.unwrap();
        });

        rt.block_on(async {
            let graph = Arc::new(GraphStore::open(&path_str).unwrap());
            let session = Session::open(graph.clone(), AgentConfig::default()).unwrap();
            let second = session.add_task_message("TASK-TWO").await.unwrap();
            let thread = graph.conversation_thread(&second).unwrap();
            let text = thread
                .iter()
                .filter_map(|node| match &node.node_type {
                    NodeType::Interaction(data) => Some(data.content.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            assert!(text.contains("TASK-ONE"));
            assert!(text.contains("TASK-TWO"));
            assert_eq!(graph.get_agent_nodes().unwrap().len(), 1);
            let marked = thread
                .iter()
                .filter(|node| {
                    node.metadata.get("task").and_then(|value| value.as_bool()) == Some(true)
                })
                .count();
            assert_eq!(marked, 2);
        });
    }

    #[test]
    fn fresh_graph_per_session_opens_an_empty_store() {
        let shared = Arc::new(GraphStore::open_memory().unwrap());
        let marker = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "remember this".to_string(),
            token_count: None,
        }));
        let marker_id = marker.id.clone();
        shared.add_node(marker).unwrap();

        let same = open_session_graph(shared.clone(), &AgentConfig::default()).unwrap();
        assert!(Arc::ptr_eq(&same, &shared));
        assert!(same.get_node(&marker_id).is_ok());

        let dir = tempfile::tempdir().unwrap();
        let mut config = AgentConfig::default();
        config.fresh_graph_per_session = true;
        config.fresh_graph_dir = Some(dir.path().to_path_buf());
        let fresh = open_session_graph(shared, &config).unwrap();
        assert!(fresh.get_node(&marker_id).is_err());
    }

    #[test]
    fn test_session_stores_config() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig {
            name: "test-bot".to_string(),
            model: "gpt-4o".to_string(),
            ..AgentConfig::default()
        };
        let session = Session::new(graph, config).unwrap();

        assert_eq!(session.agent_config.name, "test-bot");
        assert_eq!(session.agent_config.model, "gpt-4o");
    }

    #[tokio::test]
    async fn test_session_add_user_message() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig::default();
        let session = Session::new(graph.clone(), config).unwrap();

        let msg_id = session.add_user_message("Hello!").await.unwrap();
        let node = graph.get_node(&msg_id).unwrap();
        match &node.node_type {
            NodeType::Interaction(data) => {
                assert_eq!(data.role, "user");
                assert_eq!(data.content, "Hello!");
                assert_eq!(node.label(), Some("interaction_1_1_1"));
            }
            _ => panic!("expected Interaction node"),
        }
    }

    #[tokio::test]
    async fn test_session_record_interaction_assigns_turn_position_labels() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig::default();
        let session = Session::new(graph.clone(), config).unwrap();

        let user_id = session.add_user_message("Hello!").await.unwrap();
        let assistant_id = session
            .record_interaction(GraphNode::new(NodeType::Interaction(InteractionData {
                role: "assistant".to_string(),
                content: "Hi there".to_string(),
                token_count: None,
            })))
            .await
            .unwrap();
        let tool_id = session
            .record_interaction(GraphNode::new(NodeType::Interaction(InteractionData {
                role: "tool".to_string(),
                content: "output".to_string(),
                token_count: None,
            })))
            .await
            .unwrap();

        assert_eq!(
            graph.get_node(&user_id).unwrap().label(),
            Some("interaction_1_1_1")
        );
        assert_eq!(
            graph.get_node(&assistant_id).unwrap().label(),
            Some("interaction_1_2_1")
        );
        assert_eq!(
            graph.get_node(&tool_id).unwrap().label(),
            Some("interaction_1_3_1")
        );
    }

    #[tokio::test]
    async fn test_session_add_user_message_resets_position_for_new_turn() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig::default();
        let session = Session::new(graph.clone(), config).unwrap();

        let _ = session.add_user_message("turn 1").await.unwrap();
        let _ = session
            .record_interaction(GraphNode::new(NodeType::Interaction(InteractionData {
                role: "assistant".to_string(),
                content: "first reply".to_string(),
                token_count: None,
            })))
            .await
            .unwrap();

        let second_user_id = session.add_user_message("turn 2").await.unwrap();
        assert_eq!(
            graph.get_node(&second_user_id).unwrap().label(),
            Some("interaction_2_1_1")
        );
    }

    #[tokio::test]
    async fn test_session_link_interaction_creates_responds_to_chain() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig::default();
        let session = Session::new(graph.clone(), config).unwrap();

        // Three sequential messages
        let id1 = session.add_user_message("msg1").await.unwrap();
        let id2 = session.add_user_message("msg2").await.unwrap();
        let id3 = session.add_user_message("msg3").await.unwrap();

        // msg2 should have a RespondsTo edge pointing to msg1
        let msg2_responds: Vec<_> = graph
            .neighbors(&id2, Some(EdgeType::RespondsTo), Direction::Outgoing)
            .unwrap();
        assert_eq!(msg2_responds.len(), 1);
        assert_eq!(msg2_responds[0].id, id1);

        // msg3 should have a RespondsTo edge pointing to msg2
        let msg3_responds: Vec<_> = graph
            .neighbors(&id3, Some(EdgeType::RespondsTo), Direction::Outgoing)
            .unwrap();
        assert_eq!(msg3_responds.len(), 1);
        assert_eq!(msg3_responds[0].id, id2);
    }

    #[tokio::test]
    async fn test_add_user_message_with_context_forks_responds_to() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig::default();
        let session = Session::new(graph.clone(), config).unwrap();

        let id1 = session.add_user_message("first").await.unwrap();
        let _id2 = session.add_user_message("second").await.unwrap();
        let fork = session
            .add_user_message_with_context("forked", Some(id1.clone()))
            .await
            .unwrap();

        let fork_responds: Vec<_> = graph
            .neighbors(&fork, Some(EdgeType::RespondsTo), Direction::Outgoing)
            .unwrap();
        assert_eq!(fork_responds.len(), 1);
        assert_eq!(fork_responds[0].id, id1);
    }

    #[test]
    fn session_hitl_field_is_none_by_default() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig::default();
        let session = Session::new(graph, config).unwrap();
        assert!(session.hitl.is_none());
    }

    #[test]
    fn session_with_hitl_gate_stores_it() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig::default();
        let gate = Arc::new(HitlGate::new());
        let session = Session::new(graph, config).unwrap().with_hitl(gate.clone());
        assert!(session.hitl.is_some());
    }

    #[test]
    fn test_session_memory_retriever_is_none_by_default() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig::default();
        let session = Session::new(graph, config).unwrap();
        assert!(session.memory_retriever().is_none());
    }

    #[tokio::test]
    async fn test_session_runtime_suffix_starts_empty() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig::default();
        let session = Session::new(graph, config).unwrap();
        assert_eq!(session.memory_suffix().await, "");
    }

    #[tokio::test]
    async fn test_session_set_memory_suffix() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig::default();
        let session = Session::new(graph, config).unwrap();
        session
            .set_memory_suffix("relevant context".to_string())
            .await;
        assert_eq!(session.memory_suffix().await, "relevant context");
    }

    #[test]
    fn session_stores_workspace_in_metadata() {
        use crate::config::AgentConfig;
        use graphirm_graph::GraphStore;
        use std::sync::Arc;

        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig {
            workspace_name: Some("myapp".to_string()),
            ..AgentConfig::default()
        };
        let session = Session::new(graph.clone(), config).unwrap();
        let node = graph.get_node(&session.id).unwrap();
        assert_eq!(
            node.metadata.get("workspace"),
            Some(&serde_json::json!("myapp"))
        );
    }

    #[test]
    fn session_restore_binds_to_existing_node() {
        use graphirm_graph::GraphStore;
        use graphirm_graph::nodes::{AgentData, GraphNode, NodeType};
        use std::sync::Arc;

        let graph = Arc::new(GraphStore::open_memory().unwrap());

        let agent_node = GraphNode::new(NodeType::Agent(AgentData {
            name: "restored-agent".to_string(),
            model: "deepseek-chat".to_string(),
            system_prompt: None,
            status: "idle".to_string(),
        }));
        let node_id = graph.add_node(agent_node).unwrap();
        let agent_count_before = graph.get_agent_nodes().unwrap().len();

        let config = AgentConfig::default();
        let session = Session::restore(graph.clone(), node_id.clone(), config, chrono::Utc::now());

        assert_eq!(
            graph.get_agent_nodes().unwrap().len(),
            agent_count_before,
            "restore must not create new nodes"
        );
        assert_eq!(
            session.id, node_id,
            "restored session must use the existing node id"
        );
    }

    #[tokio::test]
    async fn test_session_set_status() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig::default();
        let session = Session::new(graph.clone(), config).unwrap();

        session.set_status("completed").await.unwrap();

        let node = graph.get_node(&session.id).unwrap();
        match &node.node_type {
            NodeType::Agent(data) => assert_eq!(data.status, "completed"),
            _ => panic!("expected Agent node"),
        }
    }
}
