// Session management: create, resume, list, archive sessions

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use graphirm_graph::GraphStore;
use graphirm_graph::edges::{EdgeType, GraphEdge};
use graphirm_graph::nodes::{AgentData, GraphNode, InteractionData, NodeId, NodeType};

use crate::config::AgentConfig;
use crate::error::AgentError;
use crate::hitl::HitlGate;
use crate::knowledge::memory::MemoryRetriever;

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
}

impl SessionMetadata {
    pub fn from_agent_node_id(
        session_id: String,
        name: String,
        model: String,
        created_at: DateTime<Utc>,
        status: SessionStatus,
    ) -> Self {
        Self { session_id, name, model, created_at, status }
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
    turn_counter: AtomicU32,
    turn_pos_counter: Arc<AtomicU32>,
}

impl Session {
    pub fn new(graph: Arc<GraphStore>, config: AgentConfig) -> Result<Self, AgentError> {
        let now = Utc::now();
        let mut agent_node = GraphNode::new(NodeType::Agent(AgentData {
            name: config.name.clone(),
            model: config.model.clone(),
            system_prompt: Some(config.system_prompt.clone()),
            status: "active".to_string(),
        }));
        agent_node.set_label("agent_0_1_1");
        agent_node.metadata["session_id"] = serde_json::json!(agent_node.id.to_string());
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
            turn_counter: AtomicU32::new(0),
            turn_pos_counter: Arc::new(AtomicU32::new(0)),
        })
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
        if let NodeType::Interaction(ref data) = node.node_type {
            if data.role == "user" {
                return Some(data.content.clone());
            }
        }
        None
    }

    /// Add a user message to this session's conversation.
    /// Returns the NodeId of the created Interaction node.
    pub async fn add_user_message(&self, content: &str) -> Result<NodeId, AgentError> {
        let turn = self.turn_counter.fetch_add(1, Ordering::SeqCst) + 1;
        self.turn_pos_counter.store(1, Ordering::SeqCst);
        let mut interaction_node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: content.to_string(),
            token_count: None,
        }));
        interaction_node.metadata["session_id"] = serde_json::json!(self.id.to_string());
        interaction_node.set_label(format!("interaction_{turn}_1_1"));
        self.persist_interaction(interaction_node).await
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
        self.persist_interaction(node).await
    }

    async fn persist_interaction(&self, node: GraphNode) -> Result<NodeId, AgentError> {
        let graph = self.graph.clone();
        let node_id = tokio::task::spawn_blocking(move || graph.add_node(node))
            .await
            .map_err(|e| AgentError::Join(e.to_string()))??;
        self.link_interaction(&node_id).await?;
        Ok(node_id)
    }

    /// Record an Interaction node as part of this session's conversation chain.
    ///
    /// Adds:
    /// - A `Produces` edge from the Agent node to the new Interaction node.
    /// - A `RespondsTo` edge from the new node to the previous Interaction node
    ///   (if any), forming a traversable conversation chain.
    pub async fn link_interaction(&self, node_id: &NodeId) -> Result<(), AgentError> {
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
            let prev = last.clone();
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
                graph.add_edge(GraphEdge::new(
                    EdgeType::RespondsTo,
                    node_id_clone,
                    prev,
                ))?;
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

        assert_eq!(graph.get_node(&user_id).unwrap().label(), Some("interaction_1_1_1"));
        assert_eq!(
            graph.get_node(&assistant_id).unwrap().label(),
            Some("interaction_1_2_1")
        );
        assert_eq!(graph.get_node(&tool_id).unwrap().label(), Some("interaction_1_3_1"));
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
        session.set_memory_suffix("relevant context".to_string()).await;
        assert_eq!(session.memory_suffix().await, "relevant context");
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
