// Session management: create, resume, list, archive sessions

use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use graphirm_graph::GraphStore;
use graphirm_graph::edges::{EdgeType, GraphEdge};
use graphirm_graph::nodes::{AgentData, GraphNode, InteractionData, NodeId, NodeType};

use crate::config::AgentConfig;
use crate::error::AgentError;
use crate::hitl::HitlGate;

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
    /// ID of the most recent Interaction node in this session's conversation
    /// chain. Used to create `RespondsTo` edges that link messages into a
    /// traversable DAG (rather than a flat star off the agent node).
    last_interaction_id: Mutex<Option<NodeId>>,
}

impl Session {
    pub fn new(graph: Arc<GraphStore>, config: AgentConfig) -> Result<Self, AgentError> {
        let now = Utc::now();
        let agent_node = GraphNode::new(NodeType::Agent(AgentData {
            name: config.name.clone(),
            model: config.model.clone(),
            system_prompt: Some(config.system_prompt.clone()),
            status: "active".to_string(),
        }));
        let id = graph.add_node(agent_node)?;
        Ok(Self {
            id,
            agent_config: config,
            graph,
            created_at: now,
            hitl: None,
            last_interaction_id: Mutex::new(None),
        })
    }

    /// Attach a [`HitlGate`] to this session, enabling HITL approval flow.
    pub fn with_hitl(mut self, gate: Arc<HitlGate>) -> Self {
        self.hitl = Some(gate);
        self
    }

    /// Add a user message to this session's conversation.
    /// Returns the NodeId of the created Interaction node.
    pub fn add_user_message(&self, content: &str) -> Result<NodeId, AgentError> {
        let interaction_node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: content.to_string(),
            token_count: None,
        }));
        let msg_id = self.graph.add_node(interaction_node)?;
        self.link_interaction(&msg_id)?;
        Ok(msg_id)
    }

    /// Record an Interaction node as part of this session's conversation chain.
    ///
    /// Adds:
    /// - A `Produces` edge from the Agent node to the new Interaction node.
    /// - A `RespondsTo` edge from the new node to the previous Interaction node
    ///   (if any), forming a traversable conversation chain.
    pub fn link_interaction(&self, node_id: &NodeId) -> Result<(), AgentError> {
        // Agent → node (Produces)
        self.graph.add_edge(GraphEdge::new(
            EdgeType::Produces,
            self.id.clone(),
            node_id.clone(),
        ))?;

        // node → previous (RespondsTo) — chains the conversation for graph traversal
        let mut last = self
            .last_interaction_id
            .lock()
            .expect("last_interaction_id lock poisoned");
        if let Some(prev_id) = last.as_ref() {
            self.graph.add_edge(GraphEdge::new(
                EdgeType::RespondsTo,
                node_id.clone(),
                prev_id.clone(),
            ))?;
        }
        *last = Some(node_id.clone());
        Ok(())
    }

    /// Update the Agent node's status field in the graph.
    /// Typical values: `"active"`, `"completed"`, `"cancelled"`, `"limit_reached"`.
    pub fn set_status(&self, status: &str) -> Result<(), AgentError> {
        let mut node = self.graph.get_node(&self.id)?;
        if let NodeType::Agent(ref mut data) = node.node_type {
            data.status = status.to_string();
        }
        self.graph.update_node(&self.id, node)?;
        Ok(())
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
            NodeType::Agent(data) => assert_eq!(data.name, "graphirm"),
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

    #[test]
    fn test_session_add_user_message() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig::default();
        let session = Session::new(graph.clone(), config).unwrap();

        let msg_id = session.add_user_message("Hello!").unwrap();
        let node = graph.get_node(&msg_id).unwrap();
        match &node.node_type {
            NodeType::Interaction(data) => {
                assert_eq!(data.role, "user");
                assert_eq!(data.content, "Hello!");
            }
            _ => panic!("expected Interaction node"),
        }
    }

    #[test]
    fn test_session_link_interaction_creates_responds_to_chain() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig::default();
        let session = Session::new(graph.clone(), config).unwrap();

        // Three sequential messages
        let id1 = session.add_user_message("msg1").unwrap();
        let id2 = session.add_user_message("msg2").unwrap();
        let id3 = session.add_user_message("msg3").unwrap();

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
    fn test_session_set_status() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig::default();
        let session = Session::new(graph.clone(), config).unwrap();

        session.set_status("completed").unwrap();

        let node = graph.get_node(&session.id).unwrap();
        match &node.node_type {
            NodeType::Agent(data) => assert_eq!(data.status, "completed"),
            _ => panic!("expected Agent node"),
        }
    }
}
