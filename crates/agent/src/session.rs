// Session management: create, resume, list, archive sessions

use std::sync::Arc;

use chrono::{DateTime, Utc};
use graphirm_graph::edges::{EdgeType, GraphEdge};
use graphirm_graph::nodes::{AgentData, GraphNode, InteractionData, NodeId, NodeType};
use graphirm_graph::GraphStore;

use crate::config::AgentConfig;
use crate::error::AgentError;

pub struct Session {
    pub id: NodeId,
    pub agent_config: AgentConfig,
    pub graph: Arc<GraphStore>,
    pub created_at: DateTime<Utc>,
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
        })
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
        self.graph.add_edge(GraphEdge::new(
            EdgeType::Produces,
            self.id.clone(),
            msg_id.clone(),
        ))?;
        Ok(msg_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
