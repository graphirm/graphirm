use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub String);

impl NodeId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

impl Default for NodeId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for NodeId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionData {
    pub role: String,
    pub content: String,
    pub token_count: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentData {
    pub name: String,
    pub model: String,
    pub system_prompt: Option<String>,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentData {
    pub content_type: String,
    pub path: Option<String>,
    pub body: String,
    pub language: Option<String>,
}

/// Typed status for Task nodes. Replaces bare `String` to prevent typo-driven silent failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    #[default]
    Pending,
    Running,
    Completed,
    Failed,
}

impl fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TaskStatus::Pending => write!(f, "pending"),
            TaskStatus::Running => write!(f, "running"),
            TaskStatus::Completed => write!(f, "completed"),
            TaskStatus::Failed => write!(f, "failed"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskData {
    pub title: String,
    pub description: String,
    pub status: TaskStatus,
    pub priority: Option<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeData {
    pub entity: String,
    pub entity_type: String,
    pub summary: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum NodeType {
    Interaction(InteractionData),
    Agent(AgentData),
    Content(ContentData),
    Task(TaskData),
    Knowledge(KnowledgeData),
}

impl NodeType {
    pub fn type_name(&self) -> &'static str {
        match self {
            NodeType::Interaction(_) => "interaction",
            NodeType::Agent(_) => "agent",
            NodeType::Content(_) => "content",
            NodeType::Task(_) => "task",
            NodeType::Knowledge(_) => "knowledge",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: NodeId,
    pub node_type: NodeType,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: serde_json::Value,
}

impl GraphNode {
    pub fn new(node_type: NodeType) -> Self {
        let now = Utc::now();
        Self {
            id: NodeId::new(),
            node_type,
            created_at: now,
            updated_at: now,
            metadata: serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    pub fn label(&self) -> Option<&str> {
        self.metadata.get("label")?.as_str()
    }

    pub fn set_label(&mut self, label: impl Into<String>) {
        if !self.metadata.is_object() {
            self.metadata = serde_json::Value::Object(serde_json::Map::new());
        }
        let metadata = self
            .metadata
            .as_object_mut()
            .expect("graph node metadata should be an object");
        metadata.insert("label".to_string(), serde_json::Value::String(label.into()));
        metadata
            .entry("label_ver".to_string())
            .or_insert_with(|| serde_json::json!(1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_id_display() {
        let id = NodeId("abc-123".to_string());
        assert_eq!(id.to_string(), "abc-123");
    }

    #[test]
    fn node_id_from_str() {
        let id = NodeId::from("test-id");
        assert_eq!(id.0, "test-id");
    }

    #[test]
    fn node_id_new_is_unique() {
        let a = NodeId::new();
        let b = NodeId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn node_id_serde_roundtrip() {
        let id = NodeId("roundtrip-test".to_string());
        let json = serde_json::to_string(&id).unwrap();
        let back: NodeId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, back);
    }

    #[test]
    fn node_id_hash_eq() {
        use std::collections::HashSet;
        let id = NodeId("same".to_string());
        let mut set = HashSet::new();
        set.insert(id.clone());
        assert!(set.contains(&id));
    }

    #[test]
    fn interaction_node_serde_roundtrip() {
        let node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "Hello, world!".to_string(),
            token_count: Some(5),
        }));
        let json = serde_json::to_string(&node).unwrap();
        let back: GraphNode = serde_json::from_str(&json).unwrap();
        assert_eq!(node.id, back.id);
        match &back.node_type {
            NodeType::Interaction(data) => {
                assert_eq!(data.role, "user");
                assert_eq!(data.content, "Hello, world!");
                assert_eq!(data.token_count, Some(5));
            }
            _ => panic!("expected Interaction variant"),
        }
    }

    #[test]
    fn agent_node_serde_roundtrip() {
        let node = GraphNode::new(NodeType::Agent(AgentData {
            name: "coder".to_string(),
            model: "claude-sonnet-4-20250514".to_string(),
            system_prompt: Some("You are a coding agent.".to_string()),
            status: "running".to_string(),
        }));
        let json = serde_json::to_string(&node).unwrap();
        let back: GraphNode = serde_json::from_str(&json).unwrap();
        match &back.node_type {
            NodeType::Agent(data) => assert_eq!(data.name, "coder"),
            _ => panic!("expected Agent variant"),
        }
    }

    #[test]
    fn content_node_serde_roundtrip() {
        let node = GraphNode::new(NodeType::Content(ContentData {
            content_type: "file".to_string(),
            path: Some("src/main.rs".to_string()),
            body: "fn main() {}".to_string(),
            language: Some("rust".to_string()),
        }));
        let json = serde_json::to_string(&node).unwrap();
        let back: GraphNode = serde_json::from_str(&json).unwrap();
        match &back.node_type {
            NodeType::Content(data) => {
                assert_eq!(data.content_type, "file");
                assert_eq!(data.path, Some("src/main.rs".to_string()));
            }
            _ => panic!("expected Content variant"),
        }
    }

    #[test]
    fn task_node_serde_roundtrip() {
        let node = GraphNode::new(NodeType::Task(TaskData {
            title: "Implement login".to_string(),
            description: "Add OAuth2 login flow".to_string(),
            status: TaskStatus::Pending,
            priority: Some(1),
        }));
        let json = serde_json::to_string(&node).unwrap();
        let back: GraphNode = serde_json::from_str(&json).unwrap();
        match &back.node_type {
            NodeType::Task(data) => assert_eq!(data.title, "Implement login"),
            _ => panic!("expected Task variant"),
        }
    }

    #[test]
    fn knowledge_node_serde_roundtrip() {
        let node = GraphNode::new(NodeType::Knowledge(KnowledgeData {
            entity: "GraphStore".to_string(),
            entity_type: "concept".to_string(),
            summary: "Dual-write graph persistence layer".to_string(),
            confidence: 0.95,
        }));
        let json = serde_json::to_string(&node).unwrap();
        let back: GraphNode = serde_json::from_str(&json).unwrap();
        match &back.node_type {
            NodeType::Knowledge(data) => {
                assert_eq!(data.entity, "GraphStore");
                assert!((data.confidence - 0.95).abs() < f64::EPSILON);
            }
            _ => panic!("expected Knowledge variant"),
        }
    }

    #[test]
    fn node_type_name() {
        let interaction = NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "hi".to_string(),
            token_count: None,
        });
        assert_eq!(interaction.type_name(), "interaction");

        let task = NodeType::Task(TaskData {
            title: "t".to_string(),
            description: "d".to_string(),
            status: TaskStatus::Pending,
            priority: None,
        });
        assert_eq!(task.type_name(), "task");
    }

    #[test]
    fn graph_node_new_sets_timestamps() {
        let before = Utc::now();
        let node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "test".to_string(),
            token_count: None,
        }));
        let after = Utc::now();
        assert!(node.created_at >= before && node.created_at <= after);
        assert_eq!(node.created_at, node.updated_at);
    }

    #[test]
    fn graph_node_label_is_absent_by_default() {
        let node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "test".to_string(),
            token_count: None,
        }));
        assert_eq!(node.label(), None);
    }

    #[test]
    fn graph_node_set_label_stores_label_and_initial_version() {
        let mut node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".to_string(),
            content: "hello".to_string(),
            token_count: None,
        }));

        node.set_label("interaction_1_2_1");

        assert_eq!(node.label(), Some("interaction_1_2_1"));
        assert_eq!(node.metadata.get("label_ver"), Some(&serde_json::json!(1)));
    }

    #[test]
    fn graph_node_set_label_preserves_existing_version() {
        let mut node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "tool".to_string(),
            content: "output".to_string(),
            token_count: None,
        }));
        node.metadata = serde_json::json!({
            "label_ver": 7
        });

        node.set_label("interaction_1_3_7");

        assert_eq!(node.label(), Some("interaction_1_3_7"));
        assert_eq!(node.metadata.get("label_ver"), Some(&serde_json::json!(7)));
    }
}
