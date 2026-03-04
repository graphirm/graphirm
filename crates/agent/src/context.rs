// Context engine: graph-native relevance scoring and context window building

use std::collections::{HashMap, VecDeque};

use chrono::{Duration, Utc};
use serde::Deserialize;

use graphirm_graph::{
    Direction, EdgeType, GraphEdge, GraphNode, GraphStore, NodeId, NodeType,
    InteractionData, ContentData, KnowledgeData, AgentData, TaskData,
};
use graphirm_llm::LlmMessage;

use crate::error::AgentError;

#[derive(Debug, Clone, Deserialize)]
pub struct EdgeWeights {
    #[serde(default = "default_modifies")]
    pub modifies: f64,
    #[serde(default = "default_produces")]
    pub produces: f64,
    #[serde(default = "default_reads")]
    pub reads: f64,
    #[serde(default = "default_relates_to")]
    pub relates_to: f64,
    #[serde(default = "default_responds_to")]
    pub responds_to: f64,
    #[serde(default = "default_other")]
    pub other: f64,
}

fn default_modifies() -> f64 { 1.0 }
fn default_produces() -> f64 { 0.8 }
fn default_reads() -> f64 { 0.6 }
fn default_relates_to() -> f64 { 0.4 }
fn default_responds_to() -> f64 { 0.3 }
fn default_other() -> f64 { 0.1 }

impl Default for EdgeWeights {
    fn default() -> Self {
        Self {
            modifies: 1.0,
            produces: 0.8,
            reads: 0.6,
            relates_to: 0.4,
            responds_to: 0.3,
            other: 0.1,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContextConfig {
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_system_prompt")]
    pub system_prompt: String,
    #[serde(default = "default_guaranteed_recent_turns")]
    pub guaranteed_recent_turns: usize,
    #[serde(default = "default_max_content_nodes")]
    pub max_content_nodes: usize,
    #[serde(default = "default_recency_decay")]
    pub recency_decay: f64,
    #[serde(default)]
    pub edge_weights: EdgeWeights,
    #[serde(default = "default_enable_compaction")]
    pub enable_compaction: bool,
}

fn default_max_tokens() -> usize { 128_000 }
fn default_system_prompt() -> String { "You are a helpful coding assistant.".to_string() }
fn default_guaranteed_recent_turns() -> usize { 4 }
fn default_max_content_nodes() -> usize { 20 }
fn default_recency_decay() -> f64 { 0.1 }
fn default_enable_compaction() -> bool { false }

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_tokens: 128_000,
            system_prompt: "You are a helpful coding assistant.".to_string(),
            guaranteed_recent_turns: 4,
            max_content_nodes: 20,
            recency_decay: 0.1,
            edge_weights: EdgeWeights::default(),
            enable_compaction: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScoredNode {
    pub node: GraphNode,
    pub score: f64,
    pub token_estimate: usize,
}

#[derive(Debug, Clone)]
pub struct ContextWindow {
    pub system: LlmMessage,
    pub messages: Vec<LlmMessage>,
    pub total_tokens: usize,
}

/// Extract the primary text content from any GraphNode variant.
pub fn get_text_content(node: &GraphNode) -> &str {
    match &node.node_type {
        NodeType::Interaction(data) => &data.content,
        NodeType::Content(data) => &data.body,
        NodeType::Knowledge(data) => &data.summary,
        NodeType::Agent(data) => data.system_prompt.as_deref().unwrap_or(""),
        NodeType::Task(data) => &data.description,
    }
}

/// Convert a GraphNode (Interaction) into an LlmMessage.
/// Returns None for non-Interaction nodes or unknown roles.
pub fn node_to_message(node: &GraphNode) -> Option<LlmMessage> {
    match &node.node_type {
        NodeType::Interaction(data) => {
            let msg = match data.role.as_str() {
                "user" => LlmMessage::human(&data.content),
                "assistant" => LlmMessage::assistant(&data.content),
                "system" => LlmMessage::system(&data.content),
                _ => return None,
            };
            Some(msg)
        }
        NodeType::Content(data) => {
            let label = match &data.path {
                Some(path) => format!("[File: {}]\n{}", path, data.body),
                None => format!("[Content: {}]\n{}", data.content_type, data.body),
            };
            Some(LlmMessage::human(label))
        }
        NodeType::Knowledge(data) => {
            Some(LlmMessage::human(format!(
                "[Knowledge: {} ({})]\n{}",
                data.entity, data.entity_type, data.summary
            )))
        }
        _ => None,
    }
}

/// Estimate token count for a text string using word-based heuristic.
/// Approximation: tokens ≈ words / 0.75 (1 token ≈ 0.75 words).
pub fn estimate_tokens_str(text: &str) -> usize {
    let word_count = text.split_whitespace().count();
    if word_count == 0 {
        return 0;
    }
    (word_count as f64 / 0.75).ceil() as usize
}

/// Estimate token count for a GraphNode based on its text content.
pub fn estimate_tokens(node: &GraphNode) -> usize {
    estimate_tokens_str(get_text_content(node))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_config_default_values() {
        let config = ContextConfig::default();
        assert_eq!(config.max_tokens, 128_000);
        assert_eq!(config.guaranteed_recent_turns, 4);
        assert_eq!(config.max_content_nodes, 20);
        assert!((config.recency_decay - 0.1).abs() < f64::EPSILON);
        assert!(!config.enable_compaction);
    }

    #[test]
    fn edge_weights_default_values() {
        let w = EdgeWeights::default();
        assert!((w.modifies - 1.0).abs() < f64::EPSILON);
        assert!((w.produces - 0.8).abs() < f64::EPSILON);
        assert!((w.reads - 0.6).abs() < f64::EPSILON);
        assert!((w.relates_to - 0.4).abs() < f64::EPSILON);
        assert!((w.responds_to - 0.3).abs() < f64::EPSILON);
        assert!((w.other - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn context_config_from_toml() {
        let toml_str = r#"
            max_tokens = 64000
            system_prompt = "You are a Rust expert."
            guaranteed_recent_turns = 6
            max_content_nodes = 10
            recency_decay = 0.05
            enable_compaction = true

            [edge_weights]
            modifies = 2.0
            produces = 1.5
            reads = 1.0
            relates_to = 0.5
            responds_to = 0.2
            other = 0.05
        "#;
        let config: ContextConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.max_tokens, 64_000);
        assert_eq!(config.system_prompt, "You are a Rust expert.");
        assert_eq!(config.guaranteed_recent_turns, 6);
        assert_eq!(config.max_content_nodes, 10);
        assert!((config.recency_decay - 0.05).abs() < f64::EPSILON);
        assert!(config.enable_compaction);
        assert!((config.edge_weights.modifies - 2.0).abs() < f64::EPSILON);
        assert!((config.edge_weights.produces - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn context_config_from_toml_with_defaults() {
        let toml_str = r#"
            max_tokens = 32000
        "#;
        let config: ContextConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.max_tokens, 32_000);
        assert_eq!(config.guaranteed_recent_turns, 4);
        assert!((config.edge_weights.modifies - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn scored_node_construction() {
        let node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "Hello world".to_string(),
            token_count: None,
        }));
        let scored = ScoredNode {
            node: node.clone(),
            score: 0.85,
            token_estimate: 4,
        };
        assert!((scored.score - 0.85).abs() < f64::EPSILON);
        assert_eq!(scored.token_estimate, 4);
        assert_eq!(scored.node.id, node.id);
    }

    #[test]
    fn context_window_construction() {
        let window = ContextWindow {
            system: LlmMessage::system("You are helpful."),
            messages: vec![
                LlmMessage::human("Hi"),
                LlmMessage::assistant("Hello!"),
            ],
            total_tokens: 100,
        };
        assert_eq!(window.messages.len(), 2);
        assert_eq!(window.total_tokens, 100);
    }

    #[test]
    fn get_text_content_interaction() {
        let node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "What is Rust?".to_string(),
            token_count: None,
        }));
        assert_eq!(get_text_content(&node), "What is Rust?");
    }

    #[test]
    fn get_text_content_content_node() {
        let node = GraphNode::new(NodeType::Content(ContentData {
            content_type: "file".to_string(),
            path: Some("main.rs".to_string()),
            body: "fn main() {}".to_string(),
            language: Some("rust".to_string()),
        }));
        assert_eq!(get_text_content(&node), "fn main() {}");
    }

    #[test]
    fn get_text_content_knowledge_node() {
        let node = GraphNode::new(NodeType::Knowledge(KnowledgeData {
            entity: "GraphStore".to_string(),
            entity_type: "struct".to_string(),
            summary: "Dual-write graph persistence.".to_string(),
            confidence: 0.9,
        }));
        assert_eq!(get_text_content(&node), "Dual-write graph persistence.");
    }

    #[test]
    fn node_to_message_user() {
        let node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "Hello!".to_string(),
            token_count: None,
        }));
        let msg = node_to_message(&node).unwrap();
        assert_eq!(msg.role, graphirm_llm::Role::Human);
    }

    #[test]
    fn node_to_message_assistant() {
        let node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".to_string(),
            content: "Hi there!".to_string(),
            token_count: None,
        }));
        let msg = node_to_message(&node).unwrap();
        assert_eq!(msg.role, graphirm_llm::Role::Assistant);
    }

    #[test]
    fn node_to_message_content() {
        let node = GraphNode::new(NodeType::Content(ContentData {
            content_type: "file".to_string(),
            path: Some("lib.rs".to_string()),
            body: "pub mod graph;".to_string(),
            language: Some("rust".to_string()),
        }));
        let msg = node_to_message(&node).unwrap();
        assert_eq!(msg.role, graphirm_llm::Role::Human);
    }

    #[test]
    fn node_to_message_task_returns_none() {
        let node = GraphNode::new(NodeType::Task(TaskData {
            title: "Fix bug".to_string(),
            description: "Something is broken".to_string(),
            status: "pending".to_string(),
            priority: Some(1),
        }));
        assert!(node_to_message(&node).is_none());
    }

    #[test]
    fn estimate_tokens_str_known_strings() {
        // "Hello world" = 2 words → 2 / 0.75 = 2.67 → ceil → 3
        assert_eq!(estimate_tokens_str("Hello world"), 3);

        // Empty string = 0 tokens
        assert_eq!(estimate_tokens_str(""), 0);

        // 10 words → 10 / 0.75 = 13.33 → ceil → 14
        assert_eq!(
            estimate_tokens_str("one two three four five six seven eight nine ten"),
            14
        );

        // Single word → 1 / 0.75 = 1.33 → ceil → 2
        assert_eq!(estimate_tokens_str("hello"), 2);
    }

    #[test]
    fn estimate_tokens_node() {
        let node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "What is the meaning of life the universe and everything".to_string(),
            token_count: None,
        }));
        // 10 words → 14 tokens
        assert_eq!(estimate_tokens(&node), 14);
    }

    #[test]
    fn estimate_tokens_code_content() {
        let node = GraphNode::new(NodeType::Content(ContentData {
            content_type: "file".to_string(),
            path: Some("main.rs".to_string()),
            body: "fn main() {\n    println!(\"Hello, world!\");\n}".to_string(),
            language: Some("rust".to_string()),
        }));
        let tokens = estimate_tokens(&node);
        assert!(tokens > 0);
        assert!(tokens < 100);
    }
}
