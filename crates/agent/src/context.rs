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
}
