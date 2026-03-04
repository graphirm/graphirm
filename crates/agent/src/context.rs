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

impl EdgeWeights {
    /// Get the weight for a specific edge type.
    pub fn weight_for(&self, edge_type: EdgeType) -> f64 {
        match edge_type {
            EdgeType::Modifies => self.modifies,
            EdgeType::Produces => self.produces,
            EdgeType::Reads => self.reads,
            EdgeType::RelatesTo => self.relates_to,
            EdgeType::RespondsTo => self.responds_to,
            _ => self.other,
        }
    }
}

/// Score a node based on the types and counts of its edges.
/// For each edge type, counts both outgoing and incoming edges,
/// then multiplies by the configured weight for that type.
pub fn score_edge_weights(
    node_id: &NodeId,
    graph: &GraphStore,
    weights: &EdgeWeights,
) -> Result<f64, AgentError> {
    let weighted_types = [
        EdgeType::Modifies,
        EdgeType::Produces,
        EdgeType::Reads,
        EdgeType::RelatesTo,
        EdgeType::RespondsTo,
        EdgeType::DelegatesTo,
        EdgeType::DependsOn,
        EdgeType::Summarizes,
        EdgeType::Contains,
        EdgeType::FollowsUp,
        EdgeType::Steers,
        EdgeType::SpawnedBy,
    ];

    let mut total = 0.0;
    for et in &weighted_types {
        let out_count = graph
            .neighbors(node_id, Some(*et), Direction::Outgoing)
            .map_err(|e| AgentError::Context(e.to_string()))?
            .len();
        let in_count = graph
            .neighbors(node_id, Some(*et), Direction::Incoming)
            .map_err(|e| AgentError::Context(e.to_string()))?
            .len();
        total += weights.weight_for(*et) * (out_count + in_count) as f64;
    }

    Ok(total)
}

/// Compute BFS shortest distances from a start node, following all edge types
/// in both directions, up to max_depth hops.
pub fn bfs_distances(
    graph: &GraphStore,
    start: &NodeId,
    max_depth: usize,
) -> Result<HashMap<NodeId, usize>, AgentError> {
    let mut distances = HashMap::new();
    distances.insert(start.clone(), 0);
    let mut queue = VecDeque::new();
    queue.push_back((start.clone(), 0_usize));

    while let Some((current, depth)) = queue.pop_front() {
        if depth >= max_depth {
            continue;
        }

        let outgoing = graph
            .neighbors(&current, None, Direction::Outgoing)
            .map_err(|e| AgentError::Context(e.to_string()))?;
        let incoming = graph
            .neighbors(&current, None, Direction::Incoming)
            .map_err(|e| AgentError::Context(e.to_string()))?;

        for neighbor in outgoing.iter().chain(incoming.iter()) {
            if !distances.contains_key(&neighbor.id) {
                distances.insert(neighbor.id.clone(), depth + 1);
                queue.push_back((neighbor.id.clone(), depth + 1));
            }
        }
    }

    Ok(distances)
}

/// Score a node based on graph distance from the current turn.
/// Formula: 1.0 / (1.0 + hops). Returns 0.0 if unreachable.
pub fn score_graph_distance(node_id: &NodeId, distances: &HashMap<NodeId, usize>) -> f64 {
    match distances.get(node_id) {
        Some(&hops) => 1.0 / (1.0 + hops as f64),
        None => 0.0,
    }
}

/// Select the highest-scored nodes that fit within a token budget.
/// Greedy approach: sort by score descending, take nodes until budget exhausted.
/// Skips individual nodes that don't fit, continues to try smaller ones.
pub fn fit_to_budget(mut scored: Vec<ScoredNode>, budget: usize) -> Vec<ScoredNode> {
    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    let mut selected = Vec::new();
    let mut remaining = budget;

    for node in scored {
        if node.token_estimate <= remaining {
            remaining -= node.token_estimate;
            selected.push(node);
        }
    }

    selected
}

const W_RECENCY: f64 = 0.3;
const W_EDGE: f64 = 0.2;
const W_DISTANCE: f64 = 0.3;
const W_PAGERANK: f64 = 0.2;

/// Compute composite relevance score for a node.
/// Combines four signals: recency, edge type weights, graph distance, and PageRank.
/// Returns a value in approximately [0, 1].
pub fn score_node(
    node: &GraphNode,
    pagerank_scores: &HashMap<NodeId, f64>,
    distances: &HashMap<NodeId, usize>,
    config: &ContextConfig,
    graph: &GraphStore,
) -> Result<f64, AgentError> {
    let recency = score_recency(node, config.recency_decay);

    let edge_raw = score_edge_weights(&node.id, graph, &config.edge_weights)?;
    let edge_normalized = edge_raw / (1.0 + edge_raw);

    let distance = score_graph_distance(&node.id, distances);

    let pagerank = pagerank_scores.get(&node.id).copied().unwrap_or(0.0);

    Ok(
        W_RECENCY * recency
            + W_EDGE * edge_normalized
            + W_DISTANCE * distance
            + W_PAGERANK * pagerank.min(1.0),
    )
}

/// Compute recency score for a node using exponential decay.
/// Formula: e^(-decay * hours_since_creation)
/// Returns a value in (0, 1] where 1.0 means "just created".
pub fn score_recency(node: &GraphNode, decay: f64) -> f64 {
    let now = Utc::now();
    let elapsed = now.signed_duration_since(node.created_at);
    let hours = elapsed.num_seconds() as f64 / 3600.0;
    let hours = hours.max(0.0);
    (-decay * hours).exp()
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
    fn fit_to_budget_selects_top_scored() {
        let make_scored = |content: &str, score: f64, tokens: usize| -> ScoredNode {
            ScoredNode {
                node: GraphNode::new(NodeType::Interaction(InteractionData {
                    role: "user".to_string(),
                    content: content.to_string(),
                    token_count: None,
                })),
                score,
                token_estimate: tokens,
            }
        };

        let scored = vec![
            make_scored("low", 0.1, 10),
            make_scored("high", 0.9, 10),
            make_scored("medium", 0.5, 10),
            make_scored("very-high", 0.95, 10),
            make_scored("medium2", 0.4, 10),
        ];

        // Budget for 30 tokens → fits 3 nodes of 10 tokens each
        let selected = fit_to_budget(scored, 30);
        assert_eq!(selected.len(), 3);

        assert!(selected[0].score > selected[1].score);
        assert!(selected[1].score > selected[2].score);

        assert!((selected[0].score - 0.95).abs() < f64::EPSILON);
        assert!((selected[1].score - 0.9).abs() < f64::EPSILON);
        assert!((selected[2].score - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn fit_to_budget_respects_token_limit() {
        let make_scored = |tokens: usize, score: f64| -> ScoredNode {
            ScoredNode {
                node: GraphNode::new(NodeType::Interaction(InteractionData {
                    role: "user".to_string(),
                    content: "x".to_string(),
                    token_count: None,
                })),
                score,
                token_estimate: tokens,
            }
        };

        let scored = vec![
            make_scored(50, 0.9),  // takes 50/100
            make_scored(40, 0.8),  // takes 90/100
            make_scored(30, 0.7),  // would need 120 → skip
            make_scored(10, 0.6),  // takes 100/100
            make_scored(5, 0.5),   // would need 105 → skip
        ];

        let selected = fit_to_budget(scored, 100);
        assert_eq!(selected.len(), 3);
        let total: usize = selected.iter().map(|s| s.token_estimate).sum();
        assert!(total <= 100, "Total tokens {total} should fit in budget 100");
        assert_eq!(total, 100); // 50 + 40 + 10
    }

    #[test]
    fn fit_to_budget_empty_input() {
        let selected = fit_to_budget(vec![], 1000);
        assert!(selected.is_empty());
    }

    #[test]
    fn fit_to_budget_zero_budget() {
        let scored = vec![ScoredNode {
            node: GraphNode::new(NodeType::Interaction(InteractionData {
                role: "user".to_string(),
                content: "test".to_string(),
                token_count: None,
            })),
            score: 1.0,
            token_estimate: 10,
        }];
        let selected = fit_to_budget(scored, 0);
        assert!(selected.is_empty());
    }

    #[test]
    fn score_node_composite_ranking() {
        let graph = GraphStore::open_memory().unwrap();
        let config = ContextConfig::default();

        let current_turn = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "Fix the bug in main.rs".to_string(),
            token_count: None,
        }));
        let ct_id = current_turn.id.clone();
        graph.add_node(current_turn.clone()).unwrap();

        let mut old_msg = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".to_string(),
            content: "I can help with that.".to_string(),
            token_count: None,
        }));
        old_msg.created_at = Utc::now() - Duration::hours(10);
        old_msg.updated_at = old_msg.created_at;
        let old_id = old_msg.id.clone();
        graph.add_node(old_msg.clone()).unwrap();
        graph
            .add_edge(GraphEdge::new(EdgeType::RespondsTo, ct_id.clone(), old_id.clone()))
            .unwrap();

        let file_node = GraphNode::new(NodeType::Content(ContentData {
            content_type: "file".to_string(),
            path: Some("main.rs".to_string()),
            body: "fn main() { panic!() }".to_string(),
            language: Some("rust".to_string()),
        }));
        let file_id = file_node.id.clone();
        graph.add_node(file_node.clone()).unwrap();
        graph
            .add_edge(GraphEdge::new(EdgeType::Modifies, ct_id.clone(), file_id.clone()))
            .unwrap();

        let pr_vec = graph.pagerank().unwrap();
        let pagerank_scores: HashMap<NodeId, f64> = pr_vec.into_iter().collect();
        let distances = bfs_distances(&graph, &ct_id, 10).unwrap();

        let score_old = score_node(&old_msg, &pagerank_scores, &distances, &config, &graph).unwrap();
        let score_file = score_node(&file_node, &pagerank_scores, &distances, &config, &graph).unwrap();

        assert!(
            score_file > score_old,
            "Recent file with Modifies edge ({score_file}) should outscore \
             10h-old message with RespondsTo edge ({score_old})"
        );
    }

    #[test]
    fn score_node_all_components_contribute() {
        let graph = GraphStore::open_memory().unwrap();
        let config = ContextConfig::default();

        let node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "test".to_string(),
            token_count: None,
        }));
        let id = node.id.clone();
        graph.add_node(node.clone()).unwrap();

        let mut pagerank_scores = HashMap::new();
        pagerank_scores.insert(id.clone(), 0.5);

        let mut distances = HashMap::new();
        distances.insert(id.clone(), 0);

        let score = score_node(&node, &pagerank_scores, &distances, &config, &graph).unwrap();

        assert!(score > 0.5, "Score should be substantial, got {score}");
        assert!(score < 1.0, "Score should be below 1.0, got {score}");
    }

    #[test]
    fn bfs_distances_from_start() {
        let graph = GraphStore::open_memory().unwrap();

        let make_node = |content: &str| {
            GraphNode::new(NodeType::Interaction(InteractionData {
                role: "user".to_string(),
                content: content.to_string(),
                token_count: None,
            }))
        };

        let a = make_node("a"); let a_id = a.id.clone();
        let b = make_node("b"); let b_id = b.id.clone();
        let c = make_node("c"); let c_id = c.id.clone();
        let d = make_node("d"); let d_id = d.id.clone();

        graph.add_node(a).unwrap();
        graph.add_node(b).unwrap();
        graph.add_node(c).unwrap();
        graph.add_node(d).unwrap();

        graph.add_edge(GraphEdge::new(EdgeType::RespondsTo, a_id.clone(), b_id.clone())).unwrap();
        graph.add_edge(GraphEdge::new(EdgeType::RespondsTo, b_id.clone(), c_id.clone())).unwrap();
        graph.add_edge(GraphEdge::new(EdgeType::RespondsTo, c_id.clone(), d_id.clone())).unwrap();

        let distances = bfs_distances(&graph, &a_id, 10).unwrap();
        assert_eq!(distances[&a_id], 0);
        assert_eq!(distances[&b_id], 1);
        assert_eq!(distances[&c_id], 2);
        assert_eq!(distances[&d_id], 3);
    }

    #[test]
    fn bfs_distances_respects_max_depth() {
        let graph = GraphStore::open_memory().unwrap();

        let make_node = |c: &str| {
            GraphNode::new(NodeType::Interaction(InteractionData {
                role: "user".to_string(),
                content: c.to_string(),
                token_count: None,
            }))
        };

        let a = make_node("a"); let a_id = a.id.clone();
        let b = make_node("b"); let b_id = b.id.clone();
        let c = make_node("c"); let c_id = c.id.clone();
        graph.add_node(a).unwrap();
        graph.add_node(b).unwrap();
        graph.add_node(c).unwrap();
        graph.add_edge(GraphEdge::new(EdgeType::RespondsTo, a_id.clone(), b_id.clone())).unwrap();
        graph.add_edge(GraphEdge::new(EdgeType::RespondsTo, b_id.clone(), c_id.clone())).unwrap();

        let distances = bfs_distances(&graph, &a_id, 1).unwrap();
        assert!(distances.contains_key(&a_id));
        assert!(distances.contains_key(&b_id));
        assert!(!distances.contains_key(&c_id), "C should be beyond max_depth=1");
    }

    #[test]
    fn score_graph_distance_values() {
        let mut distances = HashMap::new();
        let id_0 = NodeId::from("n0");
        let id_1 = NodeId::from("n1");
        let id_3 = NodeId::from("n3");
        let id_missing = NodeId::from("missing");

        distances.insert(id_0.clone(), 0);
        distances.insert(id_1.clone(), 1);
        distances.insert(id_3.clone(), 3);

        assert!((score_graph_distance(&id_0, &distances) - 1.0).abs() < f64::EPSILON);
        assert!((score_graph_distance(&id_1, &distances) - 0.5).abs() < f64::EPSILON);
        assert!((score_graph_distance(&id_3, &distances) - 0.25).abs() < f64::EPSILON);
        assert!((score_graph_distance(&id_missing, &distances) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn score_graph_distance_neighbor_beats_distant() {
        let mut distances = HashMap::new();
        let near = NodeId::from("near");
        let far = NodeId::from("far");
        distances.insert(near.clone(), 1);
        distances.insert(far.clone(), 5);

        assert!(
            score_graph_distance(&near, &distances) > score_graph_distance(&far, &distances),
            "Neighbor should score higher than 5-hop node"
        );
    }

    #[test]
    fn score_edge_weights_modifies_higher_than_reads() {
        let graph = GraphStore::open_memory().unwrap();
        let weights = EdgeWeights::default();

        let a = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".to_string(),
            content: "I'll edit that file.".to_string(),
            token_count: None,
        }));
        let content = GraphNode::new(NodeType::Content(ContentData {
            content_type: "file".to_string(),
            path: Some("main.rs".to_string()),
            body: "fn main() {}".to_string(),
            language: Some("rust".to_string()),
        }));
        let a_id = a.id.clone();
        let content_id = content.id.clone();
        graph.add_node(a).unwrap();
        graph.add_node(content.clone()).unwrap();
        graph
            .add_edge(GraphEdge::new(EdgeType::Modifies, a_id.clone(), content_id.clone()))
            .unwrap();

        let b = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".to_string(),
            content: "Let me read that.".to_string(),
            token_count: None,
        }));
        let content2 = GraphNode::new(NodeType::Content(ContentData {
            content_type: "file".to_string(),
            path: Some("lib.rs".to_string()),
            body: "pub mod store;".to_string(),
            language: Some("rust".to_string()),
        }));
        let b_id = b.id.clone();
        let c2_id = content2.id.clone();
        graph.add_node(b).unwrap();
        graph.add_node(content2).unwrap();
        graph
            .add_edge(GraphEdge::new(EdgeType::Reads, b_id.clone(), c2_id.clone()))
            .unwrap();

        let score_a = score_edge_weights(&a_id, &graph, &weights).unwrap();
        let score_b = score_edge_weights(&b_id, &graph, &weights).unwrap();
        assert!(
            score_a > score_b,
            "Modifies ({score_a}) should score higher than Reads ({score_b})"
        );
    }

    #[test]
    fn score_edge_weights_no_edges_scores_zero() {
        let graph = GraphStore::open_memory().unwrap();
        let weights = EdgeWeights::default();

        let node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "isolated".to_string(),
            token_count: None,
        }));
        let id = node.id.clone();
        graph.add_node(node).unwrap();

        let score = score_edge_weights(&id, &graph, &weights).unwrap();
        assert!((score - 0.0).abs() < f64::EPSILON, "Isolated node should score 0.0");
    }

    #[test]
    fn score_edge_weights_multiple_edges_sum() {
        let graph = GraphStore::open_memory().unwrap();
        let weights = EdgeWeights::default();

        let agent = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".to_string(),
            content: "working".to_string(),
            token_count: None,
        }));
        let agent_id = agent.id.clone();
        graph.add_node(agent).unwrap();

        // Agent modifies two files and reads one
        for i in 0..3 {
            let content = GraphNode::new(NodeType::Content(ContentData {
                content_type: "file".to_string(),
                path: Some(format!("file{i}.rs")),
                body: "code".to_string(),
                language: Some("rust".to_string()),
            }));
            let cid = content.id.clone();
            graph.add_node(content).unwrap();
            let et = if i < 2 { EdgeType::Modifies } else { EdgeType::Reads };
            graph
                .add_edge(GraphEdge::new(et, agent_id.clone(), cid))
                .unwrap();
        }

        let score = score_edge_weights(&agent_id, &graph, &weights).unwrap();
        // 2 * Modifies(1.0) + 1 * Reads(0.6) = 2.6
        assert!(
            (score - 2.6).abs() < f64::EPSILON,
            "Expected 2.6, got {score}"
        );
    }

    #[test]
    fn score_recency_recent_node_scores_high() {
        let node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "just now".to_string(),
            token_count: None,
        }));
        // Node was just created → hours ≈ 0 → e^0 = 1.0
        let score = score_recency(&node, 0.1);
        assert!(score > 0.99, "Recent node should score near 1.0, got {score}");
    }

    #[test]
    fn score_recency_old_node_scores_low() {
        let mut node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "old message".to_string(),
            token_count: None,
        }));
        // Set created_at to 24 hours ago
        node.created_at = Utc::now() - Duration::hours(24);
        node.updated_at = node.created_at;

        let score = score_recency(&node, 0.1);
        // e^(-0.1 * 24) = e^(-2.4) ≈ 0.0907
        assert!(score < 0.15, "24h-old node with decay=0.1 should score low, got {score}");
        assert!(score > 0.05, "Score should be positive, got {score}");
    }

    #[test]
    fn score_recency_one_hour_beats_twenty_four_hours() {
        let mut recent = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "1h ago".to_string(),
            token_count: None,
        }));
        recent.created_at = Utc::now() - Duration::hours(1);

        let mut old = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "24h ago".to_string(),
            token_count: None,
        }));
        old.created_at = Utc::now() - Duration::hours(24);

        let score_recent = score_recency(&recent, 0.1);
        let score_old = score_recency(&old, 0.1);
        assert!(
            score_recent > score_old,
            "1h ago ({score_recent}) should score higher than 24h ago ({score_old})"
        );
    }

    #[test]
    fn score_recency_higher_decay_penalizes_more() {
        let mut node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "test".to_string(),
            token_count: None,
        }));
        node.created_at = Utc::now() - Duration::hours(5);

        let score_gentle = score_recency(&node, 0.05);
        let score_harsh = score_recency(&node, 0.5);
        assert!(
            score_gentle > score_harsh,
            "Lower decay ({score_gentle}) should give higher score than higher decay ({score_harsh})"
        );
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
