# Phase 6: Context Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace Phase 4's simple linear context builder with a graph-native relevance engine that uses PageRank, edge type weights, recency decay, token budgets, and LLM-driven compaction to build optimal context windows.

**Architecture:** The context engine lives in `crates/agent/src/context.rs` (scoring and building) and `crates/agent/src/compact.rs` (summarization). It replaces Phase 4's `build_context()` which simply walked `Produces` edges. The new engine computes a composite relevance score for every candidate node (Interaction, Content, Knowledge) using four signals: recency (exponential decay), edge type weights (Modifies > Produces > Reads), graph distance (BFS hops from current turn), and PageRank centrality. A greedy knapsack packs the highest-scored nodes into the token budget, with guaranteed slots for the N most recent turns. When sessions grow large, auto-compaction summarizes old nodes via a lightweight LLM call, creating Knowledge nodes with `Summarizes` edges.

**Knowledge nodes as first-class relevance inputs:** `Knowledge` nodes written by Phase 9's background consciousness loop are scored and ranked alongside `Interaction` and `Content` nodes — they are not injected as a separate block. A Knowledge node about "this codebase uses newtypes for all IDs" competes on the same relevance score as the most recent conversation turn. High-PageRank Knowledge nodes (referenced across many sessions) naturally surface in every context window without any special-casing. This is how agent identity emerges from the graph: there is no `identity.md` or hardcoded system prompt — the accumulated weight of Knowledge nodes *is* the identity, shaped by what the agent has actually done and learned.

**Tech Stack:** `graphirm-graph` (GraphStore, PageRank, traversals), `graphirm-llm` (MockProvider for compaction tests, LlmProvider for production), `chrono` (time deltas), `serde` (config deserialization), `tokio` (async compaction)

---

## Prerequisites (expected APIs from Phases 1–4)

Phase 6 depends on types and methods from earlier phases. If actual signatures differ after Phases 1–4 land, adapt the code below — the logic and test structure remain the same.

### From `graphirm-graph` (Phase 1)

```rust
use graphirm_graph::nodes::{
    NodeId, GraphNode, NodeType, InteractionData, ContentData, KnowledgeData, AgentData, TaskData,
};
use graphirm_graph::edges::{EdgeType, GraphEdge};
use graphirm_graph::store::GraphStore;
use graphirm_graph::Direction;

// GraphStore methods used by Phase 6:
impl GraphStore {
    pub fn open_memory() -> Result<Self, GraphError>;
    pub fn add_node(&self, node: GraphNode) -> Result<NodeId, GraphError>;
    pub fn get_node(&self, id: &NodeId) -> Result<GraphNode, GraphError>;
    pub fn update_node(&self, id: &NodeId, node: GraphNode) -> Result<(), GraphError>;
    pub fn add_edge(&self, edge: GraphEdge) -> Result<EdgeId, GraphError>;
    pub fn neighbors(&self, id: &NodeId, edge_type: Option<EdgeType>, direction: Direction) -> Result<Vec<GraphNode>, GraphError>;
    pub fn conversation_thread(&self, leaf_id: &NodeId) -> Result<Vec<GraphNode>, GraphError>; // newest-first
    pub fn pagerank(&self) -> Result<Vec<(NodeId, f64)>, GraphError>; // sorted desc
}
```

### From `graphirm-llm` (Phase 2)

```rust
use graphirm_llm::{
    LlmMessage, LlmProvider, MockProvider, MockResponse,
    CompletionConfig, ContentPart, Role, TokenUsage,
};

// LlmMessage constructors:
LlmMessage::system("text")    // Role::System
LlmMessage::human("text")     // Role::Human
LlmMessage::assistant("text") // Role::Assistant

// MockProvider for testing:
MockProvider::fixed("canned response text")
```

### From `graphirm-agent` (Phase 4)

```rust
// Phase 4's context.rs has a simple build_context() that we're replacing.
// Phase 4's compact.rs is a stub file.
// Phase 4's AgentError has a Context(String) variant.
// Phase 4's Cargo.toml already has chrono, serde, tokio, etc.
```

### Re-exports needed

Ensure `crates/graph/src/lib.rs` re-exports the types Phase 6 needs:

```rust
pub use nodes::{NodeId, GraphNode, NodeType, InteractionData, ContentData, KnowledgeData, AgentData, TaskData};
pub use edges::{EdgeId, EdgeType, GraphEdge};
pub use store::GraphStore;
pub use error::GraphError;
pub use petgraph::Direction;
```

If these aren't re-exported yet, either add them or adjust the `use` paths in the code below (e.g., `graphirm_graph::nodes::NodeId` instead of `graphirm_graph::NodeId`).

## Branch

```bash
git checkout -b phase-6/context-engine main
```

---

## Task 1: Define ContextConfig and EdgeWeights

- [x] Complete

**Files:**
- Modify: `crates/agent/src/context.rs`
- Test: `crates/agent/src/context.rs` (in-module `#[cfg(test)]`)

### Step 1: Write the structs with Deserialize and Default

Replace the contents of `crates/agent/src/context.rs` with:

```rust
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
```

### Step 2: Run tests

Run: `cargo test -p graphirm-agent context::tests -- --nocapture 2>&1`
Expected: All 4 tests pass.

### Step 3: Commit

```bash
git add crates/agent/src/context.rs
git commit -m "feat(agent): define ContextConfig and EdgeWeights with serde defaults"
```

---

## Task 2: Define ScoredNode and ContextWindow

- [x] Complete

**Files:**
- Modify: `crates/agent/src/context.rs`
- Test: `crates/agent/src/context.rs` (in-module `#[cfg(test)]`)

### Step 1: Add data structures and helpers

Add above the `#[cfg(test)]` block in `crates/agent/src/context.rs`:

```rust
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
```

### Step 2: Write tests

Add to the `#[cfg(test)] mod tests` block:

```rust
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
```

### Step 3: Run tests

Run: `cargo test -p graphirm-agent context::tests -- --nocapture 2>&1`
Expected: All 13 tests pass (4 from Task 1 + 9 new).

### Step 4: Commit

```bash
git add crates/agent/src/context.rs
git commit -m "feat(agent): define ScoredNode, ContextWindow, and node conversion helpers"
```

---

## Task 3: Implement estimate_tokens()

- [x] Complete

**Files:**
- Modify: `crates/agent/src/context.rs`
- Test: `crates/agent/src/context.rs` (in-module `#[cfg(test)]`)

### Step 1: Write the failing tests

Add to the `tests` module:

```rust
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
```

### Step 2: Run tests to verify they fail

Run: `cargo test -p graphirm-agent context::tests::estimate_tokens -- --nocapture 2>&1`
Expected: FAIL — `estimate_tokens` and `estimate_tokens_str` don't exist.

### Step 3: Implement

Add above the `#[cfg(test)]` block:

```rust
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
```

### Step 4: Run tests

Run: `cargo test -p graphirm-agent context::tests::estimate_tokens -- --nocapture 2>&1`
Expected: All 3 tests pass.

### Step 5: Commit

```bash
git add crates/agent/src/context.rs
git commit -m "feat(agent): implement estimate_tokens with word-based heuristic"
```

---

## Task 4: Implement recency scoring

- [x] Complete

**Files:**
- Modify: `crates/agent/src/context.rs`
- Test: `crates/agent/src/context.rs` (in-module `#[cfg(test)]`)

### Step 1: Write the failing tests

Add to the `tests` module:

```rust
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
```

### Step 2: Run tests to verify they fail

Run: `cargo test -p graphirm-agent context::tests::score_recency -- --nocapture 2>&1`
Expected: FAIL — `score_recency` doesn't exist.

### Step 3: Implement

Add above the `#[cfg(test)]` block:

```rust
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
```

### Step 4: Run tests

Run: `cargo test -p graphirm-agent context::tests::score_recency -- --nocapture 2>&1`
Expected: All 4 tests pass.

### Step 5: Commit

```bash
git add crates/agent/src/context.rs
git commit -m "feat(agent): implement score_recency with exponential decay"
```

---

## Task 5: Implement edge weight scoring

- [x] Complete

**Files:**
- Modify: `crates/agent/src/context.rs`
- Test: `crates/agent/src/context.rs` (in-module `#[cfg(test)]`)

### Step 1: Write the failing tests

Add to the `tests` module:

```rust
    #[test]
    fn score_edge_weights_modifies_higher_than_reads() {
        let graph = GraphStore::open_memory().unwrap();
        let weights = EdgeWeights::default();

        // Node A modifies a content node
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

        // Node B only reads a content node
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
```

### Step 2: Run tests to verify they fail

Run: `cargo test -p graphirm-agent context::tests::score_edge_weights -- --nocapture 2>&1`
Expected: FAIL — `score_edge_weights` doesn't exist.

### Step 3: Implement

Add above the `#[cfg(test)]` block:

```rust
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
```

### Step 4: Run tests

Run: `cargo test -p graphirm-agent context::tests::score_edge_weights -- --nocapture 2>&1`
Expected: All 3 tests pass.

### Step 5: Commit

```bash
git add crates/agent/src/context.rs
git commit -m "feat(agent): implement score_edge_weights with per-type weighted counting"
```

---

## Task 6: Implement graph distance scoring

- [x] Complete

**Files:**
- Modify: `crates/agent/src/context.rs`
- Test: `crates/agent/src/context.rs` (in-module `#[cfg(test)]`)

### Step 1: Write the failing tests

Add to the `tests` module:

```rust
    #[test]
    fn bfs_distances_from_start() {
        let graph = GraphStore::open_memory().unwrap();

        // Chain: A --RespondsTo--> B --RespondsTo--> C --RespondsTo--> D
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

        // hops=0 → 1.0/(1+0) = 1.0
        assert!((score_graph_distance(&id_0, &distances) - 1.0).abs() < f64::EPSILON);
        // hops=1 → 1.0/(1+1) = 0.5
        assert!((score_graph_distance(&id_1, &distances) - 0.5).abs() < f64::EPSILON);
        // hops=3 → 1.0/(1+3) = 0.25
        assert!((score_graph_distance(&id_3, &distances) - 0.25).abs() < f64::EPSILON);
        // not reachable → 0.0
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
```

### Step 2: Run tests to verify they fail

Run: `cargo test -p graphirm-agent context::tests::bfs_distances -- --nocapture 2>&1`
Expected: FAIL — `bfs_distances` doesn't exist.

### Step 3: Implement

Add above the `#[cfg(test)]` block:

```rust
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
```

### Step 4: Run tests

Run: `cargo test -p graphirm-agent context::tests::bfs_distances -- --nocapture 2>&1`
Run: `cargo test -p graphirm-agent context::tests::score_graph_distance -- --nocapture 2>&1`
Expected: All 4 tests pass.

### Step 5: Commit

```bash
git add crates/agent/src/context.rs
git commit -m "feat(agent): implement BFS distance scoring with inverse decay"
```

---

## Task 7: Implement composite score_node()

- [x] Complete

**Files:**
- Modify: `crates/agent/src/context.rs`
- Test: `crates/agent/src/context.rs` (in-module `#[cfg(test)]`)

### Step 1: Write the failing tests

Add to the `tests` module:

```rust
    #[test]
    fn score_node_composite_ranking() {
        let graph = GraphStore::open_memory().unwrap();
        let config = ContextConfig::default();

        // Build a small graph: current_turn → responds_to → old_msg
        //                      current_turn → modifies → file_node
        let current_turn = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "Fix the bug in main.rs".to_string(),
            token_count: None,
        }));
        let ct_id = current_turn.id.clone();
        graph.add_node(current_turn.clone()).unwrap();

        // Old message: 10 hours ago, 1 hop away via RespondsTo
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

        // File node: recent, 1 hop away via Modifies (higher weight)
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

        // Compute PageRank
        let pr_vec = graph.pagerank().unwrap();
        let pagerank_scores: HashMap<NodeId, f64> =
            pr_vec.into_iter().collect();

        // BFS distances from current turn
        let distances = bfs_distances(&graph, &ct_id, 10).unwrap();

        // Score both candidates
        let score_old = score_node(
            &old_msg, &pagerank_scores, &distances, &config, &graph,
        ).unwrap();
        let score_file = score_node(
            &file_node, &pagerank_scores, &distances, &config, &graph,
        ).unwrap();

        // File node should score higher: it's recent + has Modifies edge (weight 1.0)
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

        let score = score_node(
            &node, &pagerank_scores, &distances, &config, &graph,
        ).unwrap();

        // All signals are high (recent, distance=0, pagerank=0.5, no edges though)
        // recency ≈ 1.0, distance = 1.0, pagerank = 0.5, edges = 0.0
        // 0.3*1.0 + 0.2*0.0 + 0.3*1.0 + 0.2*0.5 = 0.3 + 0 + 0.3 + 0.1 = 0.7
        assert!(score > 0.5, "Score should be substantial, got {score}");
        assert!(score < 1.0, "Score should be below 1.0, got {score}");
    }
```

### Step 2: Run tests to verify they fail

Run: `cargo test -p graphirm-agent context::tests::score_node -- --nocapture 2>&1`
Expected: FAIL — `score_node` doesn't exist.

### Step 3: Implement

Add above the `#[cfg(test)]` block:

```rust
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
```

### Step 4: Run tests

Run: `cargo test -p graphirm-agent context::tests::score_node -- --nocapture 2>&1`
Expected: All 2 tests pass.

### Step 5: Commit

```bash
git add crates/agent/src/context.rs
git commit -m "feat(agent): implement composite score_node combining 4 relevance signals"
```

---

## Task 8: Implement fit_to_budget()

- [x] Complete

**Files:**
- Modify: `crates/agent/src/context.rs`
- Test: `crates/agent/src/context.rs` (in-module `#[cfg(test)]`)

### Step 1: Write the failing tests

Add to the `tests` module:

```rust
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

        // Verify they're sorted by score descending
        assert!(selected[0].score > selected[1].score);
        assert!(selected[1].score > selected[2].score);

        // Top 3 should be: very-high (0.95), high (0.9), medium (0.5)
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
```

### Step 2: Run tests to verify they fail

Run: `cargo test -p graphirm-agent context::tests::fit_to_budget -- --nocapture 2>&1`
Expected: FAIL — `fit_to_budget` doesn't exist.

### Step 3: Implement

Add above the `#[cfg(test)]` block:

```rust
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
```

### Step 4: Run tests

Run: `cargo test -p graphirm-agent context::tests::fit_to_budget -- --nocapture 2>&1`
Expected: All 4 tests pass.

### Step 5: Commit

```bash
git add crates/agent/src/context.rs
git commit -m "feat(agent): implement fit_to_budget with greedy score-ordered selection"
```

---

## Task 9: Implement build_context()

- [x] Complete

**Files:**
- Modify: `crates/agent/src/context.rs`
- Test: `crates/agent/src/context.rs` (in-module `#[cfg(test)]`)

### Step 1: Write the failing test

Add to the `tests` module:

```rust
    #[test]
    fn build_context_from_session() {
        let graph = GraphStore::open_memory().unwrap();

        // Create agent node
        let agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "test-agent".to_string(),
            model: "mock".to_string(),
            system_prompt: Some("You are helpful.".to_string()),
            status: "running".to_string(),
        }));
        let agent_id = agent.id.clone();
        graph.add_node(agent).unwrap();

        // Simulate a 20-message conversation
        let mut prev_id: Option<NodeId> = None;
        for i in 0..20 {
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            let mut node = GraphNode::new(NodeType::Interaction(InteractionData {
                role: role.to_string(),
                content: format!("Message {i}"),
                token_count: None,
            }));
            // Spread messages over 20 hours (oldest first)
            node.created_at = Utc::now() - Duration::hours(20 - i as i64);
            node.updated_at = node.created_at;
            let node_id = node.id.clone();
            graph.add_node(node).unwrap();

            // Agent produces this message
            graph
                .add_edge(GraphEdge::new(
                    EdgeType::Produces,
                    agent_id.clone(),
                    node_id.clone(),
                ))
                .unwrap();

            // RespondsTo chain
            if let Some(pid) = &prev_id {
                graph
                    .add_edge(GraphEdge::new(
                        EdgeType::RespondsTo,
                        node_id.clone(),
                        pid.clone(),
                    ))
                    .unwrap();
            }
            prev_id = Some(node_id);
        }

        let config = ContextConfig {
            max_tokens: 500,
            system_prompt: "You are helpful.".to_string(),
            guaranteed_recent_turns: 4,
            recency_decay: 0.1,
            enable_compaction: false,
            ..ContextConfig::default()
        };

        let window = build_context(&graph, agent_id, &config).unwrap();

        // System prompt should be present
        assert_eq!(window.system.role, graphirm_llm::Role::System);

        // Should have at least the guaranteed recent turns
        assert!(
            window.messages.len() >= 4,
            "Should include at least 4 guaranteed recent turns, got {}",
            window.messages.len()
        );

        // Total tokens should respect budget
        assert!(
            window.total_tokens <= config.max_tokens,
            "Total tokens {} should fit in budget {}",
            window.total_tokens,
            config.max_tokens
        );

        // Most recent message should be "Message 19"
        let last_msg = window.messages.last().unwrap();
        // Check it contains the last message content
        let last_content = &last_msg.content;
        let has_msg_19 = last_content.iter().any(|part| {
            matches!(part, graphirm_llm::ContentPart::Text { text } if text.contains("Message 19"))
        });
        assert!(has_msg_19, "Last message should contain 'Message 19'");
    }

    #[test]
    fn build_context_empty_conversation() {
        let graph = GraphStore::open_memory().unwrap();

        let agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "test".to_string(),
            model: "mock".to_string(),
            system_prompt: None,
            status: "running".to_string(),
        }));
        let agent_id = agent.id.clone();
        graph.add_node(agent).unwrap();

        let config = ContextConfig::default();
        let window = build_context(&graph, agent_id, &config).unwrap();

        assert_eq!(window.system.role, graphirm_llm::Role::System);
        assert!(window.messages.is_empty());
    }

    #[test]
    fn build_context_includes_content_nodes() {
        let graph = GraphStore::open_memory().unwrap();

        let agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "coder".to_string(),
            model: "mock".to_string(),
            system_prompt: None,
            status: "running".to_string(),
        }));
        let agent_id = agent.id.clone();
        graph.add_node(agent).unwrap();

        // User message
        let user_msg = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "Fix the bug".to_string(),
            token_count: None,
        }));
        let user_id = user_msg.id.clone();
        graph.add_node(user_msg).unwrap();
        graph
            .add_edge(GraphEdge::new(EdgeType::Produces, agent_id.clone(), user_id.clone()))
            .unwrap();

        // Assistant response that modifies a file
        let asst_msg = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".to_string(),
            content: "I'll fix main.rs".to_string(),
            token_count: None,
        }));
        let asst_id = asst_msg.id.clone();
        graph.add_node(asst_msg).unwrap();
        graph
            .add_edge(GraphEdge::new(EdgeType::Produces, agent_id.clone(), asst_id.clone()))
            .unwrap();
        graph
            .add_edge(GraphEdge::new(EdgeType::RespondsTo, asst_id.clone(), user_id.clone()))
            .unwrap();

        // Content node linked via Modifies
        let file_node = GraphNode::new(NodeType::Content(ContentData {
            content_type: "file".to_string(),
            path: Some("main.rs".to_string()),
            body: "fn main() { fixed() }".to_string(),
            language: Some("rust".to_string()),
        }));
        let file_id = file_node.id.clone();
        graph.add_node(file_node).unwrap();
        graph
            .add_edge(GraphEdge::new(EdgeType::Modifies, asst_id.clone(), file_id.clone()))
            .unwrap();

        let config = ContextConfig {
            max_tokens: 10_000,
            system_prompt: "Help.".to_string(),
            guaranteed_recent_turns: 2,
            ..ContextConfig::default()
        };

        let window = build_context(&graph, agent_id, &config).unwrap();

        // Should include the content node as context
        let all_text: String = window
            .messages
            .iter()
            .flat_map(|m| m.content.iter())
            .filter_map(|p| match p {
                graphirm_llm::ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ");

        assert!(
            all_text.contains("main.rs") || all_text.contains("fixed()"),
            "Context should include the modified file content"
        );
    }
```

### Step 2: Run tests to verify they fail

Run: `cargo test -p graphirm-agent context::tests::build_context -- --nocapture 2>&1`
Expected: FAIL — `build_context` doesn't exist (or the Phase 4 version has a different signature).

### Step 3: Implement

Add above the `#[cfg(test)]` block:

```rust
/// Find the most recent Interaction node linked to the agent via Produces.
fn find_current_turn(
    graph: &GraphStore,
    agent_id: &NodeId,
) -> Result<Option<GraphNode>, AgentError> {
    let neighbors = graph
        .neighbors(agent_id, Some(EdgeType::Produces), Direction::Outgoing)
        .map_err(|e| AgentError::Context(e.to_string()))?;

    let mut interactions: Vec<GraphNode> = neighbors
        .into_iter()
        .filter(|n| matches!(n.node_type, NodeType::Interaction(_)))
        .collect();

    interactions.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(interactions.into_iter().next())
}

/// Collect Content and Knowledge nodes reachable from conversation nodes
/// via Reads, Modifies, Produces, and Summarizes edges (1 hop).
fn collect_context_nodes(
    graph: &GraphStore,
    conversation_ids: &[NodeId],
) -> Result<Vec<GraphNode>, AgentError> {
    let mut seen = std::collections::HashSet::new();
    let mut context_nodes = Vec::new();

    for conv_id in conversation_ids {
        seen.insert(conv_id.clone());
    }

    let relevant_edges = [
        EdgeType::Reads,
        EdgeType::Modifies,
        EdgeType::Produces,
        EdgeType::Summarizes,
    ];

    for conv_id in conversation_ids {
        for et in &relevant_edges {
            let neighbors = graph
                .neighbors(conv_id, Some(*et), Direction::Outgoing)
                .map_err(|e| AgentError::Context(e.to_string()))?;

            for neighbor in neighbors {
                if seen.insert(neighbor.id.clone()) {
                    match &neighbor.node_type {
                        NodeType::Content(_) | NodeType::Knowledge(_) => {
                            context_nodes.push(neighbor);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    Ok(context_nodes)
}

/// Build a relevance-scored context window from the session graph.
///
/// Algorithm:
/// 1. Find current turn (latest Interaction linked to agent)
/// 2. Walk conversation thread backward via RespondsTo edges
/// 3. Reserve the N most recent turns as guaranteed
/// 4. Collect Content/Knowledge nodes reachable from the conversation
/// 5. Compute PageRank and BFS distances for scoring
/// 6. Score all non-guaranteed candidates
/// 7. Fit scored candidates into the remaining token budget
/// 8. Assemble ContextWindow with system prompt + selected messages
pub fn build_context(
    graph: &GraphStore,
    agent_id: NodeId,
    config: &ContextConfig,
) -> Result<ContextWindow, AgentError> {
    let system_msg = LlmMessage::system(&config.system_prompt);
    let system_tokens = estimate_tokens_str(&config.system_prompt);

    let current_turn = match find_current_turn(graph, &agent_id)? {
        Some(node) => node,
        None => {
            return Ok(ContextWindow {
                system: system_msg,
                messages: vec![],
                total_tokens: system_tokens,
            });
        }
    };

    // Walk conversation thread (newest-first)
    let thread = graph
        .conversation_thread(&current_turn.id)
        .map_err(|e| AgentError::Context(e.to_string()))?;

    // Split into guaranteed recent and older messages
    let guaranteed_count = config.guaranteed_recent_turns.min(thread.len());
    let guaranteed_recent: Vec<GraphNode> = thread[..guaranteed_count].to_vec();
    let older_conversation: Vec<GraphNode> = thread[guaranteed_count..].to_vec();

    // Token accounting
    let guaranteed_tokens: usize = guaranteed_recent.iter().map(|n| estimate_tokens(n)).sum();
    let remaining_budget = config
        .max_tokens
        .saturating_sub(system_tokens)
        .saturating_sub(guaranteed_tokens);

    // Collect Content/Knowledge nodes reachable from conversation
    let conversation_ids: Vec<NodeId> = thread.iter().map(|n| n.id.clone()).collect();
    let content_nodes = collect_context_nodes(graph, &conversation_ids)?;

    // Compute scoring inputs
    let pr_vec = graph
        .pagerank()
        .map_err(|e| AgentError::Context(e.to_string()))?;
    let pagerank_scores: HashMap<NodeId, f64> = pr_vec.into_iter().collect();

    let distances = bfs_distances(graph, &current_turn.id, 10)?;

    // Score older conversation nodes
    let mut candidates: Vec<ScoredNode> = Vec::new();
    for node in &older_conversation {
        let score = score_node(node, &pagerank_scores, &distances, config, graph)?;
        candidates.push(ScoredNode {
            node: node.clone(),
            score,
            token_estimate: estimate_tokens(node),
        });
    }

    // Score content/knowledge nodes (capped by max_content_nodes)
    let mut content_scored: Vec<ScoredNode> = Vec::new();
    for node in &content_nodes {
        let score = score_node(node, &pagerank_scores, &distances, config, graph)?;
        content_scored.push(ScoredNode {
            node: node.clone(),
            score,
            token_estimate: estimate_tokens(node),
        });
    }
    content_scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    content_scored.truncate(config.max_content_nodes);
    candidates.extend(content_scored);

    // Fit into remaining budget
    let selected = fit_to_budget(candidates, remaining_budget);
    let selected_tokens: usize = selected.iter().map(|s| s.token_estimate).sum();

    // Assemble messages: selected (older/context) in chronological order, then guaranteed recent
    let mut all_nodes: Vec<GraphNode> = selected.into_iter().map(|s| s.node).collect();
    // guaranteed_recent is newest-first from conversation_thread, reverse for chronological
    let mut recent_chrono: Vec<GraphNode> = guaranteed_recent;
    recent_chrono.reverse();

    all_nodes.sort_by(|a, b| a.created_at.cmp(&b.created_at));
    all_nodes.extend(recent_chrono);

    let messages: Vec<LlmMessage> = all_nodes
        .iter()
        .filter_map(|n| node_to_message(n))
        .collect();

    let total_tokens = system_tokens + guaranteed_tokens + selected_tokens;

    Ok(ContextWindow {
        system: system_msg,
        messages,
        total_tokens,
    })
}
```

### Step 4: Run tests

Run: `cargo test -p graphirm-agent context::tests::build_context -- --nocapture 2>&1`
Expected: All 3 tests pass.

### Step 5: Commit

```bash
git add crates/agent/src/context.rs
git commit -m "feat(agent): implement build_context — full relevance-scored context pipeline"
```

---

## Task 10: Implement compact_context()

- [x] Complete

**Files:**
- Modify: `crates/agent/src/compact.rs`
- Test: `crates/agent/src/compact.rs` (in-module `#[cfg(test)]`)

### Step 1: Write the full compact.rs with tests

Replace the contents of `crates/agent/src/compact.rs` with:

```rust
use std::collections::HashMap;

use graphirm_graph::{
    EdgeType, GraphEdge, GraphNode, GraphStore, NodeId, NodeType,
    InteractionData, ContentData, KnowledgeData,
};
use graphirm_llm::{CompletionConfig, LlmMessage, LlmProvider};

use crate::context::{estimate_tokens, estimate_tokens_str, get_text_content};
use crate::error::AgentError;

#[derive(Debug, Clone)]
pub struct CompactionConfig {
    pub model: String,
    pub max_summary_tokens: usize,
    pub min_nodes_to_compact: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            model: "mock".to_string(),
            max_summary_tokens: 500,
            min_nodes_to_compact: 3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompactionResult {
    pub summary_node_id: NodeId,
    pub compacted_node_ids: Vec<NodeId>,
    pub tokens_saved: usize,
}

/// Compact old context nodes by summarizing them via an LLM call.
///
/// Steps:
/// 1. Collect the text content of all nodes to compact
/// 2. Build a summarization prompt
/// 3. Call LLM with complete()
/// 4. Create a Knowledge node with the summary
/// 5. Add Summarizes edges from the Knowledge node to each compacted node
/// 6. Mark original nodes as compacted (metadata["compacted"] = true)
pub async fn compact_context(
    graph: &GraphStore,
    llm: &dyn LlmProvider,
    nodes_to_compact: Vec<NodeId>,
    config: &CompactionConfig,
) -> Result<CompactionResult, AgentError> {
    if nodes_to_compact.len() < config.min_nodes_to_compact {
        return Err(AgentError::Context(format!(
            "Need at least {} nodes to compact, got {}",
            config.min_nodes_to_compact,
            nodes_to_compact.len()
        )));
    }

    // Collect text from nodes
    let mut texts = Vec::new();
    let mut original_tokens = 0_usize;

    for node_id in &nodes_to_compact {
        let node = graph
            .get_node(node_id)
            .map_err(|e| AgentError::Context(e.to_string()))?;
        original_tokens += estimate_tokens(&node);
        let content = get_text_content(&node);
        if !content.is_empty() {
            texts.push(content.to_string());
        }
    }

    // Build summarization prompt
    let combined = texts.join("\n---\n");
    let prompt = format!(
        "Summarize the following conversation context into a concise summary \
         that preserves key information, decisions, and file changes. \
         Keep it under {} tokens.\n\n{}",
        config.max_summary_tokens, combined
    );

    let messages = vec![
        LlmMessage::system(
            "You are a concise summarizer. Produce a factual summary preserving \
             key technical details, file paths, decisions, and outcomes.",
        ),
        LlmMessage::human(prompt),
    ];

    let completion_config = CompletionConfig::new(&config.model)
        .with_max_tokens(config.max_summary_tokens as u32);

    let response = llm
        .complete(messages, &[], &completion_config)
        .await
        .map_err(|e| AgentError::Context(format!("Compaction LLM call failed: {e}")))?;

    let summary_text = response.text_content();
    let summary_tokens = estimate_tokens_str(&summary_text);

    // Create Knowledge node with summary
    let summary_node = GraphNode::new(NodeType::Knowledge(KnowledgeData {
        entity: "session_summary".to_string(),
        entity_type: "compaction".to_string(),
        summary: summary_text,
        confidence: 1.0,
    }));
    let summary_node_id = summary_node.id.clone();
    graph
        .add_node(summary_node)
        .map_err(|e| AgentError::Context(e.to_string()))?;

    // Add Summarizes edges from summary to each compacted node
    for node_id in &nodes_to_compact {
        graph
            .add_edge(GraphEdge::new(
                EdgeType::Summarizes,
                summary_node_id.clone(),
                node_id.clone(),
            ))
            .map_err(|e| AgentError::Context(e.to_string()))?;
    }

    // Mark original nodes as compacted
    for node_id in &nodes_to_compact {
        let mut node = graph
            .get_node(node_id)
            .map_err(|e| AgentError::Context(e.to_string()))?;

        if let Some(obj) = node.metadata.as_object_mut() {
            obj.insert("compacted".to_string(), serde_json::Value::Bool(true));
        }

        graph
            .update_node(node_id, node)
            .map_err(|e| AgentError::Context(e.to_string()))?;
    }

    let tokens_saved = original_tokens.saturating_sub(summary_tokens);

    Ok(CompactionResult {
        summary_node_id,
        compacted_node_ids: nodes_to_compact,
        tokens_saved,
    })
}

/// Check if a node has been compacted (excluded from future context builds).
pub fn is_compacted(node: &GraphNode) -> bool {
    node.metadata
        .get("compacted")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphirm_llm::MockProvider;

    #[test]
    fn compaction_config_defaults() {
        let config = CompactionConfig::default();
        assert_eq!(config.max_summary_tokens, 500);
        assert_eq!(config.min_nodes_to_compact, 3);
    }

    #[tokio::test]
    async fn compact_context_creates_knowledge_node() {
        let graph = GraphStore::open_memory().unwrap();

        // Create 5 interaction nodes
        let mut node_ids = Vec::new();
        for i in 0..5 {
            let node = GraphNode::new(NodeType::Interaction(InteractionData {
                role: if i % 2 == 0 { "user" } else { "assistant" }.to_string(),
                content: format!("Message {i} with some discussion about the project."),
                token_count: None,
            }));
            let id = node.id.clone();
            graph.add_node(node).unwrap();
            node_ids.push(id);
        }

        let llm = MockProvider::fixed(
            "Summary: 5 messages discussing project. Key points: \
             code review feedback, main.rs changes, test additions.",
        );

        let config = CompactionConfig {
            model: "mock".to_string(),
            max_summary_tokens: 100,
            min_nodes_to_compact: 3,
        };

        let result = compact_context(&graph, &llm, node_ids.clone(), &config)
            .await
            .unwrap();

        // Verify summary node created
        let summary_node = graph.get_node(&result.summary_node_id).unwrap();
        match &summary_node.node_type {
            NodeType::Knowledge(data) => {
                assert_eq!(data.entity_type, "compaction");
                assert!(data.summary.contains("Summary"));
            }
            other => panic!("Expected Knowledge node, got {:?}", other),
        }

        // Verify Summarizes edges
        let summarized = graph
            .neighbors(
                &result.summary_node_id,
                Some(EdgeType::Summarizes),
                graphirm_graph::Direction::Outgoing,
            )
            .unwrap();
        assert_eq!(summarized.len(), 5);

        // Verify compacted node IDs
        assert_eq!(result.compacted_node_ids.len(), 5);

        // Verify original nodes marked as compacted
        for id in &node_ids {
            let node = graph.get_node(id).unwrap();
            assert!(is_compacted(&node), "Node {id} should be marked compacted");
        }

        // Verify tokens saved is positive
        assert!(result.tokens_saved > 0, "Should save tokens");
    }

    #[tokio::test]
    async fn compact_context_rejects_too_few_nodes() {
        let graph = GraphStore::open_memory().unwrap();

        let node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "solo".to_string(),
            token_count: None,
        }));
        let id = node.id.clone();
        graph.add_node(node).unwrap();

        let llm = MockProvider::fixed("summary");
        let config = CompactionConfig::default(); // min_nodes = 3

        let result = compact_context(&graph, &llm, vec![id], &config).await;
        assert!(result.is_err());
    }

    #[test]
    fn is_compacted_false_by_default() {
        let node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "normal".to_string(),
            token_count: None,
        }));
        assert!(!is_compacted(&node));
    }

    #[test]
    fn is_compacted_true_when_marked() {
        let mut node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "compacted".to_string(),
            token_count: None,
        }));
        node.metadata = serde_json::json!({"compacted": true});
        assert!(is_compacted(&node));
    }
}
```

### Step 2: Update lib.rs re-exports

In `crates/agent/src/lib.rs`, ensure these are present:

```rust
pub use compact::{compact_context, is_compacted, CompactionConfig, CompactionResult};
pub use context::{
    build_context, estimate_tokens, fit_to_budget, score_node, score_recency,
    ContextConfig, ContextWindow, EdgeWeights, ScoredNode,
};
```

### Step 3: Run tests

Run: `cargo test -p graphirm-agent compact::tests -- --nocapture 2>&1`
Expected: All 4 tests pass.

### Step 4: Commit

```bash
git add crates/agent/src/compact.rs crates/agent/src/lib.rs
git commit -m "feat(agent): implement compact_context with LLM summarization and Summarizes edges"
```

---

## Task 11: Implement auto-compaction in build_context()

- [x] Complete

**Files:**
- Modify: `crates/agent/src/context.rs`
- Test: `crates/agent/src/context.rs` (in-module `#[cfg(test)]`)

### Step 1: Write the failing test

Add to the `tests` module in `context.rs`:

```rust
    #[test]
    fn build_context_excludes_compacted_nodes() {
        let graph = GraphStore::open_memory().unwrap();

        let agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "test".to_string(),
            model: "mock".to_string(),
            system_prompt: None,
            status: "running".to_string(),
        }));
        let agent_id = agent.id.clone();
        graph.add_node(agent).unwrap();

        // Create 10 messages, mark first 5 as compacted
        let mut prev_id: Option<NodeId> = None;
        for i in 0..10 {
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            let mut node = GraphNode::new(NodeType::Interaction(InteractionData {
                role: role.to_string(),
                content: format!("Message {i}"),
                token_count: None,
            }));
            node.created_at = Utc::now() - Duration::hours(10 - i as i64);
            node.updated_at = node.created_at;

            if i < 5 {
                node.metadata = serde_json::json!({"compacted": true});
            }

            let node_id = node.id.clone();
            graph.add_node(node).unwrap();
            graph
                .add_edge(GraphEdge::new(EdgeType::Produces, agent_id.clone(), node_id.clone()))
                .unwrap();
            if let Some(pid) = &prev_id {
                graph
                    .add_edge(GraphEdge::new(
                        EdgeType::RespondsTo,
                        node_id.clone(),
                        pid.clone(),
                    ))
                    .unwrap();
            }
            prev_id = Some(node_id);
        }

        // Add a Knowledge summary node for the compacted messages
        let summary = GraphNode::new(NodeType::Knowledge(KnowledgeData {
            entity: "session_summary".to_string(),
            entity_type: "compaction".to_string(),
            summary: "Summary of first 5 messages.".to_string(),
            confidence: 1.0,
        }));
        let summary_id = summary.id.clone();
        graph.add_node(summary).unwrap();
        graph
            .add_edge(GraphEdge::new(EdgeType::Produces, agent_id.clone(), summary_id.clone()))
            .unwrap();

        let config = ContextConfig {
            max_tokens: 10_000,
            system_prompt: "Help.".to_string(),
            guaranteed_recent_turns: 2,
            enable_compaction: false,
            ..ContextConfig::default()
        };

        let window = build_context(&graph, agent_id, &config).unwrap();

        // Compacted messages (0-4) should not appear in the context
        let all_text: String = window
            .messages
            .iter()
            .flat_map(|m| m.content.iter())
            .filter_map(|p| match p {
                graphirm_llm::ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ");

        for i in 0..5 {
            assert!(
                !all_text.contains(&format!("Message {i}")),
                "Compacted 'Message {i}' should not appear in context"
            );
        }

        // Non-compacted messages (5-9) should appear
        assert!(
            all_text.contains("Message 9"),
            "Non-compacted 'Message 9' should appear in context"
        );
    }
```

### Step 2: Run test to verify it fails

Run: `cargo test -p graphirm-agent context::tests::build_context_excludes_compacted -- --nocapture 2>&1`
Expected: FAIL — `build_context` doesn't filter compacted nodes.

### Step 3: Update build_context to filter compacted nodes

Modify the `build_context` function. Add `use crate::compact::is_compacted;` at the top of `context.rs`, then update the conversation thread filtering.

Find this section in `build_context`:

```rust
    // Walk conversation thread (newest-first)
    let thread = graph
        .conversation_thread(&current_turn.id)
        .map_err(|e| AgentError::Context(e.to_string()))?;
```

Replace it with:

```rust
    // Walk conversation thread (newest-first), excluding compacted nodes
    let full_thread = graph
        .conversation_thread(&current_turn.id)
        .map_err(|e| AgentError::Context(e.to_string()))?;
    let thread: Vec<GraphNode> = full_thread
        .into_iter()
        .filter(|n| !is_compacted(n))
        .collect();
```

Also add the import at the top of `context.rs`:

```rust
use crate::compact::is_compacted;
```

And in `collect_context_nodes`, filter out compacted nodes:

Find:

```rust
                    match &neighbor.node_type {
                        NodeType::Content(_) | NodeType::Knowledge(_) => {
                            context_nodes.push(neighbor);
                        }
                        _ => {}
                    }
```

Replace with:

```rust
                    if is_compacted(&neighbor) {
                        continue;
                    }
                    match &neighbor.node_type {
                        NodeType::Content(_) | NodeType::Knowledge(_) => {
                            context_nodes.push(neighbor);
                        }
                        _ => {}
                    }
```

### Step 4: Run tests

Run: `cargo test -p graphirm-agent context::tests::build_context -- --nocapture 2>&1`
Expected: All 4 build_context tests pass (3 from Task 9 + 1 new).

### Step 5: Commit

```bash
git add crates/agent/src/context.rs
git commit -m "feat(agent): filter compacted nodes from build_context pipeline"
```

---

## Task 12: Integration test — full session lifecycle

- [x] Complete

**Files:**
- Modify: `crates/agent/src/context.rs` (add test to existing test module)

### Step 1: Write the integration test

Add to the `tests` module in `context.rs`:

```rust
    #[test]
    fn integration_30_turn_session_with_tool_calls() {
        let graph = GraphStore::open_memory().unwrap();

        // Create agent
        let agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "coder".to_string(),
            model: "claude-sonnet".to_string(),
            system_prompt: Some("You are a Rust expert.".to_string()),
            status: "running".to_string(),
        }));
        let agent_id = agent.id.clone();
        graph.add_node(agent).unwrap();

        // Build a 30-turn conversation with tool calls producing Content nodes
        let mut prev_id: Option<NodeId> = None;
        let mut all_msg_ids = Vec::new();

        for turn in 0..30 {
            // User message
            let mut user_msg = GraphNode::new(NodeType::Interaction(InteractionData {
                role: "user".to_string(),
                content: format!("Turn {turn}: Please fix the bug in module_{}.rs", turn % 5),
                token_count: None,
            }));
            user_msg.created_at = Utc::now() - Duration::minutes((30 - turn) as i64 * 10);
            user_msg.updated_at = user_msg.created_at;
            let user_id = user_msg.id.clone();
            graph.add_node(user_msg).unwrap();
            graph
                .add_edge(GraphEdge::new(EdgeType::Produces, agent_id.clone(), user_id.clone()))
                .unwrap();
            if let Some(pid) = &prev_id {
                graph
                    .add_edge(GraphEdge::new(EdgeType::RespondsTo, user_id.clone(), pid.clone()))
                    .unwrap();
            }
            all_msg_ids.push(user_id.clone());

            // Assistant response
            let mut asst_msg = GraphNode::new(NodeType::Interaction(InteractionData {
                role: "assistant".to_string(),
                content: format!("Turn {turn}: I'll read and fix module_{}.rs now.", turn % 5),
                token_count: None,
            }));
            asst_msg.created_at = user_msg.created_at + Duration::seconds(30);
            asst_msg.updated_at = asst_msg.created_at;
            let asst_id = asst_msg.id.clone();
            graph.add_node(asst_msg).unwrap();
            graph
                .add_edge(GraphEdge::new(EdgeType::Produces, agent_id.clone(), asst_id.clone()))
                .unwrap();
            graph
                .add_edge(GraphEdge::new(EdgeType::RespondsTo, asst_id.clone(), user_id.clone()))
                .unwrap();

            // Every 3rd turn, the assistant modifies a file (Content node)
            if turn % 3 == 0 {
                let file = GraphNode::new(NodeType::Content(ContentData {
                    content_type: "file".to_string(),
                    path: Some(format!("src/module_{}.rs", turn % 5)),
                    body: format!("// Fixed in turn {turn}\nfn fixed_function() {{}}\n"),
                    language: Some("rust".to_string()),
                }));
                let file_id = file.id.clone();
                graph.add_node(file).unwrap();
                graph
                    .add_edge(GraphEdge::new(
                        EdgeType::Modifies,
                        asst_id.clone(),
                        file_id.clone(),
                    ))
                    .unwrap();
            }

            // Every 5th turn, the assistant reads a file
            if turn % 5 == 0 {
                let read_file = GraphNode::new(NodeType::Content(ContentData {
                    content_type: "file".to_string(),
                    path: Some(format!("src/module_{}.rs", turn % 5)),
                    body: format!("// Original content of module_{}\n", turn % 5),
                    language: Some("rust".to_string()),
                }));
                let rf_id = read_file.id.clone();
                graph.add_node(read_file).unwrap();
                graph
                    .add_edge(GraphEdge::new(
                        EdgeType::Reads,
                        asst_id.clone(),
                        rf_id.clone(),
                    ))
                    .unwrap();
            }

            prev_id = Some(asst_id.clone());
            all_msg_ids.push(asst_id);
        }

        // Build context at turn 30 with a modest budget
        let config = ContextConfig {
            max_tokens: 2000,
            system_prompt: "You are a Rust expert.".to_string(),
            guaranteed_recent_turns: 4,
            max_content_nodes: 5,
            recency_decay: 0.05,
            enable_compaction: false,
            ..ContextConfig::default()
        };

        let window = build_context(&graph, agent_id, &config).unwrap();

        // Verify system prompt
        assert_eq!(window.system.role, graphirm_llm::Role::System);

        // Should have messages (guaranteed recent + scored older + content)
        assert!(
            window.messages.len() >= 4,
            "Should include at least 4 guaranteed recent turns, got {}",
            window.messages.len()
        );

        // Total tokens should respect budget
        assert!(
            window.total_tokens <= config.max_tokens,
            "Total tokens {} exceeds budget {}",
            window.total_tokens,
            config.max_tokens
        );

        // Recent messages should be present (turn 29 = "Turn 29: ...")
        let all_text: String = window
            .messages
            .iter()
            .flat_map(|m| m.content.iter())
            .filter_map(|p| match p {
                graphirm_llm::ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ");

        assert!(
            all_text.contains("Turn 29"),
            "Most recent turn should be in context"
        );
        assert!(
            all_text.contains("Turn 28"),
            "Second most recent turn should be in context"
        );

        // Verify context doesn't just dump everything — budget should limit it
        assert!(
            window.messages.len() < 60,
            "Budget should limit messages, got {} (60 = all messages)",
            window.messages.len()
        );

        // Content nodes (files) should be included when they score high enough
        // Recent file modifications should appear
        let has_file_ref = all_text.contains("module_") || all_text.contains(".rs");
        assert!(
            has_file_ref,
            "Context should include relevant file references"
        );
    }
```

### Step 2: Run the integration test

Run: `cargo test -p graphirm-agent context::tests::integration_30_turn -- --nocapture 2>&1`
Expected: PASS

### Step 3: Run the full test suite

Run: `cargo test -p graphirm-agent -- --nocapture 2>&1`
Expected: All tests pass.

### Step 4: Run clippy and fmt

Run: `cargo clippy -p graphirm-agent --all-targets 2>&1`
Expected: No errors. Fix any warnings.

Run: `cargo fmt -p graphirm-agent -- --check 2>&1`
Expected: No formatting issues.

### Step 5: Commit

```bash
git add crates/agent/src/context.rs
git commit -m "test(agent): add 30-turn integration test for context engine"
```

---

## Final Checklist

After all 12 tasks, verify:

```bash
# All agent crate tests pass
cargo test -p graphirm-agent -- --nocapture

# Full workspace still compiles
cargo build

# No lint issues
cargo clippy --all-targets --all-features

# Formatting clean
cargo fmt --all -- --check
```

Expected final test count for Phase 6 additions:
- ContextConfig/EdgeWeights (4 tests)
- ScoredNode/ContextWindow/helpers (9 tests)
- estimate_tokens (3 tests)
- score_recency (4 tests)
- score_edge_weights (3 tests)
- BFS distance + graph distance (4 tests)
- score_node composite (2 tests)
- fit_to_budget (4 tests)
- build_context (4 tests)
- compact_context (4 tests)
- Integration test (1 test)
- **Total: ~42 new tests**

## Summary

| Task | Description | Files | Tests |
|------|-------------|-------|-------|
| 1 | ContextConfig + EdgeWeights with serde | context.rs | 4 |
| 2 | ScoredNode, ContextWindow, helpers | context.rs | 9 |
| 3 | estimate_tokens (word heuristic) | context.rs | 3 |
| 4 | score_recency (exponential decay) | context.rs | 4 |
| 5 | score_edge_weights (per-type counting) | context.rs | 3 |
| 6 | BFS distances + graph distance score | context.rs | 4 |
| 7 | score_node (composite 4-signal) | context.rs | 2 |
| 8 | fit_to_budget (greedy knapsack) | context.rs | 4 |
| 9 | build_context (full pipeline) | context.rs | 3 |
| 10 | compact_context (LLM summarization) | compact.rs | 4 |
| 11 | Auto-compaction filtering | context.rs | 1 |
| 12 | 30-turn integration test | context.rs | 1 |
| **Total** | | **2 files** | **~42 tests** |

## Adaptation Notes

The code references GraphStore methods from Phase 1 and LlmProvider/MockProvider from Phase 2. Key integration points:

- **GraphStore::conversation_thread()** — walks `RespondsTo` edges backward, returns newest-first
- **GraphStore::pagerank()** — iterative power method, returns `Vec<(NodeId, f64)>` sorted desc
- **GraphStore::neighbors()** — requires `Direction` (from `petgraph::Direction`)
- **GraphStore::update_node()** — used by compaction to mark metadata["compacted"]
- **MockProvider::fixed()** — returns canned text for compaction tests
- **LlmResponse::text_content()** — extracts concatenated text from `Vec<ContentPart>`

If the actual Phase 1/2 APIs differ at implementation time (field names, method signatures, re-export paths), adapt the Phase 6 code while keeping the scoring/building logic the same. The algorithm is the contract; the API calls are adapters.

## Post-Phase Follow-Ups (Not Part of This Plan)

- **Embedding-based similarity:** Add HNSW vector search for cross-session knowledge retrieval (Phase 9)
- **Streaming compaction:** Stream the summarization call instead of blocking on complete()
- **Compaction policies:** Time-based, token-based, and manual trigger options
- **Context visualization:** Show which nodes were selected/excluded in the TUI (Phase 7)
- **Configurable composite weights:** Add W_RECENCY, W_EDGE, W_DISTANCE, W_PAGERANK to ContextConfig
- **Incremental PageRank:** Cache and update PageRank scores instead of recomputing each build
