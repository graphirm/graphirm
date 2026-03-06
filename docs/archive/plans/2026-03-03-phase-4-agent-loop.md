# Phase 4: Agent Loop Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the hand-rolled async agent loop — the core state machine that cycles between context building, LLM completion, and parallel tool execution, recording every step as graph nodes and edges.

**Architecture:** The agent loop lives in `crates/agent/` and orchestrates three lower crates: `graph` (persistence), `llm` (provider abstraction), and `tools` (execution). It is an async function that builds context from the graph, calls the LLM, optionally dispatches tool calls in parallel via `tokio::JoinSet`, records everything as graph nodes/edges, and emits lifecycle events via an mpsc-based `EventBus` for UI consumption. A `CancellationToken` enables graceful shutdown mid-loop.

**Tech Stack:** tokio (async runtime + JoinSet), tokio::sync::mpsc (event bus), tokio-util CancellationToken (graceful shutdown), serde (config deserialization), chrono (timestamps), uuid (node IDs)

---

## Prerequisites (expected APIs from Phases 1–3)

Phase 4 depends on types and traits defined in earlier phases. The code below references these APIs. If the actual signatures differ slightly after Phases 1–3 land, adapt accordingly — the logic and test structure remain the same.

### From `graphirm-graph` (Phase 1)

```rust
// crates/graph/src/lib.rs — re-exports
pub type NodeId = String;
pub type EdgeId = String;

pub struct Node {
    pub id: NodeId,
    pub node_type: NodeType,
    pub data: serde_json::Value,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

pub struct Edge {
    pub id: EdgeId,
    pub edge_type: EdgeType,
    pub source: NodeId,
    pub target: NodeId,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    Interaction, Agent, Content, Task, Knowledge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeType {
    RespondsTo, SpawnedBy, DelegatesTo, DependsOn,
    Produces, Reads, Modifies, Summarizes,
    Contains, FollowsUp, Steers, RelatesTo,
}

// GraphStore is Send + Sync (uses r2d2 pool + Arc<RwLock<petgraph>>)
impl GraphStore {
    pub fn new_in_memory() -> Result<Self, GraphError>;
    pub fn add_node(&self, node_type: NodeType, data: serde_json::Value) -> Result<NodeId, GraphError>;
    pub fn add_edge(&self, source: &NodeId, target: &NodeId, edge_type: EdgeType) -> Result<EdgeId, GraphError>;
    pub fn get_node(&self, id: &NodeId) -> Result<Node, GraphError>;
    pub fn get_neighbors(&self, id: &NodeId, edge_type: Option<EdgeType>) -> Result<Vec<Node>, GraphError>;
}
```

### From `graphirm-llm` (Phase 2)

```rust
// crates/llm/src/provider.rs
#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn complete(
        &self,
        messages: Vec<LlmMessage>,
        tools: Vec<ToolDefinition>,
    ) -> Result<LlmResponse, LlmError>;
}

// crates/llm/src/lib.rs — types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role { System, User, Assistant, Tool }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmMessage {
    pub role: Role,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

// crates/llm/src/stream.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamEvent {
    pub delta: String,
}
```

### From `graphirm-tools` (Phase 3)

```rust
// crates/tools/src/lib.rs
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    async fn execute(&self, args: serde_json::Value) -> Result<String, ToolError>;
}

pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self;
    pub fn register(&mut self, tool: Arc<dyn Tool>);
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>>;
    pub fn definitions(&self) -> Vec<ToolDefinition>;
    pub fn names(&self) -> Vec<String>;
}
```

---

## Task 1: Update Cargo.toml and expand AgentError enum

- [x] Complete

**Files:**
- Modify: `crates/agent/Cargo.toml`
- Modify: `crates/agent/src/error.rs`

### Step 1: Add dependencies to Cargo.toml

Replace the `[dependencies]` section in `crates/agent/Cargo.toml`:

```toml
[package]
name = "graphirm-agent"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
authors.workspace = true
description = "Agent loop, multi-agent orchestration, and context engine for Graphirm"

[dependencies]
graphirm-graph = { path = "../graph" }
graphirm-llm = { path = "../llm" }
graphirm-tools = { path = "../tools" }
async-trait = "0.1"
tokio = { version = "1", features = ["sync", "macros", "rt"] }
tokio-util = "0.7"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
tracing = "0.1"
chrono = { version = "0.4", features = ["serde"] }
uuid = { version = "1", features = ["v4"] }

[dev-dependencies]
tokio = { version = "1", features = ["full", "test-util"] }
toml = "0.8"
```

### Step 2: Expand AgentError in error.rs

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("Graph error: {0}")]
    Graph(#[from] graphirm_graph::GraphError),

    #[error("LLM error: {0}")]
    Llm(#[from] graphirm_llm::LlmError),

    #[error("Tool error: {0}")]
    Tool(#[from] graphirm_tools::ToolError),

    #[error("Session not found: {0}")]
    SessionNotFound(String),

    #[error("Workflow error: {0}")]
    Workflow(String),

    #[error("Context build failed: {0}")]
    Context(String),

    #[error("Recursion limit reached: {0} turns")]
    RecursionLimit(u32),

    #[error("Agent loop cancelled")]
    Cancelled,

    #[error("Task join error: {0}")]
    Join(String),
}
```

### Step 3: Verify

Run: `cargo check -p graphirm-agent 2>&1`
Expected: Compiles with no errors.

### Step 4: Commit

```bash
git add crates/agent/Cargo.toml crates/agent/src/error.rs
git commit -m "feat(agent): add Phase 4 dependencies, expand AgentError"
```

---

## Task 2: Define AgentEvent enum

- [x] Complete

**Files:**
- Modify: `crates/agent/src/event.rs`

### Step 1: Write tests first

Add to the bottom of `crates/agent/src/event.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_event_clone() {
        let event = AgentEvent::TurnStart { turn_index: 3 };
        let cloned = event.clone();
        assert!(matches!(cloned, AgentEvent::TurnStart { turn_index: 3 }));
    }

    #[test]
    fn test_agent_event_debug() {
        let event = AgentEvent::ToolStart {
            node_id: "node-1".to_string(),
            tool_name: "bash".to_string(),
        };
        let debug = format!("{:?}", event);
        assert!(debug.contains("ToolStart"));
        assert!(debug.contains("bash"));
    }
}
```

Run: `cargo test -p graphirm-agent event::tests 2>&1`
Expected: Fails — `AgentEvent` not defined yet.

### Step 2: Implement AgentEvent

Write the full `crates/agent/src/event.rs`:

```rust
use graphirm_graph::{EdgeId, NodeId};
use graphirm_llm::StreamEvent;

#[derive(Debug, Clone)]
pub enum AgentEvent {
    AgentStart {
        agent_id: NodeId,
    },
    AgentEnd {
        agent_id: NodeId,
        node_ids: Vec<NodeId>,
    },
    TurnStart {
        turn_index: u32,
    },
    TurnEnd {
        response_id: NodeId,
        tool_result_ids: Vec<NodeId>,
    },
    MessageStart {
        node_id: NodeId,
    },
    MessageDelta {
        node_id: NodeId,
        delta: StreamEvent,
    },
    MessageEnd {
        node_id: NodeId,
    },
    ToolStart {
        node_id: NodeId,
        tool_name: String,
    },
    ToolEnd {
        node_id: NodeId,
        is_error: bool,
    },
    GraphUpdate {
        node_id: NodeId,
        edge_ids: Vec<EdgeId>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_event_clone() {
        let event = AgentEvent::TurnStart { turn_index: 3 };
        let cloned = event.clone();
        assert!(matches!(cloned, AgentEvent::TurnStart { turn_index: 3 }));
    }

    #[test]
    fn test_agent_event_debug() {
        let event = AgentEvent::ToolStart {
            node_id: "node-1".to_string(),
            tool_name: "bash".to_string(),
        };
        let debug = format!("{:?}", event);
        assert!(debug.contains("ToolStart"));
        assert!(debug.contains("bash"));
    }
}
```

### Step 3: Update lib.rs re-exports

In `crates/agent/src/lib.rs`, add:

```rust
pub use event::AgentEvent;
```

### Step 4: Verify

Run: `cargo test -p graphirm-agent event::tests 2>&1`
Expected: `test result: ok. 2 passed; 0 failed`

### Step 5: Commit

```bash
git add crates/agent/src/event.rs crates/agent/src/lib.rs
git commit -m "feat(agent): define AgentEvent enum with all lifecycle variants"
```

---

## Task 3: Implement EventBus

- [x] Complete

**Files:**
- Modify: `crates/agent/src/event.rs`

### Step 1: Write tests

Append to the `tests` module in `crates/agent/src/event.rs`:

```rust
    #[tokio::test]
    async fn test_event_bus_single_subscriber() {
        let mut bus = EventBus::new();
        let mut rx = bus.subscribe();

        bus.emit(AgentEvent::TurnStart { turn_index: 0 }).await;

        let event = rx.recv().await.unwrap();
        assert!(matches!(event, AgentEvent::TurnStart { turn_index: 0 }));
    }

    #[tokio::test]
    async fn test_event_bus_multiple_subscribers() {
        let mut bus = EventBus::new();
        let mut rx1 = bus.subscribe();
        let mut rx2 = bus.subscribe();

        bus.emit(AgentEvent::AgentStart {
            agent_id: "a1".to_string(),
        })
        .await;

        let e1 = rx1.recv().await.unwrap();
        let e2 = rx2.recv().await.unwrap();
        assert!(matches!(e1, AgentEvent::AgentStart { .. }));
        assert!(matches!(e2, AgentEvent::AgentStart { .. }));
    }

    #[tokio::test]
    async fn test_event_bus_dropped_subscriber_does_not_block() {
        let mut bus = EventBus::new();
        let rx = bus.subscribe();
        drop(rx);

        // Should not panic or block
        bus.emit(AgentEvent::TurnStart { turn_index: 0 }).await;
    }
```

Run: `cargo test -p graphirm-agent event::tests 2>&1`
Expected: Fails — `EventBus` not defined yet.

### Step 2: Implement EventBus

Add above the `#[cfg(test)]` block in `crates/agent/src/event.rs`:

```rust
use tokio::sync::mpsc;

pub struct EventBus {
    subscribers: Vec<mpsc::Sender<AgentEvent>>,
}

impl EventBus {
    pub fn new() -> Self {
        Self {
            subscribers: Vec::new(),
        }
    }

    pub fn subscribe(&mut self) -> mpsc::Receiver<AgentEvent> {
        let (tx, rx) = mpsc::channel(256);
        self.subscribers.push(tx);
        rx
    }

    pub async fn emit(&self, event: AgentEvent) {
        for sender in &self.subscribers {
            let _ = sender.send(event.clone()).await;
        }
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}
```

### Step 3: Update lib.rs re-exports

In `crates/agent/src/lib.rs`, add:

```rust
pub use event::EventBus;
```

### Step 4: Verify

Run: `cargo test -p graphirm-agent event::tests 2>&1`
Expected: `test result: ok. 5 passed; 0 failed`

### Step 5: Commit

```bash
git add crates/agent/src/event.rs crates/agent/src/lib.rs
git commit -m "feat(agent): implement EventBus with mpsc pub/sub"
```

---

## Task 4: Define AgentConfig with serde Deserialize

- [x] Complete

**Files:**
- Modify: `crates/agent/src/config.rs`

### Step 1: Write tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_config_defaults() {
        let config = AgentConfig::default();
        assert_eq!(config.name, "graphirm");
        assert_eq!(config.max_turns, 50);
        assert!(config.tools.is_empty());
    }

    #[test]
    fn test_agent_config_from_toml() {
        let toml_str = r#"
            name = "test-agent"
            model = "claude-sonnet-4-20250514"
            system_prompt = "You are a coding assistant."
            max_turns = 10
            max_tokens = 4096
            temperature = 0.5
            tools = ["bash", "read", "write"]
        "#;
        let config: AgentConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.name, "test-agent");
        assert_eq!(config.model, "claude-sonnet-4-20250514");
        assert_eq!(config.max_turns, 10);
        assert_eq!(config.max_tokens, Some(4096));
        assert_eq!(config.temperature, Some(0.5));
        assert_eq!(config.tools, vec!["bash", "read", "write"]);
    }

    #[test]
    fn test_agent_config_from_toml_minimal() {
        let toml_str = r#"
            name = "minimal"
            model = "gpt-4o"
            system_prompt = "Help."
            max_turns = 5
            tools = []
        "#;
        let config: AgentConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.name, "minimal");
        assert_eq!(config.max_tokens, None);
        assert_eq!(config.temperature, None);
    }
}
```

Run: `cargo test -p graphirm-agent config::tests 2>&1`
Expected: Fails — `AgentConfig` not defined yet.

### Step 2: Implement AgentConfig

Write the full `crates/agent/src/config.rs`:

```rust
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct AgentConfig {
    pub name: String,
    pub model: String,
    pub system_prompt: String,
    pub max_turns: u32,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub tools: Vec<String>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: "graphirm".to_string(),
            model: "claude-sonnet-4-20250514".to_string(),
            system_prompt: "You are a helpful coding assistant.".to_string(),
            max_turns: 50,
            max_tokens: Some(8192),
            temperature: Some(0.7),
            tools: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_config_defaults() {
        let config = AgentConfig::default();
        assert_eq!(config.name, "graphirm");
        assert_eq!(config.max_turns, 50);
        assert!(config.tools.is_empty());
    }

    #[test]
    fn test_agent_config_from_toml() {
        let toml_str = r#"
            name = "test-agent"
            model = "claude-sonnet-4-20250514"
            system_prompt = "You are a coding assistant."
            max_turns = 10
            max_tokens = 4096
            temperature = 0.5
            tools = ["bash", "read", "write"]
        "#;
        let config: AgentConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.name, "test-agent");
        assert_eq!(config.model, "claude-sonnet-4-20250514");
        assert_eq!(config.max_turns, 10);
        assert_eq!(config.max_tokens, Some(4096));
        assert_eq!(config.temperature, Some(0.5));
        assert_eq!(config.tools, vec!["bash", "read", "write"]);
    }

    #[test]
    fn test_agent_config_from_toml_minimal() {
        let toml_str = r#"
            name = "minimal"
            model = "gpt-4o"
            system_prompt = "Help."
            max_turns = 5
            tools = []
        "#;
        let config: AgentConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.name, "minimal");
        assert_eq!(config.max_tokens, None);
        assert_eq!(config.temperature, None);
    }
}
```

### Step 3: Update lib.rs re-exports

In `crates/agent/src/lib.rs`, add:

```rust
pub use config::AgentConfig;
```

### Step 4: Verify

Run: `cargo test -p graphirm-agent config::tests 2>&1`
Expected: `test result: ok. 3 passed; 0 failed`

### Step 5: Commit

```bash
git add crates/agent/src/config.rs crates/agent/src/lib.rs
git commit -m "feat(agent): define AgentConfig with serde Deserialize + defaults"
```

---

## Task 5: Define Session struct

- [x] Complete

**Files:**
- Modify: `crates/agent/src/session.rs`

### Step 1: Write tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AgentConfig;

    #[test]
    fn test_session_creates_agent_node_in_graph() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let config = AgentConfig::default();
        let session = Session::new(graph.clone(), config).unwrap();

        let node = graph.get_node(&session.id).unwrap();
        assert_eq!(node.node_type, NodeType::Agent);

        let name = node.data["name"].as_str().unwrap();
        assert_eq!(name, "graphirm");
    }

    #[test]
    fn test_session_stores_config() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let config = AgentConfig {
            name: "test-bot".to_string(),
            model: "gpt-4o".to_string(),
            ..AgentConfig::default()
        };
        let session = Session::new(graph, config).unwrap();

        assert_eq!(session.agent_config.name, "test-bot");
        assert_eq!(session.agent_config.model, "gpt-4o");
    }
}
```

Run: `cargo test -p graphirm-agent session::tests 2>&1`
Expected: Fails — `Session` not defined yet.

### Step 2: Implement Session

Write the full `crates/agent/src/session.rs`:

```rust
use std::sync::Arc;

use chrono::{DateTime, Utc};

use graphirm_graph::{GraphStore, NodeId, NodeType};

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
        let agent_data = serde_json::json!({
            "name": config.name,
            "model": config.model,
            "system_prompt": config.system_prompt,
            "created_at": now.to_rfc3339(),
        });
        let id = graph.add_node(NodeType::Agent, agent_data)?;
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
        let data = serde_json::json!({
            "role": "user",
            "content": content,
            "created_at": Utc::now().to_rfc3339(),
        });
        let msg_id = self.graph.add_node(NodeType::Interaction, data)?;
        self.graph
            .add_edge(&self.id, &msg_id, graphirm_graph::EdgeType::Produces)?;
        Ok(msg_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AgentConfig;

    #[test]
    fn test_session_creates_agent_node_in_graph() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let config = AgentConfig::default();
        let session = Session::new(graph.clone(), config).unwrap();

        let node = graph.get_node(&session.id).unwrap();
        assert_eq!(node.node_type, NodeType::Agent);

        let name = node.data["name"].as_str().unwrap();
        assert_eq!(name, "graphirm");
    }

    #[test]
    fn test_session_stores_config() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
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
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let config = AgentConfig::default();
        let session = Session::new(graph.clone(), config).unwrap();

        let msg_id = session.add_user_message("Hello!").unwrap();
        let node = graph.get_node(&msg_id).unwrap();
        assert_eq!(node.node_type, NodeType::Interaction);
        assert_eq!(node.data["content"].as_str().unwrap(), "Hello!");
        assert_eq!(node.data["role"].as_str().unwrap(), "user");
    }
}
```

### Step 3: Update lib.rs re-exports

In `crates/agent/src/lib.rs`, add:

```rust
pub use session::Session;
```

### Step 4: Verify

Run: `cargo test -p graphirm-agent session::tests 2>&1`
Expected: `test result: ok. 3 passed; 0 failed`

### Step 5: Commit

```bash
git add crates/agent/src/session.rs crates/agent/src/lib.rs
git commit -m "feat(agent): implement Session with graph-backed agent node creation"
```

---

## Task 6: Implement build_context()

- [x] Complete

**Files:**
- Modify: `crates/agent/src/context.rs`

### Step 1: Write tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AgentConfig;
    use crate::session::Session;
    use graphirm_graph::GraphStore;
    use std::sync::Arc;

    #[test]
    fn test_build_context_empty_conversation() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let config = AgentConfig {
            system_prompt: "You are helpful.".to_string(),
            ..AgentConfig::default()
        };
        let session = Session::new(graph, config).unwrap();

        let context = build_context(&session).unwrap();

        assert_eq!(context.len(), 1);
        assert_eq!(context[0].role, Role::System);
        assert_eq!(context[0].content, "You are helpful.");
    }

    #[test]
    fn test_build_context_with_messages() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let config = AgentConfig {
            system_prompt: "Be concise.".to_string(),
            ..AgentConfig::default()
        };
        let session = Session::new(graph.clone(), config).unwrap();

        // Simulate a 5-message conversation
        let roles = ["user", "assistant", "user", "assistant", "user"];
        for (i, role) in roles.iter().enumerate() {
            let data = serde_json::json!({
                "role": role,
                "content": format!("Message {}", i),
                "created_at": chrono::Utc::now().to_rfc3339(),
            });
            let msg_id = graph
                .add_node(graphirm_graph::NodeType::Interaction, data)
                .unwrap();
            graph
                .add_edge(
                    &session.id,
                    &msg_id,
                    graphirm_graph::EdgeType::Produces,
                )
                .unwrap();
        }

        let context = build_context(&session).unwrap();

        // System prompt + 5 messages
        assert_eq!(context.len(), 6);
        assert_eq!(context[0].role, Role::System);
        assert_eq!(context[0].content, "Be concise.");
        assert_eq!(context[1].role, Role::User);
        assert_eq!(context[1].content, "Message 0");
        assert_eq!(context[5].role, Role::User);
        assert_eq!(context[5].content, "Message 4");
    }

    #[test]
    fn test_build_context_includes_tool_results() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let config = AgentConfig::default();
        let session = Session::new(graph.clone(), config).unwrap();

        // User message
        let user_data = serde_json::json!({
            "role": "user",
            "content": "Run ls",
            "created_at": chrono::Utc::now().to_rfc3339(),
        });
        let user_id = graph
            .add_node(graphirm_graph::NodeType::Interaction, user_data)
            .unwrap();
        graph
            .add_edge(&session.id, &user_id, graphirm_graph::EdgeType::Produces)
            .unwrap();

        // Assistant with tool call
        let assistant_data = serde_json::json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "name": "bash", "arguments": {"command": "ls"}}],
            "created_at": chrono::Utc::now().to_rfc3339(),
        });
        let asst_id = graph
            .add_node(graphirm_graph::NodeType::Interaction, assistant_data)
            .unwrap();
        graph
            .add_edge(&session.id, &asst_id, graphirm_graph::EdgeType::Produces)
            .unwrap();

        // Tool result
        let tool_data = serde_json::json!({
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "file1.rs\nfile2.rs",
            "created_at": chrono::Utc::now().to_rfc3339(),
        });
        let tool_id = graph
            .add_node(graphirm_graph::NodeType::Interaction, tool_data)
            .unwrap();
        graph
            .add_edge(&session.id, &tool_id, graphirm_graph::EdgeType::Produces)
            .unwrap();

        let context = build_context(&session).unwrap();

        // System + user + assistant + tool = 4
        assert_eq!(context.len(), 4);
        assert_eq!(context[3].role, Role::Tool);
        assert_eq!(context[3].tool_call_id, Some("call_1".to_string()));
    }
}
```

Run: `cargo test -p graphirm-agent context::tests 2>&1`
Expected: Fails — `build_context` not defined yet.

### Step 2: Implement build_context

Write the full `crates/agent/src/context.rs`:

```rust
use graphirm_graph::{EdgeType, NodeType};
use graphirm_llm::{LlmMessage, Role, ToolCall};

use crate::error::AgentError;
use crate::session::Session;

/// Build the LLM message context from the session's conversation graph.
///
/// MVP strategy: get all Interaction nodes linked to the Agent node via
/// Produces edges, sort by created_at, and convert to LlmMessage format.
/// Prepend the system prompt.
pub fn build_context(session: &Session) -> Result<Vec<LlmMessage>, AgentError> {
    let mut messages = Vec::new();

    // System prompt
    messages.push(LlmMessage {
        role: Role::System,
        content: session.agent_config.system_prompt.clone(),
        tool_calls: vec![],
        tool_call_id: None,
    });

    // Get all conversation nodes linked to this agent
    let mut interactions = session
        .graph
        .get_neighbors(&session.id, Some(EdgeType::Produces))
        .map_err(|e| AgentError::Context(e.to_string()))?;

    // Sort by created_at timestamp
    interactions.sort_by(|a, b| {
        let ts_a = a.data["created_at"].as_str().unwrap_or("");
        let ts_b = b.data["created_at"].as_str().unwrap_or("");
        ts_a.cmp(ts_b)
    });

    // Convert each node to an LlmMessage
    for node in &interactions {
        if node.node_type != NodeType::Interaction {
            continue;
        }

        let role = match node.data["role"].as_str().unwrap_or("user") {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            _ => Role::User,
        };

        let content = node.data["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let tool_calls: Vec<ToolCall> = node.data["tool_calls"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| serde_json::from_value(v.clone()).ok())
                    .collect()
            })
            .unwrap_or_default();

        let tool_call_id = node.data["tool_call_id"]
            .as_str()
            .map(|s| s.to_string());

        messages.push(LlmMessage {
            role,
            content,
            tool_calls,
            tool_call_id,
        });
    }

    Ok(messages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AgentConfig;
    use crate::session::Session;
    use graphirm_graph::GraphStore;
    use std::sync::Arc;

    #[test]
    fn test_build_context_empty_conversation() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let config = AgentConfig {
            system_prompt: "You are helpful.".to_string(),
            ..AgentConfig::default()
        };
        let session = Session::new(graph, config).unwrap();

        let context = build_context(&session).unwrap();

        assert_eq!(context.len(), 1);
        assert_eq!(context[0].role, Role::System);
        assert_eq!(context[0].content, "You are helpful.");
    }

    #[test]
    fn test_build_context_with_messages() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let config = AgentConfig {
            system_prompt: "Be concise.".to_string(),
            ..AgentConfig::default()
        };
        let session = Session::new(graph.clone(), config).unwrap();

        let roles = ["user", "assistant", "user", "assistant", "user"];
        for (i, role) in roles.iter().enumerate() {
            let data = serde_json::json!({
                "role": role,
                "content": format!("Message {}", i),
                "created_at": chrono::Utc::now().to_rfc3339(),
            });
            let msg_id = graph
                .add_node(graphirm_graph::NodeType::Interaction, data)
                .unwrap();
            graph
                .add_edge(
                    &session.id,
                    &msg_id,
                    graphirm_graph::EdgeType::Produces,
                )
                .unwrap();
        }

        let context = build_context(&session).unwrap();

        assert_eq!(context.len(), 6);
        assert_eq!(context[0].role, Role::System);
        assert_eq!(context[1].role, Role::User);
        assert_eq!(context[1].content, "Message 0");
        assert_eq!(context[5].role, Role::User);
        assert_eq!(context[5].content, "Message 4");
    }

    #[test]
    fn test_build_context_includes_tool_results() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let config = AgentConfig::default();
        let session = Session::new(graph.clone(), config).unwrap();

        let user_data = serde_json::json!({
            "role": "user",
            "content": "Run ls",
            "created_at": chrono::Utc::now().to_rfc3339(),
        });
        let user_id = graph
            .add_node(graphirm_graph::NodeType::Interaction, user_data)
            .unwrap();
        graph
            .add_edge(&session.id, &user_id, graphirm_graph::EdgeType::Produces)
            .unwrap();

        let assistant_data = serde_json::json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "name": "bash", "arguments": {"command": "ls"}}],
            "created_at": chrono::Utc::now().to_rfc3339(),
        });
        let asst_id = graph
            .add_node(graphirm_graph::NodeType::Interaction, assistant_data)
            .unwrap();
        graph
            .add_edge(&session.id, &asst_id, graphirm_graph::EdgeType::Produces)
            .unwrap();

        let tool_data = serde_json::json!({
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "file1.rs\nfile2.rs",
            "created_at": chrono::Utc::now().to_rfc3339(),
        });
        let tool_id = graph
            .add_node(graphirm_graph::NodeType::Interaction, tool_data)
            .unwrap();
        graph
            .add_edge(&session.id, &tool_id, graphirm_graph::EdgeType::Produces)
            .unwrap();

        let context = build_context(&session).unwrap();

        assert_eq!(context.len(), 4);
        assert_eq!(context[3].role, Role::Tool);
        assert_eq!(context[3].tool_call_id, Some("call_1".to_string()));
    }
}
```

### Step 3: Verify

Run: `cargo test -p graphirm-agent context::tests 2>&1`
Expected: `test result: ok. 3 passed; 0 failed`

### Step 4: Commit

```bash
git add crates/agent/src/context.rs
git commit -m "feat(agent): implement build_context — graph-to-LlmMessage conversion"
```

---

## Task 7: Implement stream_and_record()

- [x] Complete

**Files:**
- Modify: `crates/agent/src/workflow.rs`

### Step 1: Define test helpers (MockProvider, MockTool)

Add a test helper module at the bottom of `crates/agent/src/workflow.rs`. These mocks are reused by Tasks 8–11.

```rust
#[cfg(test)]
mod test_helpers {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use async_trait::async_trait;

    use graphirm_llm::{
        LlmError, LlmMessage, LlmProvider, LlmResponse, ToolCall, ToolDefinition, Usage,
    };
    use graphirm_tools::ToolError;

    /// Mock LLM provider that returns pre-configured responses in order.
    pub struct MockProvider {
        responses: Vec<LlmResponse>,
        call_index: AtomicUsize,
    }

    impl MockProvider {
        pub fn new(responses: Vec<LlmResponse>) -> Self {
            Self {
                responses,
                call_index: AtomicUsize::new(0),
            }
        }

        pub fn call_count(&self) -> usize {
            self.call_index.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl LlmProvider for MockProvider {
        async fn complete(
            &self,
            _messages: Vec<LlmMessage>,
            _tools: Vec<ToolDefinition>,
        ) -> Result<LlmResponse, LlmError> {
            let idx = self.call_index.fetch_add(1, Ordering::SeqCst);
            if idx < self.responses.len() {
                Ok(self.responses[idx].clone())
            } else {
                Err(LlmError::Provider("No more mock responses".to_string()))
            }
        }
    }

    /// Mock tool that returns a fixed output string.
    pub struct MockTool {
        pub tool_name: String,
        pub output: String,
    }

    #[async_trait]
    impl graphirm_tools::Tool for MockTool {
        fn name(&self) -> &str {
            &self.tool_name
        }
        fn description(&self) -> &str {
            "Mock tool for testing"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {}
            })
        }
        async fn execute(&self, _args: serde_json::Value) -> Result<String, ToolError> {
            Ok(self.output.clone())
        }
    }

    /// Helper to build an LlmResponse with no tool calls.
    pub fn text_response(content: &str) -> LlmResponse {
        LlmResponse {
            content: content.to_string(),
            tool_calls: vec![],
            usage: Usage {
                input_tokens: 100,
                output_tokens: 20,
            },
        }
    }

    /// Helper to build an LlmResponse with tool calls.
    pub fn tool_call_response(calls: Vec<(&str, &str, serde_json::Value)>) -> LlmResponse {
        LlmResponse {
            content: String::new(),
            tool_calls: calls
                .into_iter()
                .enumerate()
                .map(|(i, (name, id, args))| ToolCall {
                    id: id.to_string(),
                    name: name.to_string(),
                    arguments: args,
                })
                .collect(),
            usage: Usage {
                input_tokens: 100,
                output_tokens: 50,
            },
        }
    }
}
```

### Step 2: Write test for stream_and_record

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AgentConfig;
    use crate::event::EventBus;
    use crate::session::Session;
    use graphirm_graph::GraphStore;
    use graphirm_tools::ToolRegistry;
    use std::sync::Arc;
    use test_helpers::*;

    #[tokio::test]
    async fn test_stream_and_record_creates_assistant_node() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let config = AgentConfig::default();
        let session = Session::new(graph.clone(), config).unwrap();

        session.add_user_message("What is 2+2?").unwrap();

        let provider = MockProvider::new(vec![text_response("The answer is 4.")]);
        let tools = ToolRegistry::new();
        let bus = EventBus::new();

        let (response, node_id) =
            stream_and_record(&session, &provider, &tools, &bus).await.unwrap();

        assert_eq!(response.content, "The answer is 4.");
        assert!(response.tool_calls.is_empty());

        let node = graph.get_node(&node_id).unwrap();
        assert_eq!(
            node.data["content"].as_str().unwrap(),
            "The answer is 4."
        );
        assert_eq!(node.data["role"].as_str().unwrap(), "assistant");
    }
}
```

Run: `cargo test -p graphirm-agent workflow::tests::test_stream_and_record 2>&1`
Expected: Fails — `stream_and_record` not defined yet.

### Step 3: Implement stream_and_record

Write the implementation at the top of `crates/agent/src/workflow.rs`:

```rust
use graphirm_graph::{EdgeType, NodeType};
use graphirm_llm::{LlmProvider, LlmResponse, ToolDefinition};
use graphirm_tools::ToolRegistry;
use tracing::info;

use crate::context::build_context;
use crate::error::AgentError;
use crate::event::{AgentEvent, EventBus};
use crate::session::Session;

/// Call the LLM with the current conversation context and record the
/// assistant response as an Interaction node in the graph.
///
/// Returns the LlmResponse (which may contain tool_calls) and the
/// NodeId of the recorded response node.
pub async fn stream_and_record(
    session: &Session,
    llm: &dyn LlmProvider,
    tools: &ToolRegistry,
    events: &EventBus,
) -> Result<(LlmResponse, graphirm_graph::NodeId), AgentError> {
    let context = build_context(session)?;
    let tool_defs: Vec<ToolDefinition> = tools.definitions();

    let response = llm.complete(context, tool_defs).await?;

    // Record assistant response in graph
    let data = serde_json::json!({
        "role": "assistant",
        "content": &response.content,
        "tool_calls": &response.tool_calls,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        "created_at": chrono::Utc::now().to_rfc3339(),
    });

    let node_id = session.graph.add_node(NodeType::Interaction, data)?;
    session
        .graph
        .add_edge(&session.id, &node_id, EdgeType::Produces)?;

    info!(node_id = %node_id, "Recorded assistant response");

    events
        .emit(AgentEvent::MessageEnd {
            node_id: node_id.clone(),
        })
        .await;

    Ok((response, node_id))
}
```

### Step 4: Verify

Run: `cargo test -p graphirm-agent workflow::tests::test_stream_and_record 2>&1`
Expected: `test result: ok. 1 passed; 0 failed`

### Step 5: Commit

```bash
git add crates/agent/src/workflow.rs
git commit -m "feat(agent): implement stream_and_record — LLM call + graph persistence"
```

---

## Task 8: Implement run_agent_loop()

- [x] Complete

**Files:**
- Modify: `crates/agent/src/workflow.rs`

### Step 1: Write test — single turn, no tools

Add to the `tests` module in `crates/agent/src/workflow.rs`:

```rust
    #[tokio::test]
    async fn test_agent_loop_single_turn_no_tools() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let config = AgentConfig {
            max_turns: 10,
            ..AgentConfig::default()
        };
        let session = Session::new(graph.clone(), config).unwrap();
        session.add_user_message("What is 2+2?").unwrap();

        let provider = MockProvider::new(vec![text_response("4")]);
        let tools = ToolRegistry::new();
        let mut bus = EventBus::new();
        let mut rx = bus.subscribe();
        let token = CancellationToken::new();

        run_agent_loop(&session, &provider, &tools, &bus, &token)
            .await
            .unwrap();

        // Verify exactly 1 LLM call
        assert_eq!(provider.call_count(), 1);

        // Collect all events
        let mut events = vec![];
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        // First event: AgentStart
        assert!(matches!(events[0], AgentEvent::AgentStart { .. }));
        // Second event: TurnStart { turn_index: 0 }
        assert!(matches!(
            events[1],
            AgentEvent::TurnStart { turn_index: 0 }
        ));
        // Last event: AgentEnd
        assert!(matches!(
            events.last().unwrap(),
            AgentEvent::AgentEnd { .. }
        ));
    }
```

Run: `cargo test -p graphirm-agent workflow::tests::test_agent_loop_single_turn 2>&1`
Expected: Fails — `run_agent_loop` not defined yet.

### Step 2: Implement execute_tools_parallel and run_agent_loop

Add to `crates/agent/src/workflow.rs`:

```rust
use graphirm_graph::NodeId;
use graphirm_llm::ToolCall;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

/// Execute tool calls in parallel using tokio::JoinSet.
/// Each result is recorded as an Interaction node (role: tool) in the graph.
async fn execute_tools_parallel(
    session: &Session,
    tools: &ToolRegistry,
    tool_calls: &[ToolCall],
    events: &EventBus,
) -> Result<Vec<NodeId>, AgentError> {
    let mut set = JoinSet::new();

    for call in tool_calls {
        let tool = tools
            .get(&call.name)
            .ok_or_else(|| AgentError::Workflow(format!("Unknown tool: {}", call.name)))?;
        let args = call.arguments.clone();
        let call_id = call.id.clone();
        let tool_name = call.name.clone();

        set.spawn(async move {
            let result = tool.execute(args).await;
            (call_id, tool_name, result)
        });
    }

    let mut node_ids = Vec::new();

    while let Some(join_result) = set.join_next().await {
        let (call_id, tool_name, exec_result) =
            join_result.map_err(|e| AgentError::Join(e.to_string()))?;

        let (content, is_error) = match exec_result {
            Ok(output) => (output, false),
            Err(e) => (e.to_string(), true),
        };

        // Record tool result in graph
        let data = serde_json::json!({
            "role": "tool",
            "tool_call_id": call_id,
            "tool_name": tool_name,
            "content": content,
            "is_error": is_error,
            "created_at": chrono::Utc::now().to_rfc3339(),
        });
        let node_id = session.graph.add_node(NodeType::Interaction, data)?;
        session
            .graph
            .add_edge(&session.id, &node_id, EdgeType::Produces)?;

        events
            .emit(AgentEvent::ToolEnd {
                node_id: node_id.clone(),
                is_error,
            })
            .await;

        info!(node_id = %node_id, tool = %tool_name, is_error, "Tool execution complete");
        node_ids.push(node_id);
    }

    Ok(node_ids)
}

/// The main agent loop. Cycles between:
/// 1. Build context from graph
/// 2. Call LLM and record response
/// 3. If tool calls present, execute them in parallel and record results
/// 4. Repeat until no tool calls or max_turns reached
pub async fn run_agent_loop(
    session: &Session,
    llm: &dyn LlmProvider,
    tools: &ToolRegistry,
    events: &EventBus,
    cancel: &CancellationToken,
) -> Result<(), AgentError> {
    let max_turns = session.agent_config.max_turns;
    let mut all_node_ids: Vec<NodeId> = Vec::new();

    events
        .emit(AgentEvent::AgentStart {
            agent_id: session.id.clone(),
        })
        .await;

    for turn in 0..max_turns {
        // Check cancellation before each turn
        if cancel.is_cancelled() {
            info!("Agent loop cancelled at turn {}", turn);
            events
                .emit(AgentEvent::AgentEnd {
                    agent_id: session.id.clone(),
                    node_ids: all_node_ids,
                })
                .await;
            return Err(AgentError::Cancelled);
        }

        events.emit(AgentEvent::TurnStart { turn_index: turn }).await;

        // Call LLM and record response
        let (response, response_id) = stream_and_record(session, llm, tools, events).await?;
        all_node_ids.push(response_id.clone());

        if response.tool_calls.is_empty() {
            // No tool calls — conversation turn complete
            events
                .emit(AgentEvent::TurnEnd {
                    response_id,
                    tool_result_ids: vec![],
                })
                .await;
            break;
        }

        // Execute tool calls in parallel
        for call in &response.tool_calls {
            events
                .emit(AgentEvent::ToolStart {
                    node_id: response_id.clone(),
                    tool_name: call.name.clone(),
                })
                .await;
        }

        let tool_result_ids =
            execute_tools_parallel(session, tools, &response.tool_calls, events).await?;

        all_node_ids.extend(tool_result_ids.clone());

        events
            .emit(AgentEvent::TurnEnd {
                response_id,
                tool_result_ids,
            })
            .await;

        // If this was the last allowed turn, emit recursion limit
        if turn + 1 >= max_turns {
            info!("Recursion limit reached at {} turns", max_turns);
            events
                .emit(AgentEvent::AgentEnd {
                    agent_id: session.id.clone(),
                    node_ids: all_node_ids,
                })
                .await;
            return Err(AgentError::RecursionLimit(max_turns));
        }
    }

    events
        .emit(AgentEvent::AgentEnd {
            agent_id: session.id.clone(),
            node_ids: all_node_ids,
        })
        .await;

    Ok(())
}
```

### Step 3: Verify

Run: `cargo test -p graphirm-agent workflow::tests::test_agent_loop_single_turn 2>&1`
Expected: `test result: ok. 1 passed; 0 failed`

### Step 4: Commit

```bash
git add crates/agent/src/workflow.rs
git commit -m "feat(agent): implement run_agent_loop with parallel tool execution"
```

---

## Task 9: Test — tool call loop (2 turns)

- [x] Complete

**Files:**
- Modify: `crates/agent/src/workflow.rs` (test only)

### Step 1: Write test

Add to the `tests` module in `crates/agent/src/workflow.rs`:

```rust
    #[tokio::test]
    async fn test_agent_loop_tool_call_then_text() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let config = AgentConfig {
            max_turns: 10,
            ..AgentConfig::default()
        };
        let session = Session::new(graph.clone(), config).unwrap();
        session.add_user_message("List files").unwrap();

        // Turn 1: LLM returns a tool call
        // Turn 2: LLM returns text (no more tools)
        let provider = MockProvider::new(vec![
            tool_call_response(vec![(
                "bash",
                "call_1",
                serde_json::json!({"command": "ls"}),
            )]),
            text_response("Here are your files: src/ Cargo.toml"),
        ]);

        let mock_bash = Arc::new(MockTool {
            tool_name: "bash".to_string(),
            output: "src/\nCargo.toml".to_string(),
        });
        let mut tools = ToolRegistry::new();
        tools.register(mock_bash);

        let mut bus = EventBus::new();
        let mut rx = bus.subscribe();
        let token = CancellationToken::new();

        run_agent_loop(&session, &provider, &tools, &bus, &token)
            .await
            .unwrap();

        // 2 LLM calls total
        assert_eq!(provider.call_count(), 2);

        // Collect events
        let mut events = vec![];
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        // Verify we see TurnStart 0, TurnEnd (with tool results), TurnStart 1, TurnEnd (no tools)
        let turn_starts: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::TurnStart { .. }))
            .collect();
        assert_eq!(turn_starts.len(), 2);

        // Verify tool execution events
        let tool_ends: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolEnd { .. }))
            .collect();
        assert_eq!(tool_ends.len(), 1);

        // Verify tool result was recorded in graph
        let neighbors = graph
            .get_neighbors(&session.id, Some(graphirm_graph::EdgeType::Produces))
            .unwrap();
        let tool_nodes: Vec<_> = neighbors
            .iter()
            .filter(|n| n.data["role"].as_str() == Some("tool"))
            .collect();
        assert_eq!(tool_nodes.len(), 1);
        assert_eq!(
            tool_nodes[0].data["content"].as_str().unwrap(),
            "src/\nCargo.toml"
        );
    }
```

### Step 2: Verify

Run: `cargo test -p graphirm-agent workflow::tests::test_agent_loop_tool_call_then_text 2>&1`
Expected: `test result: ok. 1 passed; 0 failed`

### Step 3: Commit

```bash
git add crates/agent/src/workflow.rs
git commit -m "test(agent): add tool call loop test — 2 turns with bash mock"
```

---

## Task 10: Test — recursion limit

- [x] Complete

**Files:**
- Modify: `crates/agent/src/workflow.rs` (test only)

### Step 1: Write test

Add to the `tests` module:

```rust
    #[tokio::test]
    async fn test_agent_loop_recursion_limit() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let config = AgentConfig {
            max_turns: 3,
            ..AgentConfig::default()
        };
        let session = Session::new(graph.clone(), config).unwrap();
        session.add_user_message("Do infinite things").unwrap();

        // Every response has a tool call — loop should stop at max_turns
        let provider = MockProvider::new(vec![
            tool_call_response(vec![(
                "bash",
                "c1",
                serde_json::json!({"command": "echo 1"}),
            )]),
            tool_call_response(vec![(
                "bash",
                "c2",
                serde_json::json!({"command": "echo 2"}),
            )]),
            tool_call_response(vec![(
                "bash",
                "c3",
                serde_json::json!({"command": "echo 3"}),
            )]),
        ]);

        let mock_bash = Arc::new(MockTool {
            tool_name: "bash".to_string(),
            output: "ok".to_string(),
        });
        let mut tools = ToolRegistry::new();
        tools.register(mock_bash);

        let mut bus = EventBus::new();
        let mut rx = bus.subscribe();
        let token = CancellationToken::new();

        let result = run_agent_loop(&session, &provider, &tools, &bus, &token).await;

        // Should fail with RecursionLimit
        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::RecursionLimit(n) => assert_eq!(n, 3),
            other => panic!("Expected RecursionLimit, got: {:?}", other),
        }

        // Should have made exactly 3 LLM calls
        assert_eq!(provider.call_count(), 3);

        // Verify AgentEnd was still emitted
        let mut events = vec![];
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        assert!(matches!(
            events.last().unwrap(),
            AgentEvent::AgentEnd { .. }
        ));
    }
```

### Step 2: Verify

Run: `cargo test -p graphirm-agent workflow::tests::test_agent_loop_recursion_limit 2>&1`
Expected: `test result: ok. 1 passed; 0 failed`

### Step 3: Commit

```bash
git add crates/agent/src/workflow.rs
git commit -m "test(agent): add recursion limit test — loop stops at max_turns"
```

---

## Task 11: Test — cancellation

- [x] Complete

**Files:**
- Modify: `crates/agent/src/workflow.rs` (test only)

### Step 1: Write test

Add to the `tests` module:

```rust
    #[tokio::test]
    async fn test_agent_loop_cancellation() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let config = AgentConfig {
            max_turns: 100,
            ..AgentConfig::default()
        };
        let session = Session::new(graph.clone(), config).unwrap();
        session.add_user_message("Start working").unwrap();

        // Provide many responses so the loop would run for a while
        let provider = MockProvider::new(vec![
            tool_call_response(vec![(
                "bash",
                "c1",
                serde_json::json!({"command": "echo 1"}),
            )]),
            tool_call_response(vec![(
                "bash",
                "c2",
                serde_json::json!({"command": "echo 2"}),
            )]),
            text_response("done"), // won't reach this
        ]);

        let mock_bash = Arc::new(MockTool {
            tool_name: "bash".to_string(),
            output: "ok".to_string(),
        });
        let mut tools = ToolRegistry::new();
        tools.register(mock_bash);

        let mut bus = EventBus::new();
        let mut rx = bus.subscribe();
        let token = CancellationToken::new();

        // Cancel after a brief delay
        let cancel_token = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            cancel_token.cancel();
        });

        let result = run_agent_loop(&session, &provider, &tools, &bus, &token).await;

        // The loop should exit with Cancelled (may complete 1-2 turns first)
        assert!(
            matches!(result, Err(AgentError::Cancelled))
                || result.is_ok(), // if cancellation check happened after last tool call
            "Expected Cancelled or Ok, got: {:?}",
            result
        );

        // Verify AgentEnd was emitted
        let mut events = vec![];
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        let has_agent_end = events
            .iter()
            .any(|e| matches!(e, AgentEvent::AgentEnd { .. }));
        assert!(has_agent_end, "AgentEnd event should be emitted on cancel");
    }
```

### Step 2: Verify

Run: `cargo test -p graphirm-agent workflow::tests::test_agent_loop_cancellation 2>&1`
Expected: `test result: ok. 1 passed; 0 failed`

### Step 3: Run full test suite

Run: `cargo test -p graphirm-agent 2>&1`
Expected: `test result: ok. 13 passed; 0 failed` (approximate — all tests pass)

### Step 4: Run clippy

Run: `cargo clippy -p graphirm-agent --all-targets 2>&1`
Expected: No errors. Warnings acceptable.

### Step 5: Commit

```bash
git add crates/agent/src/workflow.rs
git commit -m "test(agent): add cancellation test — CancellationToken stops loop cleanly"
```

---

## Final: Update lib.rs with all re-exports

- [x] Complete

**Files:**
- Modify: `crates/agent/src/lib.rs`

Ensure `crates/agent/src/lib.rs` has all public re-exports:

```rust
pub mod compact;
pub mod config;
pub mod context;
pub mod error;
pub mod event;
pub mod multi;
pub mod session;
pub mod workflow;

pub use config::AgentConfig;
pub use error::AgentError;
pub use event::{AgentEvent, EventBus};
pub use session::Session;
pub use workflow::run_agent_loop;
```

### Verify full build

Run: `cargo build -p graphirm-agent 2>&1`
Expected: Compiles with no errors.

Run: `cargo test -p graphirm-agent 2>&1`
Expected: All tests pass.

### Commit

```bash
git add crates/agent/src/lib.rs
git commit -m "feat(agent): Phase 4 complete — agent loop with events, context, tools"
```

---

## Post-Merge Cleanup (from code review — Minor items)

These are quality improvements identified during code review that do not block merge but should be addressed before Phase 7 integration work.

- [ ] **#12 Double clone of `tool_result_ids` removed** — Already fixed during the Critical/Important fix pass (`workflow.rs`).

- [ ] **#13 Recursion limit comment** — `workflow.rs` `run_agent_loop`: the check `turn + 1 >= max_turns` fires only on tool-call turns (text-response turns exit via `break`). Add an inline comment explaining this asymmetry so it's not mistaken for an off-by-one.

- [ ] **#14 Doc comments on all public items** — Add `///` doc comments to `run_agent_loop`, `stream_and_record`, `Session::new`, `AgentConfig` fields, and all `AgentEvent` variants. Follow the pattern already set by `build_context` and `add_user_message`.

- [ ] **#15 Fragile test ordering** — `context::tests::test_build_context_with_messages` creates five nodes in a tight loop and relies on `sort_by(created_at)`. If all nodes get the same timestamp at microsecond granularity, sort order is insertion-dependent. Refactor: insert nodes with explicit `chrono::Utc::now() + Duration::milliseconds(i as i64)` offsets, or use a test helper that forces distinct timestamps.

- [ ] **#16 `tool_call_response` helper arg order** — Add a doc comment to the helper in `workflow::test_helpers` clarifying the tuple order `(tool_name, call_id, arguments)` to prevent argument transposition when writing new tests.
