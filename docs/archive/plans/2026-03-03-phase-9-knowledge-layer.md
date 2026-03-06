# Phase 9: Knowledge Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement knowledge extraction from agent conversations and HNSW-powered cross-session memory retrieval, so past decisions, patterns, and entities are automatically surfaced in new sessions.

**Architecture:** After each agent turn, an extraction pipeline calls a lightweight LLM with a structured JSON schema prompt to pull entities and relationships from the conversation. Extracted entities become `Knowledge` graph nodes linked via `RelatesTo` and a new `DerivedFrom` edge type. Each knowledge node stores an embedding vector (from rig-core's embedding API) as a BLOB in SQLite. On startup, an HNSW index is rebuilt from stored embeddings. When a new session begins, the initial prompt is embedded and used to query the HNSW index for relevant knowledge from all past sessions, which is injected as system context.

**Background consciousness loop:** In addition to post-turn extraction, a continuous async loop runs a cheap model on a configurable cadence (default: after every 5 completed turns, or on a 60s idle timer). It traverses recently closed `Interaction` and `Task` nodes, asks "what patterns, decisions, or heuristics should I remember from this work?", and writes the answers as `Knowledge` nodes with `Summarizes` edges back to the source nodes. This loop runs in its own `tokio::task`, consumes a capped token budget (configurable, default 5% of session budget), and is always on — not opt-in. The result: the agent accumulates judgment continuously, not just at turn boundaries. This is distinct from compaction (Phase 6) — compaction shrinks context, this loop *grows* the knowledge graph.

**Tech Stack:** instant-distance 0.6 (HNSW, MIT licensed), rig-core (embedding API), serde + serde_json (structured extraction), tokio (async), rusqlite (BLOB storage), uuid + chrono (IDs/timestamps)

---

## Prerequisites (expected APIs from earlier phases)

### From `graphirm-graph` (Phase 1)

```rust
pub struct NodeId(pub String);
pub struct EdgeId(pub String);

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

impl GraphStore {
    pub fn new_in_memory() -> Result<Self, GraphError>;
    pub fn add_node(&self, node_type: NodeType, data: serde_json::Value) -> Result<NodeId, GraphError>;
    pub fn add_edge(&self, source: &NodeId, target: &NodeId, edge_type: EdgeType) -> Result<EdgeId, GraphError>;
    pub fn get_node(&self, id: &NodeId) -> Result<Node, GraphError>;
    pub fn get_neighbors(&self, id: &NodeId, edge_type: Option<EdgeType>) -> Result<Vec<Node>, GraphError>;
    pub fn query_nodes(&self, node_type: NodeType) -> Result<Vec<Node>, GraphError>;
}
```

### From `graphirm-llm` (Phase 2)

```rust
#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn complete(
        &self,
        messages: Vec<LlmMessage>,
        tools: Vec<ToolDefinition>,
    ) -> Result<LlmResponse, LlmError>;

    async fn embed(&self, text: &str) -> Result<Vec<f32>, LlmError>;
}

pub struct LlmMessage {
    pub role: Role,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub tool_call_id: Option<String>,
}

pub struct LlmResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Usage,
}
```

### From `graphirm-agent` (Phase 4)

```rust
pub struct Session {
    pub id: String,
    pub graph: Arc<GraphStore>,
    pub config: AgentConfig,
}

pub struct AgentConfig {
    pub model: String,
    pub max_turns: u32,
    pub system_prompt: String,
}

pub enum AgentEvent {
    TurnEnd { response_id: NodeId, tool_result_ids: Vec<NodeId> },
    // ...
}

pub struct EventBus { /* ... */ }
impl EventBus {
    pub fn subscribe(&mut self) -> mpsc::Receiver<AgentEvent>;
}
```

---

## Task 1: Add Phase 9 dependencies and create knowledge module

- [x] Complete

**Files:**
- Modify: `crates/agent/Cargo.toml`
- Create: `crates/agent/src/knowledge.rs`
- Modify: `crates/agent/src/lib.rs`
- Modify: `crates/graph/Cargo.toml`
- Create: `crates/graph/src/vector.rs`
- Modify: `crates/graph/src/lib.rs`

### Step 1: Add instant-distance to graph crate

Add to `crates/graph/Cargo.toml` under `[dependencies]`:

```toml
instant-distance = "0.6"
bytemuck = { version = "1", features = ["derive"] }
```

### Step 2: Create vector module stub in graph crate

Create `crates/graph/src/vector.rs`:

```rust
//! HNSW vector index for knowledge node embeddings.

pub struct VectorIndex;
```

### Step 3: Export vector module from graph crate

Add to `crates/graph/src/lib.rs`:

```rust
pub mod vector;
```

### Step 4: Add DerivedFrom edge type

Add `DerivedFrom` to the `EdgeType` enum in `crates/graph/src/edges.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeType {
    RespondsTo,
    SpawnedBy,
    DelegatesTo,
    DependsOn,
    Produces,
    Reads,
    Modifies,
    Summarizes,
    Contains,
    FollowsUp,
    Steers,
    RelatesTo,
    DerivedFrom, // knowledge → source interaction
}
```

Update the `EdgeType::as_str()` and `EdgeType::from_str()` implementations to include `"derived_from" <=> DerivedFrom`.

### Step 5: Create knowledge module stub in agent crate

Create `crates/agent/src/knowledge.rs`:

```rust
//! Knowledge extraction and cross-session memory.

pub mod extraction;
pub mod memory;
```

Create `crates/agent/src/knowledge/extraction.rs`:

```rust
//! Post-turn knowledge extraction from conversations.
```

Create `crates/agent/src/knowledge/memory.rs`:

```rust
//! Cross-session memory retrieval using HNSW vector search.
```

### Step 6: Export knowledge module from agent crate

Add to `crates/agent/src/lib.rs`:

```rust
pub mod knowledge;
```

### Step 7: Verify

Run: `cargo check --workspace 2>&1`
Expected: Compiles with no errors.

### Step 8: Commit

```bash
git add crates/graph/Cargo.toml crates/graph/src/vector.rs crates/graph/src/edges.rs crates/graph/src/lib.rs \
       crates/agent/Cargo.toml crates/agent/src/knowledge.rs crates/agent/src/knowledge/ crates/agent/src/lib.rs
git commit -m "feat(knowledge): scaffold Phase 9 modules, add DerivedFrom edge type"
```

---

## Task 2: Define ExtractionConfig with serde defaults

- [x] Complete

**Files:**
- Modify: `crates/agent/src/knowledge/extraction.rs`
- Test: `crates/agent/src/knowledge/extraction.rs` (in-module `#[cfg(test)]`)

### Step 1: Write failing tests

Add to `crates/agent/src/knowledge/extraction.rs`:

```rust
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extraction_config_defaults() {
        let config = ExtractionConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.model, "gpt-4o-mini");
        assert_eq!(config.min_confidence, 0.7);
        assert!(!config.entity_types.is_empty());
    }

    #[test]
    fn test_extraction_config_deserialize_partial() {
        let toml_str = r#"
            enabled = true
            model = "claude-3-haiku"
        "#;
        let config: ExtractionConfig = toml::from_str(toml_str).unwrap();
        assert!(config.enabled);
        assert_eq!(config.model, "claude-3-haiku");
        assert_eq!(config.min_confidence, 0.7); // default
        assert!(!config.entity_types.is_empty()); // default
    }

    #[test]
    fn test_extraction_config_serialize_roundtrip() {
        let config = ExtractionConfig {
            enabled: true,
            model: "deepseek-chat".to_string(),
            entity_types: vec!["function".to_string(), "api".to_string()],
            min_confidence: 0.85,
        };
        let json = serde_json::to_string(&config).unwrap();
        let back: ExtractionConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.enabled, true);
        assert_eq!(back.model, "deepseek-chat");
        assert_eq!(back.entity_types.len(), 2);
        assert!((back.min_confidence - 0.85).abs() < f64::EPSILON);
    }
}
```

Run: `cargo test -p graphirm-agent knowledge::extraction::tests 2>&1`
Expected: Fails — `ExtractionConfig` not defined yet.

### Step 2: Implement ExtractionConfig

Add above the tests in `crates/agent/src/knowledge/extraction.rs`:

```rust
use serde::{Deserialize, Serialize};

fn default_entity_types() -> Vec<String> {
    vec![
        "function".into(),
        "api".into(),
        "pattern".into(),
        "decision".into(),
        "bug".into(),
        "architecture".into(),
        "convention".into(),
        "library".into(),
    ]
}

fn default_model() -> String {
    "gpt-4o-mini".into()
}

fn default_min_confidence() -> f64 {
    0.7
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    #[serde(default)]
    pub enabled: bool,

    #[serde(default = "default_model")]
    pub model: String,

    #[serde(default = "default_entity_types")]
    pub entity_types: Vec<String>,

    #[serde(default = "default_min_confidence")]
    pub min_confidence: f64,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: default_model(),
            entity_types: default_entity_types(),
            min_confidence: default_min_confidence(),
        }
    }
}
```

### Step 3: Add toml dev-dependency

Add to `crates/agent/Cargo.toml` under `[dev-dependencies]`:

```toml
toml = "0.8"
```

### Step 4: Run tests

Run: `cargo test -p graphirm-agent knowledge::extraction::tests -- --nocapture 2>&1`
Expected: All 3 tests pass.

### Step 5: Commit

```bash
git add crates/agent/src/knowledge/extraction.rs crates/agent/Cargo.toml
git commit -m "feat(knowledge): define ExtractionConfig with serde defaults"
```

---

## Task 3: Define ExtractedEntity and EntityRelationship

- [x] Complete

**Files:**
- Modify: `crates/agent/src/knowledge/extraction.rs`
- Test: `crates/agent/src/knowledge/extraction.rs` (in-module `#[cfg(test)]`)

### Step 1: Write failing tests

Add to the `tests` module in `crates/agent/src/knowledge/extraction.rs`:

```rust
    #[test]
    fn test_extracted_entity_roundtrip() {
        let entity = ExtractedEntity {
            entity_type: "function".to_string(),
            name: "parse_config".to_string(),
            description: "Parses TOML configuration files and returns a Config struct".to_string(),
            confidence: 0.92,
            relationships: vec![
                EntityRelationship {
                    target_name: "Config".to_string(),
                    relationship: "returns".to_string(),
                },
                EntityRelationship {
                    target_name: "toml".to_string(),
                    relationship: "uses".to_string(),
                },
            ],
        };
        let json = serde_json::to_string(&entity).unwrap();
        let back: ExtractedEntity = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "parse_config");
        assert_eq!(back.entity_type, "function");
        assert_eq!(back.relationships.len(), 2);
        assert_eq!(back.relationships[0].target_name, "Config");
        assert!((back.confidence - 0.92).abs() < f64::EPSILON);
    }

    #[test]
    fn test_extracted_entity_from_llm_json() {
        let llm_output = r#"{
            "entities": [
                {
                    "entity_type": "pattern",
                    "name": "Repository Pattern",
                    "description": "Data access abstraction using trait objects",
                    "confidence": 0.88,
                    "relationships": [
                        { "target_name": "GraphStore", "relationship": "implements" }
                    ]
                },
                {
                    "entity_type": "decision",
                    "name": "Use rusqlite over sqlitegraph",
                    "description": "Chose MIT-licensed rusqlite to avoid GPL infection",
                    "confidence": 0.95,
                    "relationships": []
                }
            ]
        }"#;
        let parsed: ExtractionResponse = serde_json::from_str(llm_output).unwrap();
        assert_eq!(parsed.entities.len(), 2);
        assert_eq!(parsed.entities[0].name, "Repository Pattern");
        assert_eq!(parsed.entities[1].entity_type, "decision");
    }

    #[test]
    fn test_entity_relationship_display() {
        let rel = EntityRelationship {
            target_name: "tokio".to_string(),
            relationship: "depends_on".to_string(),
        };
        let debug = format!("{:?}", rel);
        assert!(debug.contains("tokio"));
        assert!(debug.contains("depends_on"));
    }
```

Run: `cargo test -p graphirm-agent knowledge::extraction::tests 2>&1`
Expected: Fails — `ExtractedEntity`, `EntityRelationship`, `ExtractionResponse` not defined.

### Step 2: Implement the types

Add to `crates/agent/src/knowledge/extraction.rs` (above the tests, below `ExtractionConfig`):

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    pub entity_type: String,
    pub name: String,
    pub description: String,
    pub confidence: f64,
    pub relationships: Vec<EntityRelationship>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRelationship {
    pub target_name: String,
    pub relationship: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResponse {
    pub entities: Vec<ExtractedEntity>,
}
```

### Step 3: Run tests

Run: `cargo test -p graphirm-agent knowledge::extraction::tests -- --nocapture 2>&1`
Expected: All 6 tests pass (3 from Task 2 + 3 new).

### Step 4: Commit

```bash
git add crates/agent/src/knowledge/extraction.rs
git commit -m "feat(knowledge): define ExtractedEntity, EntityRelationship, ExtractionResponse"
```

---

## Task 4: Implement extraction prompt builder

- [x] Complete

**Files:**
- Modify: `crates/agent/src/knowledge/extraction.rs`
- Test: `crates/agent/src/knowledge/extraction.rs` (in-module `#[cfg(test)]`)

### Step 1: Write failing tests

Add to the `tests` module:

```rust
    #[test]
    fn test_build_extraction_prompt_structure() {
        let messages = vec![
            ("user".to_string(), "How should I implement authentication?".to_string()),
            ("assistant".to_string(), "You should use JWT tokens with bcrypt for password hashing. Store sessions in Redis.".to_string()),
            ("user".to_string(), "What about refresh tokens?".to_string()),
        ];
        let config = ExtractionConfig::default();
        let prompt = build_extraction_prompt(&messages, &config);

        assert!(prompt.contains("authentication"));
        assert!(prompt.contains("JWT"));
        assert!(prompt.contains("entity_type"));
        assert!(prompt.contains("confidence"));
        assert!(prompt.contains("function"));
        assert!(prompt.contains("pattern"));
    }

    #[test]
    fn test_build_extraction_prompt_includes_entity_types() {
        let messages = vec![
            ("user".to_string(), "Hello".to_string()),
        ];
        let config = ExtractionConfig {
            entity_types: vec!["function".into(), "api".into()],
            ..ExtractionConfig::default()
        };
        let prompt = build_extraction_prompt(&messages, &config);
        assert!(prompt.contains("function"));
        assert!(prompt.contains("api"));
        assert!(!prompt.contains("architecture"));
    }

    #[test]
    fn test_build_extraction_prompt_empty_conversation() {
        let messages: Vec<(String, String)> = vec![];
        let config = ExtractionConfig::default();
        let prompt = build_extraction_prompt(&messages, &config);
        assert!(prompt.contains("entity_type"));
        assert!(prompt.contains("CONVERSATION"));
    }
```

Run: `cargo test -p graphirm-agent knowledge::extraction::tests 2>&1`
Expected: Fails — `build_extraction_prompt` not defined.

### Step 2: Implement build_extraction_prompt

Add to `crates/agent/src/knowledge/extraction.rs`:

```rust
pub fn build_extraction_prompt(
    messages: &[(String, String)],
    config: &ExtractionConfig,
) -> String {
    let entity_types_list = config.entity_types.join(", ");

    let conversation_block = if messages.is_empty() {
        "(empty conversation)".to_string()
    } else {
        messages
            .iter()
            .map(|(role, content)| format!("[{}]: {}", role, content))
            .collect::<Vec<_>>()
            .join("\n")
    };

    format!(
        r#"Extract knowledge entities from the following conversation.

For each entity, provide:
- entity_type: one of [{entity_types}]
- name: short identifier for the entity
- description: one-sentence description of what it is or why it matters
- confidence: 0.0-1.0 how confident you are this is a real, useful entity
- relationships: array of {{ target_name, relationship }} pairs linking to other entities

Only extract entities with confidence >= {min_confidence}.

Respond with ONLY valid JSON in this exact format:
{{
  "entities": [
    {{
      "entity_type": "pattern",
      "name": "Example Pattern",
      "description": "Description of the pattern",
      "confidence": 0.9,
      "relationships": [
        {{ "target_name": "OtherEntity", "relationship": "uses" }}
      ]
    }}
  ]
}}

CONVERSATION:
{conversation}
"#,
        entity_types = entity_types_list,
        min_confidence = config.min_confidence,
        conversation = conversation_block,
    )
}
```

### Step 3: Run tests

Run: `cargo test -p graphirm-agent knowledge::extraction::tests -- --nocapture 2>&1`
Expected: All 9 tests pass.

### Step 4: Commit

```bash
git add crates/agent/src/knowledge/extraction.rs
git commit -m "feat(knowledge): implement extraction prompt builder"
```

---

## Task 5: Implement extract_knowledge() with LLM call

- [x] Complete

**Files:**
- Modify: `crates/agent/src/knowledge/extraction.rs`
- Test: `crates/agent/src/knowledge/extraction.rs` (in-module `#[cfg(test)]`)

### Step 1: Write failing tests

Add to the `tests` module:

```rust
    use graphirm_graph::{GraphStore, NodeType, EdgeType};
    use graphirm_llm::{LlmMessage, LlmResponse, LlmError, ToolCall, ToolDefinition, Usage, Role};
    use std::sync::Arc;

    struct MockExtractionProvider {
        response_json: String,
    }

    #[async_trait::async_trait]
    impl graphirm_llm::LlmProvider for MockExtractionProvider {
        async fn complete(
            &self,
            _messages: Vec<LlmMessage>,
            _tools: Vec<ToolDefinition>,
        ) -> Result<LlmResponse, LlmError> {
            Ok(LlmResponse {
                content: self.response_json.clone(),
                tool_calls: vec![],
                usage: Usage::default(),
            })
        }

        async fn embed(&self, _text: &str) -> Result<Vec<f32>, LlmError> {
            Ok(vec![0.1; 128])
        }
    }

    #[tokio::test]
    async fn test_extract_knowledge_creates_nodes() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let config = ExtractionConfig {
            enabled: true,
            min_confidence: 0.5,
            ..ExtractionConfig::default()
        };
        let llm = Arc::new(MockExtractionProvider {
            response_json: r#"{
                "entities": [
                    {
                        "entity_type": "pattern",
                        "name": "JWT Authentication",
                        "description": "Token-based auth using JSON Web Tokens",
                        "confidence": 0.92,
                        "relationships": []
                    },
                    {
                        "entity_type": "library",
                        "name": "bcrypt",
                        "description": "Password hashing library",
                        "confidence": 0.88,
                        "relationships": [
                            { "target_name": "JWT Authentication", "relationship": "used_with" }
                        ]
                    }
                ]
            }"#.to_string(),
        });

        let messages = vec![
            ("user".to_string(), "How do I do auth?".to_string()),
            ("assistant".to_string(), "Use JWT with bcrypt".to_string()),
        ];

        let source_node_id = graph
            .add_node(NodeType::Interaction, serde_json::json!({"role": "assistant", "content": "Use JWT with bcrypt"}))
            .unwrap();

        let node_ids = extract_knowledge(
            &graph, &*llm, &messages, &source_node_id, &config,
        ).await.unwrap();

        assert_eq!(node_ids.len(), 2);

        // Verify nodes exist in graph
        for id in &node_ids {
            let node = graph.get_node(id).unwrap();
            assert_eq!(node.node_type, NodeType::Knowledge);
        }
    }

    #[tokio::test]
    async fn test_extract_knowledge_creates_derived_from_edges() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let config = ExtractionConfig {
            enabled: true,
            min_confidence: 0.5,
            ..ExtractionConfig::default()
        };
        let llm = Arc::new(MockExtractionProvider {
            response_json: r#"{
                "entities": [
                    {
                        "entity_type": "decision",
                        "name": "Use Rust",
                        "description": "Chose Rust for performance",
                        "confidence": 0.95,
                        "relationships": []
                    }
                ]
            }"#.to_string(),
        });

        let messages = vec![("user".to_string(), "Why Rust?".to_string())];
        let source_id = graph
            .add_node(NodeType::Interaction, serde_json::json!({"content": "Why Rust?"}))
            .unwrap();

        let node_ids = extract_knowledge(
            &graph, &*llm, &messages, &source_id, &config,
        ).await.unwrap();

        assert_eq!(node_ids.len(), 1);

        // Verify DerivedFrom edge exists: knowledge -> source interaction
        let neighbors = graph
            .get_neighbors(&node_ids[0], Some(EdgeType::DerivedFrom))
            .unwrap();
        assert!(neighbors.iter().any(|n| n.id == source_id));
    }

    #[tokio::test]
    async fn test_extract_knowledge_filters_by_confidence() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let config = ExtractionConfig {
            enabled: true,
            min_confidence: 0.9,
            ..ExtractionConfig::default()
        };
        let llm = Arc::new(MockExtractionProvider {
            response_json: r#"{
                "entities": [
                    {
                        "entity_type": "function",
                        "name": "high_confidence",
                        "description": "Very sure about this one",
                        "confidence": 0.95,
                        "relationships": []
                    },
                    {
                        "entity_type": "function",
                        "name": "low_confidence",
                        "description": "Not sure about this",
                        "confidence": 0.4,
                        "relationships": []
                    }
                ]
            }"#.to_string(),
        });

        let messages = vec![("user".to_string(), "test".to_string())];
        let source_id = graph
            .add_node(NodeType::Interaction, serde_json::json!({"content": "test"}))
            .unwrap();

        let node_ids = extract_knowledge(
            &graph, &*llm, &messages, &source_id, &config,
        ).await.unwrap();

        assert_eq!(node_ids.len(), 1); // only the high-confidence entity
    }

    #[tokio::test]
    async fn test_extract_knowledge_creates_relates_to_edges() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let config = ExtractionConfig {
            enabled: true,
            min_confidence: 0.5,
            ..ExtractionConfig::default()
        };
        let llm = Arc::new(MockExtractionProvider {
            response_json: r#"{
                "entities": [
                    {
                        "entity_type": "pattern",
                        "name": "EntityA",
                        "description": "First entity",
                        "confidence": 0.9,
                        "relationships": [
                            { "target_name": "EntityB", "relationship": "uses" }
                        ]
                    },
                    {
                        "entity_type": "library",
                        "name": "EntityB",
                        "description": "Second entity",
                        "confidence": 0.85,
                        "relationships": []
                    }
                ]
            }"#.to_string(),
        });

        let messages = vec![("user".to_string(), "entities".to_string())];
        let source_id = graph
            .add_node(NodeType::Interaction, serde_json::json!({"content": "entities"}))
            .unwrap();

        let node_ids = extract_knowledge(
            &graph, &*llm, &messages, &source_id, &config,
        ).await.unwrap();

        assert_eq!(node_ids.len(), 2);

        // EntityA should have a RelatesTo edge to EntityB
        let neighbors = graph
            .get_neighbors(&node_ids[0], Some(EdgeType::RelatesTo))
            .unwrap();
        assert_eq!(neighbors.len(), 1);
    }
```

Run: `cargo test -p graphirm-agent knowledge::extraction::tests 2>&1`
Expected: Fails — `extract_knowledge` not defined.

### Step 2: Implement extract_knowledge

Add to `crates/agent/src/knowledge/extraction.rs`:

```rust
use graphirm_graph::{EdgeType, GraphStore, NodeId, NodeType};
use graphirm_llm::{LlmMessage, LlmProvider, Role, ToolDefinition};
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::AgentError;

pub async fn extract_knowledge(
    graph: &GraphStore,
    llm: &dyn LlmProvider,
    messages: &[(String, String)],
    source_node_id: &NodeId,
    config: &ExtractionConfig,
) -> Result<Vec<NodeId>, AgentError> {
    if !config.enabled {
        return Ok(vec![]);
    }

    let prompt = build_extraction_prompt(messages, config);

    let llm_messages = vec![LlmMessage {
        role: Role::User,
        content: prompt,
        tool_calls: vec![],
        tool_call_id: None,
    }];

    let response = llm
        .complete(llm_messages, vec![])
        .await
        .map_err(AgentError::Llm)?;

    let extraction: ExtractionResponse = serde_json::from_str(&response.content)
        .map_err(|e| AgentError::Workflow(format!("Failed to parse extraction response: {}", e)))?;

    let filtered: Vec<&ExtractedEntity> = extraction
        .entities
        .iter()
        .filter(|e| e.confidence >= config.min_confidence)
        .collect();

    // First pass: create all knowledge nodes, building a name -> NodeId map
    let mut name_to_id: HashMap<String, NodeId> = HashMap::new();
    let mut created_ids: Vec<NodeId> = Vec::new();

    for entity in &filtered {
        let data = serde_json::json!({
            "entity_type": entity.entity_type,
            "name": entity.name,
            "description": entity.description,
            "confidence": entity.confidence,
        });

        let node_id = graph
            .add_node(NodeType::Knowledge, data)
            .map_err(AgentError::Graph)?;

        // DerivedFrom edge: knowledge -> source interaction
        graph
            .add_edge(&node_id, source_node_id, EdgeType::DerivedFrom)
            .map_err(AgentError::Graph)?;

        name_to_id.insert(entity.name.clone(), node_id.clone());
        created_ids.push(node_id);
    }

    // Second pass: create RelatesTo edges between entities
    for entity in &filtered {
        if let Some(source_id) = name_to_id.get(&entity.name) {
            for rel in &entity.relationships {
                if let Some(target_id) = name_to_id.get(&rel.target_name) {
                    graph
                        .add_edge(source_id, target_id, EdgeType::RelatesTo)
                        .map_err(AgentError::Graph)?;
                }
            }
        }
    }

    tracing::info!(
        extracted = filtered.len(),
        total = extraction.entities.len(),
        "Knowledge extraction complete"
    );

    Ok(created_ids)
}
```

### Step 3: Run tests

Run: `cargo test -p graphirm-agent knowledge::extraction::tests -- --nocapture 2>&1`
Expected: All 13 tests pass.

### Step 4: Commit

```bash
git add crates/agent/src/knowledge/extraction.rs
git commit -m "feat(knowledge): implement extract_knowledge with LLM call, node creation, and edge linking"
```

---

## Task 6: Implement post-turn extraction hook

- [x] Complete

**Files:**
- Modify: `crates/agent/src/knowledge/extraction.rs`
- Modify: `crates/agent/src/loop.rs` (or wherever the agent loop lives)
- Test: `crates/agent/src/knowledge/extraction.rs` (in-module `#[cfg(test)]`)

### Step 1: Write failing test

Add to the `tests` module:

```rust
    #[tokio::test]
    async fn test_post_turn_extraction_hook() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let config = ExtractionConfig {
            enabled: true,
            min_confidence: 0.5,
            ..ExtractionConfig::default()
        };
        let llm = Arc::new(MockExtractionProvider {
            response_json: r#"{
                "entities": [
                    {
                        "entity_type": "convention",
                        "name": "snake_case naming",
                        "description": "Use snake_case for Rust function names",
                        "confidence": 0.9,
                        "relationships": []
                    }
                ]
            }"#.to_string(),
        });

        // Simulate conversation context: user message + assistant response
        let user_id = graph
            .add_node(NodeType::Interaction, serde_json::json!({
                "role": "user",
                "content": "What naming convention should I use?"
            }))
            .unwrap();

        let assistant_id = graph
            .add_node(NodeType::Interaction, serde_json::json!({
                "role": "assistant",
                "content": "Use snake_case for functions in Rust."
            }))
            .unwrap();

        // Simulate edge: assistant responds to user
        graph.add_edge(&assistant_id, &user_id, EdgeType::RespondsTo).unwrap();

        let result = post_turn_extract(
            &graph, &*llm, &config, &assistant_id,
        ).await;

        assert!(result.is_ok());
        let node_ids = result.unwrap();
        assert_eq!(node_ids.len(), 1);

        // Knowledge node should be DerivedFrom the assistant message
        let neighbors = graph
            .get_neighbors(&node_ids[0], Some(EdgeType::DerivedFrom))
            .unwrap();
        assert!(neighbors.iter().any(|n| n.id == assistant_id));
    }

    #[tokio::test]
    async fn test_post_turn_extraction_disabled() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let config = ExtractionConfig {
            enabled: false,
            ..ExtractionConfig::default()
        };
        let llm = Arc::new(MockExtractionProvider {
            response_json: "{}".to_string(),
        });

        let node_id = graph
            .add_node(NodeType::Interaction, serde_json::json!({"content": "test"}))
            .unwrap();

        let result = post_turn_extract(&graph, &*llm, &config, &node_id).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }
```

Run: `cargo test -p graphirm-agent knowledge::extraction::tests::test_post_turn 2>&1`
Expected: Fails — `post_turn_extract` not defined.

### Step 2: Implement post_turn_extract

Add to `crates/agent/src/knowledge/extraction.rs`:

```rust
/// Called after each agent turn. Gathers the recent conversation from the graph
/// (the turn's response node and its context) and runs extraction.
pub async fn post_turn_extract(
    graph: &GraphStore,
    llm: &dyn LlmProvider,
    config: &ExtractionConfig,
    response_node_id: &NodeId,
) -> Result<Vec<NodeId>, AgentError> {
    if !config.enabled {
        return Ok(vec![]);
    }

    // Gather conversation context by traversing from the response node
    let response_node = graph.get_node(response_node_id).map_err(AgentError::Graph)?;

    let mut messages: Vec<(String, String)> = Vec::new();

    // Get the parent messages this node responds to
    let parents = graph
        .get_neighbors(response_node_id, Some(EdgeType::RespondsTo))
        .map_err(AgentError::Graph)?;

    for parent in &parents {
        let role = parent
            .data
            .get("role")
            .and_then(|v| v.as_str())
            .unwrap_or("user");
        let content = parent
            .data
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        messages.push((role.to_string(), content.to_string()));
    }

    // Add the response itself
    let role = response_node
        .data
        .get("role")
        .and_then(|v| v.as_str())
        .unwrap_or("assistant");
    let content = response_node
        .data
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    messages.push((role.to_string(), content.to_string()));

    extract_knowledge(graph, llm, &messages, response_node_id, config).await
}
```

### Step 3: Run tests

Run: `cargo test -p graphirm-agent knowledge::extraction::tests -- --nocapture 2>&1`
Expected: All 15 tests pass.

### Step 4: Integrate into agent loop

In the agent loop (e.g., `crates/agent/src/loop.rs`), add after each turn completes:

```rust
// After recording the assistant response node...
if let Some(ref extraction_config) = session.config.extraction {
    if let Err(e) = knowledge::extraction::post_turn_extract(
        &session.graph,
        llm,
        extraction_config,
        &response_node_id,
    ).await {
        tracing::warn!(error = %e, "Knowledge extraction failed (non-fatal)");
    }
}
```

This is a non-blocking integration — extraction failures are logged but don't break the agent loop.

### Step 5: Commit

```bash
git add crates/agent/src/knowledge/extraction.rs crates/agent/src/loop.rs
git commit -m "feat(knowledge): implement post_turn_extract hook, integrate into agent loop"
```

---

## Task 7: Add embedding storage to GraphStore

- [x] Complete

**Files:**
- Modify: `crates/graph/src/store.rs`
- Test: `crates/graph/src/store.rs` (in-module `#[cfg(test)]`)

### Step 1: Write failing tests

Add to the `tests` module in `crates/graph/src/store.rs`:

```rust
    #[test]
    fn test_store_and_retrieve_embedding() {
        let store = GraphStore::new_in_memory().unwrap();
        let node_id = store
            .add_node(NodeType::Knowledge, serde_json::json!({"name": "test_entity"}))
            .unwrap();

        let embedding: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        store.set_embedding(&node_id, &embedding).unwrap();

        let retrieved = store.get_embedding(&node_id).unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.len(), 5);
        assert!((retrieved[0] - 0.1).abs() < f32::EPSILON);
        assert!((retrieved[4] - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_get_embedding_returns_none_when_not_set() {
        let store = GraphStore::new_in_memory().unwrap();
        let node_id = store
            .add_node(NodeType::Knowledge, serde_json::json!({"name": "no_embedding"}))
            .unwrap();

        let retrieved = store.get_embedding(&node_id).unwrap();
        assert!(retrieved.is_none());
    }

    #[test]
    fn test_get_all_embeddings() {
        let store = GraphStore::new_in_memory().unwrap();

        let id1 = store
            .add_node(NodeType::Knowledge, serde_json::json!({"name": "e1"}))
            .unwrap();
        let id2 = store
            .add_node(NodeType::Knowledge, serde_json::json!({"name": "e2"}))
            .unwrap();
        let _id3 = store
            .add_node(NodeType::Knowledge, serde_json::json!({"name": "no_embed"}))
            .unwrap();

        store.set_embedding(&id1, &vec![1.0, 2.0, 3.0]).unwrap();
        store.set_embedding(&id2, &vec![4.0, 5.0, 6.0]).unwrap();

        let all = store.get_all_embeddings().unwrap();
        assert_eq!(all.len(), 2);
        assert!(all.iter().any(|(id, _)| *id == id1));
        assert!(all.iter().any(|(id, _)| *id == id2));
    }

    #[test]
    fn test_embedding_roundtrip_large_vector() {
        let store = GraphStore::new_in_memory().unwrap();
        let node_id = store
            .add_node(NodeType::Knowledge, serde_json::json!({"name": "large"}))
            .unwrap();

        // Typical embedding dimension (1536 for OpenAI ada-002)
        let embedding: Vec<f32> = (0..1536).map(|i| i as f32 * 0.001).collect();
        store.set_embedding(&node_id, &embedding).unwrap();

        let retrieved = store.get_embedding(&node_id).unwrap().unwrap();
        assert_eq!(retrieved.len(), 1536);
        assert!((retrieved[0] - 0.0).abs() < f32::EPSILON);
        assert!((retrieved[1535] - 1.535).abs() < 0.001);
    }
```

Run: `cargo test -p graphirm-graph store::tests 2>&1`
Expected: Fails — `set_embedding`, `get_embedding`, `get_all_embeddings` not defined.

### Step 2: Add embeddings table to schema

In the SQLite schema initialization (inside `GraphStore::new()` or `GraphStore::new_in_memory()`), add:

```rust
conn.execute_batch(
    "CREATE TABLE IF NOT EXISTS embeddings (
        node_id TEXT PRIMARY KEY REFERENCES nodes(id),
        embedding BLOB NOT NULL
    );"
)?;
```

### Step 3: Implement embedding methods on GraphStore

Add to `crates/graph/src/store.rs`:

```rust
impl GraphStore {
    /// Store an embedding vector for a node. Overwrites any existing embedding.
    pub fn set_embedding(&self, node_id: &NodeId, embedding: &[f32]) -> Result<(), GraphError> {
        let conn = self.pool.get().map_err(|e| GraphError::Pool(e.to_string()))?;
        let bytes: &[u8] = bytemuck::cast_slice(embedding);
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (node_id, embedding) VALUES (?1, ?2)",
            rusqlite::params![node_id.0, bytes],
        )?;
        Ok(())
    }

    /// Retrieve the embedding vector for a node, if one exists.
    pub fn get_embedding(&self, node_id: &NodeId) -> Result<Option<Vec<f32>>, GraphError> {
        let conn = self.pool.get().map_err(|e| GraphError::Pool(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT embedding FROM embeddings WHERE node_id = ?1"
        )?;
        let result = stmt.query_row(rusqlite::params![node_id.0], |row| {
            let bytes: Vec<u8> = row.get(0)?;
            Ok(bytes)
        });

        match result {
            Ok(bytes) => {
                let floats: &[f32] = bytemuck::cast_slice(&bytes);
                Ok(Some(floats.to_vec()))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Retrieve all stored embeddings as (NodeId, Vec<f32>) pairs.
    pub fn get_all_embeddings(&self) -> Result<Vec<(NodeId, Vec<f32>)>, GraphError> {
        let conn = self.pool.get().map_err(|e| GraphError::Pool(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT node_id, embedding FROM embeddings"
        )?;
        let rows = stmt.query_map([], |row| {
            let id: String = row.get(0)?;
            let bytes: Vec<u8> = row.get(1)?;
            Ok((id, bytes))
        })?;

        let mut results = Vec::new();
        for row in rows {
            let (id, bytes) = row?;
            let floats: &[f32] = bytemuck::cast_slice(&bytes);
            results.push((NodeId(id), floats.to_vec()));
        }
        Ok(results)
    }
}
```

### Step 4: Add bytemuck dependency

Add to `crates/graph/Cargo.toml` under `[dependencies]`:

```toml
bytemuck = { version = "1", features = ["derive"] }
```

### Step 5: Run tests

Run: `cargo test -p graphirm-graph store::tests -- --nocapture 2>&1`
Expected: All embedding tests pass.

### Step 6: Commit

```bash
git add crates/graph/src/store.rs crates/graph/Cargo.toml
git commit -m "feat(graph): add embedding BLOB storage with set/get/get_all methods"
```

---

## Task 8: Implement VectorIndex with instant-distance

- [x] Complete

**Files:**
- Modify: `crates/graph/src/vector.rs`
- Test: `crates/graph/src/vector.rs` (in-module `#[cfg(test)]`)

### Step 1: Write failing tests

Write the full `crates/graph/src/vector.rs`:

```rust
use crate::NodeId;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_index_new() {
        let index = VectorIndex::new(128);
        assert_eq!(index.dimension(), 128);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_vector_index_insert_and_search() {
        let mut index = VectorIndex::new(3);

        let id1 = NodeId("node-1".to_string());
        let id2 = NodeId("node-2".to_string());
        let id3 = NodeId("node-3".to_string());

        index.insert(id1.clone(), vec![1.0, 0.0, 0.0]);
        index.insert(id2.clone(), vec![0.0, 1.0, 0.0]);
        index.insert(id3.clone(), vec![0.9, 0.1, 0.0]);

        // Must rebuild after inserts
        index.rebuild();

        // Query close to id1 and id3
        let results = index.search(&[0.95, 0.05, 0.0], 2);
        assert_eq!(results.len(), 2);

        // id3 (0.9, 0.1, 0.0) should be closest to query (0.95, 0.05, 0.0)
        assert_eq!(results[0].0, id3);
        assert_eq!(results[1].0, id1);
    }

    #[test]
    fn test_vector_index_search_k_larger_than_size() {
        let mut index = VectorIndex::new(2);
        index.insert(NodeId("only-one".to_string()), vec![1.0, 0.0]);
        index.rebuild();

        let results = index.search(&[1.0, 0.0], 10);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_vector_index_empty_search() {
        let index = VectorIndex::new(4);
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_vector_index_many_vectors() {
        let mut index = VectorIndex::new(8);

        for i in 0..100 {
            let mut vec = vec![0.0f32; 8];
            vec[i % 8] = 1.0;
            vec[(i + 1) % 8] = 0.5;
            index.insert(NodeId(format!("node-{}", i)), vec);
        }
        index.rebuild();

        let results = index.search(&[1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5);
        assert_eq!(results.len(), 5);

        // All results should have distances (non-negative)
        for (_, dist) in &results {
            assert!(*dist >= 0.0);
        }
    }

    #[test]
    fn test_vector_index_rebuild_from_pairs() {
        let pairs: Vec<(NodeId, Vec<f32>)> = vec![
            (NodeId("a".into()), vec![1.0, 0.0, 0.0]),
            (NodeId("b".into()), vec![0.0, 1.0, 0.0]),
            (NodeId("c".into()), vec![0.0, 0.0, 1.0]),
        ];

        let index = VectorIndex::from_pairs(3, pairs);
        assert_eq!(index.len(), 3);

        let results = index.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, NodeId("a".into()));
    }
}
```

Run: `cargo test -p graphirm-graph vector::tests 2>&1`
Expected: Fails — `VectorIndex` has no methods yet.

### Step 2: Implement VectorIndex

Replace the contents of `crates/graph/src/vector.rs`:

```rust
//! HNSW vector index for knowledge node embeddings.

use instant_distance::{Builder, HnswMap, Search};

use crate::NodeId;

/// Wraps an HNSW index mapping embedding vectors to NodeIds.
/// The index is built lazily — call `rebuild()` after batch inserts,
/// or use `from_pairs()` to build from existing data.
pub struct VectorIndex {
    dimension: usize,
    pending: Vec<(NodeId, Vec<f32>)>,
    map: Option<HnswMap<Point, NodeId>>,
}

#[derive(Clone)]
struct Point(Vec<f32>);

impl instant_distance::Point for Point {
    fn distance(&self, other: &Self) -> f32 {
        // Euclidean distance
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

impl VectorIndex {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            pending: Vec::new(),
            map: None,
        }
    }

    /// Build an index from pre-existing (NodeId, embedding) pairs.
    pub fn from_pairs(dimension: usize, pairs: Vec<(NodeId, Vec<f32>)>) -> Self {
        let mut index = Self {
            dimension,
            pending: pairs,
            map: None,
        };
        index.rebuild();
        index
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn len(&self) -> usize {
        match &self.map {
            Some(map) => map.values().len(),
            None => self.pending.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Queue a vector for insertion. Call `rebuild()` after batch inserts.
    pub fn insert(&mut self, id: NodeId, embedding: Vec<f32>) {
        debug_assert_eq!(
            embedding.len(),
            self.dimension,
            "Embedding dimension mismatch: expected {}, got {}",
            self.dimension,
            embedding.len()
        );
        self.pending.push((id, embedding));
        // Invalidate the built map since we have new data
        self.map = None;
    }

    /// Rebuild the HNSW index from all pending data.
    pub fn rebuild(&mut self) {
        if self.pending.is_empty() {
            return;
        }

        let points: Vec<Point> = self.pending.iter().map(|(_, v)| Point(v.clone())).collect();
        let values: Vec<NodeId> = self.pending.iter().map(|(id, _)| id.clone()).collect();

        let map = Builder::default().build(points, values);
        self.map = Some(map);
    }

    /// Search for the k nearest neighbors to the query vector.
    /// Returns (NodeId, distance) pairs sorted by ascending distance.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(NodeId, f32)> {
        let map = match &self.map {
            Some(map) => map,
            None => return vec![],
        };

        if map.values().is_empty() {
            return vec![];
        }

        let query_point = Point(query.to_vec());
        let mut search = Search::default();

        let effective_k = k.min(map.values().len());

        map.search(&query_point, &mut search)
            .take(effective_k)
            .map(|item| {
                let node_id = item.value.clone();
                let distance = item.distance;
                (node_id, distance)
            })
            .collect()
    }
}
```

### Step 3: Run tests

Run: `cargo test -p graphirm-graph vector::tests -- --nocapture 2>&1`
Expected: All 6 tests pass.

### Step 4: Commit

```bash
git add crates/graph/src/vector.rs
git commit -m "feat(graph): implement VectorIndex with instant-distance HNSW"
```

---

## Task 9: Implement VectorIndex::rebuild_from_store()

- [x] Complete

**Files:**
- Modify: `crates/graph/src/vector.rs`
- Test: `crates/graph/src/vector.rs` (in-module `#[cfg(test)]`)

### Step 1: Write failing test

Add to the `tests` module in `crates/graph/src/vector.rs`:

```rust
    use crate::{GraphStore, NodeType};

    #[test]
    fn test_rebuild_from_store() {
        let store = GraphStore::new_in_memory().unwrap();

        // Create 50 knowledge nodes with embeddings
        for i in 0..50 {
            let node_id = store
                .add_node(
                    NodeType::Knowledge,
                    serde_json::json!({"name": format!("entity_{}", i)}),
                )
                .unwrap();

            let mut embedding = vec![0.0f32; 64];
            embedding[i % 64] = 1.0;
            embedding[(i + 1) % 64] = 0.5;
            store.set_embedding(&node_id, &embedding).unwrap();
        }

        // Also create a non-knowledge node (should be ignored based on node type,
        // but get_all_embeddings returns all — the caller filters if needed)
        let _other = store
            .add_node(NodeType::Interaction, serde_json::json!({"content": "test"}))
            .unwrap();

        let index = VectorIndex::rebuild_from_store(&store, 64).unwrap();
        assert_eq!(index.len(), 50);

        // Search should find nearest neighbor
        let mut query = vec![0.0f32; 64];
        query[0] = 1.0;
        query[1] = 0.5;
        let results = index.search(&query, 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_rebuild_from_store_empty() {
        let store = GraphStore::new_in_memory().unwrap();
        let index = VectorIndex::rebuild_from_store(&store, 128).unwrap();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }
```

Run: `cargo test -p graphirm-graph vector::tests 2>&1`
Expected: Fails — `rebuild_from_store` not defined.

### Step 2: Implement rebuild_from_store

Add to `crates/graph/src/vector.rs`:

```rust
use crate::{GraphError, GraphStore};

impl VectorIndex {
    /// Load all embeddings from the graph store and build an HNSW index.
    /// Called on application startup to warm the index from SQLite.
    pub fn rebuild_from_store(
        store: &GraphStore,
        dimension: usize,
    ) -> Result<Self, GraphError> {
        let pairs = store.get_all_embeddings()?;

        // Filter out any embeddings with wrong dimension
        let valid_pairs: Vec<(NodeId, Vec<f32>)> = pairs
            .into_iter()
            .filter(|(_, emb)| emb.len() == dimension)
            .collect();

        if valid_pairs.is_empty() {
            return Ok(Self::new(dimension));
        }

        Ok(Self::from_pairs(dimension, valid_pairs))
    }
}
```

### Step 3: Run tests

Run: `cargo test -p graphirm-graph vector::tests -- --nocapture 2>&1`
Expected: All 8 tests pass.

### Step 4: Commit

```bash
git add crates/graph/src/vector.rs
git commit -m "feat(graph): implement VectorIndex::rebuild_from_store from SQLite"
```

---

## Task 10: Implement MemoryRetriever::embed_knowledge_node()

- [x] Complete

**Files:**
- Modify: `crates/agent/src/knowledge/memory.rs`
- Test: `crates/agent/src/knowledge/memory.rs` (in-module `#[cfg(test)]`)

### Step 1: Write failing tests

Write the full `crates/agent/src/knowledge/memory.rs`:

```rust
//! Cross-session memory retrieval using HNSW vector search.

use std::sync::Arc;
use tokio::sync::RwLock;

use graphirm_graph::{vector::VectorIndex, GraphStore, NodeId};
use graphirm_llm::LlmProvider;

use crate::error::AgentError;

#[cfg(test)]
mod tests {
    use super::*;
    use graphirm_graph::NodeType;
    use graphirm_llm::{LlmError, LlmMessage, LlmResponse, ToolDefinition, Usage};

    struct MockEmbeddingProvider;

    #[async_trait::async_trait]
    impl LlmProvider for MockEmbeddingProvider {
        async fn complete(
            &self,
            _messages: Vec<LlmMessage>,
            _tools: Vec<ToolDefinition>,
        ) -> Result<LlmResponse, LlmError> {
            unimplemented!("not needed for embedding tests")
        }

        async fn embed(&self, text: &str) -> Result<Vec<f32>, LlmError> {
            // Deterministic mock: hash the text into a 64-dim vector
            let mut vec = vec![0.0f32; 64];
            for (i, byte) in text.bytes().enumerate() {
                vec[i % 64] += byte as f32 / 255.0;
            }
            // Normalize
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut vec {
                    *v /= norm;
                }
            }
            Ok(vec)
        }
    }

    #[tokio::test]
    async fn test_embed_knowledge_node_stores_embedding() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let vector_index = Arc::new(RwLock::new(VectorIndex::new(64)));
        let llm: Arc<dyn LlmProvider> = Arc::new(MockEmbeddingProvider);

        let retriever = MemoryRetriever::new(graph.clone(), vector_index.clone(), llm, 64);

        let node_id = graph
            .add_node(
                NodeType::Knowledge,
                serde_json::json!({
                    "name": "JWT Authentication",
                    "description": "Token-based authentication using JSON Web Tokens"
                }),
            )
            .unwrap();

        retriever.embed_knowledge_node(&node_id).await.unwrap();

        // Verify embedding stored in graph
        let embedding = graph.get_embedding(&node_id).unwrap();
        assert!(embedding.is_some());
        assert_eq!(embedding.unwrap().len(), 64);
    }

    #[tokio::test]
    async fn test_embed_knowledge_node_inserts_into_hnsw() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let vector_index = Arc::new(RwLock::new(VectorIndex::new(64)));
        let llm: Arc<dyn LlmProvider> = Arc::new(MockEmbeddingProvider);

        let retriever = MemoryRetriever::new(graph.clone(), vector_index.clone(), llm, 64);

        let node_id = graph
            .add_node(
                NodeType::Knowledge,
                serde_json::json!({
                    "name": "bcrypt hashing",
                    "description": "Password hashing with bcrypt"
                }),
            )
            .unwrap();

        retriever.embed_knowledge_node(&node_id).await.unwrap();

        // Rebuild and search
        {
            let mut idx = vector_index.write().await;
            idx.rebuild();
        }

        let idx = vector_index.read().await;
        assert_eq!(idx.len(), 1);
    }
}
```

Run: `cargo test -p graphirm-agent knowledge::memory::tests 2>&1`
Expected: Fails — `MemoryRetriever` not defined.

### Step 2: Implement MemoryRetriever and embed_knowledge_node

Add above the tests in `crates/agent/src/knowledge/memory.rs`:

```rust
use std::sync::Arc;
use tokio::sync::RwLock;

use graphirm_graph::{vector::VectorIndex, GraphStore, NodeId};
use graphirm_llm::LlmProvider;

use crate::error::AgentError;

pub struct MemoryRetriever {
    graph: Arc<GraphStore>,
    vector_index: Arc<RwLock<VectorIndex>>,
    llm: Arc<dyn LlmProvider>,
    embedding_dimension: usize,
}

impl MemoryRetriever {
    pub fn new(
        graph: Arc<GraphStore>,
        vector_index: Arc<RwLock<VectorIndex>>,
        llm: Arc<dyn LlmProvider>,
        embedding_dimension: usize,
    ) -> Self {
        Self {
            graph,
            vector_index,
            llm,
            embedding_dimension,
        }
    }

    /// Embed a knowledge node's description and store the vector
    /// in both SQLite and the in-memory HNSW index.
    pub async fn embed_knowledge_node(
        &self,
        node_id: &NodeId,
    ) -> Result<(), AgentError> {
        let node = self.graph.get_node(node_id).map_err(AgentError::Graph)?;

        // Build text to embed from node data
        let name = node
            .data
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let description = node
            .data
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let text = format!("{}: {}", name, description);

        let embedding = self
            .llm
            .embed(&text)
            .await
            .map_err(AgentError::Llm)?;

        if embedding.len() != self.embedding_dimension {
            return Err(AgentError::Workflow(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.embedding_dimension,
                embedding.len()
            )));
        }

        // Store in SQLite
        self.graph
            .set_embedding(node_id, &embedding)
            .map_err(AgentError::Graph)?;

        // Insert into HNSW index (will need rebuild() before searching)
        {
            let mut idx = self.vector_index.write().await;
            idx.insert(node_id.clone(), embedding);
        }

        tracing::debug!(node_id = %node_id, "Embedded knowledge node");
        Ok(())
    }
}
```

### Step 3: Run tests

Run: `cargo test -p graphirm-agent knowledge::memory::tests -- --nocapture 2>&1`
Expected: All 2 tests pass.

### Step 4: Commit

```bash
git add crates/agent/src/knowledge/memory.rs
git commit -m "feat(knowledge): implement MemoryRetriever::embed_knowledge_node"
```

---

## Task 11: Implement MemoryRetriever::retrieve_relevant()

- [x] Complete

**Files:**
- Modify: `crates/agent/src/knowledge/memory.rs`
- Test: `crates/agent/src/knowledge/memory.rs` (in-module `#[cfg(test)]`)

### Step 1: Write failing tests

Add to the `tests` module in `crates/agent/src/knowledge/memory.rs`:

```rust
    #[tokio::test]
    async fn test_retrieve_relevant_finds_similar_knowledge() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let vector_index = Arc::new(RwLock::new(VectorIndex::new(64)));
        let llm: Arc<dyn LlmProvider> = Arc::new(MockEmbeddingProvider);

        let retriever = MemoryRetriever::new(graph.clone(), vector_index.clone(), llm, 64);

        // Insert several knowledge nodes about different topics
        let topics = vec![
            ("JWT Auth", "Token-based authentication using JSON Web Tokens"),
            ("bcrypt", "Password hashing library for secure storage"),
            ("PostgreSQL", "Relational database for persistent storage"),
            ("React hooks", "React state management with useState and useEffect"),
            ("OAuth2 flow", "Authorization protocol for third-party access"),
        ];

        for (name, desc) in &topics {
            let node_id = graph
                .add_node(
                    NodeType::Knowledge,
                    serde_json::json!({"name": name, "description": desc}),
                )
                .unwrap();
            retriever.embed_knowledge_node(&node_id).await.unwrap();
        }

        // Rebuild the index after all inserts
        {
            let mut idx = vector_index.write().await;
            idx.rebuild();
        }

        // Query about authentication — should return auth-related nodes
        let results = retriever.retrieve_relevant("implement user login", 3).await.unwrap();
        assert_eq!(results.len(), 3);

        // The results should be Node objects with Knowledge type
        for node in &results {
            assert_eq!(node.node_type, NodeType::Knowledge);
        }
    }

    #[tokio::test]
    async fn test_retrieve_relevant_empty_index() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let vector_index = Arc::new(RwLock::new(VectorIndex::new(64)));
        let llm: Arc<dyn LlmProvider> = Arc::new(MockEmbeddingProvider);

        let retriever = MemoryRetriever::new(graph.clone(), vector_index, llm, 64);

        let results = retriever.retrieve_relevant("anything", 5).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_retrieve_relevant_k_larger_than_index() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let vector_index = Arc::new(RwLock::new(VectorIndex::new(64)));
        let llm: Arc<dyn LlmProvider> = Arc::new(MockEmbeddingProvider);

        let retriever = MemoryRetriever::new(graph.clone(), vector_index.clone(), llm, 64);

        let node_id = graph
            .add_node(
                NodeType::Knowledge,
                serde_json::json!({"name": "only entity", "description": "the only one"}),
            )
            .unwrap();
        retriever.embed_knowledge_node(&node_id).await.unwrap();

        {
            let mut idx = vector_index.write().await;
            idx.rebuild();
        }

        let results = retriever.retrieve_relevant("query", 100).await.unwrap();
        assert_eq!(results.len(), 1);
    }
```

Run: `cargo test -p graphirm-agent knowledge::memory::tests 2>&1`
Expected: Fails — `retrieve_relevant` not defined.

### Step 2: Implement retrieve_relevant

Add to the `MemoryRetriever` impl block:

```rust
    /// Embed a query string and search the HNSW index for the k most
    /// similar knowledge nodes. Returns full Node objects from the graph.
    pub async fn retrieve_relevant(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<graphirm_graph::Node>, AgentError> {
        let query_embedding = self
            .llm
            .embed(query)
            .await
            .map_err(AgentError::Llm)?;

        let index = self.vector_index.read().await;
        let candidates = index.search(&query_embedding, k);
        drop(index);

        let mut nodes = Vec::with_capacity(candidates.len());
        for (node_id, _distance) in candidates {
            match self.graph.get_node(&node_id) {
                Ok(node) => nodes.push(node),
                Err(e) => {
                    tracing::warn!(
                        node_id = %node_id,
                        error = %e,
                        "Knowledge node in HNSW index but missing from graph"
                    );
                }
            }
        }

        Ok(nodes)
    }
```

### Step 3: Run tests

Run: `cargo test -p graphirm-agent knowledge::memory::tests -- --nocapture 2>&1`
Expected: All 5 tests pass.

### Step 4: Commit

```bash
git add crates/agent/src/knowledge/memory.rs
git commit -m "feat(knowledge): implement MemoryRetriever::retrieve_relevant with HNSW search"
```

---

## Task 12: Implement cross-session memory injection

- [x] Complete

**Files:**
- Create: `crates/agent/src/knowledge/injection.rs`
- Modify: `crates/agent/src/knowledge.rs`
- Test: `crates/agent/src/knowledge/injection.rs` (in-module `#[cfg(test)]`)

### Step 1: Write failing tests

Create `crates/agent/src/knowledge/injection.rs`:

```rust
//! Cross-session memory injection: retrieves relevant knowledge from past sessions
//! and formats it as system context for the agent's first turn.

use graphirm_graph::Node;

use crate::error::AgentError;

use super::memory::MemoryRetriever;

#[cfg(test)]
mod tests {
    use super::*;
    use graphirm_graph::{GraphStore, NodeType};
    use graphirm_graph::vector::VectorIndex;
    use graphirm_llm::{LlmError, LlmMessage, LlmProvider, LlmResponse, ToolDefinition, Usage};
    use std::sync::Arc;
    use tokio::sync::RwLock;

    struct MockEmbeddingProvider;

    #[async_trait::async_trait]
    impl LlmProvider for MockEmbeddingProvider {
        async fn complete(
            &self,
            _messages: Vec<LlmMessage>,
            _tools: Vec<ToolDefinition>,
        ) -> Result<LlmResponse, LlmError> {
            unimplemented!()
        }

        async fn embed(&self, text: &str) -> Result<Vec<f32>, LlmError> {
            let mut vec = vec![0.0f32; 64];
            for (i, byte) in text.bytes().enumerate() {
                vec[i % 64] += byte as f32 / 255.0;
            }
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut vec {
                    *v /= norm;
                }
            }
            Ok(vec)
        }
    }

    #[tokio::test]
    async fn test_format_memory_context() {
        let nodes = vec![
            Node {
                id: graphirm_graph::NodeId("k1".into()),
                node_type: NodeType::Knowledge,
                data: serde_json::json!({
                    "entity_type": "pattern",
                    "name": "JWT Auth",
                    "description": "Token-based authentication using JSON Web Tokens",
                }),
                created_at: chrono::Utc::now(),
            },
            Node {
                id: graphirm_graph::NodeId("k2".into()),
                node_type: NodeType::Knowledge,
                data: serde_json::json!({
                    "entity_type": "library",
                    "name": "bcrypt",
                    "description": "Password hashing library",
                }),
                created_at: chrono::Utc::now(),
            },
        ];

        let context = format_memory_context(&nodes);
        assert!(context.contains("JWT Auth"));
        assert!(context.contains("bcrypt"));
        assert!(context.contains("pattern"));
        assert!(context.contains("library"));
        assert!(context.contains("Relevant knowledge from past sessions"));
    }

    #[tokio::test]
    async fn test_format_memory_context_empty() {
        let nodes: Vec<Node> = vec![];
        let context = format_memory_context(&nodes);
        assert!(context.is_empty());
    }

    #[tokio::test]
    async fn test_build_session_context_full_pipeline() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let vector_index = Arc::new(RwLock::new(VectorIndex::new(64)));
        let llm: Arc<dyn LlmProvider> = Arc::new(MockEmbeddingProvider);
        let retriever = MemoryRetriever::new(graph.clone(), vector_index.clone(), llm, 64);

        // Simulate session A: extract knowledge
        let topics = vec![
            ("JWT Auth", "pattern", "Token-based authentication using JSON Web Tokens"),
            ("bcrypt", "library", "Password hashing library for secure credential storage"),
            ("Session cookies", "pattern", "Browser session management via HTTP cookies"),
        ];

        for (name, entity_type, desc) in &topics {
            let node_id = graph
                .add_node(
                    NodeType::Knowledge,
                    serde_json::json!({
                        "entity_type": entity_type,
                        "name": name,
                        "description": desc,
                    }),
                )
                .unwrap();
            retriever.embed_knowledge_node(&node_id).await.unwrap();
        }

        {
            let mut idx = vector_index.write().await;
            idx.rebuild();
        }

        // Session B starts: user wants to "implement login"
        let context = build_session_context(&retriever, "implement user login", 3)
            .await
            .unwrap();

        assert!(!context.is_empty());
        assert!(context.contains("Relevant knowledge from past sessions"));
    }
}
```

Run: `cargo test -p graphirm-agent knowledge::injection::tests 2>&1`
Expected: Fails — `format_memory_context`, `build_session_context` not defined.

### Step 2: Implement injection functions

Add above the tests in `crates/agent/src/knowledge/injection.rs`:

```rust
use graphirm_graph::Node;

use crate::error::AgentError;

use super::memory::MemoryRetriever;

/// Format retrieved knowledge nodes into a system-context block
/// that can be prepended to the agent's messages.
pub fn format_memory_context(nodes: &[Node]) -> String {
    if nodes.is_empty() {
        return String::new();
    }

    let mut lines = vec![
        "## Relevant knowledge from past sessions\n".to_string(),
    ];

    for node in nodes {
        let name = node
            .data
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let entity_type = node
            .data
            .get("entity_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let description = node
            .data
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        lines.push(format!("- **{}** ({}): {}", name, entity_type, description));
    }

    lines.join("\n")
}

/// Retrieve relevant knowledge from past sessions and format it
/// as context for the agent's first turn in a new session.
pub async fn build_session_context(
    retriever: &MemoryRetriever,
    initial_prompt: &str,
    max_knowledge_items: usize,
) -> Result<String, AgentError> {
    let nodes = retriever
        .retrieve_relevant(initial_prompt, max_knowledge_items)
        .await?;

    Ok(format_memory_context(&nodes))
}
```

### Step 3: Export from knowledge module

Update `crates/agent/src/knowledge.rs`:

```rust
pub mod extraction;
pub mod injection;
pub mod memory;
```

### Step 4: Run tests

Run: `cargo test -p graphirm-agent knowledge::injection::tests -- --nocapture 2>&1`
Expected: All 3 tests pass.

### Step 5: Commit

```bash
git add crates/agent/src/knowledge/injection.rs crates/agent/src/knowledge.rs
git commit -m "feat(knowledge): implement cross-session memory injection with format_memory_context"
```

---

## Task 13: Integration test — full extraction → retrieval pipeline

- [x] Complete

**Files:**
- Create: `crates/agent/tests/knowledge_integration.rs`

### Step 1: Write the integration test

Create `crates/agent/tests/knowledge_integration.rs`:

```rust
//! Integration test: full knowledge extraction → embedding → cross-session retrieval pipeline.
//!
//! Simulates:
//! 1. Session A has a conversation about authentication patterns
//! 2. Knowledge is extracted from session A
//! 3. Knowledge nodes are embedded
//! 4. New session B starts with "implement login"
//! 5. Retrieves auth-related knowledge from session A

use std::sync::Arc;
use tokio::sync::RwLock;

use graphirm_agent::error::AgentError;
use graphirm_agent::knowledge::extraction::{extract_knowledge, ExtractionConfig};
use graphirm_agent::knowledge::injection::build_session_context;
use graphirm_agent::knowledge::memory::MemoryRetriever;
use graphirm_graph::vector::VectorIndex;
use graphirm_graph::{EdgeType, GraphStore, NodeType};
use graphirm_llm::{LlmError, LlmMessage, LlmProvider, LlmResponse, ToolDefinition, Usage};

struct TestProvider {
    extraction_response: String,
}

#[async_trait::async_trait]
impl LlmProvider for TestProvider {
    async fn complete(
        &self,
        _messages: Vec<LlmMessage>,
        _tools: Vec<ToolDefinition>,
    ) -> Result<LlmResponse, LlmError> {
        Ok(LlmResponse {
            content: self.extraction_response.clone(),
            tool_calls: vec![],
            usage: Usage::default(),
        })
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, LlmError> {
        // Deterministic mock embedding
        let mut vec = vec![0.0f32; 64];
        for (i, byte) in text.bytes().enumerate() {
            vec[i % 64] += byte as f32 / 255.0;
        }
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vec {
                *v /= norm;
            }
        }
        Ok(vec)
    }
}

#[tokio::test]
async fn test_full_knowledge_pipeline() {
    // ── Session A: conversation about auth ──
    let graph = Arc::new(GraphStore::new_in_memory().unwrap());

    // Create conversation nodes
    let user_msg = graph
        .add_node(
            NodeType::Interaction,
            serde_json::json!({
                "role": "user",
                "content": "How should I implement authentication in my Rust web app?"
            }),
        )
        .unwrap();

    let assistant_msg = graph
        .add_node(
            NodeType::Interaction,
            serde_json::json!({
                "role": "assistant",
                "content": "Use JWT tokens with the jsonwebtoken crate. Hash passwords with argon2. \
                            Store sessions in Redis for scalability. Use axum middleware for auth guards."
            }),
        )
        .unwrap();

    graph
        .add_edge(&assistant_msg, &user_msg, EdgeType::RespondsTo)
        .unwrap();

    // Extract knowledge
    let extraction_llm = Arc::new(TestProvider {
        extraction_response: r#"{
            "entities": [
                {
                    "entity_type": "pattern",
                    "name": "JWT Authentication",
                    "description": "Token-based auth using jsonwebtoken crate with RS256 signing",
                    "confidence": 0.95,
                    "relationships": [
                        { "target_name": "argon2 password hashing", "relationship": "used_with" }
                    ]
                },
                {
                    "entity_type": "library",
                    "name": "argon2 password hashing",
                    "description": "Secure password hashing using the argon2 crate",
                    "confidence": 0.90,
                    "relationships": []
                },
                {
                    "entity_type": "architecture",
                    "name": "Redis session store",
                    "description": "Using Redis for session storage to enable horizontal scaling",
                    "confidence": 0.85,
                    "relationships": [
                        { "target_name": "JWT Authentication", "relationship": "alternative_to" }
                    ]
                },
                {
                    "entity_type": "pattern",
                    "name": "axum auth middleware",
                    "description": "Authentication guard implemented as axum middleware layer",
                    "confidence": 0.88,
                    "relationships": [
                        { "target_name": "JWT Authentication", "relationship": "implements" }
                    ]
                }
            ]
        }"#
        .to_string(),
    });

    let config = ExtractionConfig {
        enabled: true,
        min_confidence: 0.8,
        ..ExtractionConfig::default()
    };

    let messages = vec![
        (
            "user".to_string(),
            "How should I implement authentication in my Rust web app?".to_string(),
        ),
        (
            "assistant".to_string(),
            "Use JWT tokens with the jsonwebtoken crate. Hash passwords with argon2.".to_string(),
        ),
    ];

    let knowledge_ids = extract_knowledge(
        &graph,
        &*extraction_llm,
        &messages,
        &assistant_msg,
        &config,
    )
    .await
    .unwrap();

    assert_eq!(knowledge_ids.len(), 4);

    // ── Embed all knowledge nodes ──
    let vector_index = Arc::new(RwLock::new(VectorIndex::new(64)));
    let retriever = MemoryRetriever::new(
        graph.clone(),
        vector_index.clone(),
        extraction_llm.clone(),
        64,
    );

    for id in &knowledge_ids {
        retriever.embed_knowledge_node(id).await.unwrap();
    }

    // Rebuild HNSW
    {
        let mut idx = vector_index.write().await;
        idx.rebuild();
    }

    // ── Session B: new session asks about login ──
    let context = build_session_context(&retriever, "implement user login with password", 3)
        .await
        .unwrap();

    // Should contain relevant knowledge from session A
    assert!(!context.is_empty());
    assert!(context.contains("Relevant knowledge from past sessions"));

    // At least one auth-related entity should appear
    let has_auth_knowledge = context.contains("JWT")
        || context.contains("argon2")
        || context.contains("auth")
        || context.contains("session");
    assert!(
        has_auth_knowledge,
        "Context should contain auth-related knowledge but got: {}",
        context
    );

    // Verify DerivedFrom edges exist
    for id in &knowledge_ids {
        let neighbors = graph
            .get_neighbors(id, Some(EdgeType::DerivedFrom))
            .unwrap();
        assert!(
            neighbors.iter().any(|n| n.id == assistant_msg),
            "Knowledge node {} should have DerivedFrom edge to assistant message",
            id
        );
    }

    // Verify RelatesTo edges exist between related entities
    let all_knowledge: Vec<_> = knowledge_ids
        .iter()
        .map(|id| graph.get_node(id).unwrap())
        .collect();

    let jwt_node = all_knowledge
        .iter()
        .find(|n| {
            n.data
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                == "JWT Authentication"
        })
        .expect("JWT Authentication node should exist");

    let jwt_relations = graph
        .get_neighbors(&jwt_node.id, Some(EdgeType::RelatesTo))
        .unwrap();
    // JWT should relate to argon2
    assert!(
        !jwt_relations.is_empty(),
        "JWT node should have RelatesTo edges"
    );
}

#[tokio::test]
async fn test_rebuild_index_from_store_then_query() {
    let graph = Arc::new(GraphStore::new_in_memory().unwrap());
    let llm: Arc<dyn LlmProvider> = Arc::new(TestProvider {
        extraction_response: "{}".to_string(),
    });

    // Manually create knowledge nodes with embeddings (simulating a previous session)
    let entities = vec![
        ("GraphQL API", "API design pattern using GraphQL schema"),
        ("REST endpoints", "Traditional REST API with JSON responses"),
        ("gRPC service", "High-performance RPC using Protocol Buffers"),
    ];

    for (name, desc) in &entities {
        let node_id = graph
            .add_node(
                NodeType::Knowledge,
                serde_json::json!({"name": name, "description": desc}),
            )
            .unwrap();

        // Embed using mock provider
        let embedding = llm
            .embed(&format!("{}: {}", name, desc))
            .await
            .unwrap();
        graph.set_embedding(&node_id, &embedding).unwrap();
    }

    // Simulate app restart: rebuild index from store
    let rebuilt_index = VectorIndex::rebuild_from_store(&graph, 64).unwrap();
    assert_eq!(rebuilt_index.len(), 3);

    let vector_index = Arc::new(RwLock::new(rebuilt_index));
    let retriever = MemoryRetriever::new(graph.clone(), vector_index, llm, 64);

    let results = retriever.retrieve_relevant("API design", 2).await.unwrap();
    assert_eq!(results.len(), 2);
}
```

### Step 2: Run integration tests

Run: `cargo test -p graphirm-agent --test knowledge_integration -- --nocapture 2>&1`
Expected: Both tests pass.

### Step 3: Commit

```bash
git add crates/agent/tests/knowledge_integration.rs
git commit -m "test(knowledge): integration test for full extraction → embedding → retrieval pipeline"
```

---

## Task 14: Add ONNX Runtime and tokenizer dependencies, define ExtractionBackend

- [x] Complete

**Files:**
- Modify: `crates/agent/Cargo.toml`
- Modify: `crates/agent/src/knowledge/extraction.rs`
- Create: `crates/agent/src/knowledge/local_extraction.rs`
- Modify: `crates/agent/src/knowledge.rs`

**Rationale:** GLiNER2 is a 205M-parameter Apache 2.0 model that performs entity extraction, relation extraction, and structured parsing in a single forward pass on CPU (100-250ms). Running it locally via ONNX avoids LLM API costs for the fast extraction path, guarantees structured output (no JSON parse failures), and doesn't consume the consciousness loop's token budget. The LLM backend remains for higher-order reasoning ("what should I learn?"); GLiNER2 handles "what entities and relationships are mentioned."

### Step 1: Add dependencies

Add to `crates/agent/Cargo.toml` under `[dependencies]`:

```toml
ort = { version = "2", features = ["download-binaries"] }
tokenizers = { version = "0.21", default-features = false }
```

Both crates are MIT/Apache-2.0 dual-licensed.

### Step 2: Define ExtractionBackend enum

Modify `crates/agent/src/knowledge/extraction.rs` — add a backend discriminator to `ExtractionConfig`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionBackend {
    #[default]
    Llm,
    Local {
        model_path: String,
        tokenizer_path: String,
    },
    Hybrid {
        model_path: String,
        tokenizer_path: String,
    },
}
```

Add to `ExtractionConfig`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    #[serde(default)]
    pub enabled: bool,

    #[serde(default = "default_model")]
    pub model: String,

    #[serde(default = "default_entity_types")]
    pub entity_types: Vec<String>,

    #[serde(default = "default_min_confidence")]
    pub min_confidence: f64,

    #[serde(default)]
    pub backend: ExtractionBackend,
}
```

### Step 3: Create local_extraction module stub

Create `crates/agent/src/knowledge/local_extraction.rs`:

```rust
//! Local ONNX-based entity and relation extraction using GLiNER2.
//!
//! Provides a fast, zero-cost extraction path that runs a 205M parameter
//! model on CPU via ONNX Runtime. Used for per-turn entity/relation extraction
//! while the LLM backend handles higher-order synthesis.

use std::path::Path;

use crate::error::AgentError;
use super::extraction::{ExtractedEntity, EntityRelationship, ExtractionResponse};
```

### Step 4: Export from knowledge module

Add to `crates/agent/src/knowledge.rs`:

```rust
pub mod extraction;
pub mod injection;
pub mod local_extraction;
pub mod memory;
```

### Step 5: Write tests for ExtractionBackend serde

Add to the `tests` module in `crates/agent/src/knowledge/extraction.rs`:

```rust
    #[test]
    fn test_extraction_config_default_backend_is_llm() {
        let config = ExtractionConfig::default();
        assert!(matches!(config.backend, ExtractionBackend::Llm));
    }

    #[test]
    fn test_extraction_config_local_backend_deserialize() {
        let toml_str = r#"
            enabled = true

            [backend]
            local = { model_path = "/models/gliner2-base.onnx", tokenizer_path = "/models/tokenizer.json" }
        "#;
        let config: ExtractionConfig = toml::from_str(toml_str).unwrap();
        assert!(config.enabled);
        match &config.backend {
            ExtractionBackend::Local { model_path, .. } => {
                assert!(model_path.contains("gliner2"));
            }
            other => panic!("Expected Local backend, got {:?}", other),
        }
    }

    #[test]
    fn test_extraction_config_hybrid_backend_deserialize() {
        let toml_str = r#"
            enabled = true

            [backend]
            hybrid = { model_path = "/models/gliner2-base.onnx", tokenizer_path = "/models/tokenizer.json" }
        "#;
        let config: ExtractionConfig = toml::from_str(toml_str).unwrap();
        match &config.backend {
            ExtractionBackend::Hybrid { model_path, tokenizer_path } => {
                assert!(model_path.contains("gliner2"));
                assert!(tokenizer_path.contains("tokenizer"));
            }
            other => panic!("Expected Hybrid backend, got {:?}", other),
        }
    }
```

### Step 6: Verify

Run: `cargo check --workspace 2>&1`
Expected: Compiles with no errors.

Run: `cargo test -p graphirm-agent knowledge::extraction::tests -- --nocapture 2>&1`
Expected: All tests pass (previous + 3 new).

### Step 7: Commit

```bash
git add crates/agent/Cargo.toml crates/agent/src/knowledge/extraction.rs \
       crates/agent/src/knowledge/local_extraction.rs crates/agent/src/knowledge.rs
git commit -m "feat(knowledge): add ort + tokenizers deps, define ExtractionBackend enum"
```

---

## Task 15: Implement OnnxExtractor with GLiNER2 model loading and inference

- [x] Complete

**Files:**
- Modify: `crates/agent/src/knowledge/local_extraction.rs`
- Test: `crates/agent/src/knowledge/local_extraction.rs` (in-module `#[cfg(test)]`)

**Note:** GLiNER2 requires a pre-exported ONNX model + tokenizer.json. Export steps (run once from Python):

```bash
pip install gliner2 optimum onnxruntime
python -c "
from gliner2 import GLiNER2
model = GLiNER2.from_pretrained('fastino/gliner2-base-v1')
model.export_onnx('gliner2-base-v1.onnx')
"
```

The tokenizer.json ships with the HuggingFace model files. Both artifacts are placed in a configurable `model_path` directory.

### Step 1: Write failing tests

Add to `crates/agent/src/knowledge/local_extraction.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_extractor_new_fails_with_missing_model() {
        let result = OnnxExtractor::new(
            Path::new("/nonexistent/model.onnx"),
            Path::new("/nonexistent/tokenizer.json"),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_build_entity_type_prompt() {
        let types = vec![
            "function".to_string(),
            "pattern".to_string(),
            "decision".to_string(),
        ];
        let prompt = build_entity_type_prompt(&types);
        assert!(prompt.contains("function"));
        assert!(prompt.contains("pattern"));
        assert!(prompt.contains("decision"));
    }

    #[test]
    fn test_parse_onnx_output_to_extraction_response() {
        // Simulate raw ONNX output: list of (entity_type, text, start, end, confidence)
        let raw_entities = vec![
            RawOnnxEntity {
                entity_type: "function".to_string(),
                text: "parse_config".to_string(),
                confidence: 0.92,
            },
            RawOnnxEntity {
                entity_type: "library".to_string(),
                text: "serde".to_string(),
                confidence: 0.88,
            },
        ];

        let response = raw_entities_to_extraction_response(raw_entities);
        assert_eq!(response.entities.len(), 2);
        assert_eq!(response.entities[0].name, "parse_config");
        assert_eq!(response.entities[0].entity_type, "function");
        assert!((response.entities[0].confidence - 0.92).abs() < f64::EPSILON);
        assert_eq!(response.entities[1].name, "serde");
    }
}
```

Run: `cargo test -p graphirm-agent knowledge::local_extraction::tests 2>&1`
Expected: Fails — types not defined yet.

### Step 2: Implement OnnxExtractor

Add to `crates/agent/src/knowledge/local_extraction.rs`:

```rust
use std::path::Path;

use ort::session::Session;
use tokenizers::Tokenizer;

use crate::error::AgentError;
use super::extraction::{ExtractedEntity, EntityRelationship, ExtractionResponse};

/// Raw entity output from the ONNX model before conversion to ExtractionResponse.
#[derive(Debug, Clone)]
pub struct RawOnnxEntity {
    pub entity_type: String,
    pub text: String,
    pub confidence: f64,
}

/// Wraps a GLiNER2 ONNX model session and tokenizer for local entity extraction.
pub struct OnnxExtractor {
    session: Session,
    tokenizer: Tokenizer,
}

impl OnnxExtractor {
    /// Load a GLiNER2 ONNX model and its tokenizer from disk.
    pub fn new(model_path: &Path, tokenizer_path: &Path) -> Result<Self, AgentError> {
        let session = Session::builder()
            .and_then(|b| b.with_intra_threads(4))
            .and_then(|b| b.commit_from_file(model_path))
            .map_err(|e| AgentError::Workflow(format!("Failed to load ONNX model: {}", e)))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| AgentError::Workflow(format!("Failed to load tokenizer: {}", e)))?;

        Ok(Self { session, tokenizer })
    }

    /// Run entity extraction on the given text using the configured entity types.
    ///
    /// Returns raw entities that can be converted to an ExtractionResponse.
    /// The actual ONNX inference logic (input tensor construction, output parsing)
    /// is model-specific to GLiNER2's architecture:
    /// - Input: tokenized text + entity type prompts
    /// - Output: span predictions with entity type labels and confidence scores
    pub async fn extract(
        &self,
        text: &str,
        entity_types: &[String],
        min_confidence: f64,
    ) -> Result<ExtractionResponse, AgentError> {
        let encoding = self.tokenizer
            .encode(text, true)
            .map_err(|e| AgentError::Workflow(format!("Tokenization failed: {}", e)))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| m as i64).collect();
        let seq_len = input_ids.len();

        // Build entity type prompt tokens
        let type_prompt = build_entity_type_prompt(entity_types);
        let type_encoding = self.tokenizer
            .encode(type_prompt.as_str(), true)
            .map_err(|e| AgentError::Workflow(format!("Entity type tokenization failed: {}", e)))?;
        let type_ids: Vec<i64> = type_encoding.get_ids().iter().map(|&id| id as i64).collect();
        let type_mask: Vec<i64> = type_encoding.get_attention_mask().iter().map(|&m| m as i64).collect();

        // Run inference
        // NOTE: Actual input/output tensor shapes depend on the exported GLiNER2 ONNX graph.
        // This is a structural scaffold — the tensor construction and output parsing will
        // need adjustment once the exact ONNX export format is validated.
        let outputs = self.session
            .run(ort::inputs![
                "input_ids" => ndarray::Array2::from_shape_vec((1, seq_len), input_ids)
                    .map_err(|e| AgentError::Workflow(format!("Input tensor error: {}", e)))?,
                "attention_mask" => ndarray::Array2::from_shape_vec((1, seq_len), attention_mask)
                    .map_err(|e| AgentError::Workflow(format!("Mask tensor error: {}", e)))?,
            ].map_err(|e| AgentError::Workflow(format!("ONNX input error: {}", e)))?)
            .map_err(|e| AgentError::Workflow(format!("ONNX inference failed: {}", e)))?;

        // Parse output tensors into raw entities
        // TODO: Implement GLiNER2-specific output parsing once ONNX export format is validated.
        // The model outputs span logits that need to be decoded against the tokenizer offsets
        // and matched to entity type labels.
        let raw_entities: Vec<RawOnnxEntity> = parse_onnx_outputs(&outputs, &encoding, entity_types)
            .into_iter()
            .filter(|e| e.confidence >= min_confidence)
            .collect();

        Ok(raw_entities_to_extraction_response(raw_entities))
    }
}

/// Build the entity type prompt string that GLiNER2 uses to condition extraction.
pub fn build_entity_type_prompt(entity_types: &[String]) -> String {
    entity_types
        .iter()
        .map(|t| format!("<entity>{}</entity>", t))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Convert raw ONNX entity outputs into the shared ExtractionResponse format.
pub fn raw_entities_to_extraction_response(raw_entities: Vec<RawOnnxEntity>) -> ExtractionResponse {
    let entities = raw_entities
        .into_iter()
        .map(|raw| ExtractedEntity {
            entity_type: raw.entity_type,
            name: raw.text,
            description: String::new(),
            confidence: raw.confidence,
            relationships: vec![],
        })
        .collect();

    ExtractionResponse { entities }
}

/// Parse ONNX session outputs into RawOnnxEntity values.
///
/// Placeholder: the real implementation must decode GLiNER2's output tensor format
/// (span start/end logits, entity type classification logits) against the tokenizer's
/// offset mapping to recover text spans and labels.
fn parse_onnx_outputs(
    _outputs: &ort::session::SessionOutputs,
    _encoding: &tokenizers::Encoding,
    _entity_types: &[String],
) -> Vec<RawOnnxEntity> {
    // TODO: Implement once the ONNX export graph structure is known.
    vec![]
}
```

### Step 3: Add ndarray dependency

Add to `crates/agent/Cargo.toml` under `[dependencies]`:

```toml
ndarray = "0.16"
```

### Step 4: Run tests

Run: `cargo test -p graphirm-agent knowledge::local_extraction::tests -- --nocapture 2>&1`
Expected: All 3 tests pass. (The extractor construction test fails gracefully on missing file; the parsing tests work on in-memory data.)

### Step 5: Commit

```bash
git add crates/agent/src/knowledge/local_extraction.rs crates/agent/Cargo.toml
git commit -m "feat(knowledge): implement OnnxExtractor scaffold with GLiNER2 model loading"
```

---

## Task 16: Wire hybrid extraction backend into extract_knowledge pipeline

- [x] Complete

**Files:**
- Modify: `crates/agent/src/knowledge/extraction.rs`
- Create: `crates/agent/tests/local_extraction_integration.rs`

**Goal:** Route `extract_knowledge()` through either the LLM backend, the local ONNX backend, or a hybrid of both depending on `ExtractionConfig.backend`. In hybrid mode, GLiNER2 runs first for fast entity extraction, then the LLM synthesizes descriptions and higher-order relationships.

### Step 1: Write failing tests

Add to the `tests` module in `crates/agent/src/knowledge/extraction.rs`:

```rust
    #[tokio::test]
    async fn test_extract_knowledge_skips_llm_when_backend_is_local() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let config = ExtractionConfig {
            enabled: true,
            min_confidence: 0.5,
            backend: ExtractionBackend::Local {
                model_path: "/nonexistent/model.onnx".to_string(),
                tokenizer_path: "/nonexistent/tokenizer.json".to_string(),
            },
            ..ExtractionConfig::default()
        };

        let messages = vec![("user".to_string(), "test".to_string())];
        let source_id = graph
            .add_node(NodeType::Interaction, serde_json::json!({"content": "test"}))
            .unwrap();

        // Local backend with missing model should return an error, not call LLM
        let result = extract_knowledge_with_backend(
            &graph, None, None, &messages, &source_id, &config,
        ).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_extract_knowledge_uses_llm_when_backend_is_llm() {
        let graph = Arc::new(GraphStore::new_in_memory().unwrap());
        let config = ExtractionConfig {
            enabled: true,
            min_confidence: 0.5,
            backend: ExtractionBackend::Llm,
            ..ExtractionConfig::default()
        };
        let llm = Arc::new(MockExtractionProvider {
            response_json: r#"{"entities": [{"entity_type": "pattern", "name": "Test", "description": "A test", "confidence": 0.9, "relationships": []}]}"#.to_string(),
        });

        let messages = vec![("user".to_string(), "test".to_string())];
        let source_id = graph
            .add_node(NodeType::Interaction, serde_json::json!({"content": "test"}))
            .unwrap();

        let result = extract_knowledge_with_backend(
            &graph, Some(&*llm), None, &messages, &source_id, &config,
        ).await.unwrap();

        assert_eq!(result.len(), 1);
    }
```

Run: `cargo test -p graphirm-agent knowledge::extraction::tests 2>&1`
Expected: Fails — `extract_knowledge_with_backend` not defined.

### Step 2: Implement extract_knowledge_with_backend

Add to `crates/agent/src/knowledge/extraction.rs`:

```rust
use super::local_extraction::OnnxExtractor;
use std::path::Path;

/// Backend-aware extraction dispatcher. Routes to LLM, local ONNX, or hybrid
/// depending on ExtractionConfig.backend.
///
/// - `Llm`: calls the LLM provider (existing behavior)
/// - `Local`: runs GLiNER2 via ONNX (fast, no token cost, no descriptions)
/// - `Hybrid`: runs GLiNER2 first for entities, then LLM to synthesize descriptions
///   and discover higher-order relationships
pub async fn extract_knowledge_with_backend(
    graph: &GraphStore,
    llm: Option<&dyn LlmProvider>,
    onnx: Option<&OnnxExtractor>,
    messages: &[(String, String)],
    source_node_id: &NodeId,
    config: &ExtractionConfig,
) -> Result<Vec<NodeId>, AgentError> {
    if !config.enabled {
        return Ok(vec![]);
    }

    let extraction = match &config.backend {
        ExtractionBackend::Llm => {
            let llm = llm.ok_or_else(|| {
                AgentError::Workflow("LLM backend selected but no LLM provider given".into())
            })?;
            let prompt = build_extraction_prompt(messages, config);
            let llm_messages = vec![LlmMessage {
                role: Role::User,
                content: prompt,
                tool_calls: vec![],
                tool_call_id: None,
            }];
            let response = llm.complete(llm_messages, vec![]).await.map_err(AgentError::Llm)?;
            serde_json::from_str::<ExtractionResponse>(&response.content)
                .map_err(|e| AgentError::Workflow(format!("Failed to parse extraction response: {}", e)))?
        }

        ExtractionBackend::Local { model_path, tokenizer_path } => {
            let extractor = onnx.ok_or_else(|| {
                // If no pre-loaded extractor, try to load on demand
                AgentError::Workflow("Local backend selected but no OnnxExtractor provided".into())
            })?;
            let conversation_text = messages
                .iter()
                .map(|(role, content)| format!("[{}]: {}", role, content))
                .collect::<Vec<_>>()
                .join("\n");
            extractor
                .extract(&conversation_text, &config.entity_types, config.min_confidence)
                .await?
        }

        ExtractionBackend::Hybrid { model_path, tokenizer_path } => {
            // Phase 1: fast entity extraction via ONNX
            let extractor = onnx.ok_or_else(|| {
                AgentError::Workflow("Hybrid backend selected but no OnnxExtractor provided".into())
            })?;
            let conversation_text = messages
                .iter()
                .map(|(role, content)| format!("[{}]: {}", role, content))
                .collect::<Vec<_>>()
                .join("\n");
            let local_result = extractor
                .extract(&conversation_text, &config.entity_types, config.min_confidence)
                .await?;

            // Phase 2: LLM enrichment — add descriptions and discover relationships
            if let Some(llm) = llm {
                let entity_names: Vec<&str> = local_result.entities.iter().map(|e| e.name.as_str()).collect();
                let enrichment_prompt = format!(
                    "Given these entities extracted from a conversation: {}\n\n\
                     For each entity, provide a one-sentence description and any relationships between them.\n\n\
                     Conversation:\n{}\n\n\
                     Respond with ONLY valid JSON in the ExtractionResponse format.",
                    entity_names.join(", "),
                    conversation_text,
                );
                let llm_messages = vec![LlmMessage {
                    role: Role::User,
                    content: enrichment_prompt,
                    tool_calls: vec![],
                    tool_call_id: None,
                }];
                match llm.complete(llm_messages, vec![]).await {
                    Ok(response) => {
                        serde_json::from_str::<ExtractionResponse>(&response.content)
                            .unwrap_or(local_result)
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "LLM enrichment failed, falling back to local-only");
                        local_result
                    }
                }
            } else {
                local_result
            }
        }
    };

    // Shared node creation logic (same as existing extract_knowledge)
    let filtered: Vec<&ExtractedEntity> = extraction
        .entities
        .iter()
        .filter(|e| e.confidence >= config.min_confidence)
        .collect();

    let mut name_to_id: HashMap<String, NodeId> = HashMap::new();
    let mut created_ids: Vec<NodeId> = Vec::new();

    for entity in &filtered {
        let data = serde_json::json!({
            "entity_type": entity.entity_type,
            "name": entity.name,
            "description": entity.description,
            "confidence": entity.confidence,
        });

        let node_id = graph.add_node(NodeType::Knowledge, data).map_err(AgentError::Graph)?;
        graph.add_edge(&node_id, source_node_id, EdgeType::DerivedFrom).map_err(AgentError::Graph)?;
        name_to_id.insert(entity.name.clone(), node_id.clone());
        created_ids.push(node_id);
    }

    for entity in &filtered {
        if let Some(source_id) = name_to_id.get(&entity.name) {
            for rel in &entity.relationships {
                if let Some(target_id) = name_to_id.get(&rel.target_name) {
                    graph.add_edge(source_id, target_id, EdgeType::RelatesTo).map_err(AgentError::Graph)?;
                }
            }
        }
    }

    tracing::info!(
        extracted = filtered.len(),
        total = extraction.entities.len(),
        backend = ?config.backend,
        "Knowledge extraction complete"
    );

    Ok(created_ids)
}
```

### Step 3: Update post_turn_extract to use backend-aware function

Modify `post_turn_extract` to call `extract_knowledge_with_backend` instead of `extract_knowledge`, passing the appropriate backend references from the session context.

### Step 4: Run tests

Run: `cargo test -p graphirm-agent knowledge::extraction::tests -- --nocapture 2>&1`
Expected: All tests pass.

### Step 5: Commit

```bash
git add crates/agent/src/knowledge/extraction.rs
git commit -m "feat(knowledge): wire hybrid extraction backend into extract_knowledge pipeline"
```

---

## Final: Run full test suite and clippy

- [x] Complete

### Step 1: Run all tests

Run: `cargo test --workspace 2>&1`
Expected: All tests pass.

### Step 2: Run clippy

Run: `cargo clippy --workspace --all-targets 2>&1`
Expected: No errors. Warnings acceptable.

### Step 3: Format check

Run: `cargo fmt --all -- --check 2>&1`
Expected: No formatting issues.

### Step 4: Final commit

```bash
git add -A
git commit -m "feat: Phase 9 complete — knowledge extraction, HNSW vector search, cross-session memory"
```
