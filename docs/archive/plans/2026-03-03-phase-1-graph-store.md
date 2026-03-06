# Phase 1: Graph Store Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the graph persistence layer on rusqlite + petgraph with full CRUD, typed traversals, conversation threading, subgraph extraction, and PageRank scoring.

**Architecture:** Dual-write architecture — every mutation writes to both SQLite (durable storage) and petgraph (in-memory graph for fast traversals/algorithms). SQLite is the source of truth; petgraph is rebuilt from SQLite on `open()`. Thread safety via `r2d2` connection pool for SQLite and `Arc<RwLock<>>` for the petgraph instance and node index map. All public methods take `&self` (interior mutability).

**Tech Stack:** rusqlite (bundled SQLite), petgraph (graph algorithms), r2d2 + r2d2_sqlite (connection pooling), serde + serde_json (serialization), chrono (timestamps), uuid (IDs), thiserror (errors), tempfile (test fixtures)

---

## Task 1: Define NodeId and EdgeId newtypes

- [x] Complete

**Files:**
- Modify: `crates/graph/src/nodes.rs`
- Test: `crates/graph/src/nodes.rs` (in-module `#[cfg(test)]`)

**Step 1: Write the failing tests**

Add to `crates/graph/src/nodes.rs`:

```rust
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub String);

impl NodeId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
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
}
```

**Step 2: Run tests to verify they pass**

Run: `cargo test -p graphirm-graph nodes::tests -- --nocapture 2>&1`
Expected: All 5 tests pass.

**Step 3: Add EdgeId to edges.rs**

Add to `crates/graph/src/edges.rs`:

```rust
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EdgeId(pub String);

impl EdgeId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

impl fmt::Display for EdgeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for EdgeId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_id_display() {
        let id = EdgeId("edge-456".to_string());
        assert_eq!(id.to_string(), "edge-456");
    }

    #[test]
    fn edge_id_serde_roundtrip() {
        let id = EdgeId("edge-rt".to_string());
        let json = serde_json::to_string(&id).unwrap();
        let back: EdgeId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, back);
    }
}
```

**Step 4: Run all tests**

Run: `cargo test -p graphirm-graph -- --nocapture 2>&1`
Expected: All 7 tests pass.

**Step 5: Commit**

```bash
git add crates/graph/src/nodes.rs crates/graph/src/edges.rs
git commit -m "feat(graph): define NodeId and EdgeId newtypes with Display, From, Serialize"
```

---

## Task 2: Define node data structs and NodeType enum

- [x] Complete

**Files:**
- Modify: `crates/graph/src/nodes.rs`
- Test: `crates/graph/src/nodes.rs` (in-module `#[cfg(test)]`)

**Step 1: Write the data structs and enum**

Add to `crates/graph/src/nodes.rs` (above the `#[cfg(test)]` block):

```rust
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionData {
    pub role: String,       // "user", "assistant", "system"
    pub content: String,
    pub token_count: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentData {
    pub name: String,
    pub model: String,
    pub system_prompt: Option<String>,
    pub status: String,     // "running", "idle", "completed", "failed"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentData {
    pub content_type: String,  // "file", "command_output", "diff", "snippet"
    pub path: Option<String>,
    pub body: String,
    pub language: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskData {
    pub title: String,
    pub description: String,
    pub status: String,     // "pending", "in_progress", "completed", "failed", "cancelled"
    pub priority: Option<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeData {
    pub entity: String,
    pub entity_type: String,  // "function", "file", "concept", "pattern", "error"
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
}
```

**Step 2: Write serialization roundtrip tests**

Add these tests inside the existing `#[cfg(test)] mod tests` block:

```rust
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
            status: "pending".to_string(),
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
            status: "pending".to_string(),
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
```

**Step 3: Run tests**

Run: `cargo test -p graphirm-graph nodes::tests -- --nocapture 2>&1`
Expected: All 12 tests pass (5 from Task 1 + 7 new).

**Step 4: Commit**

```bash
git add crates/graph/src/nodes.rs
git commit -m "feat(graph): define node data structs (Interaction, Agent, Content, Task, Knowledge) and GraphNode"
```

---

## Task 3: Define EdgeType enum and GraphEdge struct

- [x] Complete

**Files:**
- Modify: `crates/graph/src/edges.rs`
- Test: `crates/graph/src/edges.rs` (in-module `#[cfg(test)]`)

**Step 1: Write the EdgeType enum and GraphEdge struct**

Add to `crates/graph/src/edges.rs` (below `EdgeId`, above `#[cfg(test)]`):

```rust
use chrono::{DateTime, Utc};
use crate::nodes::NodeId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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
}

impl EdgeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            EdgeType::RespondsTo => "responds_to",
            EdgeType::SpawnedBy => "spawned_by",
            EdgeType::DelegatesTo => "delegates_to",
            EdgeType::DependsOn => "depends_on",
            EdgeType::Produces => "produces",
            EdgeType::Reads => "reads",
            EdgeType::Modifies => "modifies",
            EdgeType::Summarizes => "summarizes",
            EdgeType::Contains => "contains",
            EdgeType::FollowsUp => "follows_up",
            EdgeType::Steers => "steers",
            EdgeType::RelatesTo => "relates_to",
        }
    }

    pub fn all() -> &'static [EdgeType] {
        &[
            EdgeType::RespondsTo,
            EdgeType::SpawnedBy,
            EdgeType::DelegatesTo,
            EdgeType::DependsOn,
            EdgeType::Produces,
            EdgeType::Reads,
            EdgeType::Modifies,
            EdgeType::Summarizes,
            EdgeType::Contains,
            EdgeType::FollowsUp,
            EdgeType::Steers,
            EdgeType::RelatesTo,
        ]
    }
}

impl fmt::Display for EdgeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub id: EdgeId,
    pub edge_type: EdgeType,
    pub source: NodeId,
    pub target: NodeId,
    pub weight: f64,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
}

impl GraphEdge {
    pub fn new(edge_type: EdgeType, source: NodeId, target: NodeId) -> Self {
        Self {
            id: EdgeId::new(),
            edge_type,
            source,
            target,
            weight: 1.0,
            metadata: serde_json::Value::Object(serde_json::Map::new()),
            created_at: Utc::now(),
        }
    }

    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }
}
```

**Step 2: Write tests**

Add to the existing `#[cfg(test)] mod tests` block in `edges.rs`:

```rust
    use crate::nodes::NodeId;

    #[test]
    fn edge_type_display() {
        assert_eq!(EdgeType::RespondsTo.to_string(), "responds_to");
        assert_eq!(EdgeType::DelegatesTo.to_string(), "delegates_to");
        assert_eq!(EdgeType::FollowsUp.to_string(), "follows_up");
    }

    #[test]
    fn edge_type_serde_roundtrip() {
        for et in EdgeType::all() {
            let json = serde_json::to_string(et).unwrap();
            let back: EdgeType = serde_json::from_str(&json).unwrap();
            assert_eq!(*et, back);
        }
    }

    #[test]
    fn edge_type_all_has_twelve_variants() {
        assert_eq!(EdgeType::all().len(), 12);
    }

    #[test]
    fn graph_edge_new_defaults() {
        let src = NodeId::from("src");
        let tgt = NodeId::from("tgt");
        let edge = GraphEdge::new(EdgeType::RespondsTo, src.clone(), tgt.clone());
        assert_eq!(edge.source, src);
        assert_eq!(edge.target, tgt);
        assert_eq!(edge.edge_type, EdgeType::RespondsTo);
        assert!((edge.weight - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn graph_edge_with_weight() {
        let edge = GraphEdge::new(
            EdgeType::DependsOn,
            NodeId::from("a"),
            NodeId::from("b"),
        )
        .with_weight(0.75);
        assert!((edge.weight - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn graph_edge_serde_roundtrip() {
        let edge = GraphEdge::new(
            EdgeType::Produces,
            NodeId::from("agent-1"),
            NodeId::from("content-1"),
        )
        .with_weight(0.9)
        .with_metadata(serde_json::json!({"tool": "bash"}));

        let json = serde_json::to_string(&edge).unwrap();
        let back: GraphEdge = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, edge.id);
        assert_eq!(back.edge_type, EdgeType::Produces);
        assert_eq!(back.source.0, "agent-1");
        assert_eq!(back.target.0, "content-1");
        assert!((back.weight - 0.9).abs() < f64::EPSILON);
        assert_eq!(back.metadata["tool"], "bash");
    }
```

**Step 3: Run tests**

Run: `cargo test -p graphirm-graph edges::tests -- --nocapture 2>&1`
Expected: All 8 tests pass (2 from Task 1 + 6 new).

**Step 4: Commit**

```bash
git add crates/graph/src/edges.rs
git commit -m "feat(graph): define EdgeType enum (12 variants) and GraphEdge struct"
```

---

## Task 4: Implement GraphStore::open() and open_memory() with SQLite schema

- [x] Complete

**Files:**
- Modify: `crates/graph/src/store.rs`
- Modify: `crates/graph/src/lib.rs` (add re-exports if needed)
- Test: `crates/graph/src/store.rs` (in-module `#[cfg(test)]`)

**Step 1: Write the failing test**

Add to `crates/graph/src/store.rs`:

```rust
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use petgraph::stable_graph::{NodeIndex, StableGraph};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::params;

use crate::edges::{EdgeId, EdgeType, GraphEdge};
use crate::error::GraphError;
use crate::nodes::{GraphNode, NodeId, NodeType};

pub struct GraphStore {
    pool: Pool<SqliteConnectionManager>,
    graph: Arc<RwLock<StableGraph<NodeId, EdgeId>>>,
    node_indices: Arc<RwLock<HashMap<NodeId, NodeIndex>>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_memory_creates_tables() {
        let store = GraphStore::open_memory().unwrap();
        let conn = store.pool.get().unwrap();

        // Verify nodes table exists
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='nodes'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);

        // Verify edges table exists
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='edges'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p graphirm-graph store::tests::open_memory_creates_tables -- --nocapture 2>&1`
Expected: FAIL — `open_memory` method doesn't exist yet.

**Step 3: Implement open_memory() and open()**

Add these `impl` blocks above the `#[cfg(test)]` block in `store.rs`:

```rust
impl GraphStore {
    pub fn open(path: &str) -> Result<Self, GraphError> {
        let manager = SqliteConnectionManager::file(path);
        let pool = Pool::builder()
            .max_size(4)
            .build(manager)
            .map_err(GraphError::Pool)?;

        let store = Self {
            pool,
            graph: Arc::new(RwLock::new(StableGraph::new())),
            node_indices: Arc::new(RwLock::new(HashMap::new())),
        };
        store.init_schema()?;
        store.rebuild_graph()?;
        Ok(store)
    }

    pub fn open_memory() -> Result<Self, GraphError> {
        let manager = SqliteConnectionManager::memory();
        let pool = Pool::builder()
            .max_size(1)
            .build(manager)
            .map_err(GraphError::Pool)?;

        let store = Self {
            pool,
            graph: Arc::new(RwLock::new(StableGraph::new())),
            node_indices: Arc::new(RwLock::new(HashMap::new())),
        };
        store.init_schema()?;
        Ok(store)
    }

    fn init_schema(&self) -> Result<(), GraphError> {
        let conn = self.pool.get()?;
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                data TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                edge_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
            CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
            CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);

            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;
            ",
        )?;
        Ok(())
    }

    fn rebuild_graph(&self) -> Result<(), GraphError> {
        let conn = self.pool.get()?;
        let mut graph = self.graph.write().map_err(|_| GraphError::LockPoisoned)?;
        let mut indices = self
            .node_indices
            .write()
            .map_err(|_| GraphError::LockPoisoned)?;

        graph.clear();
        indices.clear();

        // Load all node IDs
        let mut stmt = conn.prepare("SELECT id FROM nodes")?;
        let node_ids: Vec<NodeId> = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                Ok(NodeId(id))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        for node_id in node_ids {
            let idx = graph.add_node(node_id.clone());
            indices.insert(node_id, idx);
        }

        // Load all edges
        let mut stmt = conn.prepare("SELECT id, source_id, target_id FROM edges")?;
        let edges: Vec<(EdgeId, NodeId, NodeId)> = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let source: String = row.get(1)?;
                let target: String = row.get(2)?;
                Ok((EdgeId(id), NodeId(source), NodeId(target)))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        for (edge_id, source, target) in edges {
            if let (Some(&src_idx), Some(&tgt_idx)) = (indices.get(&source), indices.get(&target))
            {
                graph.add_edge(src_idx, tgt_idx, edge_id);
            }
        }

        Ok(())
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p graphirm-graph store::tests::open_memory_creates_tables -- --nocapture 2>&1`
Expected: PASS

**Step 5: Add test for file-based open**

Add to the test module:

```rust
    #[test]
    fn open_file_creates_tables() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = GraphStore::open(db_path.to_str().unwrap()).unwrap();
        let conn = store.pool.get().unwrap();

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='nodes'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }
```

**Step 6: Run all store tests**

Run: `cargo test -p graphirm-graph store::tests -- --nocapture 2>&1`
Expected: Both tests pass.

**Step 7: Commit**

```bash
git add crates/graph/src/store.rs
git commit -m "feat(graph): implement GraphStore with SQLite schema and petgraph rebuild"
```

---

## Task 5: Implement add_node() and get_node()

- [x] Complete

**Files:**
- Modify: `crates/graph/src/store.rs`

**Step 1: Write the failing test**

Add to the test module in `store.rs`:

```rust
    use crate::nodes::{InteractionData, NodeType};

    #[test]
    fn add_and_get_node_roundtrip() {
        let store = GraphStore::open_memory().unwrap();
        let node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "Hello!".to_string(),
            token_count: Some(3),
        }));
        let id = node.id.clone();

        let returned_id = store.add_node(node).unwrap();
        assert_eq!(returned_id, id);

        let fetched = store.get_node(&id).unwrap();
        assert_eq!(fetched.id, id);
        match &fetched.node_type {
            NodeType::Interaction(data) => {
                assert_eq!(data.role, "user");
                assert_eq!(data.content, "Hello!");
                assert_eq!(data.token_count, Some(3));
            }
            _ => panic!("expected Interaction"),
        }
    }

    #[test]
    fn get_nonexistent_node_returns_error() {
        let store = GraphStore::open_memory().unwrap();
        let result = store.get_node(&NodeId::from("nonexistent"));
        assert!(result.is_err());
        match result.unwrap_err() {
            GraphError::NodeNotFound(id) => assert_eq!(id, "nonexistent"),
            other => panic!("expected NodeNotFound, got: {other}"),
        }
    }

    #[test]
    fn add_node_updates_petgraph() {
        let store = GraphStore::open_memory().unwrap();
        let node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "test".to_string(),
            token_count: None,
        }));
        let id = node.id.clone();
        store.add_node(node).unwrap();

        let indices = store.node_indices.read().unwrap();
        assert!(indices.contains_key(&id));

        let graph = store.graph.read().unwrap();
        assert_eq!(graph.node_count(), 1);
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p graphirm-graph store::tests::add_and_get -- --nocapture 2>&1`
Expected: FAIL — `add_node` and `get_node` don't exist.

**Step 3: Implement add_node() and get_node()**

Add to the `impl GraphStore` block in `store.rs`:

```rust
    pub fn add_node(&self, node: GraphNode) -> Result<NodeId, GraphError> {
        let conn = self.pool.get()?;
        let data = serde_json::to_string(&node.node_type)?;
        let metadata = serde_json::to_string(&node.metadata)?;

        conn.execute(
            "INSERT INTO nodes (id, node_type, data, metadata, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                node.id.0,
                node.node_type.type_name(),
                data,
                metadata,
                node.created_at.to_rfc3339(),
                node.updated_at.to_rfc3339(),
            ],
        )?;

        // Update petgraph
        let mut graph = self.graph.write().map_err(|_| GraphError::LockPoisoned)?;
        let mut indices = self
            .node_indices
            .write()
            .map_err(|_| GraphError::LockPoisoned)?;
        let idx = graph.add_node(node.id.clone());
        indices.insert(node.id.clone(), idx);

        Ok(node.id)
    }

    pub fn get_node(&self, id: &NodeId) -> Result<GraphNode, GraphError> {
        let conn = self.pool.get()?;
        let mut stmt = conn.prepare(
            "SELECT id, data, metadata, created_at, updated_at FROM nodes WHERE id = ?1",
        )?;

        let node = stmt
            .query_row(params![id.0], |row| {
                let id: String = row.get(0)?;
                let data: String = row.get(1)?;
                let metadata: String = row.get(2)?;
                let created_at: String = row.get(3)?;
                let updated_at: String = row.get(4)?;
                Ok((id, data, metadata, created_at, updated_at))
            })
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    GraphError::NodeNotFound(id.0.clone())
                }
                other => GraphError::Sqlite(other),
            })?;

        let (id_str, data, metadata, created_at, updated_at) = node;
        let node_type: NodeType = serde_json::from_str(&data)?;
        let metadata: serde_json::Value = serde_json::from_str(&metadata)?;
        let created_at = chrono::DateTime::parse_from_rfc3339(&created_at)
            .map_err(|e| GraphError::NodeNotFound(format!("bad timestamp: {e}")))?
            .with_timezone(&chrono::Utc);
        let updated_at = chrono::DateTime::parse_from_rfc3339(&updated_at)
            .map_err(|e| GraphError::NodeNotFound(format!("bad timestamp: {e}")))?
            .with_timezone(&chrono::Utc);

        Ok(GraphNode {
            id: NodeId(id_str),
            node_type,
            metadata,
            created_at,
            updated_at,
        })
    }
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p graphirm-graph store::tests -- --nocapture 2>&1`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add crates/graph/src/store.rs
git commit -m "feat(graph): implement add_node() and get_node() with SQLite + petgraph dual-write"
```

---

## Task 6: Implement update_node() and delete_node()

- [x] Complete

**Files:**
- Modify: `crates/graph/src/store.rs`

**Step 1: Write the failing tests**

Add to the test module in `store.rs`:

```rust
    #[test]
    fn update_node_changes_fields() {
        let store = GraphStore::open_memory().unwrap();
        let mut node = GraphNode::new(NodeType::Task(crate::nodes::TaskData {
            title: "Original".to_string(),
            description: "First version".to_string(),
            status: "pending".to_string(),
            priority: Some(1),
        }));
        let id = node.id.clone();
        store.add_node(node.clone()).unwrap();

        node.node_type = NodeType::Task(crate::nodes::TaskData {
            title: "Updated".to_string(),
            description: "Second version".to_string(),
            status: "in_progress".to_string(),
            priority: Some(2),
        });
        store.update_node(&id, node).unwrap();

        let fetched = store.get_node(&id).unwrap();
        match &fetched.node_type {
            NodeType::Task(data) => {
                assert_eq!(data.title, "Updated");
                assert_eq!(data.status, "in_progress");
            }
            _ => panic!("expected Task"),
        }
        assert!(fetched.updated_at > fetched.created_at);
    }

    #[test]
    fn delete_node_removes_from_store() {
        let store = GraphStore::open_memory().unwrap();
        let node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "deleteme".to_string(),
            token_count: None,
        }));
        let id = node.id.clone();
        store.add_node(node).unwrap();

        store.delete_node(&id).unwrap();

        assert!(store.get_node(&id).is_err());

        let indices = store.node_indices.read().unwrap();
        assert!(!indices.contains_key(&id));

        let graph = store.graph.read().unwrap();
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn update_nonexistent_node_returns_error() {
        let store = GraphStore::open_memory().unwrap();
        let node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "ghost".to_string(),
            token_count: None,
        }));
        let result = store.update_node(&NodeId::from("nonexistent"), node);
        assert!(result.is_err());
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p graphirm-graph store::tests::update_node -- --nocapture 2>&1`
Expected: FAIL — `update_node` doesn't exist.

**Step 3: Implement update_node() and delete_node()**

Add to the `impl GraphStore` block:

```rust
    pub fn update_node(&self, id: &NodeId, mut node: GraphNode) -> Result<(), GraphError> {
        node.updated_at = chrono::Utc::now();
        let conn = self.pool.get()?;
        let data = serde_json::to_string(&node.node_type)?;
        let metadata = serde_json::to_string(&node.metadata)?;

        let rows = conn.execute(
            "UPDATE nodes SET node_type = ?1, data = ?2, metadata = ?3, updated_at = ?4
             WHERE id = ?5",
            params![
                node.node_type.type_name(),
                data,
                metadata,
                node.updated_at.to_rfc3339(),
                id.0,
            ],
        )?;

        if rows == 0 {
            return Err(GraphError::NodeNotFound(id.0.clone()));
        }
        Ok(())
    }

    pub fn delete_node(&self, id: &NodeId) -> Result<(), GraphError> {
        let conn = self.pool.get()?;

        // Delete edges referencing this node from SQLite
        conn.execute(
            "DELETE FROM edges WHERE source_id = ?1 OR target_id = ?1",
            params![id.0],
        )?;

        // Delete the node from SQLite
        let rows = conn.execute("DELETE FROM nodes WHERE id = ?1", params![id.0])?;
        if rows == 0 {
            return Err(GraphError::NodeNotFound(id.0.clone()));
        }

        // Remove from petgraph
        let mut graph = self.graph.write().map_err(|_| GraphError::LockPoisoned)?;
        let mut indices = self
            .node_indices
            .write()
            .map_err(|_| GraphError::LockPoisoned)?;

        if let Some(idx) = indices.remove(id) {
            graph.remove_node(idx);
        }

        Ok(())
    }
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p graphirm-graph store::tests -- --nocapture 2>&1`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add crates/graph/src/store.rs
git commit -m "feat(graph): implement update_node() and delete_node() with cascade"
```

---

## Task 7: Implement add_edge() and get_edge()

- [x] Complete

**Files:**
- Modify: `crates/graph/src/store.rs`

**Step 1: Write the failing tests**

Add to the test module in `store.rs`:

```rust
    #[test]
    fn add_and_get_edge_roundtrip() {
        let store = GraphStore::open_memory().unwrap();

        let n1 = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "Hello".to_string(),
            token_count: None,
        }));
        let n2 = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".to_string(),
            content: "Hi!".to_string(),
            token_count: None,
        }));
        let n1_id = n1.id.clone();
        let n2_id = n2.id.clone();
        store.add_node(n1).unwrap();
        store.add_node(n2).unwrap();

        let edge = GraphEdge::new(EdgeType::RespondsTo, n2_id.clone(), n1_id.clone())
            .with_weight(0.8);
        let edge_id = edge.id.clone();
        let returned_id = store.add_edge(edge).unwrap();
        assert_eq!(returned_id, edge_id);

        let fetched = store.get_edge(&edge_id).unwrap();
        assert_eq!(fetched.edge_type, EdgeType::RespondsTo);
        assert_eq!(fetched.source, n2_id);
        assert_eq!(fetched.target, n1_id);
        assert!((fetched.weight - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn get_nonexistent_edge_returns_error() {
        let store = GraphStore::open_memory().unwrap();
        let result = store.get_edge(&EdgeId::from("nonexistent"));
        assert!(result.is_err());
    }

    #[test]
    fn add_edge_updates_petgraph() {
        let store = GraphStore::open_memory().unwrap();

        let n1 = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "a".to_string(),
            token_count: None,
        }));
        let n2 = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".to_string(),
            content: "b".to_string(),
            token_count: None,
        }));
        let n1_id = n1.id.clone();
        let n2_id = n2.id.clone();
        store.add_node(n1).unwrap();
        store.add_node(n2).unwrap();

        store
            .add_edge(GraphEdge::new(
                EdgeType::RespondsTo,
                n2_id.clone(),
                n1_id.clone(),
            ))
            .unwrap();

        let graph = store.graph.read().unwrap();
        assert_eq!(graph.edge_count(), 1);
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p graphirm-graph store::tests::add_and_get_edge -- --nocapture 2>&1`
Expected: FAIL — `add_edge` doesn't exist.

**Step 3: Implement add_edge() and get_edge()**

Add to the `impl GraphStore` block:

```rust
    pub fn add_edge(&self, edge: GraphEdge) -> Result<EdgeId, GraphError> {
        let conn = self.pool.get()?;
        let metadata = serde_json::to_string(&edge.metadata)?;

        conn.execute(
            "INSERT INTO edges (id, edge_type, source_id, target_id, weight, metadata, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                edge.id.0,
                edge.edge_type.as_str(),
                edge.source.0,
                edge.target.0,
                edge.weight,
                metadata,
                edge.created_at.to_rfc3339(),
            ],
        )?;

        // Update petgraph
        let graph = self.graph.read().map_err(|_| GraphError::LockPoisoned)?;
        let indices = self
            .node_indices
            .read()
            .map_err(|_| GraphError::LockPoisoned)?;

        let src_idx = indices
            .get(&edge.source)
            .ok_or_else(|| GraphError::NodeNotFound(edge.source.0.clone()))?;
        let tgt_idx = indices
            .get(&edge.target)
            .ok_or_else(|| GraphError::NodeNotFound(edge.target.0.clone()))?;

        drop(indices);
        drop(graph);

        let mut graph = self.graph.write().map_err(|_| GraphError::LockPoisoned)?;
        let indices = self
            .node_indices
            .read()
            .map_err(|_| GraphError::LockPoisoned)?;

        let src_idx = *indices
            .get(&edge.source)
            .ok_or_else(|| GraphError::NodeNotFound(edge.source.0.clone()))?;
        let tgt_idx = *indices
            .get(&edge.target)
            .ok_or_else(|| GraphError::NodeNotFound(edge.target.0.clone()))?;

        graph.add_edge(src_idx, tgt_idx, edge.id.clone());

        Ok(edge.id)
    }

    pub fn get_edge(&self, id: &EdgeId) -> Result<GraphEdge, GraphError> {
        let conn = self.pool.get()?;
        let mut stmt = conn.prepare(
            "SELECT id, edge_type, source_id, target_id, weight, metadata, created_at
             FROM edges WHERE id = ?1",
        )?;

        let row = stmt
            .query_row(params![id.0], |row| {
                let id: String = row.get(0)?;
                let edge_type: String = row.get(1)?;
                let source: String = row.get(2)?;
                let target: String = row.get(3)?;
                let weight: f64 = row.get(4)?;
                let metadata: String = row.get(5)?;
                let created_at: String = row.get(6)?;
                Ok((id, edge_type, source, target, weight, metadata, created_at))
            })
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    GraphError::EdgeNotFound(id.0.clone())
                }
                other => GraphError::Sqlite(other),
            })?;

        let (id_str, edge_type_str, source, target, weight, metadata, created_at) = row;

        let edge_type: EdgeType = serde_json::from_str(&format!("\"{}\"", edge_type_str))?;
        let metadata: serde_json::Value = serde_json::from_str(&metadata)?;
        let created_at = chrono::DateTime::parse_from_rfc3339(&created_at)
            .map_err(|e| GraphError::EdgeNotFound(format!("bad timestamp: {e}")))?
            .with_timezone(&chrono::Utc);

        Ok(GraphEdge {
            id: EdgeId(id_str),
            edge_type,
            source: NodeId(source),
            target: NodeId(target),
            weight,
            metadata,
            created_at,
        })
    }
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p graphirm-graph store::tests -- --nocapture 2>&1`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add crates/graph/src/store.rs
git commit -m "feat(graph): implement add_edge() and get_edge() with dual-write"
```

---

## Task 8: Test delete_node() cascades edges

- [x] Complete

**Files:**
- Modify: `crates/graph/src/store.rs`

**Step 1: Write the cascade test**

Add to the test module in `store.rs`:

```rust
    #[test]
    fn delete_node_cascades_edges() {
        let store = GraphStore::open_memory().unwrap();

        let n1 = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "msg1".to_string(),
            token_count: None,
        }));
        let n2 = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".to_string(),
            content: "msg2".to_string(),
            token_count: None,
        }));
        let n3 = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "msg3".to_string(),
            token_count: None,
        }));
        let n1_id = n1.id.clone();
        let n2_id = n2.id.clone();
        let n3_id = n3.id.clone();
        store.add_node(n1).unwrap();
        store.add_node(n2).unwrap();
        store.add_node(n3).unwrap();

        let e1 = GraphEdge::new(EdgeType::RespondsTo, n2_id.clone(), n1_id.clone());
        let e2 = GraphEdge::new(EdgeType::RespondsTo, n3_id.clone(), n2_id.clone());
        let e1_id = e1.id.clone();
        let e2_id = e2.id.clone();
        store.add_edge(e1).unwrap();
        store.add_edge(e2).unwrap();

        // Delete the middle node — both edges should be removed
        store.delete_node(&n2_id).unwrap();

        // Both edges gone from SQLite
        assert!(store.get_edge(&e1_id).is_err());
        assert!(store.get_edge(&e2_id).is_err());

        // Petgraph should have 2 nodes, 0 edges
        let graph = store.graph.read().unwrap();
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 0);
    }
```

**Step 2: Run the test**

Run: `cargo test -p graphirm-graph store::tests::delete_node_cascades_edges -- --nocapture 2>&1`
Expected: PASS (delete_node already handles cascade from Task 6).

**Step 3: Commit**

```bash
git add crates/graph/src/store.rs
git commit -m "test(graph): verify delete_node cascades edge removal"
```

---

## Task 9: Implement neighbors()

- [x] Complete

**Files:**
- Modify: `crates/graph/src/store.rs`

**Step 1: Write the failing test**

Add to the test module in `store.rs`:

```rust
    #[test]
    fn neighbors_outgoing() {
        let store = GraphStore::open_memory().unwrap();

        let agent = GraphNode::new(NodeType::Agent(crate::nodes::AgentData {
            name: "coder".to_string(),
            model: "claude".to_string(),
            system_prompt: None,
            status: "running".to_string(),
        }));
        let content1 = GraphNode::new(NodeType::Content(crate::nodes::ContentData {
            content_type: "file".to_string(),
            path: Some("a.rs".to_string()),
            body: "code".to_string(),
            language: Some("rust".to_string()),
        }));
        let content2 = GraphNode::new(NodeType::Content(crate::nodes::ContentData {
            content_type: "file".to_string(),
            path: Some("b.rs".to_string()),
            body: "more code".to_string(),
            language: Some("rust".to_string()),
        }));
        let agent_id = agent.id.clone();
        let c1_id = content1.id.clone();
        let c2_id = content2.id.clone();
        store.add_node(agent).unwrap();
        store.add_node(content1).unwrap();
        store.add_node(content2).unwrap();

        store
            .add_edge(GraphEdge::new(
                EdgeType::Reads,
                agent_id.clone(),
                c1_id.clone(),
            ))
            .unwrap();
        store
            .add_edge(GraphEdge::new(
                EdgeType::Modifies,
                agent_id.clone(),
                c2_id.clone(),
            ))
            .unwrap();

        // All outgoing neighbors (any edge type)
        let all = store
            .neighbors(&agent_id, None, Direction::Outgoing)
            .unwrap();
        assert_eq!(all.len(), 2);

        // Filter by Reads only
        let reads = store
            .neighbors(&agent_id, Some(EdgeType::Reads), Direction::Outgoing)
            .unwrap();
        assert_eq!(reads.len(), 1);
        assert_eq!(reads[0].id, c1_id);

        // Incoming to content1
        let incoming = store
            .neighbors(&c1_id, None, Direction::Incoming)
            .unwrap();
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0].id, agent_id);
    }
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p graphirm-graph store::tests::neighbors_outgoing -- --nocapture 2>&1`
Expected: FAIL — `neighbors` doesn't exist.

**Step 3: Implement neighbors()**

Add `use petgraph::Direction;` at the top of `store.rs`, then add to the `impl GraphStore` block:

```rust
    pub fn neighbors(
        &self,
        id: &NodeId,
        edge_type: Option<EdgeType>,
        direction: Direction,
    ) -> Result<Vec<GraphNode>, GraphError> {
        let conn = self.pool.get()?;

        let (query, param) = match direction {
            Direction::Outgoing => (
                match edge_type {
                    Some(et) => format!(
                        "SELECT target_id FROM edges WHERE source_id = ?1 AND edge_type = '{}'",
                        et.as_str()
                    ),
                    None => "SELECT target_id FROM edges WHERE source_id = ?1".to_string(),
                },
                &id.0,
            ),
            Direction::Incoming => (
                match edge_type {
                    Some(et) => format!(
                        "SELECT source_id FROM edges WHERE target_id = ?1 AND edge_type = '{}'",
                        et.as_str()
                    ),
                    None => "SELECT source_id FROM edges WHERE target_id = ?1".to_string(),
                },
                &id.0,
            ),
        };

        let mut stmt = conn.prepare(&query)?;
        let neighbor_ids: Vec<NodeId> = stmt
            .query_map(params![param], |row| {
                let nid: String = row.get(0)?;
                Ok(NodeId(nid))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        let mut neighbors = Vec::new();
        for nid in neighbor_ids {
            neighbors.push(self.get_node(&nid)?);
        }

        Ok(neighbors)
    }
```

Also add the re-export in `crates/graph/src/lib.rs`:

```rust
pub use petgraph::Direction;
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p graphirm-graph store::tests::neighbors_outgoing -- --nocapture 2>&1`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/graph/src/store.rs crates/graph/src/lib.rs
git commit -m "feat(graph): implement neighbors() with optional edge type filter and direction"
```

---

## Task 10: Implement traverse() with BFS

- [x] Complete

**Files:**
- Modify: `crates/graph/src/store.rs`

**Step 1: Write the failing test**

Add to the test module in `store.rs`:

```rust
    #[test]
    fn traverse_bfs_with_depth() {
        let store = GraphStore::open_memory().unwrap();

        // Build a chain: A --Contains--> B --Contains--> C --Contains--> D
        let make_content = |name: &str| {
            GraphNode::new(NodeType::Content(crate::nodes::ContentData {
                content_type: "file".to_string(),
                path: Some(name.to_string()),
                body: "".to_string(),
                language: None,
            }))
        };

        let a = make_content("a"); let a_id = a.id.clone();
        let b = make_content("b"); let b_id = b.id.clone();
        let c = make_content("c"); let c_id = c.id.clone();
        let d = make_content("d"); let d_id = d.id.clone();

        store.add_node(a).unwrap();
        store.add_node(b).unwrap();
        store.add_node(c).unwrap();
        store.add_node(d).unwrap();

        store.add_edge(GraphEdge::new(EdgeType::Contains, a_id.clone(), b_id.clone())).unwrap();
        store.add_edge(GraphEdge::new(EdgeType::Contains, b_id.clone(), c_id.clone())).unwrap();
        store.add_edge(GraphEdge::new(EdgeType::Contains, c_id.clone(), d_id.clone())).unwrap();

        // Traverse depth=1 from A: should get B only
        let depth1 = store.traverse(&a_id, &[EdgeType::Contains], 1).unwrap();
        assert_eq!(depth1.len(), 1);
        assert_eq!(depth1[0].id, b_id);

        // Traverse depth=2 from A: B and C
        let depth2 = store.traverse(&a_id, &[EdgeType::Contains], 2).unwrap();
        assert_eq!(depth2.len(), 2);

        // Traverse depth=10 from A: B, C, D (only 3 reachable)
        let all = store.traverse(&a_id, &[EdgeType::Contains], 10).unwrap();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn traverse_filters_edge_types() {
        let store = GraphStore::open_memory().unwrap();

        let a = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(), content: "a".to_string(), token_count: None,
        }));
        let b = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".to_string(), content: "b".to_string(), token_count: None,
        }));
        let c = GraphNode::new(NodeType::Content(crate::nodes::ContentData {
            content_type: "file".to_string(), path: None, body: "c".to_string(), language: None,
        }));

        let a_id = a.id.clone();
        let b_id = b.id.clone();
        let c_id = c.id.clone();
        store.add_node(a).unwrap();
        store.add_node(b).unwrap();
        store.add_node(c).unwrap();

        // a --RespondsTo--> b
        store.add_edge(GraphEdge::new(EdgeType::RespondsTo, a_id.clone(), b_id.clone())).unwrap();
        // a --Reads--> c
        store.add_edge(GraphEdge::new(EdgeType::Reads, a_id.clone(), c_id.clone())).unwrap();

        // Traverse only RespondsTo edges
        let result = store.traverse(&a_id, &[EdgeType::RespondsTo], 5).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, b_id);
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p graphirm-graph store::tests::traverse -- --nocapture 2>&1`
Expected: FAIL — `traverse` doesn't exist.

**Step 3: Implement traverse()**

Add to the `impl GraphStore` block:

```rust
    pub fn traverse(
        &self,
        start: &NodeId,
        edge_types: &[EdgeType],
        max_depth: usize,
    ) -> Result<Vec<GraphNode>, GraphError> {
        use std::collections::{HashSet, VecDeque};

        let mut visited = HashSet::new();
        visited.insert(start.clone());
        let mut queue = VecDeque::new();
        queue.push_back((start.clone(), 0_usize));
        let mut result = Vec::new();

        let edge_type_strs: Vec<&str> = edge_types.iter().map(|et| et.as_str()).collect();

        while let Some((current_id, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            let conn = self.pool.get()?;
            let placeholders: String = edge_type_strs
                .iter()
                .map(|_| "?")
                .collect::<Vec<_>>()
                .join(", ");
            let query = format!(
                "SELECT target_id FROM edges WHERE source_id = ?1 AND edge_type IN ({})",
                placeholders
            );

            let mut stmt = conn.prepare(&query)?;

            let mut param_idx = 1;
            let neighbor_ids: Vec<NodeId> = {
                use rusqlite::types::ToSql;
                let mut params_vec: Vec<Box<dyn ToSql>> = Vec::new();
                params_vec.push(Box::new(current_id.0.clone()));
                for et in &edge_type_strs {
                    params_vec.push(Box::new(et.to_string()));
                }
                let params_refs: Vec<&dyn ToSql> =
                    params_vec.iter().map(|p| p.as_ref()).collect();

                stmt.query_map(params_refs.as_slice(), |row| {
                    let nid: String = row.get(0)?;
                    Ok(NodeId(nid))
                })?
                .collect::<Result<Vec<_>, _>>()?
            };

            for nid in neighbor_ids {
                if visited.insert(nid.clone()) {
                    let node = self.get_node(&nid)?;
                    result.push(node);
                    queue.push_back((nid, depth + 1));
                }
            }
        }

        Ok(result)
    }
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p graphirm-graph store::tests::traverse -- --nocapture 2>&1`
Expected: Both traverse tests pass.

**Step 5: Commit**

```bash
git add crates/graph/src/store.rs
git commit -m "feat(graph): implement traverse() with BFS and edge type filtering"
```

---

## Task 11: Implement conversation_thread()

- [x] Complete

**Files:**
- Modify: `crates/graph/src/store.rs`

**Step 1: Write the failing test**

Add to the test module in `store.rs`:

```rust
    #[test]
    fn conversation_thread_walks_backwards() {
        let store = GraphStore::open_memory().unwrap();

        // Build: msg1 <--RespondsTo-- msg2 <--RespondsTo-- msg3 <--RespondsTo-- msg4 <--RespondsTo-- msg5
        let make_msg = |role: &str, content: &str| {
            GraphNode::new(NodeType::Interaction(InteractionData {
                role: role.to_string(),
                content: content.to_string(),
                token_count: None,
            }))
        };

        let msg1 = make_msg("user", "Hello");
        let msg2 = make_msg("assistant", "Hi there!");
        let msg3 = make_msg("user", "How are you?");
        let msg4 = make_msg("assistant", "I'm good!");
        let msg5 = make_msg("user", "Great");

        let ids: Vec<NodeId> = vec![
            msg1.id.clone(),
            msg2.id.clone(),
            msg3.id.clone(),
            msg4.id.clone(),
            msg5.id.clone(),
        ];

        store.add_node(msg1).unwrap();
        store.add_node(msg2).unwrap();
        store.add_node(msg3).unwrap();
        store.add_node(msg4).unwrap();
        store.add_node(msg5).unwrap();

        // Each message responds to the previous
        store.add_edge(GraphEdge::new(EdgeType::RespondsTo, ids[1].clone(), ids[0].clone())).unwrap();
        store.add_edge(GraphEdge::new(EdgeType::RespondsTo, ids[2].clone(), ids[1].clone())).unwrap();
        store.add_edge(GraphEdge::new(EdgeType::RespondsTo, ids[3].clone(), ids[2].clone())).unwrap();
        store.add_edge(GraphEdge::new(EdgeType::RespondsTo, ids[4].clone(), ids[3].clone())).unwrap();

        // Walk from the latest message (msg5) backwards
        let thread = store.conversation_thread(&ids[4]).unwrap();

        // Should return [msg5, msg4, msg3, msg2, msg1] (newest first)
        assert_eq!(thread.len(), 5);
        assert_eq!(thread[0].id, ids[4]);
        assert_eq!(thread[1].id, ids[3]);
        assert_eq!(thread[2].id, ids[2]);
        assert_eq!(thread[3].id, ids[1]);
        assert_eq!(thread[4].id, ids[0]);
    }

    #[test]
    fn conversation_thread_single_message() {
        let store = GraphStore::open_memory().unwrap();
        let msg = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "lonely".to_string(),
            token_count: None,
        }));
        let id = msg.id.clone();
        store.add_node(msg).unwrap();

        let thread = store.conversation_thread(&id).unwrap();
        assert_eq!(thread.len(), 1);
        assert_eq!(thread[0].id, id);
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p graphirm-graph store::tests::conversation_thread -- --nocapture 2>&1`
Expected: FAIL — `conversation_thread` doesn't exist.

**Step 3: Implement conversation_thread()**

Add to the `impl GraphStore` block:

```rust
    /// Walk `RespondsTo` edges backwards from `leaf_id` to the root of the conversation.
    /// Returns nodes newest-first: [leaf, ..., root].
    pub fn conversation_thread(&self, leaf_id: &NodeId) -> Result<Vec<GraphNode>, GraphError> {
        let mut thread = Vec::new();
        let mut current_id = leaf_id.clone();

        loop {
            let node = self.get_node(&current_id)?;
            thread.push(node);

            // Follow the outgoing RespondsTo edge (this node responds to its parent)
            let conn = self.pool.get()?;
            let parent_id: Option<String> = conn
                .query_row(
                    "SELECT target_id FROM edges WHERE source_id = ?1 AND edge_type = 'responds_to' LIMIT 1",
                    params![current_id.0],
                    |row| row.get(0),
                )
                .ok();

            match parent_id {
                Some(pid) => current_id = NodeId(pid),
                None => break,
            }
        }

        Ok(thread)
    }
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p graphirm-graph store::tests::conversation_thread -- --nocapture 2>&1`
Expected: Both tests pass.

**Step 5: Commit**

```bash
git add crates/graph/src/store.rs
git commit -m "feat(graph): implement conversation_thread() — walk RespondsTo edges to root"
```

---

## Task 12: Implement subgraph() extraction

- [x] Complete

**Files:**
- Modify: `crates/graph/src/store.rs`

**Step 1: Write the failing test**

Add to the test module in `store.rs`:

```rust
    #[test]
    fn subgraph_returns_nodes_and_edges_within_depth() {
        let store = GraphStore::open_memory().unwrap();

        let a = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(), content: "a".to_string(), token_count: None,
        }));
        let b = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".to_string(), content: "b".to_string(), token_count: None,
        }));
        let c = GraphNode::new(NodeType::Content(crate::nodes::ContentData {
            content_type: "file".to_string(), path: None, body: "c".to_string(), language: None,
        }));
        let d = GraphNode::new(NodeType::Content(crate::nodes::ContentData {
            content_type: "file".to_string(), path: None, body: "d".to_string(), language: None,
        }));

        let a_id = a.id.clone();
        let b_id = b.id.clone();
        let c_id = c.id.clone();
        let d_id = d.id.clone();

        store.add_node(a).unwrap();
        store.add_node(b).unwrap();
        store.add_node(c).unwrap();
        store.add_node(d).unwrap();

        // a --> b, b --> c, c --> d (all RespondsTo)
        store.add_edge(GraphEdge::new(EdgeType::RespondsTo, a_id.clone(), b_id.clone())).unwrap();
        store.add_edge(GraphEdge::new(EdgeType::RespondsTo, b_id.clone(), c_id.clone())).unwrap();
        store.add_edge(GraphEdge::new(EdgeType::RespondsTo, c_id.clone(), d_id.clone())).unwrap();

        // depth=1 from a: should get a + b, and the a->b edge
        let (nodes, edges) = store.subgraph(&a_id, 1).unwrap();
        assert_eq!(nodes.len(), 2); // a and b
        assert_eq!(edges.len(), 1); // a->b

        // depth=2 from a: a, b, c and edges a->b, b->c
        let (nodes, edges) = store.subgraph(&a_id, 2).unwrap();
        assert_eq!(nodes.len(), 3);
        assert_eq!(edges.len(), 2);
    }
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p graphirm-graph store::tests::subgraph_returns -- --nocapture 2>&1`
Expected: FAIL — `subgraph` doesn't exist.

**Step 3: Implement subgraph()**

Add to the `impl GraphStore` block:

```rust
    /// Extract a subgraph rooted at `root` up to `depth` hops (following all outgoing edges).
    /// Returns the root node plus all reachable nodes, and all edges between them.
    pub fn subgraph(
        &self,
        root: &NodeId,
        depth: usize,
    ) -> Result<(Vec<GraphNode>, Vec<GraphEdge>), GraphError> {
        use std::collections::{HashSet, VecDeque};

        let mut visited = HashSet::new();
        visited.insert(root.clone());
        let mut queue = VecDeque::new();
        queue.push_back((root.clone(), 0_usize));
        let mut nodes = vec![self.get_node(root)?];

        while let Some((current_id, d)) = queue.pop_front() {
            if d >= depth {
                continue;
            }

            let conn = self.pool.get()?;
            let mut stmt =
                conn.prepare("SELECT target_id FROM edges WHERE source_id = ?1")?;
            let neighbor_ids: Vec<NodeId> = stmt
                .query_map(params![current_id.0], |row| {
                    let nid: String = row.get(0)?;
                    Ok(NodeId(nid))
                })?
                .collect::<Result<Vec<_>, _>>()?;

            for nid in neighbor_ids {
                if visited.insert(nid.clone()) {
                    nodes.push(self.get_node(&nid)?);
                    queue.push_back((nid, d + 1));
                }
            }
        }

        // Collect edges where both source and target are in the visited set
        let conn = self.pool.get()?;
        let mut edges = Vec::new();
        for node_id in &visited {
            let mut stmt = conn.prepare(
                "SELECT id FROM edges WHERE source_id = ?1",
            )?;
            let edge_ids: Vec<EdgeId> = stmt
                .query_map(params![node_id.0], |row| {
                    let eid: String = row.get(0)?;
                    Ok(EdgeId(eid))
                })?
                .collect::<Result<Vec<_>, _>>()?;

            for eid in edge_ids {
                let edge = self.get_edge(&eid)?;
                if visited.contains(&edge.target) {
                    edges.push(edge);
                }
            }
        }

        Ok((nodes, edges))
    }
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p graphirm-graph store::tests::subgraph_returns -- --nocapture 2>&1`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/graph/src/store.rs
git commit -m "feat(graph): implement subgraph() extraction with BFS and edge collection"
```

---

## Task 13: Implement pagerank() via petgraph

- [x] Complete

**Files:**
- Modify: `crates/graph/src/store.rs`

**Step 1: Write the failing test**

Add to the test module in `store.rs`:

```rust
    #[test]
    fn pagerank_known_topology() {
        let store = GraphStore::open_memory().unwrap();

        // Star topology: B, C, D all point to A
        // A should have the highest PageRank
        let a = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(), content: "hub".to_string(), token_count: None,
        }));
        let b = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(), content: "spoke1".to_string(), token_count: None,
        }));
        let c = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(), content: "spoke2".to_string(), token_count: None,
        }));
        let d = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(), content: "spoke3".to_string(), token_count: None,
        }));

        let a_id = a.id.clone();
        let b_id = b.id.clone();
        let c_id = c.id.clone();
        let d_id = d.id.clone();

        store.add_node(a).unwrap();
        store.add_node(b).unwrap();
        store.add_node(c).unwrap();
        store.add_node(d).unwrap();

        store.add_edge(GraphEdge::new(EdgeType::RespondsTo, b_id.clone(), a_id.clone())).unwrap();
        store.add_edge(GraphEdge::new(EdgeType::RespondsTo, c_id.clone(), a_id.clone())).unwrap();
        store.add_edge(GraphEdge::new(EdgeType::RespondsTo, d_id.clone(), a_id.clone())).unwrap();

        let scores = store.pagerank().unwrap();
        assert!(!scores.is_empty());

        // A should have the highest score
        let a_score = scores.iter().find(|(id, _)| *id == a_id).unwrap().1;
        let b_score = scores.iter().find(|(id, _)| *id == b_id).unwrap().1;
        assert!(a_score > b_score, "hub should rank higher than spokes: a={a_score} b={b_score}");
    }

    #[test]
    fn pagerank_empty_graph() {
        let store = GraphStore::open_memory().unwrap();
        let scores = store.pagerank().unwrap();
        assert!(scores.is_empty());
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p graphirm-graph store::tests::pagerank -- --nocapture 2>&1`
Expected: FAIL — `pagerank` doesn't exist.

**Step 3: Implement pagerank()**

Add to the `impl GraphStore` block:

```rust
    /// Compute PageRank scores for all nodes in the graph.
    /// Returns a vector of (NodeId, score) pairs sorted by score descending.
    pub fn pagerank(&self) -> Result<Vec<(NodeId, f64)>, GraphError> {
        let graph = self.graph.read().map_err(|_| GraphError::LockPoisoned)?;
        let indices = self.node_indices.read().map_err(|_| GraphError::LockPoisoned)?;

        if graph.node_count() == 0 {
            return Ok(Vec::new());
        }

        let damping = 0.85_f64;
        let iterations = 100;
        let n = graph.node_count() as f64;
        let initial = 1.0 / n;

        // Map NodeIndex -> score
        let node_indices_vec: Vec<petgraph::graph::NodeIndex> =
            graph.node_indices().collect();
        let mut scores: HashMap<petgraph::graph::NodeIndex, f64> = node_indices_vec
            .iter()
            .map(|&idx| (idx, initial))
            .collect();

        for _ in 0..iterations {
            let mut new_scores: HashMap<petgraph::graph::NodeIndex, f64> = node_indices_vec
                .iter()
                .map(|&idx| (idx, (1.0 - damping) / n))
                .collect();

            for &idx in &node_indices_vec {
                let out_degree = graph
                    .neighbors_directed(idx, petgraph::Direction::Outgoing)
                    .count();
                if out_degree == 0 {
                    // Distribute dangling node's rank equally
                    let share = scores[&idx] / n;
                    for &other in &node_indices_vec {
                        *new_scores.get_mut(&other).unwrap() += damping * share;
                    }
                } else {
                    let share = scores[&idx] / out_degree as f64;
                    for neighbor in graph.neighbors_directed(idx, petgraph::Direction::Outgoing) {
                        *new_scores.get_mut(&neighbor).unwrap() += damping * share;
                    }
                }
            }

            scores = new_scores;
        }

        // Map back to NodeIds
        let idx_to_id: HashMap<petgraph::graph::NodeIndex, NodeId> = indices
            .iter()
            .map(|(nid, &idx)| (idx, nid.clone()))
            .collect();

        let mut result: Vec<(NodeId, f64)> = scores
            .into_iter()
            .filter_map(|(idx, score)| idx_to_id.get(&idx).map(|nid| (nid.clone(), score)))
            .collect();

        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(result)
    }
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p graphirm-graph store::tests::pagerank -- --nocapture 2>&1`
Expected: Both tests pass.

**Step 5: Commit**

```bash
git add crates/graph/src/store.rs
git commit -m "feat(graph): implement pagerank() using iterative power method on petgraph"
```

---

## Task 14: Thread safety tests — Send + Sync + concurrent access

- [x] Complete

**Files:**
- Modify: `crates/graph/src/store.rs`

**Step 1: Write the compile-time Send + Sync test**

Add to the test module in `store.rs`:

```rust
    #[test]
    fn graph_store_is_send_and_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<GraphStore>();
        assert_sync::<GraphStore>();
    }
```

**Step 2: Run the test**

Run: `cargo test -p graphirm-graph store::tests::graph_store_is_send_and_sync -- --nocapture 2>&1`
Expected: PASS (r2d2::Pool and Arc<RwLock<>> are Send + Sync).

**Step 3: Write a concurrent read/write test**

Add to the test module:

```rust
    #[test]
    fn concurrent_reads_and_writes() {
        use std::sync::Arc;
        use std::thread;

        let store = Arc::new(GraphStore::open_memory().unwrap());
        let mut handles = Vec::new();

        // Spawn 10 writers
        for i in 0..10 {
            let store = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                let node = GraphNode::new(NodeType::Interaction(InteractionData {
                    role: "user".to_string(),
                    content: format!("message {i}"),
                    token_count: None,
                }));
                store.add_node(node).unwrap();
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Verify all 10 nodes exist
        let graph = store.graph.read().unwrap();
        assert_eq!(graph.node_count(), 10);

        // Spawn 10 concurrent readers
        let indices: Vec<NodeId> = store
            .node_indices
            .read()
            .unwrap()
            .keys()
            .cloned()
            .collect();

        let mut read_handles = Vec::new();
        for id in indices {
            let store = Arc::clone(&store);
            read_handles.push(thread::spawn(move || {
                let node = store.get_node(&id).unwrap();
                assert!(matches!(node.node_type, NodeType::Interaction(_)));
            }));
        }

        for h in read_handles {
            h.join().unwrap();
        }
    }
```

**Step 4: Run the test**

Run: `cargo test -p graphirm-graph store::tests::concurrent -- --nocapture 2>&1`
Expected: PASS

**Step 5: Run full test suite**

Run: `cargo test -p graphirm-graph -- --nocapture 2>&1`
Expected: All tests pass (nodes, edges, store — should be 30+ tests total).

**Step 6: Run clippy and fmt**

Run: `cargo clippy -p graphirm-graph --all-targets 2>&1`
Expected: No errors. Fix any warnings.

Run: `cargo fmt -p graphirm-graph -- --check 2>&1`
Expected: No formatting issues.

**Step 7: Commit**

```bash
git add crates/graph/src/store.rs
git commit -m "test(graph): add Send+Sync assertion and concurrent read/write stress test"
```

---

## Final Checklist

After all 14 tasks, verify:

```bash
# All graph crate tests pass
cargo test -p graphirm-graph -- --nocapture

# Full workspace still compiles
cargo build

# No lint issues
cargo clippy --all-targets --all-features

# Formatting clean
cargo fmt --all -- --check
```

Expected final test count for `graphirm-graph`: ~35 tests covering:
- NodeId/EdgeId newtypes (7 tests)
- Node data structs and serde (7 tests)
- EdgeType and GraphEdge (8 tests)
- GraphStore CRUD (8+ tests)
- Traversal and algorithms (6+ tests)
- Thread safety (2 tests)
