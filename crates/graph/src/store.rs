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

        let row = stmt
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

        let (id_str, data, metadata, created_at, updated_at) = row;
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

        conn.execute(
            "DELETE FROM edges WHERE source_id = ?1 OR target_id = ?1",
            params![id.0],
        )?;

        let rows = conn.execute("DELETE FROM nodes WHERE id = ?1", params![id.0])?;
        if rows == 0 {
            return Err(GraphError::NodeNotFound(id.0.clone()));
        }

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_memory_creates_tables() {
        let store = GraphStore::open_memory().unwrap();
        let conn = store.pool.get().unwrap();

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='nodes'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='edges'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

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

    #[test]
    fn update_node_changes_fields() {
        let store = GraphStore::open_memory().unwrap();
        let node = GraphNode::new(NodeType::Task(crate::nodes::TaskData {
            title: "Original".to_string(),
            description: "First version".to_string(),
            status: "pending".to_string(),
            priority: Some(1),
        }));
        let id = node.id.clone();
        store.add_node(node.clone()).unwrap();

        let mut updated = node;
        updated.node_type = NodeType::Task(crate::nodes::TaskData {
            title: "Updated".to_string(),
            description: "Second version".to_string(),
            status: "in_progress".to_string(),
            priority: Some(2),
        });
        store.update_node(&id, updated).unwrap();

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
}
