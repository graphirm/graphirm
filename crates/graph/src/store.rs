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
