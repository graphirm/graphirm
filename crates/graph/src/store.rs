use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use petgraph::stable_graph::{NodeIndex, StableGraph};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::params;

use petgraph::Direction;

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
        let manager = SqliteConnectionManager::file(path).with_init(|conn| {
            conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
            Ok(())
        });
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
        let manager = SqliteConnectionManager::memory().with_init(|conn| {
            conn.execute_batch("PRAGMA foreign_keys=ON;")?;
            Ok(())
        });
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

            CREATE TABLE IF NOT EXISTS embeddings (
                node_id TEXT PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE,
                embedding BLOB NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
            CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
            CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
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
            if let (Some(&src_idx), Some(&tgt_idx)) = (indices.get(&source), indices.get(&target)) {
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
                rusqlite::Error::QueryReturnedNoRows => GraphError::NodeNotFound(id.0.clone()),
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

    /// Return all Agent nodes from the graph, ordered by creation time (newest first).
    ///
    /// Used during server startup to restore sessions from the persistent graph.
    pub fn get_agent_nodes(&self) -> Result<Vec<(GraphNode, crate::nodes::AgentData)>, GraphError> {
        let conn = self.pool.get()?;
        let mut stmt = conn.prepare(
            "SELECT id, data, metadata, created_at, updated_at
             FROM nodes
             WHERE node_type = 'agent'
             ORDER BY created_at DESC",
        )?;

        let rows: Vec<_> = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let data: String = row.get(1)?;
                let metadata: String = row.get(2)?;
                let created_at: String = row.get(3)?;
                let updated_at: String = row.get(4)?;
                Ok((id, data, metadata, created_at, updated_at))
            })?
            .collect::<Result<_, _>>()?;

        let mut result = Vec::new();
        for (id, data, metadata, created_at_str, updated_at_str) in rows {
            let node_type: NodeType = serde_json::from_str(&data)?;

            let agent_data = match &node_type {
                NodeType::Agent(d) => d.clone(),
                _ => continue,
            };

            let metadata: serde_json::Value = serde_json::from_str(&metadata)
                .unwrap_or(serde_json::Value::Object(Default::default()));

            let created_at = chrono::DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .unwrap_or_else(|_| chrono::Utc::now());

            let updated_at = chrono::DateTime::parse_from_rfc3339(&updated_at_str)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .unwrap_or_else(|_| chrono::Utc::now());

            let node = GraphNode {
                id: NodeId(id),
                node_type,
                metadata,
                created_at,
                updated_at,
            };

            result.push((node, agent_data));
        }

        Ok(result)
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
                rusqlite::Error::QueryReturnedNoRows => GraphError::EdgeNotFound(id.0.clone()),
                other => GraphError::Sqlite(other),
            })?;

        let (id_str, edge_type_str, source, target, weight, metadata, created_at) = row;

        let edge_type: EdgeType = serde_json::from_str(&format!("\"{edge_type_str}\""))?;
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

    /// Return all edges (with full metadata) that touch `id` in either direction.
    ///
    /// Use this instead of calling `neighbors` once per edge type when you need
    /// to aggregate over all edge types — it issues a single DB query instead of
    /// one per edge type × direction.
    pub fn edges_for_node(&self, id: &NodeId) -> Result<Vec<GraphEdge>, GraphError> {
        let conn = self.pool.get()?;
        let mut stmt = conn.prepare(
            "SELECT id, edge_type, source_id, target_id, weight, metadata, created_at
             FROM edges
             WHERE source_id = ?1 OR target_id = ?1",
        )?;

        let edges = stmt
            .query_map(params![id.0], |row| {
                let id: String = row.get(0)?;
                let edge_type: String = row.get(1)?;
                let source: String = row.get(2)?;
                let target: String = row.get(3)?;
                let weight: f64 = row.get(4)?;
                let metadata: String = row.get(5)?;
                let created_at: String = row.get(6)?;
                Ok((id, edge_type, source, target, weight, metadata, created_at))
            })?
            .filter_map(|r| r.ok())
            .filter_map(|(id_str, et_str, src, tgt, w, meta, ts)| {
                let edge_type: EdgeType = serde_json::from_str(&format!("\"{et_str}\"")).ok()?;
                let metadata: serde_json::Value = serde_json::from_str(&meta).ok()?;
                let created_at = chrono::DateTime::parse_from_rfc3339(&ts)
                    .ok()?
                    .with_timezone(&chrono::Utc);
                Some(GraphEdge {
                    id: EdgeId(id_str),
                    edge_type,
                    source: NodeId(src),
                    target: NodeId(tgt),
                    weight: w,
                    metadata,
                    created_at,
                })
            })
            .collect();

        Ok(edges)
    }

    pub fn neighbors(
        &self,
        id: &NodeId,
        edge_type: Option<EdgeType>,
        direction: Direction,
    ) -> Result<Vec<GraphNode>, GraphError> {
        let neighbor_ids = {
            let conn = self.pool.get()?;

            let (query, has_edge_filter) = match (&direction, &edge_type) {
                (Direction::Outgoing, Some(_)) => (
                    "SELECT target_id FROM edges WHERE source_id = ?1 AND edge_type = ?2",
                    true,
                ),
                (Direction::Outgoing, None) => {
                    ("SELECT target_id FROM edges WHERE source_id = ?1", false)
                }
                (Direction::Incoming, Some(_)) => (
                    "SELECT source_id FROM edges WHERE target_id = ?1 AND edge_type = ?2",
                    true,
                ),
                (Direction::Incoming, None) => {
                    ("SELECT source_id FROM edges WHERE target_id = ?1", false)
                }
            };

            let mut stmt = conn.prepare(query)?;
            if has_edge_filter {
                let et_str = edge_type.unwrap().as_str();
                stmt.query_map(params![id.0, et_str], |row| {
                    let nid: String = row.get(0)?;
                    Ok(NodeId(nid))
                })?
                .collect::<Result<Vec<_>, _>>()?
            } else {
                stmt.query_map(params![id.0], |row| {
                    let nid: String = row.get(0)?;
                    Ok(NodeId(nid))
                })?
                .collect::<Result<Vec<_>, _>>()?
            }
        };

        let mut neighbors = Vec::new();
        for nid in neighbor_ids {
            neighbors.push(self.get_node(&nid)?);
        }

        Ok(neighbors)
    }

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

            let neighbor_ids = {
                let conn = self.pool.get()?;
                let placeholders: String = edge_type_strs
                    .iter()
                    .map(|_| "?")
                    .collect::<Vec<_>>()
                    .join(", ");
                let query = format!(
                    "SELECT target_id FROM edges WHERE source_id = ?1 AND edge_type IN ({placeholders})"
                );

                let mut stmt = conn.prepare(&query)?;

                use rusqlite::types::ToSql;
                let mut params_vec: Vec<Box<dyn ToSql>> = Vec::new();
                params_vec.push(Box::new(current_id.0.clone()));
                for et in &edge_type_strs {
                    params_vec.push(Box::new(et.to_string()));
                }
                let params_refs: Vec<&dyn ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();

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

    /// Walk `RespondsTo` edges backwards from `leaf_id` to the root of the conversation.
    /// Returns nodes newest-first: [leaf, ..., root].
    pub fn conversation_thread(&self, leaf_id: &NodeId) -> Result<Vec<GraphNode>, GraphError> {
        let mut thread = Vec::new();
        let mut current_id = leaf_id.clone();

        loop {
            let node = self.get_node(&current_id)?;
            thread.push(node);

            let parent_id: Option<String> = {
                let conn = self.pool.get()?;
                conn.query_row(
                    "SELECT target_id FROM edges WHERE source_id = ?1 AND edge_type = 'responds_to' LIMIT 1",
                    params![current_id.0],
                    |row| row.get(0),
                )
                .ok()
            };

            match parent_id {
                Some(pid) => current_id = NodeId(pid),
                None => break,
            }
        }

        Ok(thread)
    }

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

            let neighbor_ids = {
                let conn = self.pool.get()?;
                let mut stmt = conn.prepare("SELECT target_id FROM edges WHERE source_id = ?1")?;
                stmt.query_map(params![current_id.0], |row| {
                    let nid: String = row.get(0)?;
                    Ok(NodeId(nid))
                })?
                .collect::<Result<Vec<_>, _>>()?
            };

            for nid in neighbor_ids {
                if visited.insert(nid.clone()) {
                    nodes.push(self.get_node(&nid)?);
                    queue.push_back((nid, d + 1));
                }
            }
        }

        let mut edges = Vec::new();
        for node_id in &visited {
            let edge_ids = {
                let conn = self.pool.get()?;
                let mut stmt = conn.prepare("SELECT id FROM edges WHERE source_id = ?1")?;
                stmt.query_map(params![node_id.0], |row| {
                    let eid: String = row.get(0)?;
                    Ok(EdgeId(eid))
                })?
                .collect::<Result<Vec<_>, _>>()?
            };

            for eid in edge_ids {
                let edge = self.get_edge(&eid)?;
                if visited.contains(&edge.target) {
                    edges.push(edge);
                }
            }
        }

        Ok((nodes, edges))
    }

    /// Return total node count.
    pub fn node_count_db(&self) -> Result<u64, GraphError> {
        let conn = self.pool.get()?;
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM nodes", [], |r| r.get(0))?;
        Ok(count as u64)
    }

    /// Return total edge count.
    pub fn edge_count_db(&self) -> Result<u64, GraphError> {
        let conn = self.pool.get()?;
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM edges", [], |r| r.get(0))?;
        Ok(count as u64)
    }

    /// Return node counts grouped by type.
    pub fn node_counts_by_type(&self) -> Result<Vec<(String, u64)>, GraphError> {
        let conn = self.pool.get()?;
        let mut stmt = conn.prepare(
            "SELECT node_type, COUNT(*) FROM nodes GROUP BY node_type ORDER BY COUNT(*) DESC",
        )?;
        let rows = stmt
            .query_map([], |row| {
                let t: String = row.get(0)?;
                let c: i64 = row.get(1)?;
                Ok((t, c as u64))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Return the `limit` most recently created nodes, newest first.
    pub fn list_recent_nodes(&self, limit: usize) -> Result<Vec<GraphNode>, GraphError> {
        let conn = self.pool.get()?;
        let mut stmt = conn.prepare(
            "SELECT id, data, metadata, created_at, updated_at \
             FROM nodes ORDER BY created_at DESC LIMIT ?1",
        )?;

        let rows: Vec<(String, String, String, String, String)> = stmt
            .query_map([limit as i64], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        let mut nodes = Vec::with_capacity(rows.len());
        for (id_str, data, metadata, created_at_str, updated_at_str) in rows {
            let node_type: NodeType = serde_json::from_str(&data)?;
            let metadata: serde_json::Value = serde_json::from_str(&metadata)
                .unwrap_or(serde_json::Value::Object(Default::default()));
            let created_at = chrono::DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .unwrap_or_else(|_| chrono::Utc::now());
            let updated_at = chrono::DateTime::parse_from_rfc3339(&updated_at_str)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .unwrap_or_else(|_| chrono::Utc::now());
            nodes.push(GraphNode {
                id: NodeId(id_str),
                node_type,
                metadata,
                created_at,
                updated_at,
            });
        }
        Ok(nodes)
    }

    /// Store an embedding vector for a node. Overwrites any existing embedding.
    // Bytes are stored in native-endian order via bytemuck::cast_slice.
    // This is correct for single-machine SQLite but will silently corrupt
    // embeddings if the .db file is moved between architectures.
    pub fn set_embedding(&self, node_id: &NodeId, embedding: &[f32]) -> Result<(), GraphError> {
        let conn = self.pool.get()?;
        let bytes: &[u8] = bytemuck::cast_slice(embedding);
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (node_id, embedding) VALUES (?1, ?2)",
            params![node_id.0, bytes],
        )?;
        Ok(())
    }

    /// Retrieve the embedding vector for a node, if one exists.
    pub fn get_embedding(&self, node_id: &NodeId) -> Result<Option<Vec<f32>>, GraphError> {
        let conn = self.pool.get()?;
        let mut stmt = conn.prepare("SELECT embedding FROM embeddings WHERE node_id = ?1")?;
        let result = stmt.query_row(params![node_id.0], |row| {
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
        let conn = self.pool.get()?;
        let mut stmt = conn.prepare("SELECT node_id, embedding FROM embeddings")?;
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

    /// Compute PageRank scores for all nodes in the graph.
    /// Returns a vector of (NodeId, score) pairs sorted by score descending.
    pub fn pagerank(&self) -> Result<Vec<(NodeId, f64)>, GraphError> {
        let graph = self.graph.read().map_err(|_| GraphError::LockPoisoned)?;
        let indices = self
            .node_indices
            .read()
            .map_err(|_| GraphError::LockPoisoned)?;

        if graph.node_count() == 0 {
            return Ok(Vec::new());
        }

        let damping = 0.85_f64;
        let iterations = 100;
        let n = graph.node_count() as f64;
        let initial = 1.0 / n;

        let node_indices_vec: Vec<petgraph::graph::NodeIndex> = graph.node_indices().collect();
        let mut scores: HashMap<petgraph::graph::NodeIndex, f64> =
            node_indices_vec.iter().map(|&idx| (idx, initial)).collect();

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
            status: crate::nodes::TaskStatus::Pending,
            priority: Some(1),
        }));
        let id = node.id.clone();
        store.add_node(node.clone()).unwrap();

        let mut updated = node;
        updated.node_type = NodeType::Task(crate::nodes::TaskData {
            title: "Updated".to_string(),
            description: "Second version".to_string(),
            status: crate::nodes::TaskStatus::Running,
            priority: Some(2),
        });
        store.update_node(&id, updated).unwrap();

        let fetched = store.get_node(&id).unwrap();
        match &fetched.node_type {
            NodeType::Task(data) => {
                assert_eq!(data.title, "Updated");
                assert_eq!(data.status, crate::nodes::TaskStatus::Running);
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

        let edge =
            GraphEdge::new(EdgeType::RespondsTo, n2_id.clone(), n1_id.clone()).with_weight(0.8);
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

    #[test]
    fn pagerank_known_topology() {
        let store = GraphStore::open_memory().unwrap();

        let a = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "hub".to_string(),
            token_count: None,
        }));
        let b = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "spoke1".to_string(),
            token_count: None,
        }));
        let c = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "spoke2".to_string(),
            token_count: None,
        }));
        let d = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "spoke3".to_string(),
            token_count: None,
        }));

        let a_id = a.id.clone();
        let b_id = b.id.clone();
        let c_id = c.id.clone();
        let d_id = d.id.clone();

        store.add_node(a).unwrap();
        store.add_node(b).unwrap();
        store.add_node(c).unwrap();
        store.add_node(d).unwrap();

        store
            .add_edge(GraphEdge::new(
                EdgeType::RespondsTo,
                b_id.clone(),
                a_id.clone(),
            ))
            .unwrap();
        store
            .add_edge(GraphEdge::new(
                EdgeType::RespondsTo,
                c_id.clone(),
                a_id.clone(),
            ))
            .unwrap();
        store
            .add_edge(GraphEdge::new(
                EdgeType::RespondsTo,
                d_id.clone(),
                a_id.clone(),
            ))
            .unwrap();

        let scores = store.pagerank().unwrap();
        assert!(!scores.is_empty());

        let a_score = scores.iter().find(|(id, _)| *id == a_id).unwrap().1;
        let b_score = scores.iter().find(|(id, _)| *id == b_id).unwrap().1;
        assert!(
            a_score > b_score,
            "hub should rank higher than spokes: a={a_score} b={b_score}"
        );
    }

    #[test]
    fn pagerank_empty_graph() {
        let store = GraphStore::open_memory().unwrap();
        let scores = store.pagerank().unwrap();
        assert!(scores.is_empty());
    }

    #[test]
    fn subgraph_returns_nodes_and_edges_within_depth() {
        let store = GraphStore::open_memory().unwrap();

        let a = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "a".to_string(),
            token_count: None,
        }));
        let b = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".to_string(),
            content: "b".to_string(),
            token_count: None,
        }));
        let c = GraphNode::new(NodeType::Content(crate::nodes::ContentData {
            content_type: "file".to_string(),
            path: None,
            body: "c".to_string(),
            language: None,
        }));
        let d = GraphNode::new(NodeType::Content(crate::nodes::ContentData {
            content_type: "file".to_string(),
            path: None,
            body: "d".to_string(),
            language: None,
        }));

        let a_id = a.id.clone();
        let b_id = b.id.clone();
        let c_id = c.id.clone();
        let d_id = d.id.clone();

        store.add_node(a).unwrap();
        store.add_node(b).unwrap();
        store.add_node(c).unwrap();
        store.add_node(d).unwrap();

        store
            .add_edge(GraphEdge::new(
                EdgeType::RespondsTo,
                a_id.clone(),
                b_id.clone(),
            ))
            .unwrap();
        store
            .add_edge(GraphEdge::new(
                EdgeType::RespondsTo,
                b_id.clone(),
                c_id.clone(),
            ))
            .unwrap();
        store
            .add_edge(GraphEdge::new(
                EdgeType::RespondsTo,
                c_id.clone(),
                d_id.clone(),
            ))
            .unwrap();

        let (nodes, edges) = store.subgraph(&a_id, 1).unwrap();
        assert_eq!(nodes.len(), 2);
        assert_eq!(edges.len(), 1);

        let (nodes, edges) = store.subgraph(&a_id, 2).unwrap();
        assert_eq!(nodes.len(), 3);
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn conversation_thread_walks_backwards() {
        let store = GraphStore::open_memory().unwrap();

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

        store
            .add_edge(GraphEdge::new(
                EdgeType::RespondsTo,
                ids[1].clone(),
                ids[0].clone(),
            ))
            .unwrap();
        store
            .add_edge(GraphEdge::new(
                EdgeType::RespondsTo,
                ids[2].clone(),
                ids[1].clone(),
            ))
            .unwrap();
        store
            .add_edge(GraphEdge::new(
                EdgeType::RespondsTo,
                ids[3].clone(),
                ids[2].clone(),
            ))
            .unwrap();
        store
            .add_edge(GraphEdge::new(
                EdgeType::RespondsTo,
                ids[4].clone(),
                ids[3].clone(),
            ))
            .unwrap();

        let thread = store.conversation_thread(&ids[4]).unwrap();

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

    #[test]
    fn traverse_bfs_with_depth() {
        let store = GraphStore::open_memory().unwrap();

        let make_content = |name: &str| {
            GraphNode::new(NodeType::Content(crate::nodes::ContentData {
                content_type: "file".to_string(),
                path: Some(name.to_string()),
                body: "".to_string(),
                language: None,
            }))
        };

        let a = make_content("a");
        let a_id = a.id.clone();
        let b = make_content("b");
        let b_id = b.id.clone();
        let c = make_content("c");
        let c_id = c.id.clone();
        let d = make_content("d");
        let d_id = d.id.clone();

        store.add_node(a).unwrap();
        store.add_node(b).unwrap();
        store.add_node(c).unwrap();
        store.add_node(d).unwrap();

        store
            .add_edge(GraphEdge::new(
                EdgeType::Contains,
                a_id.clone(),
                b_id.clone(),
            ))
            .unwrap();
        store
            .add_edge(GraphEdge::new(
                EdgeType::Contains,
                b_id.clone(),
                c_id.clone(),
            ))
            .unwrap();
        store
            .add_edge(GraphEdge::new(
                EdgeType::Contains,
                c_id.clone(),
                d_id.clone(),
            ))
            .unwrap();

        let depth1 = store.traverse(&a_id, &[EdgeType::Contains], 1).unwrap();
        assert_eq!(depth1.len(), 1);
        assert_eq!(depth1[0].id, b_id);

        let depth2 = store.traverse(&a_id, &[EdgeType::Contains], 2).unwrap();
        assert_eq!(depth2.len(), 2);

        let all = store.traverse(&a_id, &[EdgeType::Contains], 10).unwrap();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn traverse_filters_edge_types() {
        let store = GraphStore::open_memory().unwrap();

        let a = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "a".to_string(),
            token_count: None,
        }));
        let b = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".to_string(),
            content: "b".to_string(),
            token_count: None,
        }));
        let c = GraphNode::new(NodeType::Content(crate::nodes::ContentData {
            content_type: "file".to_string(),
            path: None,
            body: "c".to_string(),
            language: None,
        }));

        let a_id = a.id.clone();
        let b_id = b.id.clone();
        let c_id = c.id.clone();
        store.add_node(a).unwrap();
        store.add_node(b).unwrap();
        store.add_node(c).unwrap();

        store
            .add_edge(GraphEdge::new(
                EdgeType::RespondsTo,
                a_id.clone(),
                b_id.clone(),
            ))
            .unwrap();
        store
            .add_edge(GraphEdge::new(EdgeType::Reads, a_id.clone(), c_id.clone()))
            .unwrap();

        let result = store.traverse(&a_id, &[EdgeType::RespondsTo], 5).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, b_id);
    }

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

        let all = store
            .neighbors(&agent_id, None, Direction::Outgoing)
            .unwrap();
        assert_eq!(all.len(), 2);

        let reads = store
            .neighbors(&agent_id, Some(EdgeType::Reads), Direction::Outgoing)
            .unwrap();
        assert_eq!(reads.len(), 1);
        assert_eq!(reads[0].id, c1_id);

        let incoming = store.neighbors(&c1_id, None, Direction::Incoming).unwrap();
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0].id, agent_id);
    }

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

        store.delete_node(&n2_id).unwrap();

        assert!(store.get_edge(&e1_id).is_err());
        assert!(store.get_edge(&e2_id).is_err());

        let graph = store.graph.read().unwrap();
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn graph_store_is_send_and_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<GraphStore>();
        assert_sync::<GraphStore>();
    }

    #[test]
    fn concurrent_reads_and_writes() {
        use std::sync::Arc;
        use std::thread;

        let store = Arc::new(GraphStore::open_memory().unwrap());
        let mut handles = Vec::new();

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

        let graph = store.graph.read().unwrap();
        assert_eq!(graph.node_count(), 10);

        let indices: Vec<NodeId> = store.node_indices.read().unwrap().keys().cloned().collect();

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

    use crate::nodes::KnowledgeData;

    #[test]
    fn test_store_and_retrieve_embedding() {
        let store = GraphStore::open_memory().unwrap();
        let node_id = store
            .add_node(GraphNode::new(NodeType::Knowledge(KnowledgeData {
                entity: "test_entity".to_string(),
                entity_type: "function".to_string(),
                summary: "test".to_string(),
                confidence: 0.9,
            })))
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
        let store = GraphStore::open_memory().unwrap();
        let node_id = store
            .add_node(GraphNode::new(NodeType::Knowledge(KnowledgeData {
                entity: "no_embedding".to_string(),
                entity_type: "function".to_string(),
                summary: "test".to_string(),
                confidence: 0.9,
            })))
            .unwrap();

        let retrieved = store.get_embedding(&node_id).unwrap();
        assert!(retrieved.is_none());
    }

    #[test]
    fn test_get_all_embeddings() {
        let store = GraphStore::open_memory().unwrap();

        let id1 = store
            .add_node(GraphNode::new(NodeType::Knowledge(KnowledgeData {
                entity: "e1".to_string(),
                entity_type: "function".to_string(),
                summary: "test".to_string(),
                confidence: 0.9,
            })))
            .unwrap();
        let id2 = store
            .add_node(GraphNode::new(NodeType::Knowledge(KnowledgeData {
                entity: "e2".to_string(),
                entity_type: "function".to_string(),
                summary: "test".to_string(),
                confidence: 0.9,
            })))
            .unwrap();
        let _id3 = store
            .add_node(GraphNode::new(NodeType::Knowledge(KnowledgeData {
                entity: "no_embed".to_string(),
                entity_type: "function".to_string(),
                summary: "test".to_string(),
                confidence: 0.9,
            })))
            .unwrap();

        store.set_embedding(&id1, &[1.0, 2.0, 3.0]).unwrap();
        store.set_embedding(&id2, &[4.0, 5.0, 6.0]).unwrap();

        let all = store.get_all_embeddings().unwrap();
        assert_eq!(all.len(), 2);
        assert!(all.iter().any(|(id, _)| *id == id1));
        assert!(all.iter().any(|(id, _)| *id == id2));
    }

    #[test]
    fn test_embedding_roundtrip_large_vector() {
        let store = GraphStore::open_memory().unwrap();
        let node_id = store
            .add_node(GraphNode::new(NodeType::Knowledge(KnowledgeData {
                entity: "large".to_string(),
                entity_type: "function".to_string(),
                summary: "test".to_string(),
                confidence: 0.9,
            })))
            .unwrap();

        // 1536 dimensions matches OpenAI ada-002
        let embedding: Vec<f32> = (0..1536).map(|i| i as f32 * 0.001).collect();
        store.set_embedding(&node_id, &embedding).unwrap();

        let retrieved = store.get_embedding(&node_id).unwrap().unwrap();
        assert_eq!(retrieved.len(), 1536);
        assert!((retrieved[0] - 0.0).abs() < f32::EPSILON);
        assert!((retrieved[1535] - 1.535).abs() < 0.001);
    }

    #[test]
    fn test_delete_node_removes_embedding() {
        let store = GraphStore::open_memory().unwrap();
        let node_id = store
            .add_node(GraphNode::new(NodeType::Knowledge(KnowledgeData {
                entity: "cascade_test".to_string(),
                entity_type: "function".to_string(),
                summary: "will be deleted".to_string(),
                confidence: 0.8,
            })))
            .unwrap();

        store.set_embedding(&node_id, &[1.0, 2.0, 3.0]).unwrap();

        assert!(store.get_embedding(&node_id).unwrap().is_some());

        store.delete_node(&node_id).unwrap();

        let all = store.get_all_embeddings().unwrap();
        assert!(all.iter().all(|(id, _)| *id != node_id));
    }
}
