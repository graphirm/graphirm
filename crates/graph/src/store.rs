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
                rusqlite::Error::QueryReturnedNoRows => {
                    GraphError::EdgeNotFound(id.0.clone())
                }
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

    pub fn neighbors(
        &self,
        id: &NodeId,
        edge_type: Option<EdgeType>,
        direction: Direction,
    ) -> Result<Vec<GraphNode>, GraphError> {
        let neighbor_ids = {
            let conn = self.pool.get()?;

            let (query, param) = match direction {
                Direction::Outgoing => (
                    match edge_type {
                        Some(et) => format!(
                            "SELECT target_id FROM edges WHERE source_id = ?1 AND edge_type = '{}'",
                            et.as_str()
                        ),
                        None => {
                            "SELECT target_id FROM edges WHERE source_id = ?1".to_string()
                        }
                    },
                    &id.0,
                ),
                Direction::Incoming => (
                    match edge_type {
                        Some(et) => format!(
                            "SELECT source_id FROM edges WHERE target_id = ?1 AND edge_type = '{}'",
                            et.as_str()
                        ),
                        None => {
                            "SELECT source_id FROM edges WHERE target_id = ?1".to_string()
                        }
                    },
                    &id.0,
                ),
            };

            let mut stmt = conn.prepare(&query)?;
            stmt.query_map(params![param], |row| {
                let nid: String = row.get(0)?;
                Ok(NodeId(nid))
            })?
            .collect::<Result<Vec<_>, _>>()?
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
                let mut stmt =
                    conn.prepare("SELECT target_id FROM edges WHERE source_id = ?1")?;
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
            .add_edge(GraphEdge::new(
                EdgeType::Reads,
                a_id.clone(),
                c_id.clone(),
            ))
            .unwrap();

        let result = store
            .traverse(&a_id, &[EdgeType::RespondsTo], 5)
            .unwrap();
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

        let incoming = store
            .neighbors(&c1_id, None, Direction::Incoming)
            .unwrap();
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
