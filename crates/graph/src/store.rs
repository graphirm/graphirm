/// GraphStore: rusqlite + petgraph dual-write graph persistence.
///
/// Will hold an r2d2 connection pool for SQLite and an
/// Arc<RwLock<petgraph::Graph>> for in-memory traversals.
pub struct GraphStore;
