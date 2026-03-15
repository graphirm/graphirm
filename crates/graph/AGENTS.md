# graphirm-graph

Graph persistence layer. SQLite-backed storage via `rusqlite` + `r2d2` connection pool, with an
in-memory `petgraph::StableGraph` for traversals (PageRank, BFS, subgraph extraction). HNSW vector
index via `instant-distance` for knowledge node similarity search.

---

## Key Components

| File | What |
|------|------|
| `store.rs` | `GraphStore` — connection pool, node/edge CRUD, session management, PageRank, BFS |
| `nodes.rs` | `GraphNode`, `NodeType` (5 variants), `NodeId` newtype |
| `edges.rs` | `GraphEdge`, `EdgeType` (12 variants), `EdgeId` newtype |
| `query.rs` | Subgraph extraction, traversal helpers |
| `vector.rs` | `VectorIndex` — HNSW wrapper, similarity search for Knowledge nodes |
| `corpus.rs` | `export_corpus_to_jsonl`, `CorpusTurn` — corpus export for eval/analysis |
| `error.rs` | `GraphError` enum |

**Node types:** `Interaction`, `Agent`, `Content`, `Task`, `Knowledge`

**Edge types:** `RespondsTo`, `SpawnedBy`, `DelegatesTo`, `DependsOn`, `Produces`, `Reads`,
`Modifies`, `Summarizes`, `Contains`, `FollowsUp`, `Steers`, `RelatesTo`

---

## Integration Points

**Used by:** `graphirm-agent` (context scoring, session persistence), `graphirm-tools` (Content
node creation on file ops), `graphirm-server` (graph query API), `graphirm-tui` (graph explorer panel)

**Depends on:** `rusqlite`, `petgraph`, `r2d2` + `r2d2_sqlite`, `instant-distance`, `uuid`, `chrono`

**Important:** `Arc<RwLock<StableGraph>>` for in-memory graph — acquire read/write locks briefly,
never hold across an await point.

---

## How to Test

```bash
cargo test -p graphirm-graph
```

Integration test: `tests/test_session_restore.rs` — creates a graph, writes nodes/edges, drops
and reopens the DB, verifies data survives.
