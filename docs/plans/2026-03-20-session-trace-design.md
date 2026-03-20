# Session Flow Traces — Design

> **For Claude:** use `executing-plans` skill after the implementation plan is written.

## Problem

The agent can search for *facts* (Knowledge nodes via `graph_query semantic`) but cannot search
for *decision patterns* — the actual sequence of tool calls, file reads/edits, and reasoning that
led to a past outcome. "How did I debug the auth bug last time?" is unanswerable today.

No other agent has queryable decision history because no other agent stores tool calls as a
traversable graph.

## Solution

A `session_trace` non-destructive tool with two modes:

- **`search`** — semantic (or keyword fallback) query across all sessions; returns ranked
  decision traces anchored by Knowledge nodes
- **`replay`** — full decision trace for a specific session

### Retrieval: Knowledge-Anchored Trace Retrieval

Uses the *existing* Knowledge nodes and HNSW index as the search anchor, then walks backwards
from matched Knowledge nodes to their parent Interaction chains via graph edges. Knowledge nodes
are already embedded, tagged with `session_id`, and describe what the agent did. Instead of
returning the Knowledge node itself (like `graph_query semantic`), follow
`Contains`/`Produces`/`Modifies` edges back to Interaction nodes and reconstruct the local
decision trace.

**Why this over alternatives:**
- Reuses existing HNSW index — no new embedding pipeline
- Works today with any configured embedding provider
- Knowledge nodes are the "what happened" summary layer; this adds "show me the *how*"
- Graceful fallback to keyword search when no embeddings are configured

## Tool Interface

### Parameters

```json
{
  "mode": { "type": "string", "enum": ["search", "replay"] },
  "query": { "type": "string", "description": "Natural language query (required for search)" },
  "session_id": { "type": "string", "description": "Session to replay (required for replay)" },
  "detail": { "type": "string", "enum": ["compact", "full"], "default": "compact" },
  "limit": { "type": "integer", "default": 5, "description": "Max traces to return (search mode)" },
  "context_turns": { "type": "integer", "default": 3, "description": "Turns around each match (search mode)" }
}
```

### Output Format

**Compact** — one line per turn:

```
=== Session "Debug auth middleware" (2026-03-18, sim=0.87) ===
  turn 1: [user] "the JWT validation is failing on refresh tokens"
  turn 2: [assistant] read src/auth/jwt.rs → grep "refresh" → read src/auth/middleware.rs
  turn 3: [assistant] edit src/auth/jwt.rs (lines 42-58) → bash "cargo test"
  turn 4: [assistant] "Fixed: refresh token expiry was compared against issued_at..."
```

**Full** — includes assistant reasoning text, tool arguments, and file content snippets.

## Architecture

### Search Mode Flow

1. Query existing HNSW index via `KnowledgeRetriever` (or fall back to `GraphStore::search_knowledge`)
2. Group matched Knowledge nodes by `session_id` from metadata
3. For each session, walk from Knowledge node → parent Interaction via reverse edge traversal
4. From that Interaction, walk `RespondsTo` chain ±`context_turns` hops to get the local trace
5. For each turn, extract tool calls from `metadata["tool_calls"]` and any `Reads`/`Modifies`/`Produces` edges to Content nodes
6. Format traces, rank sessions by aggregate similarity score
7. Return top-k

### Replay Mode Flow

1. Call `GraphStore::get_session_chain(session_id)` — all Interaction nodes for the session, chronological
2. For each Interaction, extract tool calls from metadata and linked Content nodes
3. Format the full trace

### New GraphStore Helper

`get_session_chain(session_id: &str) -> Result<Vec<GraphNode>, GraphError>`

```sql
SELECT * FROM nodes
WHERE node_type = 'interaction'
  AND json_extract(metadata, '$.session_id') = ?
ORDER BY created_at ASC
```

Avoids walking `RespondsTo` from the leaf (which requires knowing the leaf). Direct SQL query
is simpler and handles sessions where the chain might be incomplete.

### Trace Formatting

`format_turn_compact(node, graph)` — extracts:
- Role from `InteractionData.role`
- Tool calls from `metadata["tool_calls"]` → `name` + key args (file paths)
- User/assistant text truncated to ~80 chars

`format_turn_full(node, graph)` — adds:
- Full assistant text
- Tool arguments
- Linked Content node bodies (truncated)

## Key Files

| File | Change |
|------|--------|
| `crates/tools/src/session_trace.rs` | New — `SessionTraceTool` implementing `Tool` |
| `crates/tools/src/lib.rs` | Add `pub mod session_trace;` |
| `crates/graph/src/store.rs` | Add `get_session_chain(session_id)` |
| `src/main.rs` | Register in `build_tool_registry()` |

## Fallback Behavior

- No embedding provider → `search` falls back to keyword search on Knowledge nodes, output includes a note
- No Knowledge nodes for a session → trace reconstructed from Interaction chain alone
- Empty results → `ToolOutput::success("(no matching traces)")`, not an error
- `get_session_chain` for nonexistent session → empty vec, not an error

## Testing

- **`session_trace.rs` unit tests:** mode validation, compact formatting, full formatting, empty graph, limit, context_turns
- **Mock `KnowledgeRetriever`** for search mode (same pattern as `graph_query.rs` tests)
- **`store.rs` unit test:** `get_session_chain` returns chronological Interactions filtered by session_id

## Decisions

**Approaches considered:**
1. Graph-only keyword match on Interaction content — simple but no semantic matching
2. HNSW semantic search on Interaction embeddings — powerful but requires new embedding pipeline and index growth
3. Knowledge-anchored trace retrieval (chosen) — reuses existing infrastructure, delivers semantic search via Knowledge nodes as anchors

**Why #3:** It's the only approach that provides semantic search without new embedding infrastructure. The one-hop cost from Knowledge → Interaction is negligible. Knowledge extraction quality is the limiting factor, but it's already good enough for cross-session linking (Phase 16) which validates the approach.

**Detail levels:** Configurable (`compact`/`full`) defaulting to `compact` because the agent usually needs the pattern, not the full text. `full` exists for "show me exactly what happened."

**Two modes vs one:** `replay` comes nearly for free once the formatting logic exists. It adds value for "what happened in session X?" without the search overhead.
