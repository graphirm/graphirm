# Session Restoration

## Overview

Sessions persisted to the graph database are automatically restored on server startup. When you restart the graphirm server, all previous sessions are available via the `/api/sessions` endpoint without any manual recovery steps.

## How It Works

On startup, the server:
1. Queries the `GraphStore` for all `Agent`-type nodes
2. Reconstructs `SessionMetadata` from each node's data
3. Makes sessions available via the REST API

Restored sessions retain their status (Running, Idle, Completed, Failed) from their last recorded state.

## Architecture

### Data Model

Sessions are represented as `Agent` nodes in the graph. Each node contains:
- Session ID (unique identifier)
- Name (human-readable label)
- Model (LLM used)
- Created timestamp
- Status (Running, Idle, Completed, Failed)

### API Integration

The `/api/sessions` endpoint automatically includes restored sessions:

```bash
GET /api/sessions
```

Response:
```json
[
  {
    "session_id": "abc123",
    "name": "auth-refactor",
    "model": "claude-sonnet-4",
    "created_at": "2026-03-06T08:00:00Z",
    "status": "Completed"
  }
]
```

### Implementation

Session restoration happens in three phases:

1. **Query Phase** (`GraphStore::get_agent_nodes()`)
   - Retrieves all Agent nodes from SQLite
   - Ordered by creation time (newest first)

2. **Mapping Phase** (`restore_sessions_from_graph()`)
   - Maps internal `AgentStatus` to API `SessionStatus`
   - Constructs `SessionMetadata` from node data
   - Populates sessions registry

3. **Startup Phase** (server initialization)
   - Called after GraphStore initialization
   - Errors logged but don't crash server
   - Restored session count logged

## Guarantees

- ✅ No data loss on restart
- ✅ All previous sessions automatically available
- ✅ Session history preserved (read-only until re-attached)
- ✅ Zero manual recovery steps required

## Testing

Comprehensive tests cover:
- Empty graph (no sessions)
- Single and multiple sessions
- All status types (Running, Idle, Completed, Failed)
- End-to-end restoration flow

Run tests:
```bash
cargo test --test test_session_restore_api
cargo test --test test_e2e_session_restore
```
