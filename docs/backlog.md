# Graphirm Backlog

Single source of truth for planned work. Completed items are recorded in `docs/completion-log.md` and `AGENTS.md` — not here.

**Current state:** Phases 0–14 complete. See `AGENTS.md` → Current State table.

---

## How to use this file

- Items are grouped by theme, not phase number
- Each item has a **size** (S / M / L) and a **priority** (P1 high / P2 medium / P3 low)
- When work starts, create a plan in `docs/plans/YYYY-MM-DD-<topic>.md` and link it here
- Mark items ✅ and move details to `docs/completion-log.md` when shipped

---

## Deployment & Operations

### Coolify on always-on spoke — P1 · M
Install Coolify on the persistent Hetzner spoke, connect to GitHub, auto-deploy on push to `main`, mount `/data` volume for the SQLite graph database. Hetzner snapshot for disaster recovery.
Plan: `docs/plans/2026-03-16-coolify-spoke-deployment.md`

### Wire `workspaces_root` on running server — P1 · S
Now that per-session workspaces exist (Phase 14), set `workspaces_root = "/root/workspaces"` in the spoke's `config/default.toml` so sessions get proper filesystem isolation in production.

### CI pipeline (GitHub Actions) — P2 · S
Run `cargo test` and `cargo clippy` on every push to `main` and every PR. Currently nothing catches regressions before they reach the spoke. A single workflow file covers it.

---

## UI (web-app)

### Workspace selector in SessionBar — P2 · S
The API accepts `"workspace"` on session creation; the UI doesn't expose it. Add a small input (or dropdown of existing workspaces from `GET /api/sessions`) to the "New Session" flow so users can assign sessions to projects without using curl.

### Real-time graph updates via SSE — P2 · M
SSE currently triggers a full `GET /api/sessions/:id/graph` refresh on every `tool_end` / `message_end` event. This causes the canvas to re-layout and lose manual node positions. Instead, stream individual node/edge additions as SSE events and apply them as React Flow node/edge patches — no full re-fetch, no layout reset.

### Session rename — P3 · S
Sessions are named only at creation. Add `PATCH /api/sessions/:id` with `{ "name": "new name" }` (server) and an inline-edit on the session name in SessionBar (UI).

### Graph node search / filter — P3 · M
Add a search bar above the canvas that filters visible nodes by content keyword or type. Client-side filter first (hide non-matching nodes), then wire to `graph_query` keyword search for server-side results. Useful once sessions grow large.

### Export session as Markdown / HTML — P3 · M
`GET /api/sessions/:id/export?format=markdown` renders the conversation + code blocks + knowledge nodes as a readable document. Useful for sharing findings without requiring Graphirm.

---

## Agent Capability

### Workspace context injection at session start — P2 · S
When a session has a workspace set, inject a brief snapshot into the system prompt: the workspace path and the output of `ls -la` (or a tree up to depth 2). Gives the agent immediate awareness of what's in the project without a manual `ls` tool call.

### HTTP-level test for per-session workspaces — P2 · S
The final reviewer for Phase 14 flagged this gap: no route test exercises `workspaces_root` + `create_dir_all` end-to-end. Add a test using `tempfile::tempdir()` + a custom `test_app_state_with_workspaces_root(root)` fixture. Should verify response contains `workspace_path` and the directory exists on disk.

### Subagent workspace isolation — P3 · M
Spawned subagents inherit the parent session's `working_dir`. They could optionally get their own subdirectory (`<workspace>/subagents/<id>/`) to avoid clobbering each other's file output. Requires passing workspace config through `Coordinator` → `delegate.rs`.

### Multi-file context tool (`read_many` / `diff`) — P3 · M
A single tool call that reads multiple files or shows a `git diff` output. Currently requires the agent to call `read` N times. Reduces turn count on code review tasks.

### Semantic `graph_query` search — P3 · M
Phase 12 `graph_query` search mode is keyword-only. Add a `semantic` mode that uses HNSW embeddings to find Knowledge nodes by meaning, not literal text match. The tool interface is already designed for this extension.

---

## Infrastructure & Quality

### Cross-session knowledge extraction — P2 · L
Knowledge nodes are created per-session but not automatically linked across sessions. Add a post-turn job that embeds new Knowledge nodes and links them to similar nodes in other sessions via `RelatesTo` edges. Enables the agent to surface relevant past work without a manual `graph_query`. (Embeddings + HNSW are already in place via Phase 9.)

### Custom tool plugins — P2 · L
Users can't extend Graphirm without recompiling. Add a plugin mechanism: load script-based tools (shell or Python) from `~/.graphirm/plugins/` at startup. Each plugin exposes a name, description, and `execute` command. Subject to the same HITL gate as `bash`.

### Agent Trace ingestion (import) — P3 · M
Phase 12 exports sessions as Agent Trace JSON. The reverse — importing a trace from another agent (Claude, OpenCode, Aider) into the Graphirm graph — is not yet implemented. Useful for consolidating work done outside Graphirm.

### Performance: pagination + query caching — P3 · M
The context engine traverses the full graph on every request. On sessions with 1k+ nodes this is measurable. Add:
- Offset/limit on all list endpoints
- SQLite indices on `session_id`, `node_type`, `created_at`
- In-memory TTL cache for frequent read queries (session list, node-by-id)

---

## Enterprise / Scale

### API versioning (`/api/v1/`) — P3 · M
Current REST API is unstable — breaking changes happen freely. Add a `/api/v1/` prefix, extract versioned request/response structs, and generate an OpenAPI spec. Required before any third-party tooling builds on top.

### Multi-user support — P3 · L
No user concept exists — all sessions share one database. For teams: add OAuth2 login (GitHub), per-session ownership + sharing links, and basic permission model (owner / viewer). Foundation for a hosted SaaS tier.

### graphirm.ai hosted demo — P3 · M
A `?demo` query param loads a pre-recorded session JSON instead of calling the API, hiding the input bar. Deploy to Cloudflare Pages (static, no server needed). Gives visitors a zero-friction look at the graph without an API key.

---

## Completed (summary — details in `docs/completion-log.md`)

| Phase | What |
|-------|------|
| 0–9 | Scaffold → Knowledge layer (graph, LLM, tools, agent, multi-agent, context engine, TUI, HTTP, knowledge/HNSW) |
| 10 | Structured LLM response segments + GLiNER2 fallback + segment context filter |
| 11 | Browser web UI (vanilla JS, d3 force graph + chat) |
| 12 | `graph_query` tool (bfs, list_type, keyword search) |
| 13 | Interactive whiteboard (React + React Flow, node expansion, grouping, steer-from-node, annotations, keyboard shortcuts, auto-approve) |
| 14 | Per-session workspaces (`workspaces_root`, named dirs, graph persistence, restart restore) |
