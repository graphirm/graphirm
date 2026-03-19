# Graphirm Backlog

Single source of truth for planned work. Completed items are recorded in `docs/completion-log.md` and `AGENTS.md` — not here.

**Current state:** Phases 0–19 complete. See `AGENTS.md` → Current State table.

---

## How to use this file

- Items are grouped by theme, not phase number
- Each item has a **size** (S / M / L) and a **priority** (P1 high / P2 medium / P3 low)
- When work starts, create a plan in `docs/plans/YYYY-MM-DD-<topic>.md` and link it here
- Mark items ✅ and move details to `docs/completion-log.md` when shipped

---

## Deployment & Operations

### ✅ Coolify on always-on spoke — P1 · M
Done 2026-03-18. `https://app.graphirm.ai` live. Floating IP `5.75.217.23`, snapshot ID `367758410`.
Plan: `docs/plans/2026-03-16-coolify-spoke-deployment.md`

### ✅ Wire `workspaces_root` on running server — P1 · S
Done 2026-03-18. Set `workspaces_root = "/data/workspaces"` in `config/default.toml` — on the Docker volume so workspaces survive redeployments.

### ✅ CI pipeline (GitHub Actions) — P2 · S
Done 2026-03-18. `cargo fmt --check`, `cargo clippy --all-features -D warnings`, `cargo build`, and `cargo test` run on every push to `main` and every PR. Fixed fmt + clippy violations across the whole codebase to get the first green run.

---

## UI (web-app)

### ✅ Workspace selector in SessionBar — P2 · S
Done 2026-03-18. "+ New" button expands an inline form with session name + workspace inputs. Enter submits, Escape cancels, promise awaited before closing.

### ✅ Real-time graph updates via SSE — P2 · M
Done 2026-03-19. `GraphUpdate` SSE payload now includes full `nodes` + `edges` patch data. Web-app applies incremental patches to React Flow state — no full re-fetch, canvas positions preserved. `message_end` refreshes messages only. `agent_end` uses a 500 ms debounced reconciliation refresh. Also fixed `@dagrejs/dagre` 1.1.8 → 1.0.4 (broken graphlib packaging).
Plan: `docs/plans/2026-03-18-p2-sse-knowledge-plugins.md`

### ✅ Session rename — P3 · S
Done 2026-03-19. `PATCH /api/sessions/:id` with `{ "name": "new name" }` persists to Agent graph node (survives restart). `display_name: Arc<RwLock<String>>` in `SessionHandle`. `name` field added to `SessionResponse` (also fixes pre-existing bug where `Session.name` was always undefined in the UI). Inline double-click edit in `SessionBar` — Enter commits, Escape cancels, blur commits.
Plan: `docs/plans/2026-03-19-session-rename.md`

### ✅ Graph node search / filter — P3 · M
Done 2026-03-19. Search input + type filter pills (`I A C T K`) in Toolbar. `useGraphData` applies `hidden: true` to non-matching React Flow nodes via `applyFilterToNodes` helper; group nodes hidden when all children hidden; annotation nodes never hidden. `matchCount/total` counter shown when filter is active. Clear `✕` button. Ctrl+F (hover) focuses search input, Escape clears and blurs. No backend changes.
Plan: `docs/plans/2026-03-19-graph-node-search.md`

### Export session as Markdown / HTML — P3 · M
`GET /api/sessions/:id/export?format=markdown` renders the conversation + code blocks + knowledge nodes as a readable document. Useful for sharing findings without requiring Graphirm.

---

## Agent Capability

### ✅ Workspace context injection at session start — P2 · S
Done 2026-03-18. `build_workspace_context` in `routes.rs` lists up to 20 sorted entries and appends `## Active Workspace` block to `config.system_prompt` at session creation time. Non-fatal, warns on errors.

### ✅ HTTP-level test for per-session workspaces — P2 · S
Done 2026-03-18. `test_workspace_creation` in `crates/server/tests/integration.rs` verifies response fields and directory existence on disk via `tempfile::tempdir()`.

### ✅ Subagent workspace isolation — P3 · M
Done 2026-03-19. `spawn_subagent` accepts `parent_working_dir: Option<PathBuf>`; when set, creates `<parent>/subagents/<agent>-<short_id>/` and sets subagent `working_dir`. Delegate passes `ctx.working_dir`. Integration test verifies subagent tool runs in workspace.
Plan: `docs/plans/2026-03-19-agent-capability-subagent-ws-multifile.md`

### ✅ Multi-file context tool (`read_many` / `diff`) — P3 · M
Done 2026-03-19. `diff` tool: file compare (`file_a`/`file_b`) and git diff (optional `ref`/`path`/`cached`), non-destructive. `read_many` tool: up to 20 paths, optional `max_lines_per_file`, concatenated output with path headers; non-destructive. Both registered in `build_tool_registry`.
Plan: `docs/plans/2026-03-19-agent-capability-subagent-ws-multifile.md`

### ✅ Semantic `graph_query` search — P3 · M
Done 2026-03-19. Added `semantic` mode to `graph_query` tool. `KnowledgeRetriever` trait defined in `graphirm-tools` (no circular deps); `MemoryRetriever` implements it via `retrieve_with_scores` (correct L2→cosine conversion: `1 - d²/2`); wired via `ToolContext.knowledge_retriever`. Returns HNSW-ranked Knowledge nodes with cosine similarity scores. Graceful `ExecutionFailed` when no embedding provider is configured.
Plan: `docs/plans/2026-03-19-semantic-graph-query.md`

---

## Infrastructure & Quality

### ✅ Cross-session knowledge extraction — P2 · L
Done 2026-03-19. Knowledge nodes now store `session_id` in metadata. After each embedding, `MemoryRetriever.find_cross_session_links` queries HNSW (min 0.7 cosine similarity, top 3) and `persist_cross_session_links` writes `RelatesTo` edges with similarity as weight. Three unit tests added.
Plan: `docs/plans/2026-03-18-p2-sse-knowledge-plugins.md`

### ✅ Custom tool plugins — P2 · L
Done 2026-03-19. `ScriptTool` loads `plugin.toml` manifests from `~/.graphirm/plugins/` (or `GRAPHIRM_PLUGINS_DIR`). Each plugin defines name, description, `command`, `destructive` flag, and JSON Schema parameters. Args passed as `GRAPHIRM_ARGS` (JSON) + `GRAPHIRM_ARG_<KEY>` env vars. `is_destructive()` added to `Tool` trait; overridden in `bash`/`write`/`edit` and respected by HITL gate alongside the built-in name list. Example plugin at `examples/plugins/hello/`.
Plan: `docs/plans/2026-03-18-p2-sse-knowledge-plugins.md`

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
| 19 | Subagent workspace isolation + diff/read_many tools |
