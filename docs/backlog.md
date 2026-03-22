# Graphirm Backlog

Single source of truth for planned work. Completed items are recorded in `docs/completion-log.md` and `AGENTS.md` — not here.

**Current state:** Phases 0–34 complete. See `AGENTS.md` → Current State table.

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

### ✅ Design system + light/dark theme — P3 · S
Done 2026-03-21. Expanded design tokens in `theme.css` (spacing scale, typography scale, surface layers, semantic colors, all edge color variables). Light/dark theme via `useTheme` hook (`localStorage` + `prefers-color-scheme`, sets `data-theme` on `<html>`). Toggle button (☀/◉) in Toolbar. Edge colors DRYed — `EDGE_COLORS` map removed from `LabelledEdge.tsx`, replaced with `getEdgeColor()` reading CSS variables via `getComputedStyle` with theme-aware cache. Clean build, 100% agent.
Plan: `docs/plans/2026-03-22-research-driven-improvements.md`

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

### ✅ Export session as Markdown / HTML — P3 · M
Done 2026-03-19. `GET /api/sessions/:id/export?format=markdown` returns conversation (user + assistant turns, tool interactions excluded) + extracted knowledge table as a `.md` download (`Content-Disposition: attachment`). `format=html` → 400. New `crates/server/src/export.rs` with `render_session_markdown` (5 unit tests). "↓ Export" button in `SessionBar` triggers browser download via `window.open`.
Plan: `docs/plans/2026-03-19-export-session.md`

---

## Agent Capability

### ✅ Context auto-compaction trigger — P2 · S
Done 2026-03-21. `select_nodes_for_compaction` in `compact.rs` — walks conversation thread, filters already-compacted nodes, compares token estimate to `compaction_threshold` (default 0.80), skips `guaranteed_recent_turns` newest, returns oldest eligible IDs when `>= min_nodes_to_compact`. Hook in `stream_and_record`: runs selection in `spawn_blocking` then awaits `compact_context` synchronously; non-fatal (errors logged, skipped). `compaction_threshold: f64` added to `ContextConfig`. 4 new unit tests. Enable: `enable_compaction = true` in `[context]` config.
Plan: `docs/plans/2026-03-22-research-driven-improvements.md`

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

### ✅ Agent Trace ingestion — Cursor import — P3 · M
Done 2026-03-22. `graphirm import-cursor <path>` imports Cursor `.txt` transcript files (one per conversation) into the graph as `Agent` + `Interaction` nodes with `Produces` and `RespondsTo` edges. Accepts a single file or a directory. Idempotent — re-importing the same file is a no-op (checked via `source_file` on the synthetic Agent node). Thinking blocks preserved in `metadata["thinking"]`; tool call/result blocks stripped. Parser: `crates/agent/src/import/cursor.rs` — state machine, 8 unit tests, zero new deps. 100% agent (parser + write function) + manual fix (trailing-newline edge case).
Plan: `docs/plans/2026-03-21-agent-trace-ingestion.md`

### ✅ SQLite performance indices — P3 · S
Done 2026-03-21. Four indices added to `GraphStore.init_schema()` targeting the hottest query patterns: `idx_nodes_created_at`, `idx_edges_created_at`, `idx_nodes_session_id` (`json_extract(metadata, '$.session_id')`), `idx_nodes_type_created` composite on `(node_type, created_at)`. All `CREATE INDEX IF NOT EXISTS` — safe on existing databases. 71 graph tests pass. 100% agent.

### ✅ Node-by-id TTL cache — P3 · S
Done 2026-03-21. `node_cache: Arc<RwLock<HashMap<NodeId, (GraphNode, Instant)>>>` added to `GraphStore`. 60 s TTL. `get_node` checks cache before hitting SQLite; populates on miss. `update_node` evicts the entry on write. No public API changes, no new deps. 71 tests pass, clippy clean. 100% agent.

### ✅ Performance: remaining — P3 · M
Done 2026-03-22. Two improvements shipped:
- **Phase 31** — `list_nodes_by_type` fast path: when called with no filters (session_id=None, metadata_filter=None), the SQL `LIMIT` is pushed into the query (`SELECT … LIMIT ?2`) so SQLite returns only the needed rows. Filtered path gets a `limit * 10` safety cap to avoid unbounded table scans.
- **Phase 32** — `get_agent_nodes` TTL cache: `agent_nodes_cache: Arc<RwLock<Option<(Vec<…>, Instant)>>>` added to `GraphStore`; 30 s TTL; invalidated in `add_node` and `update_node` when the node type is `"agent"`. Both `open()` and `open_memory()` initialize to `None`. 71 tests pass, clippy clean. Primarily agent (Task 1 100%, Task 2 95% — one collapsible-if clippy fix applied manually).

### ✅ Pinned Knowledge nodes + CLI — P2 · S
Done 2026-03-22. `pinned` metadata flag on Knowledge nodes; `list_pinned_knowledge(limit)` in GraphStore; `build_pinned_summary` in briefing (always surfaces regardless of recency); `POST /api/knowledge` + `GET /api/knowledge/pinned` endpoints; `graphirm knowledge list/pin/unpin` CLI subcommand. Coding conventions migrated from system prompt to graph-native pinned nodes.

### ✅ Model router — P2 · M
Done 2026-03-22. Automatic per-turn cheap/smart model selection. `ModelRouter` evaluates ordered rules (`first_turn`, `error_recovery`, `high_complexity`, `tool_only_turn`, `stuck_detection`) against graph-derived session signals. Routing decisions stored on Interaction node metadata for cost analysis. Same-provider constraint (both tiers via OpenRouter). Config: `[agent.routing]` in TOML.
Plan: `docs/plans/2026-03-22-model-router.md`

---

## Refactoring / Code Health

### Clean up stale `.worktrees/` — P3 · S
Stale worktrees from old feature branches (phase-4, 5, 7, 8) pollute Nodestradamus analysis and waste disk. Prune any not actively in use via `git worktree list` + `git worktree remove`.

### Extract CLI handlers from `main.rs` into `src/commands/` — P3 · S
`main.rs` is ~1200 lines and growing with every CLI subcommand. The `match cli.command` block now has 10+ arms, each 20-80 lines. Extract each handler into `src/commands/{chat,graph,knowledge,import,serve,export}.rs` with a thin dispatch in `main()`. Mechanical refactor, good dogfood candidate.

### Break TUI circular dependencies — P3 · M
All 14 cycles detected by Nodestradamus are in `crates/tui/` between `app.rs` ↔ `ui.rs` ↔ `events.rs`. Extract a shared `AppState` struct that both `ui.rs` and `events.rs` import without importing each other. TUI works fine — this is hygiene.

---

## Enterprise / Scale

### API versioning (`/api/v1/`) — P3 · M
Current REST API is unstable — breaking changes happen freely. Add a `/api/v1/` prefix, extract versioned request/response structs, and generate an OpenAPI spec. Required before any third-party tooling builds on top.

**Refactor analysis (Nodestradamus):** `routes.rs` has 1 direct dependent (`lib.rs` → `create_router`) and 55 indirect dependents (types, state, export, error, middleware). Natural clusters for splitting: session management (`create_session`, `get_session`, `list_sessions`, `rename_session` — cohesion 0.14), streaming/SSE (`agent_event_to_sse`, `prompt_session` — cohesion 0.33), and tests. Suggests `routes::v1::` sub-module with `create_router` re-exported at top.

**Useful Nodestradamus tools:**
- `get_impact file_path=crates/server/src/routes.rs refactor_mode=true` — breaking changes + clusters
- `analyze_graph algorithm=hierarchy level=module` — verify clean layering (routes → state/types → graph/agent)
- `analyze_strings mode=refs` — find all hardcoded `/api/` path strings in web-app TS + eval harness

### Multi-user support — P3 · L
No user concept exists — all sessions share one database. For teams: add OAuth2 login (GitHub), per-session ownership + sharing links, and basic permission model (owner / viewer). Foundation for a hosted SaaS tier.

**Scoping guidance (Nodestradamus):** `state.rs` (`AppState`, `SessionHandle`) is the natural place to add `user_id` ownership — it's the existing session registry. Run impact analysis on it to see the full blast radius of adding per-user fields.

**Useful Nodestradamus tools:**
- `analyze_graph algorithm=communities` — cluster modules to find natural auth boundary
- `get_impact file_path=crates/server/src/state.rs` — blast radius of adding `user_id` to `SessionHandle`
- `analyze_graph algorithm=path source=<auth_middleware> target=<SessionHandle>` — confirm dependency path is short

### graphirm.ai hosted demo — P3 · M
A `?demo` query param loads a pre-recorded session JSON instead of calling the API, hiding the input bar. Deploy to Cloudflare Pages (static, no server needed). Gives visitors a zero-friction look at the graph without an API key.

**Useful Nodestradamus tools:**
- `analyze_strings mode=refs` — find shared API URL strings + SSE event names between web-app and server (demo shim must intercept these)
- `semantic_analysis mode=search query="how does the web app fetch session data"` — find `useSession` hook and API client (the 1–2 files needing a `?demo` branch)
- `analyze_deps package=web-app` — map which React components import the API layer

---

## Graph Intelligence & Structural Awareness

Features that make Graphirm structurally intelligent — combining Nodestradamus code graph
analysis with Graphirm's persistent session memory. Inspired by what works in the ecosystem
(e.g. GitNexus), but transformed into capabilities only a graph-native agent can provide.

### ✅ Graph-Aware Tool Execution (Pre-Edit Impact Injection) — P1 · M
Done 2026-03-20. Before destructive tool calls (`write`, `edit`, `bash`), the agent loop automatically queries for structural dependents (via `rg`) and prior Knowledge notes mentioning the target file, computes a risk score (LOW/MEDIUM/HIGH), and prepends a brief to the tool output. Bash file paths extracted via tree-sitter-bash AST. Per-turn caching avoids re-analysis. Empty briefs suppressed. Non-fatal throughout. `pre_edit_impact: bool` config flag (default true).
Plan: `docs/plans/2026-03-20-graph-aware-tool-execution.md`
Design: `docs/plans/2026-03-20-graph-aware-tool-execution-design.md`

### ✅ Graph-Diff Tool (Session-Aware Blast Radius) — P1 · S
Done 2026-03-20. `graph_diff` non-destructive tool: `git` mode (resolves changed files via `git diff --name-only`) or `paths` mode (explicit list) → per-file dependent listing (rg `--files-with-matches --fixed-strings`, capped at 20), cross-session Knowledge query with stale ⚠ warnings, risk scoring (Low/Medium/High). Reuses `compute_risk` from Phase 22. 12 unit+integration tests.
Plan: `docs/plans/2026-03-20-graph-diff-tool.md`
Design: `docs/plans/2026-03-20-graph-diff-tool-design.md`

### ✅ Repo Briefing on Session Start (Structural + Memory Onboarding) — P1 · M
Done 2026-03-20. Two-tier briefing: (1) compact summary auto-injected into every session's system prompt — language breakdown (file-extension counts via async dir walk), top files by rg mention-count, recent knowledge nodes from the graph; (2) on-demand `repo_briefing` tool returning detailed files (rg --files + dir breakdown), knowledge (10 recent nodes), and git sections (`git log --oneline -10`, unstaged diff count). `repo_briefing: bool` config flag (default true). Non-fatal throughout. 10 unit+integration tests.
Plan: `docs/plans/2026-03-20-repo-briefing.md`
Design: `docs/plans/2026-03-20-repo-briefing-design.md`

### ✅ Session Flow Traces (Queryable Agent Decision History) — P2 · M
Done 2026-03-20. `session_trace` non-destructive tool: `search` (Knowledge-anchored semantic via `KnowledgeRetriever`, keyword fallback with note when no embeddings) groups by `session_id` and prints interaction traces; `replay` full chronological chain for one session; `detail` compact/full; `get_session_chain` in GraphStore. 16+ unit tests in tools, 2 in graph.
Plan: `docs/plans/2026-03-20-session-trace.md`
Design: `docs/plans/2026-03-20-session-trace-design.md`

### Any-Repo Instant Analysis (Zero-Config, Incremental, Persistent) — P2 · L

Drop Graphirm into any repo — no configuration, no prior indexing — and it automatically runs
Nodestradamus v2 analysis on first use, persists the result as graph nodes, and makes structural
intelligence available for all future sessions. On subsequent sessions, retrieves from the graph
DB instantly. If the repo has changed (new commits), runs incremental re-analysis on changed
files only — not a full re-index.

Fully automatic and persistent. Competes with tools that require explicit setup and full
re-index on every change.

**Key files:**
- `crates/agent/src/repo_analysis.rs` — `ensure_repo_analyzed(workspace, db)`; checks stored SHA
- `crates/graph/src/store.rs` — store/retrieve `RepoAnalysis` nodes with SHA + timestamp
- `crates/agent/src/workflow.rs` — call `ensure_repo_analyzed` on session init
- `crates/agent/src/config.rs` — `auto_analyze_repo: bool` (default true when Nodestradamus available)

**Useful Nodestradamus tools:**
- `codebase_health` — validate that analysis results are consistent before persisting
- `get_impact file_path=crates/graph/src/store.rs` — blast radius of adding `RepoAnalysis` node type

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
| 20–33 | Graph search/filter, session export, graph-aware tools, graph_diff, repo briefing, session traces, lessons briefing, auto-compaction, design system, read truncate, SQLite indices, node cache, cursor import, SQL fast paths, agent cache, pinned knowledge |
