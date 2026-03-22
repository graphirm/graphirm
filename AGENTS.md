# Graphirm

Graph-native coding agent in Rust. Every interaction, tool call, file read/write, and knowledge entity
is stored as a typed node in a persistent SQLite-backed graph. The graph is the session, the memory, the
context window, and the audit trail — all at once. Single static binary, no Docker, no runtime dependencies.

See `README.md` for usage examples, screenshots, and detailed feature docs.

---

## Architecture

Cargo workspace with six crates plus an eval harness. Dependency order (bottom to top):

```
rusqlite / petgraph / instant-distance  (external)
    └── graphirm-graph      # graph store, node/edge CRUD, PageRank, BFS, HNSW
         ├── graphirm-llm   # LLM provider trait, streaming, embeddings
         ├── graphirm-tools # built-in tools (bash, read, write, edit, grep, find, ls, graph_query, diff, read_many)
         └── graphirm-agent # agent loop, context engine, multi-agent, knowledge, HITL
              ├── graphirm-tui    # ratatui TUI (chat + graph explorer)
              └── graphirm-server # axum HTTP API + SSE
src/main.rs                 # CLI entrypoint (chat, graph, serve, export-corpus, ...)
graphirm-eval/              # evaluation harness (HTTP client only, no crate deps)
graphirm-vscode/            # VS Code / Cursor extension (TypeScript)
```

**Five node types:** `Interaction` (messages), `Agent` (instances), `Content` (files/output),
`Task` (DAG work items), `Knowledge` (extracted entities)

**Fifteen edge types:** `RespondsTo`, `SpawnedBy`, `DelegatesTo`, `DependsOn`, `Produces`,
`Reads`, `Modifies`, `Summarizes`, `Contains`, `FollowsUp`, `Steers`, `RelatesTo`,
`DerivedFrom`, `ApprovedBy`, `RejectedBy`

---

## Code Layout

| Path | What |
|------|------|
| `src/main.rs` | CLI: `chat`, `graph`, `serve`, `export-corpus`, `label-explore`, `schema-suggest`, `predict-spans`, `validate-agreement` |
| `crates/graph/` | `GraphStore`, node/edge types, PageRank, BFS, HNSW vector index |
| `crates/llm/` | `LlmProvider` trait, Anthropic/OpenAI/DeepSeek/Ollama/OpenRouter impls, `MockProvider` |
| `crates/tools/` | `Tool` trait, `ToolRegistry`, parallel executor, bash/read/write/edit/grep/find/ls/graph_query/diff/read_many/cargo_check |
| `crates/agent/` | `run_agent_loop`, `build_context`, `Coordinator`, `HitlGate`, knowledge extraction |
| `crates/tui/` | `App`, chat panel, graph explorer, input handling |
| `crates/server/` | axum routes, SSE streaming, `AppState`, `SessionHandle`, SDK, static file serving |
| `graphirm-eval/` | eval harness — drives agent via HTTP, checks task correctness |
| `graphirm-vscode/` | VS Code/Cursor extension (TypeScript) |
| `web-app/` | React + React Flow interactive whiteboard UI (Vite, TypeScript) |
| `web/` | Vanilla JS browser UI (legacy fallback, still served if `web-app/dist/` not present) |
| `config/default.toml` | default model, agent, knowledge, graph, TUI, server settings |

Each significant directory has its own `AGENTS.md` with purpose, key files, integration points, and test command.

---

## Build & Test

```bash
# Standard build
cargo build --release

# With GLiNER2 local extraction (requires ONNX model download first)
cargo build --release --features local-extraction

# Run all tests
cargo test

# Single crate
cargo test -p graphirm-graph
cargo test -p graphirm-llm    # mock tests only
cargo test -p graphirm-tools
cargo test -p graphirm-agent
cargo test -p graphirm-server

# LLM integration tests (need API key)
DEEPSEEK_API_KEY=sk-... cargo test -p graphirm-llm --test integration

# Run TUI
DEEPSEEK_API_KEY=sk-... ./target/release/graphirm chat

# Run HTTP server (port 3000 by default)
# Web UI served at http://localhost:3000 — prefers web-app/dist/ over web/
DEEPSEEK_API_KEY=sk-... ./target/release/graphirm serve

# Build the React web UI (run once before serving, or after changes)
cd web-app && npm install && npm run build && cd ..

# Develop the web UI with hot reload (requires server running on :3000)
cd web-app && npm run dev   # served at http://localhost:5173

# Run eval harness (server must be running)
cargo run -p graphirm-eval -- --suite coding
```

Graph database stored at `~/.graphirm/graph.db` by default. Override with `--db /path/to/graph.db`.

---

## Key Conventions

**Rust:**
- Edition 2024, MSRV 1.88 — run `cargo fmt` and `cargo clippy` before every commit
- `thiserror` for error enums (one per crate), `anyhow` in `main.rs` only
- Never `unwrap()` in production — use `?` or `expect("context")`
- `tracing::info!` / `tracing::error!` for logging — never `println!`
- `async-trait` for async trait methods
- `Arc<RwLock<StableGraph>>` for in-memory graph — acquire locks briefly, never hold across await points

**Patterns:**
- New built-in tool → implement `Tool` trait in `crates/tools/src/<name>.rs`, register in `build_tool_registry()` in `src/main.rs`
- Script plugin → create `~/.graphirm/plugins/<name>/plugin.toml` (see `examples/plugins/hello/`); loaded automatically at startup; no recompile required
- New LLM provider → implement `LlmProvider` trait in `crates/llm/`
- `bash`, `write`, `edit` are destructive tools — subject to HITL gate (unless auto-approve is enabled)
- `read`, `grep`, `find`, `ls`, `graph_query` are non-destructive — always run without confirmation
- `read` auto-truncates files > 300 lines when no `offset`/`limit` is provided — returns first 300 lines + notice; callers should use `offset`/`limit` for targeted reads
- Auto-approve: `POST /api/sessions/{id}/auto-approve` with `{ "enabled": true }` — skips HITL gating for all destructive tools in that session
- Per-session workspaces: set `workspaces_root` in `[agent]` config; `POST /api/sessions` accepts optional `"workspace"` name (defaults to sanitized session name); workspace directory is auto-created; workspace name persisted in Agent node metadata and restored on restart; response includes `workspace` and `workspace_path` when active
- Config lives in `config/default.toml`; `AgentConfig` is loaded from it at startup; `workspaces_root` in `[agent]` — optional root; when set, each session gets an isolated subdirectory `<root>/<workspace>/`
- Pinned Knowledge nodes: `POST /api/knowledge` with `"pinned": true` creates global convention/rule nodes that always surface in `repo_briefing` regardless of recency; `list_pinned_knowledge(limit)` in GraphStore; `build_pinned_summary` in briefing
- API keys via env vars: `DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`

---

## Current State

| Phase | What | Status |
|-------|------|--------|
| 0–9 | Scaffold → Knowledge layer (graph, LLM, tools, agent, multi-agent, context engine, TUI, HTTP, knowledge/HNSW) | ✅ done |
| 10 | Structured LLM response segments (parse → persist → GLiNER2 fallback → context filter → eval) | ✅ done |
| 11 | Web UI — browser graph visualization + chat | ✅ done |
| 12 | `graph_query` tool — agent can query its own graph (bfs, list_type, keyword search) | ✅ done |
| 13 | Interactive whiteboard graph — React + React Flow, node expansion (marked + hljs), grouping, steer-from-node, canvas annotations, keyboard shortcuts | ✅ done |
| 14 | Per-session workspaces — `workspaces_root` config, named workspace directories, persisted in Agent node metadata, restored on restart | ✅ done |
| 15 | Incremental SSE graph updates — `GraphUpdate` payload carries full node/edge patch; web-app applies patches without full re-fetch or canvas re-layout | ✅ done |
| 16 | Cross-session knowledge linking — `session_id` in Knowledge metadata, HNSW-based `find_cross_session_links`, `RelatesTo` edges between sessions | ✅ done |
| 17 | Custom tool plugins — `ScriptTool` loads TOML manifests from `~/.graphirm/plugins/`, executes shell commands, `is_destructive` flag respected by HITL gate | ✅ done |
| 18 | Semantic `graph_query` mode — `KnowledgeRetriever` trait, HNSW cosine similarity search (`1-d²/2`), scores in output, graceful fallback | ✅ done |
| 19 | Subagent workspace isolation + multi-file tools — `parent_working_dir` in `spawn_subagent`, subagents get `<workspace>/subagents/<name>-<id>/`; `diff` (file + git) and `read_many` (up to 20 files) tools, non-destructive | ✅ done |
| 20 | Graph node search / filter — keyword + type filter pills in Toolbar; `applyFilterToNodes` stamps `hidden` on React Flow nodes; group nodes hidden when all children match; `matchCount` counter; Ctrl+F shortcut | ✅ done |
| 21 | Session export — `GET /api/sessions/:id/export?format=markdown`; `render_session_markdown` in `crates/server/src/export.rs`; "↓ Export" button in SessionBar | ✅ done |
| 22 | Graph-aware tool execution — `ImpactProvider` trait, tree-sitter bash path extraction, `GraphImpactProvider` (rg + Knowledge notes), risk scoring, pre-edit hook in workflow, per-turn cache | ✅ done |
| 23 | `graph_diff` tool — session-aware blast radius: `git`/`paths` → dependents (rg) + stale Knowledge + risk scoring | ✅ done |
| 24 | Repo briefing on session start — compact auto-injected summary (language breakdown, top files, recent knowledge) + on-demand `repo_briefing` tool (files/knowledge/git sections) | ✅ done |
| 25 | Session flow traces — `session_trace` tool: `search` mode (Knowledge-anchored semantic or keyword fallback → ranked interaction traces per session) + `replay` mode (full chronological chain); `get_session_chain` in GraphStore; `compact`/`full` detail | ✅ done |
| 25.5 | Lesson/convention briefing — `build_lessons_summary` queries `lesson`/`convention` Knowledge nodes, injects under `## Lessons from past sessions` in repo briefing | ✅ done |
| 26 | Context auto-compaction trigger — `select_nodes_for_compaction` in `compact.rs`, `compaction_threshold` field in `ContextConfig`, hook in `stream_and_record` (sync, non-fatal); 4 new unit tests | ✅ done |
| 27 | Web-app design system — spacing/typography/surface tokens in `theme.css`, light/dark theme via `useTheme` hook (`localStorage` + system preference), theme toggle in Toolbar, edge colors DRYed to CSS variables with theme-aware cache in `LabelledEdge.tsx` | ✅ done |
| 26 | Read auto-truncate — files > 300 lines auto-truncated when no `offset`/`limit` provided; appends "Use offset and limit" notice; `MAX_AUTO_LINES` const in `read.rs` | ✅ done |
| 28 | SQLite performance indices — `idx_nodes_created_at`, `idx_edges_created_at`, `idx_nodes_session_id` (json_extract), `idx_nodes_type_created` composite; all `CREATE INDEX IF NOT EXISTS`, safe on existing DBs | ✅ done |
| 29 | Node-by-id TTL cache — `node_cache: Arc<RwLock<HashMap<NodeId, (GraphNode, Instant)>>>` in `GraphStore`; 60 s TTL; populated in `get_node`, invalidated in `update_node`; no public API changes | ✅ done |
| 30 | Cursor transcript import — `graphirm import-cursor <path>` ingests Cursor `.txt` transcripts into the graph; state-machine parser in `crates/agent/src/import/cursor.rs`; idempotent via `source_file` metadata | ✅ done |
| 31 | `list_nodes_by_type` SQL LIMIT fast path — no-filter calls push `LIMIT ?2` into SQL; filtered path gets `limit * 10` safety cap; eliminates full-table scans on common unfiltered queries | ✅ done |
| 32 | `get_agent_nodes` TTL cache — 30 s `agent_nodes_cache` in `GraphStore`; invalidated on agent node write; reduces repeated SQLite scans during session restore | ✅ done |
| 33 | Pinned Knowledge nodes — `pinned` metadata flag, `list_pinned_knowledge` in GraphStore, `build_pinned_summary` in briefing, `POST /api/knowledge` endpoint for direct creation | ✅ done |

**Segment-aware context filter:** `segment_filter` is now fully wired — set via `POST /api/sessions` → `AgentConfig` → `ContextConfig` per turn. Filter changes which prior assistant segments are reconstructed into the LLM context window.

**Segment feature summary (Phase 10):**
- `SegmentConfig` in `AgentConfig` — enable per-session via `POST /api/sessions` with `enable_segments: true`
- LLM responses parsed into typed `Content` nodes (`code`, `reasoning`, `observation`, `plan`, `answer`) linked via `Contains` edges
- Primary path: structured JSON output from LLM (system prompt injected by `build_segment_prompt`)
- Fallback path: GLiNER2 ONNX span detection via `try_gliner2_fallback` (uses `ExtractionConfig.backend` model dir)
- Context engine: optional `segment_filter` in `ContextConfig` to include only specific segment types
- Eval coverage: `cargo run -p graphirm-eval -- --filter segments` (uses `GraphContainsContentType` verifier)
- See `docs/plans/2026-03-10-structured-llm-responses.md` and `docs/plans/2026-03-15-structured-segments-phase5-6.md`

**Web UI summary (Phase 11 — vanilla JS, legacy):**
- Standalone browser UI at `web/` — adapted from `graphirm-vscode/media/` with `acquireVsCodeApi()` replaced by direct `fetch()` + `EventSource`
- Server serves static files via `tower-http::services::ServeDir` fallback — API routes at `/api/*` take precedence
- Auto-discovery: `find_web_dir()` checks `web-app/dist/` first, then `web/` as fallback
- Chat pane (markdown, HITL approval cards), graph pane (d3 force + timeline), session management
- No build step, no framework, no auth — vanilla JS ES modules, ~1200 lines total

**Interactive whiteboard UI summary (Phase 13):**
- `web-app/` — React 19 + TypeScript + `@xyflow/react` v12, built with Vite 6
- Node cards per type: InteractionNode, AgentNode, ContentNode, TaskNode, KnowledgeNode, AnnotationNode
- Custom `LabelledEdge` — per-type colour, SmoothStep (hierarchical) / Bezier (cross-cutting)
- Three layout modes: DAG (dagre), Timeline (X=time, Y=type band), Free (manual, localStorage)
- **Node expansion** — click ▼ to expand; Interaction renders markdown (marked), Content shows syntax-highlighted code (hljs); NodeResizer for manual resize
- **Visual grouping** — each Interaction + its produced nodes rendered inside a React Flow parent/group node with dashed boundary
- **Steer-from-node** — expand any Interaction node → "↩ Steer from here" button pre-fills chat input with context root; sent via existing `POST /api/sessions/{id}/prompt`
- **Canvas annotations** — double-click empty canvas or toolbar "+ Note" adds editable AnnotationNode; `POST /api/graph/{session_id}/annotate` persists as Knowledge node
- **Keyboard shortcuts** — `F` fit-view, `L` cycle layout, `N` new session, `/` focus chat
- MiniMap, Controls, dotted background grid — full pan/zoom/drag
- **Auto-approve toggle** — SessionBar button enables/disables HITL gating per session; green when active
- ChatPane with HITL approve/reject/modify cards, steer context banner; SessionBar with pause/resume/auto-approve
- Bundle: React Flow 194 kB, highlight 21 kB (trimmed to 20 languages), dagre 43 kB, app 289 kB — all chunks ≤ 500 kB
- Dev: `cd web-app && npm run dev` (proxies `/api` → `localhost:3000`)
- Build: `cd web-app && npm run build` → `web-app/dist/` (served automatically by `graphirm serve`)

**Subagent workspace + multi-file tools (Phase 19):**
- `graphirm_agent::workspace::sanitize_workspace_name` — shared from server; used for subagent dir names
- `spawn_subagent(..., parent_working_dir: Option<PathBuf>)` — when `Some`, creates `<parent>/subagents/<agent>-<short_task_id>/`, sets `agent_config.working_dir`; `delegate` passes `ctx.working_dir`
- `diff` tool — file mode (`file_a`/`file_b`, runs `diff -u`) and git mode (`mode: "git"`, optional `ref`/`path`/`cached`); non-destructive
- `read_many` tool — `paths: string[]` (max 20), optional `max_lines_per_file` (default 500); concatenated output with `=== path (N lines) ===` headers; partial failures reported per file; non-destructive
- Plan: `docs/plans/2026-03-19-agent-capability-subagent-ws-multifile.md`

**Semantic graph_query (Phase 18):**
- `KnowledgeRetriever` trait in `crates/tools/src/retriever.rs` — decouples tool from agent crate (avoids circular deps)
- `MemoryRetriever` implements `KnowledgeRetriever` via `retrieve_with_scores`; L2→cosine: `similarity = (1 - d²/2).clamp(0,1)`
- `ToolContext.knowledge_retriever: Option<Arc<dyn KnowledgeRetriever>>` — wired from `session.memory_retriever()` in `execute_tools_parallel`
- `graph_query` `semantic` mode: embeds query, returns top-k Knowledge nodes with `sim=X.XXX` scores ordered by similarity
- Returns `ExecutionFailed` with helpful message when no embedding provider is configured
- 5 unit tests in `graph_query.rs` (happy path, no retriever, empty query, empty results, limit); 3 in `memory.rs` (scores bounded, empty index, score formula regression)

**Custom tool plugins (Phase 17):**
- `crates/tools/src/script.rs` — `PluginManifest` (TOML) + `ScriptTool` that implements `Tool`
- Plugins live in `~/.graphirm/plugins/<name>/plugin.toml`; override dir via `GRAPHIRM_PLUGINS_DIR` env var
- At startup, `build_tool_registry()` in `src/main.rs` scans the plugins dir, calls `ScriptTool::from_dir`, and registers each valid plugin; invalid plugins are skipped with a warning
- Command execution: `bash -c <command>` in session `working_dir`; `${plugin_dir}` substituted in command string; args passed as `GRAPHIRM_ARGS` (JSON) and `GRAPHIRM_ARG_<KEY>` env vars
- `Tool::is_destructive()` trait method added (default `false`); overridden to `true` in `BashTool`, `WriteTool`, `EditTool`; `ScriptTool` returns `manifest.destructive`
- `ToolRegistry::is_destructive(name)` delegates to the registered tool's method
- HITL gate check uses both legacy name list (`write`/`edit`/`bash`) **and** `ToolRegistry::is_destructive` — plugins with `destructive = true` are gated
- Example plugin: `examples/plugins/hello/` — copy to `~/.graphirm/plugins/hello/` to try it

**Cross-session knowledge linking (Phase 16):**
- `persist_extracted_entities` stamps every new `Knowledge` node with `metadata["session_id"]` — enables HNSW results to be filtered by session without graph traversal
- `session_id` threaded through `post_turn_extract → extract_knowledge_with_backend → persist_extracted_entities`
- `MemoryRetriever.find_cross_session_links(node_id, exclude_session, k, min_similarity)` — embeds the node's text, queries HNSW with 3×k candidates, strips same-session and self matches, returns top-k `(NodeId, f64)` similarity pairs
- `MemoryRetriever.persist_cross_session_links(source, links)` — writes `RelatesTo` edges with cosine similarity as edge weight; non-fatal (logs per-edge failures)
- Wired in workflow after each successful `embed_knowledge_node` call; threshold `0.7`, top `3` per node
- Three new unit tests in `knowledge::memory::tests`: cross-session discovery, empty-index guard, edge persistence

**Incremental SSE graph updates (Phase 15):**
- `AgentEvent::GraphUpdate` now carries `recent_edges` (edges touching the response + tool-result nodes) and `patch_nodes` (recent nodes + edge endpoints) in addition to `recent_nodes`
- `agent_event_to_sse()` serialises `patch_nodes` and `recent_edges` directly into the SSE payload (`nodes`, `edges` keys) — `GraphNode` and `GraphEdge` both derive `Serialize`
- Web-app `useSession`: `graph_update` events call `patchGraphData` (merge by ID, preserving existing positions) instead of a full `GET /api/graph` re-fetch; `tool_end` has no refresh handler (graph is updated by the following `graph_update`); `message_end` refreshes messages only via `api.getMessages`
- `agent_end` / `error`: 500 ms debounced full reconciliation refresh (clears on unmount)
- Build fix: `@dagrejs/dagre` pinned to `1.0.4` (uses `@dagrejs/graphlib@2.1.13`) — v1.1.8 shipped a broken graphlib tarball missing `data/priority-queue.js`

**Per-session workspaces summary (Phase 14):**
- Set `workspaces_root = "/workspaces"` in `[agent]` section of `config/default.toml` to enable
- `POST /api/sessions` accepts optional `"workspace"` field; defaults to sanitized session name
- Server calls `tokio::fs::create_dir_all(<root>/<workspace>/)` and sets it as the session's `working_dir`
- Workspace name stored in Agent node metadata (`"workspace"` key) — survives SQLite restarts
- On startup, `restore_sessions_from_graph` reconstructs `working_dir` from stored workspace name
- `GET /api/sessions/:id` response includes `workspace` and `workspace_path` fields when active
- Backward-compatible: when `workspaces_root` is unset, all behaviour is unchanged

**Session export (Phase 21):**
- `crates/server/src/export.rs` — `render_session_markdown(name, model, created_at, nodes)` → Markdown; user + assistant turns sorted by `created_at` (tool/system excluded); Knowledge nodes as pipe table with escaped cells; 5 unit tests
- `GET /api/sessions/:id/export?format=markdown` — fetches subgraph (depth 10), renders, returns `text/markdown; charset=utf-8` with `Content-Disposition: attachment; filename="session-<name>.md"`; `format!=markdown` → 400; unknown session → 404
- `ExportQuery` in `crates/server/src/types.rs` with `format` defaulting to `"markdown"`
- "↓ Export" button in `SessionBar` — `window.open(url, '_blank')` triggers browser download

**Repo briefing on session start (Phase 24):**
- `crates/agent/src/briefing.rs` — `count_files_by_extension` (async dir walk, skips hidden/target/node_modules), `format_language_breakdown`, `collect_stems`, `find_top_files` (rg `--count --fixed-strings`, stems capped at 200), `count_mentions`, `build_knowledge_summary` (empty-string query → all nodes, `•` bullet format), `build_lessons_summary` (queries `lesson`/`convention` entity_type Knowledge nodes, merges + sorts by `created_at` DESC, formats as `- [lesson]/[convention] entity: summary`), `build_repo_briefing` (assembles all four sections including lessons, injected under `## Repo Briefing` header)
- `crates/agent/src/config.rs` — `repo_briefing: bool` (default `true`), `#[serde(default = "default_repo_briefing")]`
- `crates/server/src/routes.rs` — after workspace setup in `create_session`, calls `graphirm_agent::briefing::build_repo_briefing(&config.working_dir, state.graph.as_ref()).await` and appends result to `config.system_prompt` when `config.repo_briefing` is true
- `crates/tools/src/repo_briefing.rs` — `RepoBriefingTool` with `section` param (`all`/`files`/`knowledge`/`git`); files section uses `rg --files` + top-dir breakdown; knowledge section queries 10 recent nodes; git section runs `git log --oneline -10` + `git diff --name-only HEAD`; registered in `build_tool_registry()`
- 13 tests total: 4 formatting unit tests (empty map, sort order, truncation, stem uniqueness), 2 knowledge tests (empty store, format), 3 lessons tests (empty store, both types format, exclusion filter), 1 briefing assembly test (empty dir → None), 3 tool integration tests (name/params, knowledge empty, git section)
- Plan: `docs/plans/2026-03-20-repo-briefing.md`

**Context auto-compaction (Phase 26):**
- `crates/agent/src/compact.rs` — `select_nodes_for_compaction(graph, agent_id, max_tokens, threshold_ratio, guaranteed_recent_turns, min_nodes_to_compact)`: walks conversation thread via `conversation_thread`, filters out already-compacted nodes via `is_compacted`, compares total token estimate to threshold, skips newest `guaranteed_recent_turns` nodes, returns oldest eligible IDs
- `crates/agent/src/context.rs` — `compaction_threshold: f64` added to `ContextConfig` (`#[serde(default)]`, default `0.80`); `tracing::debug!` replaces prior `tracing::warn!` stub
- `crates/agent/src/workflow.rs` — after `build_context` returns, `stream_and_record` checks `enable_compaction`, runs selection in `spawn_blocking`, then awaits `compact_context` synchronously (non-fatal: errors are `tracing::warn!` and skipped)
- Enable via `enable_compaction = true` in `[context]` section of `config/default.toml`; tune with `compaction_threshold` (0.0–1.0)
- 4 new unit tests: below-threshold returns empty, above-threshold returns oldest, skips compacted, respects min_nodes

**`list_nodes_by_type` SQL LIMIT fast path + `get_agent_nodes` TTL cache (Phases 31–32):**
- `list_nodes_by_type` fast path: when `session_id.is_none() && metadata_filter.is_none()`, uses `SELECT … LIMIT ?2` — SQLite returns only the needed rows, avoiding full-table scans; filtered path now has `limit * 10` safety cap
- `agent_nodes_cache: Arc<RwLock<Option<(Vec<(GraphNode, AgentData)>, Instant)>>>` added to `GraphStore`; `AGENT_NODES_CACHE_TTL = 30s`; both `open()` and `open_memory()` initialize to `None`
- `get_agent_nodes`: scoped read-lock check (`if let Some((cached, ts)) = &*cache && ts.elapsed() < AGENT_NODES_CACHE_TTL`); populates on miss under write-lock
- Invalidated in `add_node` and `update_node` when `node_type.type_name() == "agent"` — uses let-chain `&&` to collapse nested if (clippy compliant)
- 71 graph tests pass; clippy clean; zero new deps

**Cursor transcript import (Phase 30):**
- `crates/agent/src/import/mod.rs` + `crates/agent/src/import/cursor.rs` — new `import` sub-module in `graphirm-agent`
- `ParsedTurn { role, content, thinking }` + `ParsedTranscript { source_file, turns }` — parser output types
- `parse_transcript(source_file, text)` — line-by-line state machine; handles `user:/<user_query>`, `A:`, `[Thinking]`/`[/Thinking]`, `[Tool call]`, `[Tool result]`; tool blocks discarded; thinking preserved; trailing whitespace normalised
- `ImportResult { agent_id, turns_written, skipped }` + `write_transcript(store, transcript)` — idempotency via `find_imported_agent` (checks `source_file` in Agent node metadata); creates synthetic `Agent` node, then per-turn `Interaction` nodes with `Produces` + `RespondsTo` edges; `session_id` set on every Interaction
- `src/main.rs` — `Commands::ImportCursor { path, dry_run }` variant; handler accepts single `.txt` file or directory; `--dry-run` prints turn counts without writing
- 8 unit tests; zero new crate dependencies; `cargo clippy -D warnings` clean
- Usage: `graphirm import-cursor ~/.cursor/projects/…/agent-transcripts/` (imports all `.txt` files); re-import is a no-op

**Node-by-id TTL cache (Phase 29):**
- `crates/graph/src/store.rs` — `node_cache: Arc<RwLock<HashMap<NodeId, (GraphNode, Instant)>>>` added to `GraphStore` struct; initialized in both `open()` and `open_memory()`
- `const NODE_CACHE_TTL: Duration = Duration::from_secs(60)` — module-level constant
- `get_node`: checks cache first (scoped read-lock + let-chain `&&` for TTL check); on miss queries SQLite and populates cache (scoped write-lock)
- `update_node`: after successful `UPDATE`, removes entry from cache — ensures no stale reads
- No public API or signature changes; no new crate dependencies; all lock errors → `GraphError::LockPoisoned`

**SQLite performance indices (Phase 28):**
- `crates/graph/src/store.rs` — four indices added to `init_schema()` after the existing `idx_nodes_type`:
  - `idx_nodes_created_at ON nodes(created_at)` — covers `ORDER BY created_at` in agent/knowledge queries
  - `idx_edges_created_at ON edges(created_at)` — same for edge timeline queries
  - `idx_nodes_session_id ON nodes(json_extract(metadata, '$.session_id'))` — covers `WHERE session_id = ?` filter used in conversation thread + context engine
  - `idx_nodes_type_created ON nodes(node_type, created_at)` — composite index for the hottest pattern: `WHERE node_type = ? ORDER BY created_at` (context engine, `list_by_type`, knowledge retrieval)
- All use `CREATE INDEX IF NOT EXISTS` — safe on existing databases, applied on next open
- No API or public function changes; additive only

**Web-app design system + light/dark theme (Phase 27):**
- `web-app/src/styles/theme.css` — spacing scale (`--space-1` through `--space-8`), typography (`--font-sans`, `--font-mono`, `--text-xs/sm/base/lg/xl`, `--line-height`), surfaces (`--surface-0` through `--surface-3`), semantic colors (`--info`, `--warning`), additional edge color variables; `[data-theme="light"]` block overrides all color tokens for light theme; `body` font-family/size updated to use variables
- `web-app/src/hooks/useTheme.ts` — `useTheme()` hook: reads `localStorage` key `graphirm-theme`, falls back to `prefers-color-scheme`, sets `data-theme` attribute on `<html>`, persists on change
- `web-app/src/components/Toolbar.tsx` — theme toggle button (☀/◉) using `useTheme`; no new deps
- `web-app/src/components/edges/LabelledEdge.tsx` — `EDGE_COLORS` constant removed; replaced with `getEdgeColor(edgeType)` that reads `--edge-<type>` CSS variable via `getComputedStyle`, caches per-theme to avoid per-render DOM queries

**Graph node search / filter (Phase 20):**
- `NodeFilter` interface (`query: string`, `types: Set<string>`) + `EMPTY_FILTER` exported from `crates/web-app/src/hooks/useGraphData.ts`
- `applyFilterToNodes(nodes, graphNodes, filter)` helper — computes `visibleIds`, stamps `hidden: true` on non-matching React Flow nodes; group nodes hidden when all children hidden; annotation nodes never hidden
- `useGraphData` accepts `filter: NodeFilter` (4th param, default `EMPTY_FILTER`); returns `matchCount: number`; filter reactively applied in second `useEffect` without re-running layout
- Toolbar: search `<input>` + five type-pill buttons (`I A C T K`), `matchCount/total` counter, clear `✕` button — all controlled by filter state in `GraphCanvasInner`
- Ctrl+F (hover over graph pane) focuses search, Escape clears + blurs; existing `/` shortcut for chat unaffected

**Pinned Knowledge nodes (Phase 33):**
- `crates/graph/src/store.rs` — `list_pinned_knowledge(limit)`: `SELECT … WHERE node_type = 'knowledge' AND json_extract(metadata, '$.pinned') = 1 ORDER BY created_at ASC LIMIT ?1`; 3 tests
- `crates/agent/src/briefing.rs` — `build_pinned_summary(store, limit)`: formats pinned nodes as `- [pinned] entity: summary`; wired into `build_repo_briefing` between knowledge and lessons sections; 3 tests
- `crates/server/src/routes.rs` — `POST /api/knowledge`: creates Knowledge nodes directly via API; `CreateKnowledgeRequest` in `types.rs` with `entity`, `entity_type`, `summary`, `confidence` (default 1.0), `pinned` (default false), `session_id` (optional); 1 deserialization test
- Pinned nodes are global (not session-scoped) and always surfaced in repo briefing regardless of recency — used for coding conventions that the agent should always follow
- Manage via API: `curl -X POST http://localhost:3000/api/knowledge -d '{"entity": "rule-name", "entity_type": "convention", "summary": "...", "pinned": true}'`

**Risk areas:**

**Graph-aware tool execution (Phase 22):**
- `ImpactProvider` trait in `crates/tools/src/impact.rs` — `ImpactBrief`, `RiskLevel`, `extract_target_paths`
- `crates/tools/src/bash_paths.rs` — tree-sitter-bash AST walker extracts literal file paths from shell commands
- `GraphImpactProvider` in `crates/agent/src/impact.rs` — `rg --files-with-matches` for dependents, graph Knowledge query for prior notes
- Pre-execution hook in `execute_tools_parallel` (HITL/destructive path only)
- Risk scoring: LOW (0–2 deps, no notes), MEDIUM (3–9 deps OR notes), HIGH (10+ deps AND notes)
- Per-turn `HashMap<PathBuf, ImpactBrief>` cache — avoids re-analysis within a turn
- Threshold gate: empty briefs (0 deps, no notes) are suppressed — no noise
- `ImpactBrief` persisted as `Content` node with `content_type: "impact_brief"`, linked via `Reads` edge
- `pre_edit_impact: bool` in `AgentConfig` (default `true`)
- `max_output_tokens: Option<u32>` in `AgentConfig` — limits LLM response tokens per turn (separate from `max_tokens` which controls context window budget); falls back to `max_tokens`, then 8192; default.toml sets 1500
- All analysis is non-fatal — tool always executes regardless of impact analysis success
- 40 unit tests + 1 integration test, all passing
- `Arc<RwLock<StableGraph>>` — no deadlocks; acquire briefly, never across await
- Rust version must match spoke/CI (stable, currently 1.88)
- `OnnxExtractor` is cached process-wide via `get_or_init_onnx_extractor(model_dir)` — call this instead of `OnnxExtractor::new` directly; sessions load once per unique directory

**Session flow traces (Phase 25):**
- `crates/tools/src/session_trace.rs` — `SessionTraceTool`: `search` (groups `KnowledgeRetriever` / `search_knowledge` results by `session_id`, loads `get_session_chain`, formats turns with tool metadata) and `replay` (full chain for one session); keyword fallback + note when no embedding provider
- `crates/graph/src/store.rs` — `get_session_chain(session_id)` — interactions with matching `metadata.session_id`, `ORDER BY created_at ASC, id ASC`
- Registered in `build_tool_registry()` in `src/main.rs`

**Graph-diff tool (Phase 23):**
- `graph_diff` non-destructive tool in `crates/tools/src/graph_diff.rs` — two modes: `git` (resolve changed files via `git diff --name-only`) and `paths` (explicit file list)
- For each changed file: lists up to 20 dependent files via `rg --files-with-matches --fixed-strings`, queries `GraphStore.search_knowledge()` for cross-session Knowledge notes, computes risk via `compute_risk`
- Output: structured Markdown with `##`/`###` headers, dependents list, stale knowledge ⚠ warnings ("may be invalidated"), per-file risk level (Low/Medium/High)
- Registered in `build_tool_registry()` alongside other non-destructive tools
- 12 tests (validation, dependents via rg, cross-session knowledge, git integration)
