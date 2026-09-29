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
         ├── graphirm-tools # built-in tools (bash, read, write, edit, grep, find, ls, fetch_url, graph_query, diff, read_many)
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
| `crates/tools/` | `Tool` trait, `ToolRegistry`, parallel executor, bash/read/write/edit/grep/find/ls/fetch_url/graph_query/diff/read_many/cargo_check |
| `crates/agent/` | `run_agent_loop`, `build_context`, `Coordinator`, `HitlGate`, knowledge extraction |
| `crates/tui/` | `App`, chat panel, graph explorer, input handling |
| `crates/server/` | axum routes, SSE streaming, `AppState`, `SessionHandle`, SDK, static file serving |
| `graphirm-eval/` | eval harness — drives agent via HTTP, checks task correctness |
| `graphirm-vscode/` | VS Code/Cursor extension (TypeScript) |
| `web-app/` | React phone-first decision chat (Chat / Review / Rules / Graph, 430px column). The React Flow whiteboard stays on the Graph tab and in the legacy two-pane layout (`localStorage` key `graphirm.layout`). Vite, TypeScript. |
| `web/` | Vanilla JS browser UI (legacy fallback, still served if `web-app/dist/` not present) |
| `config/default.toml` | default model, agent, knowledge, graph, TUI, server settings |
| `experiments/` | optional ad-hoc probes (not workspace members unless listed); see `experiments/AGENTS.md` |

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

**HTTP server security (`serve`):** set **`GRAPHIRM_API_KEY`** — required; clients use **`Authorization: Bearer <key>`** on REST calls and **`?token=<key>`** on SSE URLs. **`GRAPHIRM_ALLOWED_ORIGINS`** (comma-separated) restricts CORS for browser clients; when unset, any origin is allowed (local dev). React web-app: **`VITE_API_KEY`** in `web-app/.env.local`, then rebuild. VS Code extension: **`graphirm.apiKey`**. For **shared or public** instances, set **`disable_bash = true`** in **`[agent]`** in `config/default.toml` (or your deploy config) so the shell tool is refused and omitted from the model’s tool list; optional **`max_session_tokens`** caps cumulative LLM usage (input+output per completion) per session — when exceeded, the last assistant message is still saved and the session ends with status **`token_cap_exceeded`**. Restored sessions use the server’s current config on restart. See `config/default.toml` header comments and `docs/plans/2026-04-01-public-readiness-p1-design.md`.

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
- `bash`, `write`, `edit` are destructive tools — subject to HITL gate (unless auto-approve is enabled); optional **`disable_bash`** in **`[agent]`** disables `bash` entirely for public servers (see Build & Test → HTTP server security)
- `delegate_pi` — Pi (`@earendil-works/pi-coding-agent`) as an external coding executor; registered when **`[agent.pi].enabled`**; Pi's `bash`/`write`/`edit` are scored observe-only (`hitl_judge.action = "observed"`, not gated); hidden with `bash` when `disable_bash`
- `read`, `grep`, `find`, `ls`, `graph_query` are non-destructive — always run without confirmation
- **Planning ↔ artifacts:** `graph_query` `project` **`link_session`** links the session Agent to a planning Knowledge node (`DerivedFrom`). With **`auto_link_write_to_planning = true`** (default in `AgentConfig`), **`write`** / **`edit`** on **`file`** Content nodes add **`relates_to`** (planning → file, metadata **`artifact_link`**: `implements`) when that link exists; the same flag auto-links **delegated `Task`** nodes (`DelegatesTo` from parent agent) when **`link_session`** is present. **Manual** **`link_content`** (file) and **`link_task`** (delegation task; parent or subagent session) remain available. Set **`auto_link_write_to_planning = false`** in **`[agent]`** to disable auto-linking.
- `read` auto-truncates files > 300 lines when no `offset`/`limit` is provided — returns first 300 lines + notice; callers should use `offset`/`limit` for targeted reads
- Auto-approve: `POST /api/sessions/{id}/auto-approve` with `{ "enabled": true }` — skips HITL gating for all destructive tools in that session
- Per-session workspaces: set `workspaces_root` in `[agent]` config; `POST /api/sessions` accepts optional `"workspace"` name (defaults to sanitized session name); workspace directory is auto-created; workspace name persisted in Agent node metadata and restored on restart; response includes `workspace` and `workspace_path` when active
- Config lives in `config/default.toml`; `AgentConfig` is loaded from it at startup; `workspaces_root` in `[agent]` — optional root; when set, each session gets an isolated subdirectory `<root>/<workspace>/`
- Pinned Knowledge nodes: `POST /api/knowledge` with `"pinned": true` creates global convention/rule nodes that always surface in `repo_briefing` regardless of recency; `GET /api/knowledge/pinned` lists them (optional `?limit=N`); `list_pinned_knowledge(limit)` in GraphStore; `build_pinned_summary` in briefing
- Model routing: `[agent.routing]` in TOML with `cheap`/`smart` model strings and `[[agent.routing.rules]]`; `ModelRouter` selects per turn based on `TurnSignals` (turn number, tool errors, message complexity); both tiers must use the same provider backend; routing decisions stored as metadata on Interaction nodes
- API keys via env vars: `DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`

---

## Current State

Phases 0–55 are complete: graph store, LLM providers, tools, agent loop with
model-tier routing (`strategy/`), multi-agent delegation, PageRank + recency
context engine, compaction, TUI, HTTP server + phone-first decision chat (whiteboard remains on the Graph tab and in the legacy layout), structured
response segments with GLiNER2 fallback, cross-session memory (HNSW),
planning/tasks, trace analysis, `fetch_url`. Full per-phase table with dates:
`docs/completion-log.md` (bottom). Open work: `docs/backlog.md`.

**Governance docs — update in the same commit as the work, not afterwards:**
- Start a backlog item → create `docs/plans/YYYY-MM-DD-<topic>.md`, link it from the item
- Ship it → `### ✅` one-liner in `docs/backlog.md` + dated entry at the top of `docs/completion-log.md`
- Make a non-obvious decision (architecture, dependency, convention, rejected alternative) → dated entry in `docs/journal.md`
- Discover work you don't start → add it to `docs/backlog.md` with size + priority

Where judgement is made in the agent loop, and which of those seats a typed
decision model (Jev) fits — scored 2026-09-27/28:
`~/codeporate-connect/docs/evaluations/2026-09-27-jev-where-in-graphirm.md`.
Do not change context selection, compaction, memory ranking, or knowledge
extraction on intuition — they are measured by `graphirm-eval`.
