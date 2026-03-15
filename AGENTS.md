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
         ├── graphirm-tools # built-in tools (bash, read, write, edit, grep, find, ls)
         └── graphirm-agent # agent loop, context engine, multi-agent, knowledge, HITL
              ├── graphirm-tui    # ratatui TUI (chat + graph explorer)
              └── graphirm-server # axum HTTP API + SSE
src/main.rs                 # CLI entrypoint (chat, graph, serve, export-corpus, ...)
graphirm-eval/              # evaluation harness (HTTP client only, no crate deps)
graphirm-vscode/            # VS Code / Cursor extension (TypeScript)
```

**Five node types:** `Interaction` (messages), `Agent` (instances), `Content` (files/output),
`Task` (DAG work items), `Knowledge` (extracted entities)

**Twelve edge types:** `RespondsTo`, `SpawnedBy`, `DelegatesTo`, `DependsOn`, `Produces`,
`Reads`, `Modifies`, `Summarizes`, `Contains`, `FollowsUp`, `Steers`, `RelatesTo`

---

## Code Layout

| Path | What |
|------|------|
| `src/main.rs` | CLI: `chat`, `graph`, `serve`, `export-corpus`, `label-explore`, `schema-suggest`, `predict-spans`, `validate-agreement` |
| `crates/graph/` | `GraphStore`, node/edge types, PageRank, BFS, HNSW vector index |
| `crates/llm/` | `LlmProvider` trait, Anthropic/OpenAI/DeepSeek/Ollama/OpenRouter impls, `MockProvider` |
| `crates/tools/` | `Tool` trait, `ToolRegistry`, parallel executor, bash/read/write/edit/grep/find/ls |
| `crates/agent/` | `run_agent_loop`, `build_context`, `Coordinator`, `HitlGate`, knowledge extraction |
| `crates/tui/` | `App`, chat panel, graph explorer, input handling |
| `crates/server/` | axum routes, SSE streaming, `AppState`, `SessionHandle`, SDK |
| `graphirm-eval/` | eval harness — drives agent via HTTP, checks task correctness |
| `graphirm-vscode/` | VS Code/Cursor extension (TypeScript) |
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
DEEPSEEK_API_KEY=sk-... ./target/release/graphirm serve

# Run eval harness (server must be running)
cargo run -p graphirm-eval -- --suite coding
```

Graph database stored at `~/.graphirm/graph.db` by default. Override with `--db /path/to/graph.db`.

---

## Key Conventions

**Rust:**
- Edition 2024, MSRV 1.85 — run `cargo fmt` and `cargo clippy` before every commit
- `thiserror` for error enums (one per crate), `anyhow` in `main.rs` only
- Never `unwrap()` in production — use `?` or `expect("context")`
- `tracing::info!` / `tracing::error!` for logging — never `println!`
- `async-trait` for async trait methods
- `Arc<RwLock<StableGraph>>` for in-memory graph — acquire locks briefly, never hold across await points

**Patterns:**
- New tool → implement `Tool` trait, register in `ToolRegistry::new()`
- New LLM provider → implement `LlmProvider` trait in `crates/llm/`
- `bash`, `write`, `edit` are destructive tools — subject to HITL gate
- Config lives in `config/default.toml`; `AgentConfig` is loaded from it at startup
- API keys via env vars: `DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`

---

## Current State

| Phase | What | Status |
|-------|------|--------|
| 0–9 | Scaffold → Knowledge layer (graph, LLM, tools, agent, multi-agent, context engine, TUI, HTTP, knowledge/HNSW) | ✅ done |
| 10 | Structured LLM response segments (parse → persist → GLiNER2 fallback → context filter → eval) | ✅ done |
| 11 | Web UI — graph visualization | 🔲 pending |

**Segment feature summary (Phase 10):**
- `SegmentConfig` in `AgentConfig` — enable per-session via `POST /api/sessions` with `enable_segments: true`
- LLM responses parsed into typed `Content` nodes (`code`, `reasoning`, `observation`, `plan`, `answer`) linked via `Contains` edges
- Primary path: structured JSON output from LLM (system prompt injected by `build_segment_prompt`)
- Fallback path: GLiNER2 ONNX span detection via `try_gliner2_fallback` (uses `ExtractionConfig.backend` model dir)
- Context engine: optional `segment_filter` in `ContextConfig` to include only specific segment types
- Eval coverage: `cargo run -p graphirm-eval -- --filter segments` (uses `GraphContainsContentType` verifier)
- See `docs/plans/2026-03-10-structured-llm-responses.md` and `docs/plans/2026-03-15-structured-segments-phase5-6.md`

**Risk areas:**
- `Arc<RwLock<StableGraph>>` — no deadlocks; acquire briefly, never across await
- Rust version must match spoke/CI (stable, currently 1.85)
- `OnnxExtractor::new` is expensive (~seconds); GLiNER2 fallback constructs it per-turn — cache if latency matters
