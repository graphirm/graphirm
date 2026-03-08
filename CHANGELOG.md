# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Default LLM provider switched to DeepSeek** — all hardcoded `anthropic/claude-sonnet-4-20250514` defaults replaced with `deepseek/deepseek-chat`. Affected: `config/default.toml`, `AgentConfig::default()`, CLI `--model` flag default, and server `GRAPHIRM_MODEL` fallback. Set `DEEPSEEK_API_KEY` in your environment (or `.env`) to use the defaults out of the box. Anthropic remains a fully supported provider; pass `--model anthropic/<model>` or set `GRAPHIRM_MODEL=anthropic/<model>` to use it.

### Fixed

- **Workflow and multi-agent tests hang indefinitely** — root cause was a connection pool deadlock in `GraphStore::list_recent_nodes`. The method held the single r2d2 pool connection while calling `get_node()` in a loop, which tried to acquire a second connection from the same pool — deadlocking the calling thread. Fixed by rewriting the method as a single SQL query that fetches all five columns at once (matching the pattern already used in `get_agent_nodes`). This unblocked all 136 agent unit tests and all 59 server unit tests; `cargo test --lib` across the workspace now completes in under 0.3 s per crate.

- **VS Code/Cursor extension panel fails on new session** — two root causes corrected:
  - **Port mismatch:** Server `--port` default was `3000` but extension expected `5555`; all API calls hit connection refused. Server default changed to `5555` to align with the extension's `graphirm.serverUrl` default.
  - **Field name mismatch:** `ApiClient.createSession` was sending `{ name }` but the server reads `body.agent`, so the user-provided session name was silently discarded. Fixed to send `{ agent: name }`. Additionally, the `Session` TypeScript interface declared `name` (undefined at runtime) instead of `agent`; fixed across `ApiClient.ts`, `GraphirmPanel.ts`, and `sessions.js`. Status type also expanded to match all server variants (`completed`, `failed`, `cancelled`).
- **Broken test compilation** — `cargo test --workspace` now compiles and passes cleanly across all crates
  - Added `SessionStatus` and `SessionMetadata` types to `graphirm-agent` crate (were planned but never implemented)
  - Added `get_agent_nodes()` query to `GraphStore` for session restoration on startup
  - Made `session` and `request_log` modules public in `graphirm-server`
  - Declared and wired `middleware` module — `request_logging` middleware now active in `create_router`
  - Added `tempfile` dev-dependency to server crate for request log tests

---

## [3.0.0] - 2026-03-07

### Added

- **Agent Trace Export CLI** — Export sessions to the open-standard Agent Trace interchange format (CC BY 4.0)
  - `graphirm export <session-id>` — export to stdout
  - `graphirm export <session-id> --output trace.json` — export to file
  - Supports linking AI conversations to code changes, documentation systems, and custom tools
  - Complete tool call nesting and metadata preservation
- **Soft Escalation in Agent Loop** — Graceful intervention when repeated tool calls detected
  - Detects repeated tool calls within configurable sliding window
  - Injects synthesis directive as system message instead of hard recursion limit
  - Allows model final attempt to synthesize findings before enforcement
  - Configurable via `soft_escalation_turn` and `soft_escalation_threshold`
  - Emits `SoftEscalationTriggered` events for observability
  - New metrics endpoint: `GET /api/sessions/{id}/escalations`
  - Fixed synthesis context bug for proper LLM message formatting
  - Database pool expanded from 4 to 20 connections for concurrent sessions
- **DAG Timeline Layout** — Visual redesign of graph explorer with temporal and type-based node positioning
- **Session Restoration** — Sessions automatically survive server restarts with full history preserved
- **Landing Page** — graphirm.ai static site with installation and usage documentation
- **Phase 9: Knowledge Layer** ✅ INTEGRATED
  - GLiNER2 ONNX entity extraction for knowledge graphs
  - HNSW vector search for cross-session memory retrieval
  - Hybrid extraction backend (local + cloud options)
  - Cross-session context injection for improved relevance
  - 12 completed tasks with full test coverage
  - Merged into main on 2026-03-07

### Platform Features

- Graph-native session persistence with SQLite
- Multi-provider LLM support (Anthropic, DeepSeek, OpenAI, Ollama, 17+ via rig-core)
- VS Code / Cursor extension with two-pane chat + live graph visualization
- Single-binary deployment (no Docker, no runtime dependencies)
- REST API with SSE streaming
- Cross-session memory with knowledge layer

---

## [2.0.0] - Previous

### Added

- HTTP server with REST API and SSE events
- Knowledge layer with cross-session memory
- Web UI with graph visualization

---

## [1.0.0] - Previous

### Added

- Multi-agent framework
- Graph-based context engine
- TUI with ratatui

---

## [MVP] - Previous

### Added

- Single-agent implementation
- Graph-native data model (rusqlite + petgraph)
- LLM provider abstraction (rig-core)
- Tool execution system with tokio
- TUI interface
