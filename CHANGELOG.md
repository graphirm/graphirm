# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

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
