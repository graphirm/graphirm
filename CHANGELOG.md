# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2026-03-06

### Added

- **Agent Trace Export CLI** — Export sessions to the open-standard Agent Trace interchange format (CC BY 4.0)
  - `graphirm export <session-id>` — export to stdout
  - `graphirm export <session-id> --output trace.json` — export to file
  - Supports linking AI conversations to code changes, documentation systems, and custom tools
  - Complete tool call nesting and metadata preservation

### Features in 3.0

- Graph-native session persistence with SQLite
- Multi-provider LLM support (Anthropic, DeepSeek, OpenAI, Ollama, 17+ via rig-core)
- VS Code / Cursor extension with two-pane chat + live graph visualization
- DAG timeline layout for session visualization
- Session restoration on server restart
- Agent Trace export to standard interchange format
- Single-binary deployment (no Docker, no runtime dependencies)

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
