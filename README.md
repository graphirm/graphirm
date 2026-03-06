# graphirm

**Coding agents forget. Graphirm remembers.**

A graph-native coding agent in Rust. Every interaction, tool call, and file read is a node. Every relationship is a typed edge. The graph is the session, the memory, and the context window — all at once.

## Features

- **Cross-session memory** — knowledge persists; high-value nodes surface automatically in future sessions
- **Relevance-scored context** — PageRank + recency + graph distance, not "last N messages"
- **Multi-provider** — Anthropic, DeepSeek, OpenAI, Ollama, 17+ via rig-core
- **VS Code extension** — two-pane chat + live graph visualization
- **Single binary** — no Docker, no runtime deps

## Quick start

```bash
cargo build --release

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
./target/release/graphirm serve

# or DeepSeek
export DEEPSEEK_API_KEY=sk-...
export GRAPHIRM_MODEL=deepseek/deepseek-chat
./target/release/graphirm serve
```

Then open VS Code / Cursor → `Ctrl+Shift+P` → **Graphirm: Open Panel**.

## Architecture

```
crates/
  graph/  llm/  tools/  agent/  tui/  server/
graphirm-vscode/
```

Built with Rust, SQLite, petgraph, rig-core, axum, ratatui.

## License

MIT
