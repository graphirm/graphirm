# graphirm

**Coding agents forget. Graphirm remembers.**

A graph-native coding agent in Rust. Every interaction, tool call, and file read is a node. Every relationship is a typed edge. The graph is the session, the memory, and the context window — all at once.

## Features

- **Cross-session memory** — knowledge persists; high-value nodes surface automatically in future sessions
- **Relevance-scored context** — PageRank + recency + graph distance, not "last N messages"
- **Multi-provider** — Anthropic, DeepSeek, OpenAI, Ollama, 17+ via rig-core
- **VS Code extension** — two-pane chat + live graph visualization
- **Single binary** — no Docker, no runtime deps
- **Session persistence & restoration** — Sessions automatically survive server restarts; full history restored from SQLite on startup with zero manual steps

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

## How to use

**With the extension:** In the panel, create a session (or pick one from the dropdown), type a message and press Enter. The left pane shows the conversation; the right pane shows the live graph. Click a node to see its details; use “Expand subgraph” to load more of the graph around it.

**Terminal only:** Run `./target/release/graphirm chat` for a TUI in the current directory. Same graph backend; sessions are per process.

The graph database lives at `~/.local/share/graphirm/graph.db` by default. Override with `--db /path/to/graph.db` when starting `serve` or `chat`.

## Architecture

```
crates/
  graph/  llm/  tools/  agent/  tui/  server/
graphirm-vscode/
```

Built with Rust, SQLite, petgraph, rig-core, axum, ratatui.

## License

MIT
