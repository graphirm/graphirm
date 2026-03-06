# graphirm

Graph-native coding agent. A better OpenCode, in Rust, where the graph is the product.

Every interaction, tool call, file read, and knowledge entity is a node. Every relationship between them is a typed edge. The graph is the session, the memory, the audit trail, and the context window — all at once.

---

## Why a graph?

Every other coding agent stores conversations as linear message arrays. Graphirm stores everything as a graph. That difference unlocks:

- **Relevance-scored context** — PageRank + recency decay + edge type weights + BFS distance, not "last N messages"
- **Cross-session memory** — Knowledge nodes persist; high-PageRank nodes surface in every future session automatically
- **Task DAGs** — tasks form a dependency graph, trackable and replayable
- **Multi-agent coordination** — subagents write nodes into a shared graph; any agent can traverse them
- **Session replay** — the graph *is* the audit trail

---

## Quick start

```bash
# Build
cargo build --release

# Set your API key
export ANTHROPIC_API_KEY=sk-...

# Start a chat session
./target/release/graphirm chat

# Use a different provider/model
./target/release/graphirm chat --model deepseek/deepseek-chat
./target/release/graphirm chat --model ollama/qwen2.5:72b
```

The graph database is stored at `~/.local/share/graphirm/graph.db` by default. Override with `--db /path/to/graph.db`.

---

## Architecture

```
graphirm/
├── crates/
│   ├── graph/           # GraphStore — SQLite + petgraph, node/edge CRUD, PageRank, BFS
│   ├── llm/             # LLM provider trait — Anthropic, OpenAI, DeepSeek, Ollama, 17+ via rig-core
│   ├── tools/           # Built-in tools — bash, read, write, edit, grep, find, ls
│   ├── agent/           # Agent loop, context engine, compaction, multi-agent
│   ├── tui/             # Terminal UI — ratatui chat panel
│   └── server/          # HTTP API — axum REST + SSE
├── graphirm-vscode/     # VS Code/Cursor extension (TypeScript)
│   ├── src/             # Extension host — GraphirmPanel, ApiClient, SseSubscriber, statusBar
│   └── media/           # Webview frontend — d3-force graph, chat, session management
└── src/
    └── main.rs          # CLI entrypoint — chat, serve subcommands
```

### Graph data model

**Five node types:** `Interaction` (messages), `Agent` (agent instances), `Content` (files, command output), `Task` (DAG work items), `Knowledge` (extracted entities)

**Twelve edge types:** `RespondsTo`, `SpawnedBy`, `DelegatesTo`, `DependsOn`, `Produces`, `Reads`, `Modifies`, `Summarizes`, `Contains`, `FollowsUp`, `Steers`, `RelatesTo`

### Context engine (Phase 6)

Each context window is built by scoring every candidate node on four signals:

| Signal | Weight | Formula |
|--------|--------|---------|
| Recency | 0.3 | `e^(-decay × hours_since_creation)` |
| Edge weights | 0.2 | Σ `weight(edge_type)` per edge touching node |
| Graph distance | 0.3 | `1 / (1 + BFS_hops_from_current_turn)` |
| PageRank | 0.2 | Node importance across the full graph |

A greedy knapsack fills the token budget with the highest-scored nodes. Knowledge nodes compete on the same score as conversation turns — agent identity emerges from the graph, not from a hardcoded system prompt.

---

## Configuration

`config/default.toml`:

```toml
[model]
provider = "anthropic"
name = "claude-sonnet-4-20250514"
temperature = 0.7
max_tokens = 8192

[graph]
database_path = "~/.graphirm/graph.db"
max_connections = 4

[agent]
max_iterations = 50
parallel_tool_calls = true

[server]
host = "127.0.0.1"
port = 3000
```

---

## Tech stack

| Component | Library | Version |
|-----------|---------|---------|
| Language | Rust | stable |
| Graph DB | rusqlite + petgraph | custom layer, MIT |
| LLM | rig-core | 0.31+ (17+ providers) |
| TUI | ratatui | latest |
| HTTP | axum | latest |
| Async | tokio | latest |

No Docker. No runtime dependencies. Single static binary.

---

## Build status

| Phase | What | Status |
|-------|------|--------|
| 0 — Scaffold | Cargo workspace, CI, tooling | ✅ done |
| 1 — Graph Store | GraphStore, node/edge CRUD, PageRank, BFS | ✅ done |
| 2 — LLM Provider | rig-core integration, 17+ providers, streaming | ✅ done |
| 3 — Tool System | bash, read, write, edit, grep, find, ls | ✅ done |
| 4 — Agent Loop | Async state machine, tool execution, session | ✅ done |
| 5 — Multi-Agent | Coordinator, delegate tool, subagent spawning, TaskStatus enum | ✅ done |
| 6 — Context Engine | Relevance scoring, token budgets, compaction | ✅ done |
| 7 — TUI | ratatui chat panel | ✅ done |
| 8 — HTTP Server | axum REST + SSE, request logging middleware | ✅ done |
| 9 — Knowledge Layer | Entity extraction, background consciousness loop, HNSW | ✅ done |
| 10 — Usage Discovery | Real-LLM dogfooding, API usage analysis, context bug fix | ✅ done |
| 11 — VS Code Extension | Two-pane chat + d3-force graph, SSE streaming, VSIX installable in Cursor/code-server | ✅ done |

**MVP (Phases 0–4 + 7):** ✅ complete — single-agent with graph-tracked interactions and TUI.

**v1.0 (Phases 0–7):** ✅ complete — multi-agent coordinator with graph-based context engine.

**v2.0 (Phases 8–10):** ✅ complete — HTTP API server, knowledge layer, real-LLM dogfooding with bug fixes.

**v2.1 (Phase 11):** ✅ complete — VS Code/Cursor extension with live graph visualization, installable as .vsix.

### What Phase 10 (Usage Discovery) found

A significant context-engine bug was discovered during real DeepSeek dogfooding: `Content` nodes created by tools (e.g. `read` creating a file node) were sorted chronologically into the conversation thread, interleaving between assistant tool-call messages and their tool-result messages. OpenAI-compatible APIs (including DeepSeek) strictly require that tool results immediately follow their tool calls — any intervening message causes a 400 error. Fixed in `crates/agent/src/context.rs`.

Real-LLM turn latencies on a Hetzner spoke: **15–40s per turn** (vs. <1ms for MockProvider). These findings drove Phase 11's UX decisions.

### What Phase 11 (VS Code Extension) built

A TypeScript VS Code/Cursor extension (`graphirm-vscode/`) that connects to a locally running `graphirm serve` and provides:

- **Two-pane layout** — chat (left 45%) + live d3-force graph (right 55%), always visible
- **SSE-first** — extension host subscribes to `/api/sessions/:id/events`; on `agent.done` triggers parallel refresh of messages + graph
- **Persistent thinking state** — spinner + Abort button held until `agent.done` arrives (handles 15–40s real LLM turns)
- **Status bar item** — `idle / thinking… / failed / disconnected` always visible, click to open panel
- **Click-to-inspect nodes** — inline node detail panel with JSON data + "Expand subgraph" with depth slider
- **Session management** — create/select sessions via VS Code's native `showInputBox` (not blocked `prompt()`)

**Install:**

```bash
# Build (requires Node 20)
cd graphirm-vscode
nvm use 20 && npm install && npm run build
npx @vscode/vsce package --no-dependencies
# → graphirm-0.1.0.vsix

# Cursor: Extensions → ⋯ → Install from VSIX
# code-server: code-server --install-extension graphirm-0.1.0.vsix
```

**Use:** Start `graphirm serve`, then open Cursor command palette → `Graphirm: Open Panel`.

---

## License

MIT
