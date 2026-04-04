# graphirm

Graph-native coding agent in Rust. Every interaction, tool call, file read, and knowledge entity is stored as a typed node in a persistent graph. The graph is the session, the memory, the audit trail, and the context window — all at once.

Single static binary. No Docker. No runtime dependencies.

---

## Why a graph?

Every other coding agent stores conversations as linear message arrays. Graphirm stores everything as a graph. That difference unlocks:

- **Relevance-scored context** — PageRank + recency decay + edge type weights + BFS distance, not "last N messages"
- **Cross-session memory** — Knowledge nodes persist; high-PageRank nodes surface in every future session automatically
- **Entity extraction** — GLiNER2 ONNX or LLM-based extraction of entities and relationships for semantic search
- **Task DAGs** — tasks form a dependency graph, trackable and replayable
- **Multi-agent coordination** — subagents write nodes into a shared graph; any agent can traverse them
- **Session replay** — the graph _is_ the audit trail
- **Soft escalation** — detects repeated tool calls and gracefully prompts synthesis before hard limits

---

## Quick start

### Terminal UI

```bash
# Build
cargo build --release

# Set your API key (DeepSeek is the default provider)
export DEEPSEEK_API_KEY=sk-...

# Start a chat session
./target/release/graphirm chat

# Use a different provider/model
./target/release/graphirm chat --model anthropic/claude-opus-4-5
./target/release/graphirm chat --model openai/gpt-4o
./target/release/graphirm chat --model ollama/qwen2.5:72b

# Or set via environment variable
export GRAPHIRM_MODEL=deepseek/deepseek-chat
./target/release/graphirm chat
```

The graph database is stored at `~/.graphirm/graph.db` by default. Override with `--db /path/to/graph.db`.

### Browser UI (React whiteboard + chat)

```bash
# Build the web UI once (or after changes)
cd web-app && npm install && npm run build && cd ..

# Start the server — web UI is served automatically
./target/release/graphirm serve

# Open http://localhost:3000 in your browser
```

Features: interactive graph whiteboard (pan/zoom/drag), node expansion with markdown + syntax highlighting, three layout modes (DAG, timeline, free), canvas annotations, steer-from-node, HITL approval cards, per-session auto-approve.

#### HTTP server security

`graphirm serve` expects a shared API key on every `/api/*` route except `/api/health`:

- Set **`GRAPHIRM_API_KEY`** in the environment before starting the server.
- REST clients send **`Authorization: Bearer <key>`**. Browser **`EventSource`** cannot set headers; use **`?token=<key>`** on the SSE URL (supported for streams).
- **`GRAPHIRM_ALLOWED_ORIGINS`** — optional comma-separated CORS allowlist for browser apps. If unset, any origin is allowed (local development only).

**Disabling shell on shared hosts:** in `config/default.toml`, under **`[agent]`**, set **`disable_bash = true`**. The agent will not receive the `bash` tool in its API schema, and any direct `bash` call fails with a clear error; a short system-prompt notice explains the restriction. Subagents spawned via `delegate` inherit the same lock when the parent session has it enabled.

**Per-session LLM token cap:** set **`max_session_tokens`** (e.g. `500000`) under **`[agent]`** to limit cumulative completion usage (each turn adds `input_tokens + output_tokens` from the provider). When the cap is exceeded, the in-flight assistant message is still saved, streaming ends with **`message_end`**, and the session status becomes **`token_cap_exceeded`**. `GET /api/sessions/:id` includes **`tokens_used`** and **`max_session_tokens`**. Omit the field or set it only in deploy config for unlimited usage (default).

**Web app:** put the same secret in `web-app/.env.local` as **`VITE_API_KEY=...`**, then `npm run build` (the value is embedded at build time).

**VS Code / Cursor extension:** **Graphirm → `graphirm.apiKey`** in settings.

Design notes and production checklist: [`docs/plans/2026-04-01-public-readiness-p1-design.md`](docs/plans/2026-04-01-public-readiness-p1-design.md).

### VS Code / Cursor extension

```bash
# Start the server
./target/release/graphirm serve

# Build and install the extension
cd graphirm-vscode && npm run build
# Then: Extensions: Install from VSIX... in Cursor/VS Code
```

Open the panel: `Ctrl+Shift+P` → **Graphirm: Open Panel**

---

## Supported providers

Set `GRAPHIRM_MODEL=<provider>/<model>` or pass `--model` at runtime.

| Provider | Env var | Example model |
|----------|---------|---------------|
| DeepSeek (default) | `DEEPSEEK_API_KEY` | `deepseek/deepseek-chat` |
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic/claude-opus-4-5` |
| OpenAI | `OPENAI_API_KEY` | `openai/gpt-4o` |
| OpenRouter | `OPENROUTER_API_KEY` | `openrouter/qwen/qwen3-coder:free` |
| Ollama (local) | — | `ollama/qwen2.5:72b` |
| 17+ more via rig-core | varies | see rig-core docs |

---

## Architecture

```
graphirm/
├── crates/
│   ├── graph/          # GraphStore — SQLite + petgraph, node/edge CRUD, PageRank, BFS, HNSW
│   ├── llm/            # LLM provider trait — Anthropic, OpenAI, DeepSeek, Ollama, 17+ via rig-core
│   ├── tools/          # Built-in tools — bash, read, write, edit, grep, find, ls, graph_query
│   ├── agent/          # Agent loop, context engine, multi-agent, knowledge extraction, HITL
│   ├── tui/            # Terminal UI — ratatui chat panel + graph explorer
│   └── server/         # HTTP API — axum REST + SSE streaming
├── graphirm-vscode/    # VS Code / Cursor extension — chat + live graph visualization
├── web-app/            # React + React Flow interactive whiteboard UI
├── web/                # Vanilla JS fallback UI (no build step)
└── src/
    └── main.rs         # CLI entrypoint — chat, graph, serve subcommands
```

### Graph data model

**Five node types:** `Interaction` (messages), `Agent` (instances), `Content` (files, command output), `Task` (DAG work items), `Knowledge` (extracted entities)

**Fifteen edge types:** `RespondsTo`, `SpawnedBy`, `DelegatesTo`, `DependsOn`, `Produces`, `Reads`, `Modifies`, `Summarizes`, `Contains`, `FollowsUp`, `Steers`, `RelatesTo`, `DerivedFrom`, `ApprovedBy`, `RejectedBy`

### Context engine

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
provider = "deepseek"
name = "deepseek-chat"
temperature = 0.7
max_tokens = 8192

[graph]
database_path = "~/.graphirm/graph.db"
max_connections = 20

[agent]
max_iterations = 50
parallel_tool_calls = true
soft_escalation_turn = 8
soft_escalation_threshold = 2
# workspaces_root = "/workspaces"  # enable per-session isolated workspace directories

[server]
host = "127.0.0.1"
port = 3000
```

---

## Security model

- **Tool permissions** — Per-agent config can allow or deny tools by name. Tools not listed default to allowed. Use `Deny` for tools you do not want an agent to call (e.g. deny `bash` for a subagent).
- **Destructive tools** — `write`, `edit`, and `bash` can modify the filesystem or run arbitrary commands. Only these are subject to human-in-the-loop (HITL) gating.
- **HITL gate** — The agent loop blocks before executing any destructive tool call and waits for a decision: `Approve`, `Reject(reason)`, or `Modify(new_args)`. Per-session auto-approve available via API or UI toggle.
- **Sandboxing** — There is no process or filesystem sandbox. Run Graphirm in a restricted environment (container, dedicated user) if you need isolation.

---

## Knowledge extraction

Post-turn entity extraction supports three backends:

| Backend | Cost | Latency | Offline |
|---------|------|---------|---------|
| **LLM** (default) | API cost per turn | Higher | No |
| **Local** (GLiNER2 ONNX) | None | Low, CPU-bound | Yes |
| **Hybrid** | API for descriptions only | Medium | Partial |

Example config for local (zero-cost, offline) extraction:

```toml
[extraction]
enabled = true
backend = { local = { model_dir = "~/.cache/graphirm/gliner2" } }
entity_types = ["function", "api", "pattern", "decision"]
min_confidence = 0.7
```

Build with `--features local-extraction` to enable the GLiNER2 ONNX backend.

---

## Per-session workspaces

Set `workspaces_root` in `config/default.toml` to give each session an isolated filesystem directory:

```toml
[agent]
workspaces_root = "/workspaces"
```

Sessions get `<root>/<workspace>/` as their working directory. The workspace name is stored in the graph and restored on server restart. Pass `"workspace": "my-project"` when creating a session via the API, or let it default to the sanitized session name.

---

## Tech stack

| Component | Library |
|-----------|---------|
| Language | Rust (stable, MSRV 1.85) |
| Graph DB | rusqlite + petgraph (custom layer) |
| Vector index | instant-distance (HNSW) |
| LLM | rig-core 0.31+ (17+ providers) |
| TUI | ratatui |
| HTTP | axum + SSE |
| Async | tokio |
| Web UI | React 19 + React Flow v12 + dagre |

---

## License

MIT
