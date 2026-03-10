# graphirm

Graph-native coding agent. A better OpenCode, in Rust, where the graph is the product.

Every interaction, tool call, file read, and knowledge entity is a node. Every relationship between them is a typed edge. The graph is the session, the memory, the audit trail, and the context window — all at once.

---

## Why a graph?

Every other coding agent stores conversations as linear message arrays. Graphirm stores everything as a graph. That difference unlocks:

- **Relevance-scored context** — PageRank + recency decay + edge type weights + BFS distance, not "last N messages"
- **Cross-session memory** — Knowledge nodes persist; high-PageRank nodes surface in every future session automatically
- **Entity extraction** — GLiNER2 ONNX extracts entities and relationships for semantic search
- **Task DAGs** — tasks form a dependency graph, trackable and replayable
- **Multi-agent coordination** — subagents write nodes into a shared graph; any agent can traverse them
- **Session replay** — the graph *is* the audit trail
- **Soft escalation** — detects repeated tool calls and gracefully prompts synthesis before hard limits

---

## Quick start

### Terminal UI (TUI)

```bash
# Build
cargo build --release

# Set your API key (DeepSeek is the default provider)
export DEEPSEEK_API_KEY=sk-...

# Start a chat session (defaults to deepseek/deepseek-chat)
./target/release/graphirm chat

# Use a different provider/model
./target/release/graphirm chat --model openai/gpt-4o
./target/release/graphirm chat --model ollama/qwen2.5:72b

# Or set via environment variable
export GRAPHIRM_MODEL=deepseek/deepseek-chat
./target/release/graphirm serve
```

The graph database is stored at `~/.local/share/graphirm/graph.db` by default. Override with `--db /path/to/graph.db`.

### VS Code / Cursor extension

```bash
# Start the server (defaults to port 5555)
./target/release/graphirm serve

# Install the extension
cd graphirm-vscode && npm run build
# Then install the .vsix via "Extensions: Install from VSIX..." in Cursor/VS Code
```

Open the panel: `Ctrl+Shift+P` → **Graphirm: Open Panel**

The extension connects to `http://localhost:5555` by default. Change via `graphirm.serverUrl` in settings.

---

## Architecture

```
graphirm/
├── crates/
│   ├── graph/          # GraphStore — SQLite + petgraph, node/edge CRUD, PageRank, BFS
│   ├── llm/            # LLM provider trait — Anthropic, OpenAI, DeepSeek, Ollama, 17+ via rig-core
│   ├── tools/          # Built-in tools — bash, read, write, edit, grep, find, ls
│   ├── agent/          # Agent loop, context engine, compaction, multi-agent, HITL
│   ├── tui/            # Terminal UI — ratatui chat panel + graph explorer
│   └── server/         # HTTP API — axum REST + SSE streaming
├── graphirm-vscode/    # VS Code / Cursor extension — chat + live graph visualization
└── src/
    └── main.rs         # CLI entrypoint — chat, graph, serve subcommands
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

[server]
host = "127.0.0.1"
port = 5555
```

---

## Security model

- **Tool permissions** — Per-agent config can allow or deny tools by name (`permissions` in `AgentConfig`). Tools not listed default to allowed. Use `Deny` for tools you do not want an agent to call (e.g. deny `bash` for a subagent).
- **Destructive tools** — `write`, `edit`, and `bash` are treated as destructive (they can modify the filesystem or run arbitrary commands). Only these are subject to human-in-the-loop (HITL) when a gate is attached.
- **HITL gate** — When a session is created with an `HitlGate`, the agent loop blocks before executing any destructive tool call and waits for a decision. The server (or caller) receives the pending tool and can `Approve`, `Reject(reason)`, or `Modify(new_args)`. Without a gate, destructive tools run without confirmation.
- **Sandboxing** — There is no process or filesystem sandbox. The `bash` tool runs in the agent’s configured working directory. File tools (`read`, `write`, `edit`, `grep`, `find`, `ls`) operate on the host filesystem. Run Graphirm in a restricted environment (e.g. container or dedicated user) if you need isolation.

---

## Knowledge extraction

Post-turn entity extraction can use one of three backends (see `ExtractionConfig` in `crates/agent/src/knowledge/extraction.rs`):

| Backend | Cost | Latency | Accuracy | Offline |
|---------|------|---------|----------|---------|
| **Llm** (default) | API cost per turn | Higher | Good, flexible entity types | No |
| **Local** (GLiNER2 ONNX) | None | Low, CPU-bound | Good for configured entity types | Yes |
| **Hybrid** | API for descriptions only | Medium | GLiNER2 entities + LLM descriptions/relations | Partial |

- **When to use Llm** — Default; no setup. Best when you want maximum flexibility in entity types and don't mind API cost.
- **When to use Local** — Offline or zero-cost; run with `local-extraction` feature and set `backend: { local: { model_dir: "/path/to/gliner2" } }`. Populate the dir via the extraction API's model download.
- **When to use Hybrid** — You want fast entity spans from GLiNER2 plus richer descriptions and relationships from the LLM at lower cost than full LLM extraction.

Example (agent TOML) for local extraction:

```toml
[extraction]
enabled = true
backend = { local = { model_dir = "~/.cache/graphirm/gliner2" } }
entity_types = ["function", "api", "pattern", "decision"]
min_confidence = 0.7
```

### Structured response discovery (Phases 1–4 pipeline)

To discover what segment types (e.g. observation, reasoning, code) exist in LLM responses using GLiNER2:

1. **Export a corpus** from your graph (assistant turns only). Use `--limit` for a validation sample (e.g. 50–100 turns):
   ```bash
   graphirm export-corpus --db ~/.local/share/graphirm/graph.db -o corpus.jsonl
   graphirm export-corpus --db ~/.local/share/graphirm/graph.db --limit 100 -o sample.jsonl  # for Phase 4
   ```

2. **Run label exploration** (requires `--features local-extraction` and `GLINER2_MODEL_DIR`):
   ```bash
   cargo build --release --features local-extraction
   export GLINER2_MODEL_DIR=/path/to/gliner2-large-v1-onnx  # or after `graphirm model download`
   ./target/release/graphirm label-explore --corpus corpus.jsonl \
     --labels "observation,reasoning,code,instruction,answer" -o report.json
   ```

3. **Suggest a segment schema** (Phase 3) from the report:
   ```bash
   ./target/release/graphirm schema-suggest --report report.json -o schema.json
   ```
   Output: recommended segment types (real labels), merge suggestions (redundant pairs), and per-label verdicts (real / redundant / noise).

4. **Validate with human annotation** (Phase 4): get per-turn GLiNER2 spans, annotate a sample manually, then compare:
   ```bash
   ./target/release/graphirm predict-spans --corpus sample.jsonl --labels "observation,reasoning,code,answer" -o gliner_spans.jsonl
   # Annotate sample turns in human annotations JSONL (see plan for format: session_id, turn_index, segments: [{ type, start, end }]).
   ./target/release/graphirm validate-agreement --human annotations.jsonl --gliner gliner_spans.jsonl --threshold 75 -o agreement.json
   ```
   Pass criterion: agreement ≥ 75% (segment type + approximate boundary overlap).

5. Inspect the report and schema recommendation; see `docs/plans/2026-03-10-structured-llm-responses.md` for the full pipeline (Phases 1–6).

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
| 8 — HTTP Server | axum REST + SSE + VS Code/Cursor extension | ✅ done |
| 9 — Knowledge Layer | Entity extraction, background consciousness loop, HNSW | ✅ done |
| 10 — Web UI | Graph visualization | 🔲 pending |

**MVP (Phases 0–4 + 7):** ✅ complete — single-agent with graph-tracked interactions and TUI.

**v1.0 (Phases 0–7):** ✅ complete — multi-agent coordinator with graph-based context engine. Primary agent auto-injects the `delegate` tool; subagents run isolated loops with scoped tools and cancel propagation.

**v2.0 (Phases 0–9):** ✅ complete — HTTP server, SSE streaming, VS Code/Cursor extension, knowledge layer with HNSW cross-session memory.

**v3.0 (+ Phase 10):** Web UI pending.

---

## License

MIT
