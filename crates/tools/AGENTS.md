# graphirm-tools

Built-in tool implementations for the agent. Each tool implements the `Tool` trait — exposes a name,
description, JSON schema for parameters, and an async `execute` method. Tools are registered in
`build_tool_registry()` in `src/main.rs` and dispatched in parallel via `tokio::JoinSet`.

---

## Key Components

| File | What |
|------|------|
| `registry.rs` | `ToolRegistry` — registers tools, dispatches calls by name |
| `executor.rs` | Parallel tool execution — runs multiple tool calls concurrently |
| `permissions.rs` | Per-agent allow/deny list — checked before dispatch |
| `bash.rs` | Shell command execution in working directory (**destructive**) |
| `read.rs` | File reading with line numbers |
| `write.rs` | File creation/overwrite (**destructive**) |
| `edit.rs` | Exact string replacement in files (**destructive**) |
| `grep.rs` | Regex search across files |
| `find.rs` | File name pattern search |
| `ls.rs` | Directory listing |
| `graph_query.rs` | Graph query — BFS traversal, node-type enumeration, keyword search, **semantic HNSW search** |
| `retriever.rs` | `KnowledgeRetriever` trait — decouples semantic search from agent crate |
| `script.rs` | `ScriptTool` + `PluginManifest` — TOML-defined plugins executed as shell commands |
| `error.rs` | `ToolError` enum |

**Destructive tools** (`bash`, `write`, `edit`, and any script plugin with `destructive = true`) block
on the `HitlGate` when one is attached to the session. Non-destructive tools always run without confirmation.
Destructiveness is determined by `Tool::is_destructive()` (trait method, default `false`) checked via
`ToolRegistry::is_destructive(name)`, with a legacy name fallback for the three built-in destructive tools.

---

## Integration Points

**Used by:** `graphirm-agent` — `workflow.rs` calls `ToolRegistry::execute()` for each tool call
the LLM emits; creates `Content` nodes in the graph for reads/writes

**Depends on:** `graphirm-graph` (Content node creation, graph queries), `tokio`, `glob`

**Adding a new built-in tool:**
1. Create `crates/tools/src/<name>.rs`, implement the `Tool` trait
2. Register it in `build_tool_registry()` in `src/main.rs` (not `ToolRegistry::new()`)
3. Add integration test to `tests/integration.rs`

**Adding a script plugin (no recompile):**
1. Create `~/.graphirm/plugins/<name>/plugin.toml` (see `examples/plugins/hello/` for format)
2. Set `destructive = true` in the manifest if the plugin modifies files or runs dangerous commands
3. Restart graphirm — plugin is loaded automatically

---

## How to Test

```bash
cargo test -p graphirm-tools
```
