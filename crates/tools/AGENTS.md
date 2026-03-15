# graphirm-tools

Built-in tool implementations for the agent. Each tool implements the `Tool` trait — exposes a name,
description, JSON schema for parameters, and an async `execute` method. Tools are registered in
`ToolRegistry` and dispatched in parallel via `tokio::JoinSet`.

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
| `error.rs` | `ToolError` enum |

**Destructive tools** (`bash`, `write`, `edit`) block on the `HitlGate` when one is attached to the
session. Non-destructive tools (`read`, `grep`, `find`, `ls`) always run without confirmation.

---

## Integration Points

**Used by:** `graphirm-agent` — `workflow.rs` calls `ToolRegistry::execute()` for each tool call
the LLM emits; creates `Content` nodes in the graph for reads/writes

**Depends on:** `graphirm-graph` (Content node creation), `tokio`, `glob`

**Adding a new tool:**
1. Create `crates/tools/src/<name>.rs`, implement the `Tool` trait
2. Register it in `ToolRegistry::new()` in `registry.rs`
3. Add integration test to `tests/integration.rs`

---

## How to Test

```bash
cargo test -p graphirm-tools
```
