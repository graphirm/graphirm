# graphirm-eval

Automated evaluation harness for Graphirm. Drives the agent via its HTTP API and measures:

- **Task completion** — agent correctly uses tools (bash, grep, read, write, edit) and produces the expected outcome
- **Knowledge extraction** — entities from conversation turns are persisted as graph nodes and queryable
- **Cross-session memory** — high-PageRank knowledge from a previous session is recalled in a new session (requires embedding backend)
- **Graph structure** — node/edge counts and types match expectations after a run

Inspired by SWE-bench — every task has explicit, programmatically verifiable pass/fail criteria. No bash scripts, no manual inspection.

## Quick start

```bash
# Build graphirm first
cargo build --release

# List available tasks
cargo run -p graphirm-eval -- --list

# Run basic + tool-use tasks
source .env
cargo run -p graphirm-eval -- --filter basic

# Run coding tasks
cargo run -p graphirm-eval -- --filter coding

# Run all non-memory tasks
cargo run -p graphirm-eval -- --skip-memory

# Run all tasks (including memory — requires EMBEDDING_BACKEND)
export EMBEDDING_BACKEND=mistral/codestral-embed
cargo run -p graphirm-eval

# Results are written to:
#   results/latest.json   — machine-readable
#   results/latest.md     — human-readable Markdown table
```

## Task categories

| Tag | Count | What it tests |
|---|---|---|
| `basic` | 2 | Tool use — bash, grep |
| `coding` | 1 | File creation, code generation |
| `multi-turn` | 1 | State persistence across turns |
| `knowledge` | 2 | Entity extraction into graph nodes |
| `memory` | 1 | Cross-session recall (requires embedding backend) |
| `graph` | 1 | Graph node/edge structure correctness |

## Architecture

```
graphirm-eval/
├── src/
│   ├── main.rs          # CLI: --binary, --filter, --report, --skip-memory, --list
│   ├── client.rs        # GraphirmClient — thin reqwest wrapper for the REST API
│   ├── harness.rs       # TestHarness — spawns server, runs tasks, collects results
│   ├── task.rs          # EvalTask, CrossSessionTask, Verifier, TaskResult
│   ├── report.rs        # Results → Markdown + JSON
│   └── tasks/
│       ├── mod.rs       # all_tasks() → Vec<EvalTask>
│       ├── coding.rs    # Categories A + B (basic coding, multi-turn)
│       ├── knowledge.rs # Category C (entity extraction)
│       ├── memory.rs    # Category D (cross-session memory recall)
│       └── graph.rs     # Category E (graph structure integrity)
```

The harness:
1. Spawns a `graphirm serve` process against a temporary SQLite database
2. Polls `GET /api/health` until the server is ready (up to 10s)
3. For each task: creates a session, sends prompts sequentially, waits for idle
4. Applies the verifier (response substring, file contents, command exit code, API query)
5. Writes results to `results/latest.json` and `results/latest.md`
6. Kills the server on drop

## Verifier types

| Verifier | Checks |
|---|---|
| `ResponseContains` | Last assistant message contains a substring (case-insensitive) |
| `FileContains` | A file on disk exists and contains a substring |
| `CommandSucceeds` | A shell command exits with code 0 |
| `KnowledgeNodeCount` | `GET /api/graph/{session}/knowledge` returns ≥ N nodes |
| `GraphContains` | `GET /api/graph/{session}` has ≥ N nodes with a given type |
| `All` | All listed verifiers pass |

## Adding tasks

Add a new `EvalTask` to the appropriate file in `src/tasks/` and ensure `all_tasks()` in `mod.rs` picks it up. Each task is a Rust struct with no boilerplate — just id, prompts, and a verifier.

## Future extensions

- **SWE-bench adapter** — clone a repo at a failing commit, give the agent the issue, run `cargo test` to verify
- **Model comparison** — run the same suite against multiple `GRAPHIRM_MODEL` values
- **Regression CI** — GitHub Actions workflow running `--filter basic` on every PR
- **Turn efficiency metric** — track average turns per category as a model quality signal
