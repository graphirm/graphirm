# graphirm-eval

Evaluation harness. Drives a running graphirm server via HTTP, executes task suites, and measures
agent correctness across six domains: coding, knowledge extraction, cross-session memory, graph
structure, adversarial robustness, and structured response segments.

**Does not depend on any graphirm Rust crate** — communicates via HTTP only. Standalone binary.

---

## Key Components

| File | What |
|------|------|
| `main.rs` | CLI entrypoint — `--filter <tag>` to select tasks, `--list` to enumerate, `--experiment <ID>` → `results/experiments/<ID>/eval.json` |
| `client.rs` | `GraphirmClient` — thin HTTP wrapper (sessions, prompts, graph queries) |
| `harness.rs` | `TestHarness` — spawns server subprocess, runs tasks, collects results |
| `report.rs` | Result aggregation, pass/fail counts, JSON output |
| `task.rs` | `EvalTask` struct + `Verifier` enum |
| `tasks/coding.rs` | Coding tasks — file creation, editing, test execution |
| `tasks/knowledge.rs` | Knowledge extraction tasks — entity types, confidence thresholds |
| `tasks/memory.rs` | Cross-session memory tasks — knowledge persists across sessions |
| `tasks/graph.rs` | Graph structure tasks — node/edge counts, traversal correctness |
| `tasks/adversarial.rs` | Adversarial tasks — injection attempts, tool misuse |
| `tasks/segments.rs` | Segment tasks — `segment-extraction` verifies Content nodes; `segment-filter-context` smoke-tests the `segment_filter` request plumbing |

---

## Verifiers

| Variant | What it checks |
|---------|---------------|
| `ResponseContains { substring }` | Final assistant message contains substring (case-insensitive) |
| `ResponseContainsAny { substrings }` | Final message contains at least one substring |
| `ResponseNotContains { substring }` | Final message does NOT contain substring |
| `CommandSucceeds { command, args }` | Shell command exits 0 |
| `FileContains { path, substring }` | File exists and contains substring |
| `ResponseContainsCommandOutput { command, args }` | Final message contains stdout of command |
| `KnowledgeNodeCount { min_count }` | Graph has ≥ N knowledge nodes |
| `GraphContains { min_nodes, type_name }` | Graph has ≥ N nodes and at least one with `node_type.type == type_name` |
| `GraphContainsContentType { content_type }` | Graph has at least one Content node with `node_type.content_type == content_type` |
| `All(verifiers)` | Every sub-verifier must pass |

---

## Session Options

| Field | Type | Effect |
|-------|------|--------|
| `EvalTask.enable_segments` | `bool` | When `true`, creates the session with `enable_segments: true`, activating segment parsing |
| `EvalTask.segment_filter` | `Option<Vec<String>>` | When set, passes `segment_filter` to the session; restricts context window to those segment types |

Segment tasks use `GraphContainsContentType` to verify Content nodes were persisted,
and `ResponseContains` as a smoke check that the agent's output is intact.

---

## Integration Points

**Requires:** graphirm binary at `target/release/graphirm`. The harness spawns that binary on `127.0.0.1:19555` and refuses any other host. `app.graphirm.ai` is not an eval target: its limiter is shared with users. The child process gets `GRAPHIRM_RATE_LIMIT_BURST=1000`. Production leaves that variable unset, so the server default stays 60.

**Does not import:** Any `graphirm-*` crate — pure HTTP client

**Depends on:** `reqwest`, `tokio`, `serde_json`, `anyhow`, `clap`, `chrono`, `tempfile`

---

## How to Run

```bash
# Build the binary first (with local-extraction for GLiNER2 segment fallback)
cargo build --release --features local-extraction

# Run all tasks (harness spawns its own server automatically)
cargo run -p graphirm-eval

# Filter by tag
cargo run -p graphirm-eval -- --filter coding
cargo run -p graphirm-eval -- --filter knowledge
cargo run -p graphirm-eval -- --filter segments
cargo run -p graphirm-eval -- --filter graph

# List available tasks without running
cargo run -p graphirm-eval -- --list

# Skip memory tasks (require EMBEDDING_BACKEND)
cargo run -p graphirm-eval -- --skip-memory
```

Results are written to `results/latest.json`.

The harness waits on `GET /api/events/{session}` (opened before the prompt) instead of polling the session. A 429 is retried with the server's `Retry-After`, and never sooner than one second. If the retries run out, the task is an infrastructure error and is left out of the pass rate. A prompt wait that ends before any assistant message is the same kind of error: the task is retried up to three times, then excluded. A timeout after the model has already replied stays a task failure, and the result records how many assistant messages were stored. If that timeout's verifier already passes, the reason is `finished but didn't stop`. Command-output checks ignore thousands separators, so `3,537` matches `3537`. A failed agent loop is still a task failure.

Startup refuses the run unless `/api/health` reports an extraction backend of `llm`, `local`, or `hybrid`, and `memory` of `on`. The child inherits `EMBEDDING_BACKEND` (do not clear it). Each task gets its own workspace under `workspaces_root`. The harness copies `crates/` and `src/` into that directory and deletes the shared `/tmp/eval_*` files first, so a run neither writes the checkout nor inherits the previous run's files. A task that does not pass stores `final_answer` and `tool_trace` on the result before the session is deleted.
