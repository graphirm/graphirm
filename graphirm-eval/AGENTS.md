# graphirm-eval

Evaluation harness. Drives a running graphirm server via HTTP, executes task suites, and measures
agent correctness across five domains: coding, knowledge extraction, cross-session memory, graph
structure, and adversarial robustness.

**Does not depend on any graphirm Rust crate** — communicates via HTTP only. Standalone binary.

---

## Key Components

| File | What |
|------|------|
| `main.rs` | CLI entrypoint — suite selection, server URL, output format |
| `client.rs` | `GraphirmClient` — thin HTTP wrapper (creates sessions, sends prompts, reads SSE) |
| `harness.rs` | `EvalHarness` — session lifecycle, task dispatch, result collection |
| `report.rs` | Result aggregation, pass/fail counts, metrics JSON output |
| `task.rs` | `EvalTask` trait — every task implements `run(&client) -> TaskResult` |
| `tasks/coding.rs` | Coding tasks — file creation, editing, test execution |
| `tasks/knowledge.rs` | Knowledge extraction tasks — entity types, confidence thresholds |
| `tasks/memory.rs` | Cross-session memory tasks — knowledge persists across sessions |
| `tasks/graph.rs` | Graph structure tasks — node/edge counts, traversal correctness |
| `tasks/adversarial.rs` | Adversarial tasks — injection attempts, tool misuse |

---

## Integration Points

**Requires:** A running `graphirm serve` instance (default `http://localhost:3000`)

**Does not import:** Any `graphirm-*` crate — pure HTTP client

**Depends on:** `reqwest`, `tokio`, `serde_json`, `anyhow`, `clap`, `chrono`

---

## How to Run

```bash
# Start server in one terminal
DEEPSEEK_API_KEY=sk-... ./target/release/graphirm serve

# Run a suite in another
cargo run -p graphirm-eval -- --suite coding
cargo run -p graphirm-eval -- --suite knowledge
cargo run -p graphirm-eval -- --suite all

# Point at non-default server
cargo run -p graphirm-eval -- --server http://localhost:5555 --suite coding
```

Output metrics are written to JSON (e.g. `s5_metrics.json`). See `graphirm-eval/README.md` for full usage.
