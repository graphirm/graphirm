# Phase 0: Project Scaffold Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Set up the Cargo workspace with 6 crate stubs, CLI entry point, error types, CI pipeline, and config loading.

**Architecture:** Cargo workspace at the repo root with 6 member crates under `crates/` (graph, llm, tools, agent, tui, server) and a binary target in `src/main.rs`. Each crate defines its own error enum via `thiserror`. The top-level binary uses `clap` for CLI dispatch and wraps all crate errors into a unified `GraphirmError`.

**Tech Stack:** Rust 1.83+, Cargo workspaces, clap (CLI), thiserror (errors), serde + serde_json (serialization), tokio (async runtime), tracing (logging), GitHub Actions (CI)

---

## Task 1: Create workspace Cargo.toml with all 6 crate members + binary

- [x] Complete

**Files:**
- Create: `Cargo.toml`

**Step 1: Create the workspace root Cargo.toml**

```toml
[workspace]
members = [
    "crates/graph",
    "crates/llm",
    "crates/tools",
    "crates/agent",
    "crates/tui",
    "crates/server",
]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2024"
rust-version = "1.85"
license = "MIT"
authors = ["Consoul <team@consoul.dev>"]

[package]
name = "graphirm"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
authors.workspace = true
description = "Graph-native coding agent"

[[bin]]
name = "graphirm"
path = "src/main.rs"

[dependencies]
graphirm-graph = { path = "crates/graph" }
graphirm-llm = { path = "crates/llm" }
graphirm-tools = { path = "crates/tools" }
graphirm-agent = { path = "crates/agent" }
graphirm-tui = { path = "crates/tui" }
graphirm-server = { path = "crates/server" }
clap = { version = "4", features = ["derive"] }
tokio = { version = "1", features = ["full"] }
thiserror = "2"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
serde = { version = "1", features = ["derive"] }
toml = "0.8"
```

**Step 2: Verify the file is valid TOML**

This won't compile yet — the member crates don't exist. We'll verify after Task 2-7.

---

## Task 2: Create crates/graph/ with Cargo.toml + stub files

- [x] Complete

**Files:**
- Create: `crates/graph/Cargo.toml`
- Create: `crates/graph/src/lib.rs`
- Create: `crates/graph/src/store.rs`
- Create: `crates/graph/src/nodes.rs`
- Create: `crates/graph/src/edges.rs`
- Create: `crates/graph/src/query.rs`
- Create: `crates/graph/src/error.rs`

**Step 1: Create Cargo.toml**

```toml
[package]
name = "graphirm-graph"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
authors.workspace = true
description = "Graph persistence layer for Graphirm"

[dependencies]
rusqlite = { version = "0.32", features = ["bundled"] }
petgraph = "0.7"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
r2d2 = "0.8"
r2d2_sqlite = "0.25"
uuid = { version = "1", features = ["v4"] }
chrono = { version = "0.4", features = ["serde"] }

[dev-dependencies]
tempfile = "3"
```

**Step 2: Create src/lib.rs**

```rust
pub mod edges;
pub mod error;
pub mod nodes;
pub mod query;
pub mod store;

pub use error::GraphError;
pub use store::GraphStore;
```

**Step 3: Create src/store.rs**

```rust
// GraphStore: rusqlite + petgraph dual-write graph persistence
```

**Step 4: Create src/nodes.rs**

```rust
// Node type definitions: Interaction, Agent, Content, Task, Knowledge
```

**Step 5: Create src/edges.rs**

```rust
// Edge type definitions: RespondsTo, SpawnedBy, DelegatesTo, etc.
```

**Step 6: Create src/query.rs**

```rust
// Graph traversal helpers: neighbors, traverse, conversation_thread, subgraph, pagerank
```

**Step 7: Create src/error.rs**

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GraphError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("Connection pool error: {0}")]
    Pool(#[from] r2d2::Error),

    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("Node not found: {0}")]
    NodeNotFound(String),

    #[error("Edge not found: {0}")]
    EdgeNotFound(String),

    #[error("Graph lock poisoned")]
    LockPoisoned,
}
```

---

## Task 3: Create crates/llm/ with Cargo.toml + stub files

- [x] Complete

**Files:**
- Create: `crates/llm/Cargo.toml`
- Create: `crates/llm/src/lib.rs`
- Create: `crates/llm/src/provider.rs`
- Create: `crates/llm/src/stream.rs`
- Create: `crates/llm/src/tool.rs`
- Create: `crates/llm/src/error.rs`

**Step 1: Create Cargo.toml**

```toml
[package]
name = "graphirm-llm"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
authors.workspace = true
description = "LLM provider abstraction for Graphirm"

[dependencies]
async-trait = "0.1"
tokio = { version = "1", features = ["sync", "macros"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
tokio-stream = "0.1"
```

**Step 2: Create src/lib.rs**

```rust
pub mod error;
pub mod provider;
pub mod stream;
pub mod tool;

pub use error::LlmError;
```

**Step 3: Create src/provider.rs**

```rust
// LLM provider trait: completion, streaming, tool calling
```

**Step 4: Create src/stream.rs**

```rust
// Streaming response types: StreamChunk, StreamEvent
```

**Step 5: Create src/tool.rs**

```rust
// Tool call/result types: ToolCall, ToolResult, ToolDefinition
```

**Step 6: Create src/error.rs**

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("Provider error: {0}")]
    Provider(String),

    #[error("Streaming error: {0}")]
    Stream(String),

    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("Tool call error: {0}")]
    ToolCall(String),

    #[error("Request timeout")]
    Timeout,
}
```

---

## Task 4: Create crates/tools/ with Cargo.toml + stub files

- [x] Complete

**Files:**
- Create: `crates/tools/Cargo.toml`
- Create: `crates/tools/src/lib.rs`
- Create: `crates/tools/src/bash.rs`
- Create: `crates/tools/src/read.rs`
- Create: `crates/tools/src/write.rs`
- Create: `crates/tools/src/edit.rs`
- Create: `crates/tools/src/grep.rs`
- Create: `crates/tools/src/find.rs`
- Create: `crates/tools/src/ls.rs`
- Create: `crates/tools/src/error.rs`

**Step 1: Create Cargo.toml**

```toml
[package]
name = "graphirm-tools"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
authors.workspace = true
description = "Built-in tools for Graphirm agent"

[dependencies]
async-trait = "0.1"
tokio = { version = "1", features = ["process", "fs", "io-util"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
```

**Step 2: Create src/lib.rs**

```rust
pub mod bash;
pub mod edit;
pub mod error;
pub mod find;
pub mod grep;
pub mod ls;
pub mod read;
pub mod write;

pub use error::ToolError;
```

**Step 3: Create stub files for each tool**

`src/bash.rs`:
```rust
// Bash tool: execute shell commands with timeout and output capture
```

`src/read.rs`:
```rust
// Read tool: read file contents with optional line range
```

`src/write.rs`:
```rust
// Write tool: write/create files with content
```

`src/edit.rs`:
```rust
// Edit tool: search-and-replace within files
```

`src/grep.rs`:
```rust
// Grep tool: search file contents with regex patterns
```

`src/find.rs`:
```rust
// Find tool: find files by glob pattern
```

`src/ls.rs`:
```rust
// Ls tool: list directory contents with metadata
```

**Step 4: Create src/error.rs**

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Tool execution failed: {0}")]
    Execution(String),

    #[error("Tool timeout after {0}s")]
    Timeout(u64),

    #[error("Invalid arguments: {0}")]
    InvalidArgs(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),
}
```

---

## Task 5: Create crates/agent/ with Cargo.toml + stub files

- [x] Complete

**Files:**
- Create: `crates/agent/Cargo.toml`
- Create: `crates/agent/src/lib.rs`
- Create: `crates/agent/src/workflow.rs`
- Create: `crates/agent/src/config.rs`
- Create: `crates/agent/src/multi.rs`
- Create: `crates/agent/src/context.rs`
- Create: `crates/agent/src/session.rs`
- Create: `crates/agent/src/event.rs`
- Create: `crates/agent/src/compact.rs`
- Create: `crates/agent/src/error.rs`

**Step 1: Create Cargo.toml**

```toml
[package]
name = "graphirm-agent"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
authors.workspace = true
description = "Agent loop, multi-agent orchestration, and context engine for Graphirm"

[dependencies]
graphirm-graph = { path = "../graph" }
graphirm-llm = { path = "../llm" }
graphirm-tools = { path = "../tools" }
async-trait = "0.1"
tokio = { version = "1", features = ["sync", "macros", "rt"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
tracing = "0.1"
```

**Step 2: Create src/lib.rs**

```rust
pub mod compact;
pub mod config;
pub mod context;
pub mod error;
pub mod event;
pub mod multi;
pub mod session;
pub mod workflow;

pub use error::AgentError;
```

**Step 3: Create stub files**

`src/workflow.rs`:
```rust
// Agent workflow: async state machine with plan → act → observe → reflect loop
```

`src/config.rs`:
```rust
// Agent configuration: model selection, temperature, tool permissions
```

`src/multi.rs`:
```rust
// Multi-agent: coordinator pattern, subagent spawning, result aggregation
```

`src/context.rs`:
```rust
// Context engine: graph traversal for relevant context, relevance scoring
```

`src/session.rs`:
```rust
// Session management: create, resume, list, archive sessions
```

`src/event.rs`:
```rust
// Agent events: streaming events for UI consumption
```

`src/compact.rs`:
```rust
// Context compaction: summarize old context, prune graph branches
```

**Step 4: Create src/error.rs**

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("Graph error: {0}")]
    Graph(#[from] graphirm_graph::GraphError),

    #[error("LLM error: {0}")]
    Llm(#[from] graphirm_llm::LlmError),

    #[error("Tool error: {0}")]
    Tool(#[from] graphirm_tools::ToolError),

    #[error("Session not found: {0}")]
    SessionNotFound(String),

    #[error("Workflow error: {0}")]
    Workflow(String),

    #[error("Context build failed: {0}")]
    Context(String),
}
```

---

## Task 6: Create crates/tui/ with Cargo.toml + stub files

- [x] Complete

**Files:**
- Create: `crates/tui/Cargo.toml`
- Create: `crates/tui/src/lib.rs`
- Create: `crates/tui/src/app.rs`
- Create: `crates/tui/src/chat.rs`
- Create: `crates/tui/src/graph.rs`
- Create: `crates/tui/src/status.rs`

**Step 1: Create Cargo.toml**

```toml
[package]
name = "graphirm-tui"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
authors.workspace = true
description = "Terminal UI for Graphirm"

[dependencies]
graphirm-agent = { path = "../agent" }
ratatui = "0.29"
crossterm = "0.28"
tokio = { version = "1", features = ["sync", "macros"] }
serde = { version = "1", features = ["derive"] }
```

**Step 2: Create src/lib.rs**

```rust
pub mod app;
pub mod chat;
pub mod graph;
pub mod status;
```

**Step 3: Create stub files**

`src/app.rs`:
```rust
// App: main TUI application loop, event handling, layout
```

`src/chat.rs`:
```rust
// Chat view: message display, input handling, streaming text
```

`src/graph.rs`:
```rust
// Graph explorer: node/edge visualization, navigation
```

`src/status.rs`:
```rust
// Status bar: model info, session ID, token count, connection status
```

---

## Task 7: Create crates/server/ with Cargo.toml + stub files

- [x] Complete

**Files:**
- Create: `crates/server/Cargo.toml`
- Create: `crates/server/src/lib.rs`
- Create: `crates/server/src/routes.rs`
- Create: `crates/server/src/sse.rs`
- Create: `crates/server/src/sdk.rs`

**Step 1: Create Cargo.toml**

```toml
[package]
name = "graphirm-server"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
authors.workspace = true
description = "HTTP API and SDK server for Graphirm"

[dependencies]
graphirm-agent = { path = "../agent" }
axum = "0.8"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tower = "0.5"
tower-http = { version = "0.6", features = ["cors", "trace"] }
```

**Step 2: Create src/lib.rs**

```rust
pub mod routes;
pub mod sdk;
pub mod sse;
```

**Step 3: Create stub files**

`src/routes.rs`:
```rust
// HTTP routes: REST API for sessions, messages, graph queries
```

`src/sse.rs`:
```rust
// Server-Sent Events: streaming agent responses to clients
```

`src/sdk.rs`:
```rust
// SDK: client library for programmatic access to Graphirm
```

---

## Task 8: Create src/main.rs with clap CLI + src/error.rs

- [x] Complete

**Files:**
- Create: `src/main.rs`
- Create: `src/error.rs`

**Step 1: Create src/error.rs**

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GraphirmError {
    #[error("Graph error: {0}")]
    Graph(#[from] graphirm_graph::GraphError),

    #[error("LLM error: {0}")]
    Llm(#[from] graphirm_llm::LlmError),

    #[error("Tool error: {0}")]
    Tool(#[from] graphirm_tools::ToolError),

    #[error("Agent error: {0}")]
    Agent(#[from] graphirm_agent::AgentError),

    #[error("Config error: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
```

**Step 2: Create src/main.rs**

```rust
mod error;

use clap::{Parser, Subcommand};
use error::GraphirmError;

#[derive(Parser)]
#[command(name = "graphirm")]
#[command(version, about = "Graph-native coding agent")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start an interactive chat session
    Chat {
        /// Resume an existing session by ID
        #[arg(short, long)]
        session: Option<String>,

        /// Model to use (e.g., "claude-sonnet-4-20250514")
        #[arg(short, long)]
        model: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<(), GraphirmError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Chat { session, model } => {
            tracing::info!(
                session = session.as_deref().unwrap_or("new"),
                model = model.as_deref().unwrap_or("default"),
                "Starting chat session"
            );
            println!("graphirm chat — not yet implemented");
        }
    }

    Ok(())
}
```

---

## Task 9: Create config/default.toml

- [x] Complete

**Files:**
- Create: `config/default.toml`

**Step 1: Create the default configuration file**

```toml
[model]
provider = "anthropic"
name = "claude-sonnet-4-20250514"
temperature = 0.7
max_tokens = 8192

[agent]
max_iterations = 50
parallel_tool_calls = true
timeout_seconds = 300

[graph]
database_path = "~/.graphirm/graph.db"
max_connections = 4

[tui]
theme = "dark"
show_graph_panel = false
word_wrap = true

[server]
host = "127.0.0.1"
port = 3000
```

---

## Task 10: Create .github/workflows/ci.yml

- [x] Complete

**Files:**
- Create: `.github/workflows/ci.yml`

**Step 1: Create the CI workflow**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  CARGO_TERM_COLOR: always
  RUSTFLAGS: "-D warnings"

jobs:
  check:
    name: Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt
      - uses: Swatinem/rust-cache@v2
      - name: Check formatting
        run: cargo fmt --all -- --check
      - name: Clippy
        run: cargo clippy --all-targets --all-features -- -D warnings
      - name: Build
        run: cargo build --all-targets
      - name: Test
        run: cargo test --all-targets
```

---

## Task 11: Verify full workspace builds

- [x] Complete

**Step 1: Run cargo build**

Run: `cargo build 2>&1`
Expected: Compiles successfully (warnings are OK for stubs but no errors)

**Step 2: Run cargo test**

Run: `cargo test 2>&1`
Expected: `test result: ok. 0 passed; 0 failed` (no tests yet, should exit 0)

**Step 3: Run cargo clippy**

Run: `cargo clippy --all-targets --all-features 2>&1`
Expected: No errors. Warnings are acceptable at this stage.

**Step 4: Run cargo fmt check**

Run: `cargo fmt --all -- --check 2>&1`
Expected: No formatting issues.

**Step 5: Test the CLI**

Run: `cargo run -- --version 2>&1`
Expected: `graphirm 0.1.0`

Run: `cargo run -- --help 2>&1`
Expected: Shows help text with `chat` subcommand listed.

Run: `cargo run -- chat 2>&1`
Expected: `graphirm chat — not yet implemented`

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: phase 0 — cargo workspace scaffold with 6 crates, CLI, config, CI"
```
