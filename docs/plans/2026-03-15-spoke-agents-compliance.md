# Spoke Agents Compliance — AGENTS.md Integration Plan

> **Status: ✅ COMPLETE** — All 11 tasks implemented and verified 2026-03-15.

**Goal:** Make the graphirm repo fully compliant with `spoke-agents-philosophy.md` so any agent or human on any future spoke can continue work without guessing.

**Architecture:** Create a root `AGENTS.md` as the single source of truth for project context (replacing Cursor-only `.cursor/rules/` as the canonical reference), plus module-level `AGENTS.md` files for each significant crate. Add `opencode.json` for OpenCode integration.

**Tech Stack:** Markdown, JSON

---

## Gap Analysis

| Requirement | Current State | Action |
|-------------|--------------|--------|
| Root `AGENTS.md` | Missing | Create — project overview, architecture, build/test, conventions, current state |
| Module `AGENTS.md` per significant dir | Missing (0 of 7) | Create for each crate + graphirm-eval |
| `opencode.json` | Missing | Create with `instructions: ["AGENTS.md"]` |
| AGENTS.md < 150 lines each | N/A | Enforce during creation |
| Cursor rules reference AGENTS.md | Rules are standalone, duplicate portable context | Gut `graphirm-project.mdc` and `graphirm-context-always.mdc` — move portable content to AGENTS.md, keep only Cursor-specific context |

### What NOT to do

- **Don't duplicate README.md** — AGENTS.md is for agent/developer onboarding context, README.md is the public-facing project page. AGENTS.md should reference README.md for details, not copy them.
- **Don't replace `.cursor/rules/`** — those serve Cursor-specific needs (routing, skill triggers, meta). AGENTS.md is the portable layer.
- **Don't exceed 150 lines** — if a module's AGENTS.md gets long, trim to essentials.

---

## Task 1: Create `opencode.json`

**Files:**
- Create: `opencode.json`

**Step 1: Create the file**

```json
{
  "$schema": "https://opencode.ai/config.json",
  "instructions": ["AGENTS.md"]
}
```

**Step 2: Verify**

Run: `cat opencode.json | python3 -m json.tool`
Expected: Valid JSON, no errors

**Step 3: Commit**

```bash
git add opencode.json
git commit -m "chore: add opencode.json for spoke agent integration"
```

---

## Task 2: Create root `AGENTS.md`

**Files:**
- Create: `AGENTS.md`

**Step 1: Write root AGENTS.md**

Content must include these sections (see spoke-agents-philosophy.md):

1. **What this is** — one paragraph: graph-native coding agent in Rust, every interaction is a graph node, single static binary
2. **Architecture** — workspace layout, 6 crates + eval harness, how they connect (graph at the bottom, agent depends on llm+tools+graph, server/tui depend on agent)
3. **Code layout** — which directories hold what, entry points (`src/main.rs` for CLI, `crates/server/` for HTTP)
4. **Build & test** — exact commands:
   - `cargo build --release` (standard build)
   - `cargo build --release --features local-extraction` (with GLiNER2)
   - `cargo test` (all tests)
   - `cargo test -p graphirm-graph` (single crate)
   - `DEEPSEEK_API_KEY=... cargo test -p graphirm-llm` (LLM integration, needs API key)
   - `./target/release/graphirm chat` (TUI)
   - `./target/release/graphirm serve` (HTTP server, port 3000)
5. **Key conventions** — from existing `.cursor/rules/`:
   - Rust edition 2024, MSRV 1.85
   - `thiserror` for error types, one error enum per crate
   - `async-trait` for async traits
   - `tracing` for logging (not `println!` or `log`)
   - `Arc<RwLock<StableGraph>>` for in-memory graph — watch for deadlocks
   - Tools implement the `Tool` trait
   - LLM providers implement `LlmProvider` trait
   - Config in `config/default.toml`, loaded by agent
   - Five node types: Interaction, Agent, Content, Task, Knowledge
   - Twelve edge types: RespondsTo, SpawnedBy, DelegatesTo, DependsOn, Produces, Reads, Modifies, Summarizes, Contains, FollowsUp, Steers, RelatesTo
6. **Current state** — phases 0–9 complete, phase 10 (Web UI) pending. Eval harness exists. VS Code extension exists.

**Constraints:**
- Must stay under 150 lines
- Reference `README.md` for detailed usage/examples rather than duplicating
- Reference `config/default.toml` for configuration details

**Step 2: Line count check**

Run: `wc -l AGENTS.md`
Expected: ≤ 150

**Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "docs: add root AGENTS.md for spoke agent onboarding"
```

---

## Task 3: Create `crates/graph/AGENTS.md`

**Files:**
- Create: `crates/graph/AGENTS.md`

**Content (module AGENTS.md template):**

1. **Purpose** — Graph persistence layer. SQLite-backed storage with in-memory petgraph for traversals (PageRank, BFS). HNSW vector index for knowledge node similarity.
2. **Key components:**
   - `store.rs` — `GraphStore`: connection pool, CRUD, session management
   - `nodes.rs` — `GraphNode`, `NodeType` enum (5 types), `NodeId`
   - `edges.rs` — `GraphEdge`, `EdgeType` enum (12 types), `EdgeId`
   - `query.rs` — BFS, PageRank, subgraph extraction
   - `vector.rs` — HNSW index wrapper (`instant-distance`)
   - `corpus.rs` — corpus export for eval/analysis
   - `error.rs` — `GraphError` enum
3. **Integration points:**
   - Used by: `graphirm-agent` (context engine, session), `graphirm-tools` (file tracking), `graphirm-server` (graph API), `graphirm-tui` (graph explorer)
   - Depends on: `rusqlite`, `petgraph`, `instant-distance`, `r2d2`
4. **How to test:** `cargo test -p graphirm-graph`

**Constraints:** ≤ 150 lines

**Step 1: Write the file**
**Step 2: Verify line count** — `wc -l crates/graph/AGENTS.md`
**Step 3: Commit**

```bash
git add crates/graph/AGENTS.md
git commit -m "docs: add AGENTS.md for graph crate"
```

---

## Task 4: Create `crates/llm/AGENTS.md`

**Files:**
- Create: `crates/llm/AGENTS.md`

**Content:**

1. **Purpose** — LLM provider abstraction. Multi-provider support (Anthropic, OpenAI, DeepSeek, Ollama, OpenRouter) via `rig-core`. Streaming, tool calling, embeddings.
2. **Key components:**
   - `provider.rs` — `LlmProvider` trait, `LlmMessage`, `LlmResponse`, `Role`, `ContentPart`
   - `anthropic.rs`, `openai.rs`, `deepseek.rs`, `ollama.rs`, `openrouter.rs` — provider implementations
   - `stream.rs` — `StreamEvent` types for SSE
   - `tool.rs` — `ToolDefinition`, `ToolCall` types (LLM-side)
   - `factory.rs` — provider construction from config
   - `mock.rs` — `MockProvider` for testing
   - `mistral_embed.rs` — Mistral embedding provider
   - `fastembed_provider.rs` — local embeddings (feature-gated: `local-embed`)
   - `error.rs` — `LlmError` enum
3. **Integration points:**
   - Used by: `graphirm-agent` (workflow, knowledge extraction)
   - Depends on: `rig-core`, `reqwest`, `tokio-stream`
   - API keys via env vars: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`
4. **How to test:** `cargo test -p graphirm-llm` (mock tests only; integration tests need API keys)

**Step 1–3:** Same pattern as Task 3.

```bash
git add crates/llm/AGENTS.md
git commit -m "docs: add AGENTS.md for llm crate"
```

---

## Task 5: Create `crates/tools/AGENTS.md`

**Files:**
- Create: `crates/tools/AGENTS.md`

**Content:**

1. **Purpose** — Built-in tool implementations for the agent. Each tool exposes a JSON schema and async execute method.
2. **Key components:**
   - `registry.rs` — `ToolRegistry`: registers tools, dispatches calls
   - `executor.rs` — parallel tool execution via `tokio::JoinSet`
   - `bash.rs` — shell command execution (destructive)
   - `read.rs` — file reading with line numbers
   - `write.rs` — file creation/overwrite (destructive)
   - `edit.rs` — exact string replacement in files (destructive)
   - `grep.rs` — regex search across files
   - `find.rs` — file name pattern search
   - `ls.rs` — directory listing
   - `permissions.rs` — per-agent tool allow/deny
   - `error.rs` — `ToolError` enum
3. **Integration points:**
   - Used by: `graphirm-agent` (workflow loop calls tools via registry)
   - Depends on: `graphirm-graph` (for graph node creation on tool use)
4. **How to test:** `cargo test -p graphirm-tools`

**Adding a new tool:** Implement the `Tool` trait (`name()`, `description()`, `parameters()` → JSON schema, `execute()` → `ToolOutput`), then register it in `ToolRegistry::new()`.

**Step 1–3:** Same pattern.

```bash
git add crates/tools/AGENTS.md
git commit -m "docs: add AGENTS.md for tools crate"
```

---

## Task 6: Create `crates/agent/AGENTS.md`

**Files:**
- Create: `crates/agent/AGENTS.md`

**Content:**

1. **Purpose** — Core agent loop, context engine, multi-agent coordination, knowledge extraction, session management. The brain of graphirm.
2. **Key components:**
   - `workflow.rs` — `run_agent_loop`: async state machine (plan→act→observe), tool execution, streaming
   - `context.rs` — `build_context`, `score_node`: relevance scoring (recency + edge weight + distance + PageRank), token budget fitting
   - `compact.rs` — context compaction when budget exceeded
   - `session.rs` — `Session`, `SessionMetadata`: persistence, resume
   - `coordinator.rs` — `Coordinator`: multi-agent orchestration
   - `delegate.rs` — `SubagentTool`: spawn subagents as tool calls
   - `multi.rs` — `spawn_subagent`, `collect_subagent_results`
   - `escalation.rs` — soft escalation (repeated tool call detection)
   - `hitl.rs` — human-in-the-loop gate for destructive tools
   - `config.rs` — `AgentConfig`, `AgentMode`, `Permission`
   - `event.rs` — `AgentEvent`, `EventBus` for streaming
   - `knowledge/` — entity extraction (LLM, GLiNER2, hybrid), memory injection, HNSW search
   - `error.rs` — `AgentError` enum
3. **Integration points:**
   - Used by: `graphirm-server`, `graphirm-tui`, `src/main.rs`
   - Depends on: `graphirm-graph`, `graphirm-llm`, `graphirm-tools`
4. **How to test:** `cargo test -p graphirm-agent` (some tests need `local-extraction` feature)

**Step 1–3:** Same pattern.

```bash
git add crates/agent/AGENTS.md
git commit -m "docs: add AGENTS.md for agent crate"
```

---

## Task 7: Create `crates/tui/AGENTS.md`

**Files:**
- Create: `crates/tui/AGENTS.md`

**Content:**

1. **Purpose** — Terminal UI for interactive chat. Built on ratatui + crossterm.
2. **Key components:**
   - `app.rs` — `App`: main application state and event loop
   - `chat.rs` — chat panel rendering (message display, markdown)
   - `input.rs` — user input handling, keybindings
   - `graph.rs` — graph explorer panel
   - `ui.rs` — layout composition, panel arrangement
   - `status.rs` — status bar (model, session, token usage)
   - `events.rs` — terminal event handling (key, mouse, resize)
   - `types.rs` — shared TUI types
3. **Integration points:**
   - Used by: `src/main.rs` (chat subcommand)
   - Depends on: `graphirm-graph`, `graphirm-agent`, `graphirm-llm`
4. **How to test:** No automated tests. Manual: `cargo run -- chat`

**Step 1–3:** Same pattern.

```bash
git add crates/tui/AGENTS.md
git commit -m "docs: add AGENTS.md for tui crate"
```

---

## Task 8: Create `crates/server/AGENTS.md`

**Files:**
- Create: `crates/server/AGENTS.md`

**Content:**

1. **Purpose** — HTTP API server with SSE streaming. Powers the VS Code extension and programmatic access.
2. **Key components:**
   - `routes.rs` — axum route handlers (session CRUD, prompt, graph query)
   - `sse.rs` — Server-Sent Events streaming for agent responses
   - `state.rs` — `AppState`: shared server state (graph, sessions, config)
   - `session.rs` — `SessionHandle`: per-session state management
   - `types.rs` — request/response types (`CreateSessionRequest`, `PromptRequest`, etc.)
   - `sdk.rs` — HTTP client SDK for programmatic access
   - `middleware.rs` — request logging, CORS
   - `request_log.rs` — structured request logging
   - `error.rs` — `ServerError` enum
3. **Integration points:**
   - Used by: `src/main.rs` (serve subcommand), `graphirm-eval` (HTTP client), VS Code extension
   - Depends on: `graphirm-graph`, `graphirm-agent`, `graphirm-llm`, `graphirm-tools`
4. **How to test:** `cargo test -p graphirm-server` (4 test files; some need a running server)

**Step 1–3:** Same pattern.

```bash
git add crates/server/AGENTS.md
git commit -m "docs: add AGENTS.md for server crate"
```

---

## Task 9: Create `graphirm-eval/AGENTS.md`

**Files:**
- Create: `graphirm-eval/AGENTS.md`

**Content:**

1. **Purpose** — Evaluation harness. Drives the agent via HTTP API, runs task suites, measures correctness/knowledge/memory/graph structure.
2. **Key components:**
   - `main.rs` — CLI entrypoint, suite selection
   - `client.rs` — HTTP client wrapper for graphirm server
   - `harness.rs` — test orchestration, session lifecycle
   - `report.rs` — result aggregation, metrics output
   - `task.rs` — `EvalTask` trait
   - `tasks/` — task suites: `coding.rs`, `knowledge.rs`, `memory.rs`, `graph.rs`, `adversarial.rs`
3. **Integration points:**
   - Depends on: running graphirm server (`graphirm serve`)
   - Does NOT depend on any graphirm crate as a Rust dependency (HTTP only)
4. **How to test:** Start server first, then `cargo run -p graphirm-eval -- --suite coding`

**Step 1–3:** Same pattern.

```bash
git add graphirm-eval/AGENTS.md
git commit -m "docs: add AGENTS.md for eval harness"
```

---

## Task 10: Gut `graphirm-project.mdc` — replace with AGENTS.md pointer

**Files:**
- Modify: `.cursor/rules/graphirm-project.mdc`

**Why:** This rule currently contains the core thesis, tech stack, and design constraints — exactly what root AGENTS.md now owns. Keeping both creates dual-source drift. The philosophy warns: "stale docs are worse than no docs."

**Step 1: Replace the rule content**

Keep the frontmatter, replace the body with a pointer:

```markdown
---
description: Graphirm project overview — what it is and core thesis
alwaysApply: true
---

# Graphirm

> **Canonical project context lives in `AGENTS.md` at the repo root.**
> This rule exists only so Cursor always loads a project reminder.

Graph-native coding agent in Rust. Every interaction is a graph node.
See `AGENTS.md` for architecture, tech stack, conventions, and current state.
See `README.md` for usage examples and detailed documentation.
```

That's ~10 lines instead of ~40. No duplication risk.

**Step 2: Commit**

```bash
git add .cursor/rules/graphirm-project.mdc
git commit -m "refactor: slim graphirm-project.mdc, defer to AGENTS.md"
```

---

## Task 11: Gut `graphirm-context-always.mdc` — keep only Cursor-specific context

**Files:**
- Modify: `.cursor/rules/graphirm-context-always.mdc`

**Why:** This rule currently duplicates architecture, graph data model, phased build plan, and development environment. Architecture/graph model/phases now live in AGENTS.md. Only the **Cursor-specific development environment** section (local path, spoke workflow, Cursor role) has no home in AGENTS.md.

**Step 1: Replace the rule content**

```markdown
---
description: Cursor-specific workspace context — supplements AGENTS.md
alwaysApply: true
---

# Cursor Workspace Context

> **Architecture, graph model, conventions, and build commands live in `AGENTS.md`.**
> This rule covers Cursor-specific context only.

## Development Environment

Two environments share the same GitHub remote (`github.com/consoulhub/graphirm`).

### Local (this workspace)

- **Path:** `~/graphirm-repo/`
- **IDE:** Cursor
- **Role:** Primary development, planning, code review

### Remote Spoke

- **What:** Ephemeral Hetzner VM managed by Consoul (`ccx33`, 8 vCPU, 32GB RAM)
- **Project at:** `/root/project` (same repo, cloned via cloud-init)
- **IDE on spoke:** code-server + OpenCode (DeepSeek)
- **Lifespan:** Auto-deleted after 24 hours — push before teardown

### Connecting

- **Dashboard:** `https://terminimal.space` (HTTP basic auth, user: `admin`)
- **SSH:** `ssh root@<spoke-ip>` (hub SSH keys are pre-installed)
- **code-server:** SSH tunnel or spoke IP on port 8443

### Sync

```bash
git push origin main   # after local work
git pull origin main   # after spoke work
```

## Risk Reminders

- `Arc<RwLock<StableGraph>>` — watch for deadlocks under concurrent load
- Rust version: align with spoke/CI
```

That's ~35 lines instead of ~107. Architecture, graph model, and phase plan are gone (they're in AGENTS.md). Dev environment and risk reminders stay because they're Cursor-workflow-specific.

**Step 2: Commit**

```bash
git add .cursor/rules/graphirm-context-always.mdc
git commit -m "refactor: slim graphirm-context-always.mdc, defer to AGENTS.md"
```

---

## Cursor Rules — What Stays, What Changes

| Rule File | Verdict | Reason |
|-----------|---------|--------|
| `graphirm-project.mdc` | **Gut** (Task 10) | Content moves to root AGENTS.md |
| `graphirm-context-always.mdc` | **Gut** (Task 11) | Architecture/model/phases move to AGENTS.md; keep dev env + risks |
| `000-skills-first.mdc` | **Keep** | Cursor-only: skills protocol |
| `001-router.mdc` | **Keep** | Cursor-only: rule routing |
| `002-meta-generator.mdc` | **Keep** | Cursor-only: rule creation |
| `003-code-quality.mdc` | **Keep** | Cursor-only: generic code guidance |
| `100-rust-standards.mdc` | **Keep** | Cursor-only: detailed Rust patterns/examples (AGENTS.md references key conventions but doesn't duplicate full detail) |
| `102-project-management.mdc` | **Keep** | Cursor-only: project mgmt guidance |
| `103-response-quality.mdc` | **Keep** | Cursor-only: response style |
| `103-testing.mdc` | **Keep** | Cursor-only: testing guidance |
| `106-git.mdc` | **Keep** | Cursor-only: git workflow |
| `107-documentation.mdc` | **Keep** | Cursor-only: doc standards |
| `tasklist.mdc` | **Keep** | Cursor-only: task management |

**Principle:** AGENTS.md is the portable source of truth (readable by OpenCode, Claude, any agent). Cursor rules supplement with Cursor-specific behavior and detailed coding guidance. If they conflict, AGENTS.md wins.

---

## Summary

| Task | File | Action |
|------|------|--------|
| 1 | `opencode.json` | Create (4 lines) |
| 2 | `AGENTS.md` (root) | Create (~120–140 lines) |
| 3 | `crates/graph/AGENTS.md` | Create (~40–60 lines) |
| 4 | `crates/llm/AGENTS.md` | Create (~50–70 lines) |
| 5 | `crates/tools/AGENTS.md` | Create (~50–60 lines) |
| 6 | `crates/agent/AGENTS.md` | Create (~60–80 lines) |
| 7 | `crates/tui/AGENTS.md` | Create (~30–40 lines) |
| 8 | `crates/server/AGENTS.md` | Create (~50–60 lines) |
| 9 | `graphirm-eval/AGENTS.md` | Create (~35–45 lines) |
| 10 | `.cursor/rules/graphirm-project.mdc` | Gut → pointer (~10 lines, was ~40) |
| 11 | `.cursor/rules/graphirm-context-always.mdc` | Gut → Cursor-only (~35 lines, was ~107) |

**New files:** 9 (8 AGENTS.md + 1 opencode.json)
**Modified files:** 2 (Cursor rules)
**Estimated time:** 30–45 minutes with subagent-driven execution

---

## Ongoing Rules (post-integration)

After this plan is complete, these rules from `spoke-agents-philosophy.md` apply permanently:

1. **Read before edit** — read any existing AGENTS.md BEFORE editing files in that directory
2. **Update on change** — update AGENTS.md when you change what a module does or how it connects
3. **Stay under 150 lines** — trim sections that are no longer true
4. **Delete stale content** — stale docs are worse than no docs
5. **Commit together** — AGENTS.md changes ship alongside code changes
