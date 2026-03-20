# Dogfood Skill — Cursor ↔ Graphirm Local Loop

> **Status:** Phase 1 ✅ Phase 2 ✅ — skill live at `~/.cursor/skills/dogfood-graphirm/SKILL.md`
> **Goal:** Create a Cursor skill that delegates implementation tasks to a locally running graphirm agent, then reviews the results.

---

## Prerequisites

- Binary built with `cargo build --release --features local-extraction`
- API keys sourced from `~/graphirm-repo/.env` (`DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`)
- Port 3000 free
- `/data/workspaces` exists and is owned by `$USER`
- `config/default.toml` has `name = "graphirm"` and `model = "openrouter/qwen/qwen3-coder-next"` under `[agent]`

## Phase 1: Manual Validation ✅

Validated 2026-03-20. All steps work.

### Findings

| Step | Command | Result |
|------|---------|--------|
| Start server | `source .env && ./target/release/graphirm serve --host 127.0.0.1 --port 3000` | Starts, restores 104 sessions from graph, GLiNER2 loaded |
| Create session | `POST /api/sessions` with `{"name":"...","workspace":"..."}` | Returns `id`, `workspace`, `workspace_path` |
| Auto-approve | `POST /api/sessions/:id/auto-approve` with `{"enabled":true}` | Works silently |
| Simple prompt | `POST /api/sessions/:id/prompt` with `{"content":"..."}` | SSE stream, agent responds (~2s) |
| Tool-using prompt | `"list files in the current directory using ls"` | Agent calls `ls` tool, gets "(empty directory)", responds correctly |
| Read messages | `GET /api/sessions/:id/messages` | Full conversation with roles, tool calls, metadata |
| Knowledge extraction | GLiNER2 ONNX runs after each turn | 0 entities for trivial responses (expected) |

### Issues Found & Fixed

- `config/default.toml` was missing `name` and `model` under `[agent]` → config parse failed, fell back to defaults (no `workspaces_root`)
- `/data/workspaces` owned by root → `chown -R $USER:$USER /data/workspaces`
- Binary built without `--features local-extraction` → knowledge extraction warning on every turn
- `[model]` section `name = "deepseek-chat"` still present — used by provider factory, separate from `[agent].model`

### API Cheatsheet (for skill)

```bash
# Health
curl -s http://localhost:3000/api/health

# Create session with workspace
curl -s http://localhost:3000/api/sessions \
  -H 'Content-Type: application/json' \
  -d '{"name":"NAME","workspace":"WORKSPACE"}'

# Enable auto-approve (skip HITL for destructive tools)
curl -s -X POST http://localhost:3000/api/sessions/ID/auto-approve \
  -H 'Content-Type: application/json' \
  -d '{"enabled":true}'

# Send prompt (returns SSE stream)
curl -N http://localhost:3000/api/sessions/ID/prompt \
  -H 'Content-Type: application/json' \
  -d '{"content":"..."}'

# Read messages (poll for completion)
curl -s http://localhost:3000/api/sessions/ID/messages

# Session status
curl -s http://localhost:3000/api/sessions/ID

# Export session as markdown
curl -s http://localhost:3000/api/sessions/ID/export?format=markdown
```

## Phase 2: Write the Cursor Skill ✅

Written to `~/.cursor/skills/dogfood-graphirm/SKILL.md`.

### Key design decisions

- **Session context via `system_prompt`:** Sessions start in an empty `/data/workspaces/<name>/` dir. The session must be created with a `system_prompt` that tells graphirm the repo is at `/home/krs/graphirm-repo/` so it can use its read/grep/find tools to navigate there.
- **Polling strategy:** `GET /messages`, last message role is `assistant` — no explicit `status` field exists; agent completion is inferred from message presence.
- **Evaluation is a prompt:** After review, Cursor sends a structured `Evaluation:` prompt to the same session. Graphirm stores it as an Interaction node; GLiNER2 extracts entities.

### Coverage

- When to use (delegate implementation tasks)
- Server lifecycle (check health, start if needed, source `.env`)
- Session creation with `system_prompt` pointing to `~/graphirm-repo/`
- Auto-approve (skip HITL for destructive tools)
- Prompting + polling for completion
- Review protocol (`git diff`, `cargo test`, `cargo clippy`)
- Error handling (session error, tool failures, timeouts)
- Evaluation protocol (6 dimensions + hybrid storage)

## Philosophy: Cursor as Evaluator

Cursor doesn't just delegate — it evaluates every graphirm response. After each task, Cursor grades:

| Dimension | What to check | Log if |
|-----------|--------------|--------|
| Correctness | Did the output match the intent? | Wrong, incomplete, or hallucinated |
| Tool efficiency | Minimal tool calls? Right tool for the job? | Redundant reads, unnecessary bash calls, grep-then-read when grep sufficed |
| Reasoning | Planned before acting? Diagnosed errors? | Blind retries, no plan stated, ignored errors |
| Knowledge | Useful entities extracted? | Noise entities, missed obvious concepts |
| Style | Matches project conventions? | Wrong naming, missing error handling, unwrap() |
| Failure modes | Loops, escalations, silent failures | Repeated tool calls, gave up early, wrong assumptions |

### Evaluation storage (hybrid)

**1. Graph (primary):** After reviewing graphirm's work, Cursor sends a final evaluation message to the same session:

```
Evaluation: grade=pass|partial|fail
Task: <what was asked>
Findings: <specific observations>
Suggested fix: <system prompt / tool / agent loop change>
```

Graphirm stores this as an Interaction node. Knowledge extraction captures entities (e.g. "redundant reads", "tool efficiency") as Knowledge nodes linked to the session. Future sessions can surface past evaluations via `session_trace search` or `graph_query semantic`.

**2. Local log (secondary):** Cursor also appends a one-liner to `docs/dogfood-findings.md` for quick human scanning:

```
| 2026-03-20 | <session-id> | <task> | pass/partial/fail | <one-line finding> |
```

### Feedback loop

1. Cursor delegates task → graphirm executes
2. Cursor reviews output (`git diff`, `cargo test`, messages)
3. Cursor sends evaluation message to graphirm (stored in graph)
4. Cursor appends summary to `docs/dogfood-findings.md`
5. Periodically: review findings, batch improvements to system prompt / tools / agent loop
6. Graphirm can query its own past evaluations — self-aware improvement

## Phase 3: Iterate (ongoing)

Nine dogfood runs completed 2026-03-20 (2 hung). Results in `docs/dogfood-findings.md`.

### System prompt improvements discovered

| Run | Failure | Fix |
|-----|---------|-----|
| 1 (Claude) | Used `cd` in bash (not persisted) | Added "absolute paths, never cd" to system prompt |
| 2 (Qwen) | Read whole files → context overflow | Added grep-first instruction (agent ignored it) |
| 3 (Qwen) | Same context overflow | Discovered: `read` tool already had `offset`/`limit` but system prompt didn't mention them |
| 4 (Qwen) | **Pass** | Documented `offset`/`limit` in tool description + reinforced file reading discipline |
| 5 (Qwen) | Dep version conflict + ownership errors in spawn_blocking | Need to document re-exports and add spawn_blocking pattern |
| 6 (Qwen) | **Pass** | System prompt fixes validated — agent used graphirm_graph re-exports, proper Arc cloning, no petgraph/chrono added. 5 tests, 18 tool calls, all checks pass |
| 7 (Qwen) | **Partial** | cargo_check tool structure perfect (trait, registration, 9 tests, clippy/fmt clean). Two bugs: (a) early return on non-zero exit meant errors never parsed; (b) JSON struct expected top-level fields but cargo nests in `message`. Tests only covered clean path — no error path test |
| 8 (Qwen) | **Hung** | grep context_lines task. Agent read file, wrote correct param schema + execute logic to grep.rs, then LLM call hung forever. Session stuck at "running" — exposed dead `timeout_seconds` config |
| 9 (Qwen) | **Hung** | Same task, fresh session. Same hang point after file read. Confirmed: OpenRouter/Qwen streaming hangs on tool-call generation for file edits |

### Key insights

1. **Tool documentation gaps** (runs 1–4): The agent didn't need new tools — the `read` tool already supported `offset` and `limit`. The problem was purely a system prompt documentation gap. When the tool description mentioned these params and the file reading discipline section showed a concrete grep→read(offset,limit) workflow, the agent followed it perfectly.

2. **Rust-specific pitfalls** (run 5): On a harder task (250+ lines of new code), the agent understood the domain model perfectly but failed on Rust mechanics: (a) added `petgraph = "0.6"` directly when `graphirm-graph` re-exports from 0.7, causing type mismatches; (b) moved `Arc<GraphStore>` into a closure then tried to use it in a second closure. These are systematic — the system prompt should document crate re-exports and show the `Arc::clone` pattern for spawn_blocking.

3. **System prompt fixes work** (run 6): After adding "Crate dependency rules" and "Async patterns" sections to the system prompt, the agent correctly used `graphirm_graph::` re-exports and cloned `Arc` before `spawn_blocking` closures. No petgraph/chrono added to Cargo.toml. The agent also fixed String borrow issues independently (a new pattern not in the prompt), showing the crate dep section generalised well.

4. **External format knowledge gaps** (run 7): The agent implemented a tool that parses `cargo check --message-format=json` output, but didn't know the actual JSON envelope format (`{"reason":"compiler-message","message":{...}}`). It assumed diagnostic fields (`level`, `message`) were at the top level. Also assumed non-zero exit code = failure, but `cargo check` returns 101 for compilation errors (which is the tool's primary use case). Tests only validated the "no errors" path. Lesson: when a tool parses external tool output, the system prompt should document the format or instruct the agent to inspect real output first.

5. **LLM timeout is critical infrastructure** (runs 8–9): `timeout_seconds = 300` was in the config but never wired to code. The LLM call in `workflow.rs` used `tokio::select!` with only a cancellation token — no time-based arm. When OpenRouter/Qwen hung during tool-call generation, the session stayed "running" forever. Fix: added `tokio::time::sleep(llm_timeout)` as a third `select!` arm; session transitions to `"error"` on timeout. Also discovered the agent successfully wrote code to `grep.rs` before hanging — partial work was silently lost.

### Identified system prompt fixes needed

- ~~Document that `graphirm-graph` re-exports `Direction`, `GraphEdge`, `GraphNode`, `NodeId`, `NodeType`, etc.~~ ✅ Done (run 6 validated)
- ~~Add a spawn_blocking pattern example~~ ✅ Done (run 6 validated)
- ~~Add a "Crate dependency rules" section to the system prompt~~ ✅ Done (run 6 validated)

### System memory strategy (hybrid)

**Problem:** Every dogfood failure adds a new section to the system prompt. The prompt
grows linearly. Eventually the prompt itself causes context overflow.

**Solution:** Hybrid approach using GLiNER2 + planning layer.

1. **Keep system prompt minimal** — tool descriptions, absolute paths rule, project
   conventions. Only invariants that apply to every session.
2. **Store lessons as Knowledge nodes** — when Cursor sends evaluations, GLiNER2 extracts
   entities. Use `entity_type: "lesson"` and `"convention"` (planning layer entity types).
3. **Briefing injects relevant lessons** — at session start, `build_repo_briefing` queries
   recent lesson/convention Knowledge nodes and injects the top-k into context under a
   `## Lessons from past sessions` header.
4. **Agent can self-query** — `graph_query project list entity_type=lesson` surfaces past
   lessons on demand.

**GLiNER2 is the extraction layer, the graph is the storage layer, the briefing is the
retrieval layer.** No new pipelines needed — just routing conventions.

See `docs/plans/2026-03-20-planning-layer-design.md` Phase 1.5 for implementation details.

### Split-session test workflow

**Problem:** Tests are always the casualty. The agent implements the feature, burns context
fixing compile errors, then has nothing left for tests.

**Solution:** Split into two sessions linked via the planning layer.

1. **Implementation session** — implements the feature, calls `graph_query project
   link_session` to declare what planning node it worked on.
2. **Test session** — starts by querying `graph_query project list` to find what was
   implemented, reads the code, writes tests. Fresh context budget.
3. **Cross-session context** — the test session gets implementation context from the
   graph (what was built, what changed), not from re-reading the whole conversation.

This also dogfoods the planning layer's cross-session linking.

### Open items

- ~~Apply system prompt fixes from run 5 and re-test~~ ✅ Done — run 6 pass
- ~~Write unit tests for project mode~~ ✅ Done — agent wrote 5 tests in run 6
- ~~Add `cargo_check` structured error tool~~ ✅ Done — agent built structure (run 7), Cursor fixed JSON parsing bugs
- ~~Add "test the error path" heuristic to system prompt~~ ✅ Done — "Testing discipline" section added
- ~~Fix LLM timeout bug~~ ✅ Done — `timeout_seconds` wired to `tokio::select!` in workflow.rs (runs 8–9 exposed)
- Implement Phase 1.5 (lesson/convention entity types in briefing)
- Tune polling intervals
- Add support for multi-turn conversations
- Handle workspace ↔ repo sync (agent workspace vs `~/graphirm-repo/`)
