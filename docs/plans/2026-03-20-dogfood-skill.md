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

Seventeen dogfood runs completed (2 hung, 1 model-ID fail, 3 consecutive passes). Results in `docs/dogfood-findings.md`.

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
| 10 (Qwen) | **Fail** | Model ID `openrouter/qwen/qwen3-coder-next` sent to API without prefix stripping. Cause: explicit `model` in curl. Instant 400 |
| 11 (Qwen) | **Partial** | bfs-max-depth: agent bypassed GraphStore API, wrote raw SQL with private `pool` field. 2 compile errors. Fix: "Abstraction boundaries" prompt section |
| 12 (Qwen) | **Pass (assist)** | Prompt fix worked. Agent modified graph crate `traverse()`, used public API, cargo_check passed, 20/20 tests passed. rig JSON error before graph test fix (2-line human assist) |
| 13 (Qwen) | **Pass (assist)** | neighbors mode: 5 edits, 22 messages. Every destructive tool hung on `rg` stdin bug in impact provider. Context overflow (266k) before cargo_check. Human: rewrote execute_neighbors (borrow/format errors), fixed imports, found+fixed impact `rg` bug |
| 14 (Qwen) | **Near-pass** | stats mode: 6 edits, 40 messages, ~3 min. rg fix confirmed — no hangs. Ran cargo_check (pass!), tests (fail → self-diagnosed → fixed). Context overflow on final turn. Human: 2 clippy nits only. Best run yet — 95% agent |
| 15 (Qwen) | **Partial** | lesson-briefing: `build_lessons_summary` + 3 tests, wired into `build_repo_briefing`. Sort bug: `metadata.get("created_at")` instead of `GraphNode.created_at` (top-level `DateTime<Utc>`). 1-line reviewer fix. 31 tool calls, 64 msgs, no context overflow. ~95% agent |
| 16 (Qwen) | **Pass** | max-output-tokens: Added `max_output_tokens: Option<u32>` to AgentConfig (4 locations), wired fallback chain in workflow.rs, set 1500 in default.toml, updated test. Zero bugs. 28 msgs. First clean pass with no reviewer fixes. 100% agent |
| 17 (Qwen) | **Pass** | read-auto-truncate: `MAX_AUTO_LINES=300` const, detection flags, truncation logic + notice, 2 tests. Self-fixed compile error. Ran `cargo fmt` globally (reformatted unrelated `graph_query.rs` — reverted by reviewer). Notice text missing "Use offset and limit" guidance (reviewer 1-word add). Third consecutive pass. ~98% agent |

### Key insights

1. **Tool documentation gaps** (runs 1–4): The agent didn't need new tools — the `read` tool already supported `offset` and `limit`. The problem was purely a system prompt documentation gap. When the tool description mentioned these params and the file reading discipline section showed a concrete grep→read(offset,limit) workflow, the agent followed it perfectly.

2. **Rust-specific pitfalls** (run 5): On a harder task (250+ lines of new code), the agent understood the domain model perfectly but failed on Rust mechanics: (a) added `petgraph = "0.6"` directly when `graphirm-graph` re-exports from 0.7, causing type mismatches; (b) moved `Arc<GraphStore>` into a closure then tried to use it in a second closure. These are systematic — the system prompt should document crate re-exports and show the `Arc::clone` pattern for spawn_blocking.

3. **System prompt fixes work** (run 6): After adding "Crate dependency rules" and "Async patterns" sections to the system prompt, the agent correctly used `graphirm_graph::` re-exports and cloned `Arc` before `spawn_blocking` closures. No petgraph/chrono added to Cargo.toml. The agent also fixed String borrow issues independently (a new pattern not in the prompt), showing the crate dep section generalised well.

4. **External format knowledge gaps** (run 7): The agent implemented a tool that parses `cargo check --message-format=json` output, but didn't know the actual JSON envelope format (`{"reason":"compiler-message","message":{...}}`). It assumed diagnostic fields (`level`, `message`) were at the top level. Also assumed non-zero exit code = failure, but `cargo check` returns 101 for compilation errors (which is the tool's primary use case). Tests only validated the "no errors" path. Lesson: when a tool parses external tool output, the system prompt should document the format or instruct the agent to inspect real output first.

5. **Abstraction boundaries must be explicit** (runs 11–12): Without an explicit "use the public API" rule, the agent happily accessed `graph.pool` (private) and imported `rusqlite` (not a dependency). After adding an "Abstraction boundaries" section to the system prompt, the agent correctly modified `traverse()` in `graphirm-graph` to return depth info, then called the updated method from the tool. One prompt section eliminated an entire category of architectural mistakes.

6. **Impact analysis `rg` hangs when stdin is a pipe** (run 13): `count_dependents()` in `GraphImpactProvider` ran `rg --files-with-matches <pattern>` without a path argument. When the server is spawned with stdin as a pipe (not TTY), `rg` reads from stdin instead of searching the current directory, blocking forever. Every destructive tool call in the session hung until the `rg` process was manually killed. Fix: add `"."` as explicit path arg + `.stdin(Stdio::null())`. This was a latent bug in Phase 22 that only manifested when the server's stdin was a pipe (e.g. spawned from Cursor's shell tool or a script).

7. **`rig` JSON parse errors are a recurring failure mode** (runs 11–12): The `rig` Rust LLM library fails to deserialize certain OpenRouter responses (`data did not match any variant of untagged enum ApiResponse`). This kills sessions mid-work. Not a timeout — the response arrives but can't be parsed. Needs investigation: retry logic, or pinning a `rig` version that handles OpenRouter's response format.

8. **LLM timeout is critical infrastructure** (runs 8–9): `timeout_seconds = 300` was in the config but never wired to code. The LLM call in `workflow.rs` used `tokio::select!` with only a cancellation token — no time-based arm. When OpenRouter/Qwen hung during tool-call generation, the session stayed "running" forever. Fix: added `tokio::time::sleep(llm_timeout)` as a third `select!` arm; session transitions to `"error"` on timeout. Also discovered the agent successfully wrote code to `grep.rs` before hanging — partial work was silently lost.

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
- ~~Add "Abstraction boundaries" to system prompt~~ ✅ Done — run 12 validated (agent used public API)
- ~~Investigate `rig` JSON parse errors on OpenRouter responses~~ ✅ Done — root cause: `rig-core` 0.31 can't deserialize empty tool arguments (`"arguments": ""`). Fixed in rig #1437 (Feb 25). Upgraded to `rig-core` 0.33.0 (Mar 17). Zero breaking changes, all 75 LLM tests pass
- ~~Implement Phase 1.5 (lesson/convention entity types in briefing)~~ ✅ Done — `build_lessons_summary` in `briefing.rs`, wired into `build_repo_briefing`, 3 tests (dogfood run 15)
- Tune polling intervals
- Add support for multi-turn conversations
- Handle workspace ↔ repo sync (agent workspace vs `~/graphirm-repo/`)

### Ideas: context overflow mitigation (graph-native)

**Strong candidates (build in order):**

1. **Structural sub-nodes on first read** — `read` tool parses file into structural sub-nodes (one Content node per function/struct/impl/test, linked via `Contains` edges). Metadata: `symbol_name`, `symbol_type`, `line_start`, `line_end`. Agent navigates code via graph traversal, same as session graph. On re-read, checks for existing Content node for same path with newer mtime — returns outline from graph instead of re-reading disk. Agent drills in with `offset`/`limit`. Foundation for everything else. ~150 lines Rust, regex-based, no tree-sitter.

2. **Per-symbol fingerprinting** — Shazam-style: on first read, compute SHA256 per function/struct block and store as metadata on structural sub-nodes. On re-read, diff fingerprints against current file on disk. Match → "unchanged, here's the outline." Partial mismatch → return only changed sections at full resolution + outline for the rest. Agent never loads 2048 lines to discover 30 lines changed. ~150 lines Rust on top of #1.

3. **Adaptive context decay** — `build_context` uses graph edge timestamps + `Reads`/`Modifies` edges to set resolution per Content node. Recent/modified → full body. Stale (>N turns, unmodified) → signature only (from structural sub-nodes). Mechanical: `read` auto-truncates files over M lines, appending outline. This is how you tune #1 and #2, not a standalone feature.

**Quick win:**

- **File size convention** — enforce max ~500 lines per file as a project convention. Split `graph_query.rs` (2048 lines) into `graph_query/mod.rs` + per-mode files. Split `store.rs` (2293 lines) similarly. 88% token reduction for the dogfooding case with zero infrastructure. Doesn't help with external codebases but eliminates the immediate problem.

**Other ideas (scored):**

| Idea | Side | Token savings | Effort | Graph-native? | Standalone? | Score |
|------|------|---------------|--------|---------------|-------------|-------|
| `max_tokens` cap (1000-1500/turn) | Output | High (~15% of total context) | 1 line config | No | Yes | 9/10 |
| Tool call deduplication | Input | High (eliminates re-reads) | Low (~50 lines) | Yes | Yes | 9/10 |
| "Act, don't narrate" system prompt | Output | Medium (~5-10%) | 2 sentences | No | Yes | 8/10 |
| Auto-truncate `read` at 300 lines | Input | Medium (~10-15%) | Low (~30 lines) | No | Yes | 8/10 |
| Tool output compression | Input | Medium (20-30% of non-file tokens) | Low (~30 lines/tool) | No | Yes | 7/10 |
| Strip assistant prose from history | Output | Medium (~20% of accumulated history) | Medium (~60 lines in `build_context`) | No | Yes | 7/10 |
| Tool output diffing (graph-backed) | Both | High (compounds per repeated call) | Medium (~60-80 lines/tool) | Yes — diff prior Content nodes | Yes | 8/10 |
| Edit-without-read | Input | Medium (skips full read before edit) | Zero (system prompt only) | No | Fragile | 6/10 |
| Working set vs reference | Input | High (stale reads → outline) | Medium (~100 lines) | Yes | Needs #1 | 6/10 |
| `outline` tool | Input | Medium (first read) | Low (~100 lines) | No | Stepping stone | 5/10 |
| Context budget awareness | Both | Low (guidance only) | Low (inject token count) | No | Orthogonal | 4/10 |

**Prerequisite: node lifecycle management**

Tool output diffing, structural sub-nodes, and fingerprinting all create Content nodes. Without lifecycle management, the graph bloats — 100+ nodes per session, thousands across sessions, mostly stale. Node tiers:

| Tier | Examples | Lifespan | After expiry |
|------|----------|----------|-------------|
| Ephemeral | cargo_check output, grep results, ls output | Keep last N per tool per session | Delete (diff computed, raw output no longer needed) |
| Session | File reads, edit diffs, tool output diffs | Session duration | Archive or delete at session end |
| Persistent | Knowledge, Interactions, Agent nodes | Forever | Never pruned |

Options: `tier` metadata field on Content nodes (or inferred from `content_type`); keep-last-N for tool diffing (exactly 1 cargo_check node exists at any time — previous deleted after diff); session-end cleanup hook for session-tier nodes; optional TTL (`expires_at`) for time-based expiry.

**Prior art:** Aider repo-map (tree-sitter outline, graph-ranked, ~1k tokens); Claude Code compaction (3-layer: micro/auto/manual, summarizes at 95% window); OpenCode smart context (outlines classes, strips comments, import graph). Graphirm's advantage: the graph already tracks reads/modifies/timestamps — structural sub-nodes + fingerprinting make context management graph-native rather than bolted-on.
