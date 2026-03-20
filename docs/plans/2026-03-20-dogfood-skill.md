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

## Phase 3: Iterate ✅ (initial round complete)

Four dogfood runs completed 2026-03-20. Results in `docs/dogfood-findings.md`.

### System prompt improvements discovered

| Run | Failure | Fix |
|-----|---------|-----|
| 1 (Claude) | Used `cd` in bash (not persisted) | Added "absolute paths, never cd" to system prompt |
| 2 (Qwen) | Read whole files → context overflow | Added grep-first instruction (agent ignored it) |
| 3 (Qwen) | Same context overflow | Discovered: `read` tool already had `offset`/`limit` but system prompt didn't mention them |
| 4 (Qwen) | **Pass** | Documented `offset`/`limit` in tool description + reinforced file reading discipline |

### Key insight

The agent didn't need new tools — the `read` tool already supported `offset` and `limit`. The problem was purely a system prompt documentation gap. When the tool description mentioned these params and the file reading discipline section showed a concrete grep→read(offset,limit) workflow, the agent followed it perfectly.

### Open items

- Tune polling intervals
- Add support for multi-turn conversations
- Handle workspace ↔ repo sync (agent workspace vs `~/graphirm-repo/`)
- Test on a real backlog item (not just struct field additions)
