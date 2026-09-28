# Graphirm Decision Journal

Why things are the way they are. `docs/completion-log.md` records **what** shipped;
this file records **why** — decisions, rejected alternatives, and reversals.
Newest entry first. One entry per decision, not per session.

Add an entry whenever a choice is made that a future reader could not reconstruct from
the code or the plan alone: architecture, dependency, convention, measured trade-off,
something tried and abandoned. Skip routine work.

Entry template:

```markdown
## YYYY-MM-DD — <decision in one line>

**Context:** what forced the choice.
**Decision:** what was chosen.
**Alternatives:** what was considered and why it lost.
**Consequences:** what this commits us to, what to watch.
**Refs:** plan / commit / eval / transcript.
```

---

## 2026-09-28 — `bash` children run in their own process group and are SIGKILLed as a group on cancel/timeout

**Context:** `BashTool` cancelled by `task.abort()` on a `tokio::spawn`ed `wait_with_output`.
That drops the future but never signals the OS process: the `bash -c` shell and everything it
forked (`sleep 30`, `npm run dev &`, …) kept running after the agent gave up. Task A1.4 fixed
this now because `delegate_pi` (A2.5) needs the identical kill-on-cancel pattern.
**Decision:** Spawn the shell with `process_group(0)` + `kill_on_drop(true)`, `stdin(null)`;
capture `child.id()` immediately after `spawn()` as the pgid; on timeout, cancel, or an I/O error
from the reader, call `graphirm_tools::process::kill_group_and_reap(&mut child, pgid)` — an
explicit `libc::kill(-pgid, SIGKILL)`, then `start_kill()` as portable fallback, then `wait()`
under a 5 s reap timeout. `libc` is a new `[target.'cfg(unix)'.dependencies]` of
`graphirm-tools` (already in the lock file via tokio). The pgid is captured at spawn because
`child.id()` is `None` once the shell has been reaped — exactly the "shell exited, grandchild
still holds the pipe" case the group kill exists for.
**Alternatives:** (a) `start_kill()` alone — kills the shell but orphans grandchildren; proven
by `cancel_kills_the_shells_descendants`, which fails without the group kill. (b) Shelling out
to `kill -9 -- -<pgid>` (no `libc` dep) — first implementation; rejected because `kill(1)` is
absent from the `debian:bookworm-slim` runtime image (no `procps`), so the spawn failed with
ENOENT, was swallowed, and descendants still leaked in production. (c) Inherited stdin — with
the shell in a background process group a child reading the tty would get SIGTTIN and hang
until timeout; null stdin gives a deterministic EOF.
**Consequences:** bash children no longer receive the terminal's SIGINT/SIGHUP — acceptable
because the TUI is raw-mode (Ctrl-C is an event, not a signal) and `serve` cancels sessions on
SIGINT. `serve` still ignores SIGTERM, so a hard `pkill -f 'graphirm serve'` leaves live
sessions' children detached (backlog, S·P2). `process.rs` is the reference implementation for
`delegate_pi`'s kill path; A2.5 reuses it instead of adding `libc` to `graphirm-agent`.
**Refs:** `docs/plans/2026-09-28-pi-delegate-executor.md` Task A1.4; `crates/tools/src/process.rs`,
`crates/tools/src/bash.rs`; commits `136c7af` (kill(1) version) and its follow-up.

## 2026-09-28 — Pi becomes the delegated coding executor; graphirm stays the director

**Context:** Graphirm has the control plane (Jev routing in shadow, HITL judge, graph memory,
pinned rules) but its in-process `delegate` is unreachable from the server and TUI, and the
agent loop's own coding ability is what `graphirm-eval` measures. Codeporate already drives Pi
(`@earendil-works/pi-coding-agent`) via `--mode json` with a Python event mapper. A director
without a strong executor and an executor without a control plane are complementary.

**Decision:** Pi is the only external executor, run as one subprocess per delegation in
`--mode json`, observe-only (its tool calls are recorded and judged, not paused). Its work is
graph-native: Task + Pi Agent + `Interaction{role:"tool"}` nodes in the same shape in-process
tools produce. Live visibility comes from a new optional `ToolEventSink` on `ToolContext`.
Design with 15 sub-decisions: `docs/plans/2026-09-28-pi-delegate-executor-design.md`.

**Alternatives:** Hermes as executor — rejected: it is a second director with its own memory
and routing, leaving Jev no seat. Gating Pi's calls in v1 via `--mode rpc` or a Pi extension —
deferred: observe first, keep the parser process-agnostic so the switch is additive.
`GraphUpdate`-on-insert instead of a sink — rejected: the `EventBus` is per prompt and the
graph store has no channel, so there is nothing to emit from.

**Consequences:** `ToolContext` gains a field (21 mechanical edits). A new `[agent.pi]` config
block, disabled until the A4 live check passes. Track B (phone-first chat) is designed only
after A4. Nothing in context selection, compaction, memory ranking, or knowledge extraction
changes.

**Refs:** design doc above; backlog "Director / Pi executor"; Jev seat scoring
`~/codeporate-connect/docs/evaluations/2026-09-27-jev-where-in-graphirm.md`.

## 2026-09-28 — Governance docs must be updated in the same commit as the work

**Context:** Audit found the scaffolding (root + 7 scoped `AGENTS.md`, 3 Cursor rules, hooks,
1,500-line backlog, completion log) was mature but drifting: `backlog.md` header said
"Phases 0–35" while `AGENTS.md` said 0–55; `docs/README.md` was stuck at Phase 12;
`CHANGELOG.md` frozen since 2026-03-08; ~60% of `backlog.md` was completed items despite the
file's own convention to move them out. Root cause: no rule, hook, or skill obliged anyone to
update these files. The only doc-update obligation (`000-skills-first.mdc` → update
`00-execution-strategy.md`) pointed at a document abandoned in March. No decision journal existed.

**Decision:** Extend `.cursor/rules/000-skills-first.mdc` with a "Governance docs" block —
plan link on start, ✅ + completion-log entry on ship, journal entry on non-obvious decisions,
backlog entry for discovered-but-unstarted work — all in the same commit as the work. Drop the
`00-execution-strategy.md` line. Create this file. Add a pointer block to `AGENTS.md` because
`.cursor/` is gitignored and only `AGENTS.md` reaches the spoke and other clones.

**Alternatives:** (a) A separate `200-governance.mdc` rule — rejected; the existing rule already
owns "commit and progress rules", and a fourth always-apply rule adds context cost for no
isolation benefit. (b) A pre-commit hook that fails when `crates/` changes without a
`docs/` change — deferred; too blunt for refactors and hotfixes, revisit if drift recurs.
(c) ADR directory (`docs/decisions/NNNN-*.md`) instead of a single journal — rejected for now;
one file is easier to grep and append to at the current decision rate.

**Consequences:** Every agent session that ships anything now touches at least two docs. The
known drift (backlog header, `docs/README.md`, `CHANGELOG.md`, stale `src/main.rs` pointers in
`AGENTS.md`) is **not** fixed by this entry — it is separate cleanup work.

**Refs:** `.cursor/rules/000-skills-first.mdc`, `AGENTS.md` → Governance docs, commit `85ffc1a`
(same day: phase table moved out of `AGENTS.md` into `completion-log.md`).
