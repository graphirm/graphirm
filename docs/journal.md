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

## 2026-10-08 — async-trait 0.1.92 so clippy 1.99 can build

**Context:** CI runs `cargo clippy --all-targets --all-features -- -D warnings` on stable. Clippy 1.99 flags `double_must_use` on every `#[async_trait]` method because 0.1.89 stamps a bare `#[must_use]` on a future that is already must-use. Local clippy 1.93 did not have that lint, so the branch went green here and red on the pull request.

**Decision:** Bump `async-trait` from 0.1.89 to 0.1.92. That release stops emitting the attribute. The `Cargo.toml` constraint stays `0.1`.

**Alternatives:** `#[allow(clippy::double_must_use)]` on each trait. That papers over the macro in every crate that uses it, and the next trait added would fail CI again.

**Consequences:** The lockfile pulls `syn` 3 for the macro. MSRV stays 1.88 (`async-trait` 0.1.92 asks for 1.71).

**Refs:** https://github.com/graphirm/graphirm/pull/1

---

## 2026-10-08 — Piece records are nodes under the reply

**Context:** The HTML-index plan stored the cut as a `pieces` array on the assistant Interaction. A list inside one node cannot take an edge. The split exists so a person can accept, drop, or connect one part.
**Decision:** A clean cut creates Content nodes. `content_type` is `reply_part` or `reply_line`. The assistant message `Contains` each part. Each part `Contains` its lines. No new node type and no new edge type. A failed cut creates none of these nodes.
**Alternatives:** Metadata on the assistant message lost, because the graph cannot point at one entry. A sixth node type lost for now. Content already carries a `content_type` and a body.
**Consequences:** The plan `docs/plans/2026-10-08-html-piece-index.md` Task 4 writes the subgraph, not metadata. Same-turn neighbors still do not get `applies_to`. A later reply that names an earlier part gets its own nodes. The mention stays in the text.
**Refs:** `docs/plans/2026-10-08-html-piece-index.md`.

## 2026-10-08 — An HTML index is how a Pi reply becomes pieces

**Context:** Asking the model again for an outline, then once per heading, copied or dropped the reply and cost a call per heading. A single HTML page with an index of links cut cleanly: every link found its part, and the kind was the class on that part.
**Decision:** Graphirm asks Pi for that page, cuts it locally, and stores the records beside the reply. The cutter accepts a part only when the link, the id, and one of the eight kind words match. A failed cut is retried once. The reply text stays either way. Markdown replies keep `structure_segment`.
**Alternatives:** Outline-then-expand lost. Forcing the body with a provider JSON schema lost earlier (empty arrays, wrong kinds). A separate kind labeler stays optional and is not on the cut path.
**Consequences:** The kind on the record is the kind Pi wrote. Context selection, compaction, memory ranking, and knowledge extraction do not change. Plan: `docs/plans/2026-10-08-html-piece-index.md`.
**Refs:** `pi_groceries.py` cutter, `docs/plans/2026-10-05-reply-pieces.md`.

## 2026-10-05 — The Cursor tiling check is ignored unless asked for

**Context:** The tiling stress test reads live transcripts from `~/.cursor/projects`. A silent pass looks like the check ran. An empty directory used to fail, which is a reason to commit transcripts so CI can see them.
**Decision:** Mark `cursor_transcripts_tile_when_present` with `#[ignore]`. A missing home directory, a missing projects directory, or no assistant text returns without failing. Untiled text still fails when the test is run with `--ignored` and transcripts are present.
**Alternatives:** Keep the early return and let cargo report a pass. Gate it on an environment variable inside the default test run.
**Consequences:** CI does not need Cursor transcripts. The local command is `cargo test -p graphirm-agent --lib cursor_transcripts_tile_when_present -- --ignored --nocapture`.
**Refs:** `crates/agent/src/pi_delegate/pieces.rs`.

## 2026-10-05 — The piece-version snapshot is a hand-checked fixture

**Context:** A hash-prefix sample of local Cursor transcripts was committed as `cursor-subset.jsonl`. That is a random sample of session text, not a scrub. Gitleaks found no keys. The replies still contained private code, paths, and project detail. The commit had not been pushed.
**Decision:** Replace the file with 35 hand-written replies: lists, fences, headings, tables, a quote, trailing questions, and seven texts over the 16,000-character cap. Regenerate the snapshot hashes. Leave `PIECE_PARSER_VERSION` and `PIECE_BASELINE_VERSION` at `"1"`, because the rules did not change. Drop the unpushed commit that contained the raw sample so it is not on the branch.
**Alternatives:** Scrub all 135 in place. Keep the raw file and rely on the scanner.
**Consequences:** A rule change still fails the snapshot test until the matching version and the snapshot hashes are updated together. A wording edit of the fixture moves the hashes and does not by itself require a version bump.
**Refs:** `crates/agent/tests/fixtures/pieces/README.md`.

## 2026-10-05 — Labels are keyed by byte range, not piece number

**Context:** A later splitter change renumbers pieces. A label that says "piece 3 is a caveat" would then point at a different block. Labeling 100 replies also will not finish in one sitting.
**Decision:** Each label row stores the run file, the segment index, a SHA-256 of the segment text, `parser_version`, `baseline_version`, and each piece's start and end. `graphirm label-pieces` skips a row already in `piece-labels.jsonl`. `graphirm score-pieces` matches on the byte range, counts a moved range separately from a kind error, and reports single-piece replies apart from replies with several pieces.
**Alternatives:** Key the file by piece order. Pre-fill the baseline and correct it.
**Consequences:** Bump `PIECE_PARSER_VERSION` when boundaries change and `PIECE_BASELINE_VERSION` when the kind rules change. A segment whose text hash no longer matches is not scored as a kind error.
**Refs:** `docs/guides/reply-piece-labels.md`.

## 2026-10-05 — Pi recordings are capped, and the first labels are blind

**Context:** Raw Pi stdout keeps thinking, tool arguments, and file contents. It is written on every production run. The baseline labeler will be scored against hand labels, and a suggested kind is easy to accept.
**Decision:** `~/.graphirm/pi-runs` stays at or below 256 MiB by deleting the oldest `*.jsonl` files. One run stops copying at 32 MiB. `graphirm label-pieces` does not show the baseline kind unless `--show-baseline` is passed. The kind definitions live in `docs/guides/reply-piece-labels.md`.
**Alternatives:** Leave the directory uncapped until the labeled set exists. Pre-fill the baseline kind and ask the person to correct it.
**Consequences:** A long run's tail is missing from the recording after 32 MiB. The first score is only as good as the blind labels.
**Refs:** `docs/plans/2026-10-05-reply-pieces.md`, `docs/guides/reply-piece-labels.md`.

## 2026-10-05 — The parser owns the reply text

**Context:** A finished Pi reply should become ordered pieces (statement, options, steps, instructions, example, caveat, code, question) without using the question. A 0.6B structure model was tried as a schema filler and copied schema words instead of the reply.
**Decision:** `pulldown-cmark` cuts the flattened assistant text and stores UTF-8 byte offsets. The character cap is checked first and counts Unicode scalar values, the same unit as the 16,000-character interaction cap. Narration (`stopReason: toolUse`) is parsed, and every block stays a statement except fences, which stay code. Final replies get a deterministic baseline labeler, including its known misses. Pi stdout is copied to `~/.graphirm/pi-runs` (override `GRAPHIRM_PI_RUNS_DIR`, `off` disables). Cursor transcripts may stress-test tiling from `~/.cursor` and are not committed.
**Alternatives:** Asking the answering model to emit the pieces while it writes. Asking Osmosis to fill a nested pieces schema. Parsing inside a Pi TypeScript extension. Storing same-turn adjacency as `applies_to`.
**Consequences:** Cross-turn edges (`answered_by`, `executed_by`, `implements`) wait until about 100 real Pi replies are hand-labeled and the baseline has a score. The grammar-constrained labeler has to beat that score. `adjacent_to` is not stored yet.
**Refs:** `docs/plans/2026-10-05-reply-pieces.md`.

## 2026-09-29 — Phone-first chat design approved as written

**Context:** Track B B0 draft. The parent brief already locked the column, the four tabs, and the legacy switch.
**Decision:** Approve `docs/plans/2026-09-28-phone-first-decision-chat-design.md` with no row changed. Implementation plan is `docs/plans/2026-09-28-phone-first-decision-chat.md`.
**Alternatives:** None raised at approval.
**Consequences:** B1–B9 can start. Graph canvas stays mounted after first open. Review does not use `GET …/tasks`. Plan "Run" only resumes.
**Refs:** design doc Decisions table.

## 2026-09-28 — Session restore skips spawned Agents; `enabled = true` stays the product default

**Context:** Final review of Track A. `restore_sessions_from_graph` loaded every Agent node. A Pi Agent (`status = "running"`, no workspace) would become a promptable session after a spoke restart. Separately, `config/default.toml` is both the local default and the spoke file (`workspaces_root = /data/workspaces`), and A4.2 set `[agent.pi] enabled = true` while `disable_bash` stays commented out.

**Decision:** Skip any Agent with an incoming `SpawnedBy` edge (Pi and in-process subagents share that edge). Keep `enabled = true` — the live check passed and A4.2 required the flip. A public deploy of this file must set `disable_bash = true`, which already hides and refuses `delegate_pi`. Do not turn `disable_bash` on in the committed file; that would also disable `bash` on every local and spoke session.

**Alternatives:** Filter only `metadata.executor == "pi"` (misses in-process subagents). Revert `enabled = false` (undoes the approved A4.2). Set `disable_bash = true` in `default.toml` (breaks the coding agent wherever this file is used).

**Consequences:** Spoke restart no longer lists `"pi"` sessions. Deploying this branch to `app.graphirm.ai` without `disable_bash` leaves `delegate_pi` registered; it fails closed only while the image has no `pi` binary.

**Refs:** `crates/server/src/session.rs`, `config/default.toml`, design doc security row on `disable_bash`.

## 2026-09-28 — `delegate_pi`: only a turn-ending message is Pi's result; judge observe-only reuses `JudgeOutcome`; TUI lifts only `[agent.pi]`

**Context:** Assembling `PiDelegateTool` (plan A2.7) over the reviewed `process.rs` / `graph.rs`.
Three choices were not determined by the design doc.
**Decision:** (1) `last_assistant_text` is set only by an assistant `message_end` whose
`stopReason != "toolUse"`. Pi narrates before tool calls ("Let me check the directory…") as a
`toolUse` message; the plan's "nonzero exit *without a result* → failure" rule only works if that
narration does not count. Every assistant message is still recorded as a graph node. (2) The
observe-only judge verdict is written through the existing `hitl::JudgeOutcome::to_metadata()` with
`pause: false, action: JUDGE_ACTION_OBSERVED` ("observed"), so Pi tool nodes carry exactly the keys
the in-process gate writes (`version`, `p_irreversible`, `threshold`, `action`, `latency_ms`) and
read-outs can group by `action`. The judge is spawned at `tool_execution_start` and awaited at
`tool_execution_end` under `DestructiveJudge::timeout()`; one warning per run on failure; handles
without an end event are aborted. (3) `graphirm chat` has never read `config/default.toml`;
`chat.rs` now lifts only `pi` from the file so `[agent.pi].enabled` reaches the TUI without
changing its prompt/judge/routing. `serve.rs` loads config before tools via
`commands::load_agent_config()`. (4) `register_pi_delegate` is `async` (both callers are async);
a missing binary is a startup warning and the tool is still registered so the failure is visible
to the model. (5) `ToolError::Timeout(u64)` has no message slot; the timeout's partial summary
goes to `Task.metadata.failure_detail`.
**Alternatives:** Count any assistant text as the result — made `FAKE_PI_NO_END` + exit 2 a
success. A separate metadata builder for Pi verdicts — drifts from the gate's keys. Load the whole
TOML in the TUI — right long-term (backlog S·P2) but a behaviour change the task did not ask for.
`block_in_place` for a sync `register_pi_delegate` — panics on a current-thread runtime for no gain.
`ExecutionFailed("timed out …")` for the timeout — loses the typed variant the loop can match on.
**Consequences:** A Pi run whose final message is a `toolUse` turn (Pi died mid-turn) reports
"(Pi produced no final message)" on exit 0 and fails on nonzero exit. `JUDGE_ACTION_OBSERVED`
is part of the metadata contract A3.1 tests against. The TUI still ignores every other TOML
section until the backlog item lands.
**Refs:** `docs/plans/2026-09-28-pi-delegate-executor.md` A2.7 implementation notes;
`crates/agent/src/pi_delegate/tool.rs`; `crates/agent/tests/fixtures/pi/hello-run.jsonl`
(line 170 is the `stop` message; earlier assistant messages are `toolUse`).

## 2026-09-28 — Pi subprocess wrapper: drain to EOF, receiver-drop discards, handle+JoinHandle shape

**Context:** `delegate_pi` (plan A2.5) runs `pi --mode json` as a child and turns its JSONL stdout
into `PiEvent`s. Four things about Pi 0.85.1 shaped the wrapper: it auto-retries transient provider
errors (`agent_end {willRetry:true}` → `auto_retry_start` → `agent_start` …), so `agent_end` is not
a terminator; it parses a leading `@` in a positional argument as a file reference (`Error: File not
found`, exit 1); its `bash` tool can leave a backgrounded grandchild holding the stdout pipe after Pi
itself has exited; and a consumer that stops reading events would, through a bounded channel and a
full 64 KiB pipe, stall Pi until the deadline and misreport `Timeout`.
**Decision:** (1) Read stdout to EOF and then wait for exit; `AgentEnd` is forwarded like any other
event and never ends the read loop. (2) A dropped or closed event receiver flips the reader into
*discard* mode instead of stopping it, so Pi can never block on a full pipe; `PiRunHandle::wait()`
closes the receiver first — "wait" means "stop consuming". (3) The public shape is
`spawn_pi(spec, cancel) -> PiRunHandle { events: mpsc::Receiver<PiEvent>, done: JoinHandle<..> }`
rather than an async `on_event` callback; the tool layer (A2.7) awaits graph writes between
`recv()`s, and `process.rs` stays free of graph/judge concerns. Dropping the handle aborts the
driver; a `GroupKillGuard` armed in `spawn_pi` right after `child.id()` SIGKILLs the whole process
group when the driver future is dropped — armed at spawn, not in the driver, because a handle dropped
before the driver's first poll never runs driver code. (4) Tasks longer than 64 KiB *or* starting
with `@` are written to a `0600` `create_new` temp file and passed as
`-- @<path> "Carry out the task described in the attached file."`; the file is removed when the run
ends. (5) After exit, the pipes get `POST_EXIT_GRACE` (3 s, capped by the remaining deadline) to
close; on expiry the group is killed and the run still returns `Ok` with the exit code and
`pipes_lingered: true` — Pi *did* finish; a straggler is a warning, not a failure. (6) Pi's env is
the inherited env plus `PI_SKIP_VERSION_CHECK=1` minus `GRAPHIRM_API_KEY`; the provider key is Pi's
own business (`~/.pi/agent/auth.json` / provider env vars) and is never read or logged.
**Alternatives:** Stop reading at `agent_end` — wrong under auto-retry and races the exit. Stop the
reader when the receiver drops — deadlocks Pi on a full pipe; the "caller must cancel" contract was
too easy to violate (the reviewer hit it by calling `wait()` without draining). Async callback API —
forces graph I/O into the process layer and makes cancel/timeout tests awkward. Returning `Timeout`
when a straggler holds the pipe after a clean exit — loses the exit code for a condition Pi is not
responsible for. Passing `--fake-knob` env through `spawn_pi` — rejected; the fake reads knobs from
argv (`PiConfig.extra_args`) so `spawn_pi` never takes arbitrary env and tests stay parallel-safe.
**Consequences:** `PiRunHandle` has a `Drop` impl, so it cannot be destructured; consumers use
`handle.events.recv()` + `handle.wait()`. Under `pipes_lingered` the last stdout lines may be lost
(the reader is aborted); counters and the stderr tail survive via shared atomics/mutex. A pgid is
never signalled after the direct child has been reaped except through the explicit
`kill_group_and_reap` path (same pid-reuse caveat as `bash`). The `@` rule means a task that
legitimately starts with an `@mention` still works — via the file.
**Refs:** `docs/plans/2026-09-28-pi-delegate-executor-design.md` D1;
`docs/plans/2026-09-28-pi-delegate-executor.md` A2.5; `crates/agent/src/pi_delegate/process.rs`;
`crates/agent/tests/fixtures/pi/fake_pi.sh`; commits `266a892` and its review follow-up.

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
