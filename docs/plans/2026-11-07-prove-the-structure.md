# Prove the structure

**Backlog:** "Prove the structure — P1 · L" in `docs/backlog.md`.
**Direction:** `docs/plans/2026-11-01-architecture-direction.md`.
**Intent:** `docs/intent.md`.

## What we want to prove

Graphirm turns an AI's work, down to the individual claims in its replies, into a
linked record. That is the innovation. We prove it in four steps, in order. A step
starts only when the one before it passes.

0. **The structure is functional.** It does what it claims, on its own terms.
1. **The structure helps the AI.** It beats an agent that keeps a notes file.
2. **The structure pays off on large work.** It beats a plain agent as the work outgrows the window.
3. **The structure makes the AI accountable.** An auditor answers "what, why, based on what, and who approved it" from the record.

Cost is judged against steps 2 and 3, as cost per solved task and cost per answered
audit question, not against a plain agent on single small bug fixes.

## Step 0: the structure is functional

No comparison with a plain AI. Each check reads Graphirm's own output.

| Check | Question | Measure | State 2026-11-07 |
|---|---|---|---|
| 0.1 Written correctly | Are replies split into parts with the right kind label? | Label accuracy on the hand-labeled set (100 replies, `docs/guides/reply-piece-labels.md`) | Parser and baseline labeler on `feat/reply-pieces`; the set is not labeled |
| 0.2 Linked correctly | Is each citation a real edge to the right evidence? | Share of citations written as edges; share pointing at the right node on a hand-checked sample | 96% point at a real node of the right type, but are metadata, not edges, and "right node" is not checked |
| 0.3 Reachable | Can the agent read, fetch and find an earlier part? | Probe tasks that ask for a named earlier part (by reply and number, and by topic); exact-text match rate | Not possible: `graph_query` omits `Content` text, context skips `Contains` |
| 0.4 Complete | Is anything missing from the record? | Script over finished sessions: every reply has parts or a recorded reason; every tool call has its result; every `write` / `edit` has a `Modifies` edge; every gated action has `ApprovedBy` or `RejectedBy` | Fails (2026-11-07, eval binaries): tool results and `write` / `edit` records are complete; no approval records; `bash` file changes unrecorded; parts lost on persist and on text next to tool calls |

Pass thresholds are set and committed here before any 0.x measurement runs.

0.4 thresholds and the script: `docs/plans/2026-11-07-structure-completeness.md` (committed 2026-11-07, before measuring).

### Builds needed for step 0

- Make reply parts readable, linked and retrievable (backlog, P1 · M): show kind and heading, fetch one part's text, find a part by reply and number. Needed for 0.3.
- Citations as real edges (backlog, P2 · M, raised to P1 by this plan): needed for 0.2.
- Label the 100-reply set (backlog "Reply-piece parser"): needed for 0.1.
- The completeness script: new, S. Needed for 0.4.

## Step 1: the structure helps the AI

Same tasks, same model. Arms: Graphirm with readable parts; the same agent told to
keep a notes file. Measure: tasks solved, correct recall of earlier decisions, cost
per solved task. This is the existing gate on the readable-parts item.

## Step 2: the structure pays off on large work

Long sequences of related tasks across many sessions in one large codebase, where
earlier decisions matter later and do not all fit in the window. Start from
`eval/multisession/` (pilot not run). Arms: Graphirm; a plain agent with compaction;
a plain agent with a notes file. Measure: tasks solved, correct recall, contradictions
with earlier work, cost per solved task, each at two or more sizes. Passes when
Graphirm is ahead on results or cost per solved task and the gap grows with size.

## Step 3: the structure makes the AI accountable

Standard audit questions on finished sessions: why was this file changed, what
evidence supported this claim, who approved this command, rebuild exactly what the
AI saw before this decision. Answered from Graphirm's record and from a plain
agent's chat log. Measure: share answered correctly, time per answer, replay match
rate, citation accuracy, approval coverage. Needs stored packs and replay on `main`
(context budget honesty, task 3).

## Not in scope

The linear working loop and graph-scoring removal stay as they are in the backlog.
They are measured on cost; this plan supplies the benefit side they are judged against.
