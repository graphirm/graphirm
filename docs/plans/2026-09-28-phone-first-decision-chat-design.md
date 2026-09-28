# Phone-first decision chat — design (Track B, phase B0)

> **For Claude:** This is a design doc, not an implementation plan. After approval,
> use `writing-plans` to produce `docs/plans/2026-09-28-phone-first-decision-chat.md`,
> then implement B1–B9. Do not write UI code from this file alone.

**Status:** APPROVED 2026-09-29. No UI code written yet. Implementation plan:
`docs/plans/2026-09-28-phone-first-decision-chat.md`.
**Scope:** Track B only. Track A (`delegate_pi`) is on `main` (`edddd49`).
**Parent brief:** "Graphirm as director" (2026-09-28). Locked decisions are not reopened.

---

## Goal

`web-app/` becomes a chat-first, phone-first client so the control plane is visible
and correctable: typed reply blocks, live STEPS (including Pi's tool calls), a Jev
chip per turn, a confirm card, and pinned rules. The whiteboard stays as the Graph
tab and keeps working. Desktop shows the same 430px column, centred. The current
two-pane layout stays selectable until the new one has been used for real work.

Nothing in context selection, compaction, memory ranking, or knowledge extraction
changes. Jev routing behaviour does not change. `delegate_pi` does not change.

## What this is not

A new app, a native shell, a new auth path, a Hermes UI, or a rewrite of
`client.ts` / `sse.ts` / `useSession` / `GraphCanvas`. v0 (`/home/krs/jev-hooks/v0/`)
is a visual reference (430px column, bottom tabs, dark surface, amber accent, mono
kickers, typed blocks). Its copy and components are not ported.

---

## Current client

`App.tsx` is a session bar over a two-pane row: `ChatPane` beside `GraphCanvas`.
Both stay mounted. Chat can collapse. Steer-from-node, outline steer, `context_root`,
keyboard shortcuts, and popovers all live on that row.

`useSession` appends `message_delta` text into `streamingMessage`. Segments are
attached after the turn, from `Contains` children via `chatSegments.ts` /
`SegmentCard`. There is no partial-JSON parser. `HitlOverlay` dumps tool arguments
as JSON and calls approve / reject / modify with `approval.node_id`.

`[agent.segments].labels` today: `observation`, `reasoning`, `code`, `plan`, `answer`.

## Facts from Track A that this design uses

- `AwaitingApproval.node_id` is the LLM **tool call id**, not a graph node id
  (`workflow.rs` gates on `NodeId::from(call_id)`). Approve still goes to
  `POST /api/graph/{session_id}/node/{node_id}/action`.
- `AwaitingApproval` does not carry `hitl_judge`. The score is computed before the
  gate and stored on the tool-result node after the call runs.
- `GET /api/graph/{id}/tasks` walks `Agent --Produces--> Task`. A Pi Task is
  `DelegatesTo` / `Interaction --Produces--> Task`, so that endpoint is empty for
  delegations (backlog, S·P2). The session graph payload already contains those nodes.
- Pi's own `bash`/`write`/`edit` are observe-only. The UI must say "judged, not gated".
  Abort (`POST /api/sessions/{id}/abort`) kills the Pi process group.
- Jev routing is **live** (`shadow = false`, DEC-0928j(c)). The parent brief's
  "stays in shadow" line is stale. B4 displays the deciding tier. B8b records
  feedback; it does not change the router.

---

## Layout (B1)

One column, max-width **430px**, centred on viewports wider than that. Background
outside the column uses the same surface colour so the page does not look like a
phone floating on a different app.

**Tab bar**, fixed to the bottom of the column, four tabs:

| Tab | Shows |
|---|---|
| Chat | Session thread, composer, confirm card, streaming blocks |
| Review | Attention list (below) |
| Rules | Pinned knowledge |
| Graph | Existing `GraphCanvas` |

Chat is the default tab. The session bar (session switcher, new, pause/resume,
auto-approve, export) stays above the column on every tab.

**Graph stays alive.** First visit to Graph mounts `GraphCanvas`. Later tab switches
hide it (`hidden` / `inert`) and do not unmount it, so node positions, selection,
and the xyflow instance survive. Chat collapse and the two-pane shortcuts apply
only in the legacy layout.

**Legacy switch.** `localStorage` key `graphirm.layout` = `chat` (default) or
`legacy`. Legacy renders today's `App` tree unchanged. The switch is a control on
the session bar, not a tab. No server setting.

Keyboard shortcuts that move the camera or cycle graph layout no-op when the Graph
tab is not visible, instead of scrolling the chat.

## Visual tokens

Add variables to the existing theme (do not replace it):

- column width `430px`
- accent `#f2b84b` on surface `#111315`
- kicker labels in a monospace face, uppercase, tracked

Block chrome, the Jev chip, and the confirm card use these. The legacy layout does
not have to be restyled in B1.

## Blocks and STEPS (B2)

`SegmentCard` becomes a `BlockView`: mono kicker, a done mark when the block is
closed, a streaming cursor while it is open.

Display names for existing labels stay as stored (`observation`, `reasoning`,
`code`, `plan`, `answer`). Two labels are added to `[agent.segments].labels` and
the segment prompt: `caveat`, `action`. That is a prompt-label change, not a change
to how segments are extracted or ranked.

**STEPS is not a segment.** It is a row on the assistant turn, collapsed by default,
built from `tool_start` / `tool_end` during the turn and from tool-role interactions
after it. One row per tool call: name, error mark, one-line argument summary.

When the tool is `delegate_pi`, the row reads **delegated to Pi** and, once the
tool result exists, appends the tool-call count from the summary. Nested Pi tool
nodes (executor `pi` under the Task) are listed inside that row, not as peer steps
of the director. The row states **judged, not gated**. Graphirm's own `bash` /
`write` / `edit` stay normal steps; those are the ones a confirm card can pause.

## Streaming parser (B3)

On each `message_delta`, if the buffer looks like a segments envelope, parse the
longest valid prefix of `{"segments":[…]}` into `streamingMessage.segments` with
state `pending` | `streaming` | `done`. The closed objects are `done`; the last
incomplete object is `streaming`.

The chat never shows the raw JSON envelope. If the buffer is not yet an object,
show the plain text only when it does not start with `{`. If it starts with `{` and
cannot be parsed, show a single recovery row **Fixing the format…** and keep the
previous good blocks. On `message_end`, drop the partial parse and use the
persisted segments (today's path). If that client parse is unreliable in B3, the
follow-up is server `segment_start` / `segment_delta` / `segment_end`. Not in v1.

## Jev chip and sheet (B4)

A pure function of assistant Interaction metadata:

`{tier} · {strategy} · p={routing_confidence}`

from `model_tier`, `routing_strategy`, `routing_confidence`. Missing fields omit
that piece; the chip is absent when all three are missing.

The sheet opens from the chip and shows `routing_reason` as stored (live Jev
string, or `jev_fallback(…)`). **Wrong pick** and **Keep it** render disabled,
with no network call, until B8b.

## Confirm card (B5)

Replace the body of `HitlOverlay`. Approve, reject, and modify keep the current
handlers and the current endpoint. `node_id` stays the tool call id.

| Tool | Body |
|---|---|
| `bash` | command string |
| `write`, `edit` | path and a short content or diff preview |
| `delegate_pi` | the `task` string; a line that Pi will not pause for its own calls |
| other | formatted arguments, as today |

**Cancel the rest** calls the existing abort. That kills a running Pi.

`hitl_judge` is not on `AwaitingApproval` today. B5 adds an optional
`hitl_judge` field to that event and to the SSE payload, copied from the verdict
already computed in `judge_auto_approve`. The card shows `P(irreversible)` when
the field is present. Absence leaves the card unchanged. This does not change
the threshold or whether the gate pauses.

## Rules tab (B6)

Client only. `GET /api/knowledge/pinned` lists rules. Create uses
`POST /api/knowledge` with `pinned: true`. Edit uses `PATCH /api/knowledge/{id}`.
Delete uses `DELETE /api/knowledge/{id}`. No new routes.

## Review tab (B7)

Client only. One list, current session plus the sessions array already loaded:

- pending approval on the open session (same `pendingApproval` as the confirm card; tapping it switches to Chat)
- sessions with status `failed` or `token_cap_exceeded`
- sessions that are paused
- running Pi work: Task nodes in the **loaded session graph** whose metadata
  `executor` is `pi` and whose task status is not `completed` or `failed`

Do not call `GET /api/graph/{id}/tasks` for this. That endpoint misses delegated
Tasks. Fixing it stays the existing backlog item; B7 does not depend on it.

## Server seats (B8)

Rust, separate commits, offline tests through the existing decisions transport.
Fail-soft: a Jev error stores nothing and does not change the turn. Raw scores
in metadata. Thresholds live in the readers. Version constant on the payload.

### B8a — reply judging

After each assistant `message_end`, ask the five assistant questions from
`/home/krs/jev-hooks/jev_hooks/questions.py` (tagset v4), wording unchanged:
`move`, `claims_completion`, `cites_evidence`, `presents_decision`,
`presents_as_options`. `previous_message` is the last three turns, same shape as
jev-hooks `build_body`. Store the raw answers on the assistant Interaction as
`metadata.reply_judge` and include that metadata in the turn's `graph_update`.

Readers (client only, no loop change):

- CAVEAT row when `claims_completion` is high and `cites_evidence` is low
- decision card when `presents_decision` ≥ 0.6
- hint to ask for options when `presents_as_options` < 0.7

These are the jev-hooks baselines. They are display thresholds, expected to move
once graphirm replies are labelled. They do not block the turn.

On a human turn, `pin_candidate` (same question set, human side) surfaces a
**suggested rule** in the Rules tab. One tap calls the existing pin API. No
automatic pin.

### B8b — routing feedback

`POST /api/interactions/{id}/routing-feedback` with `{ "verdict": "wrong" | "keep" }`
writes metadata on that Interaction. `GET /api/routing/report` counts verdicts.
The sheet buttons from B4 call this and then enable. The router does not read
the verdict.

## Plan card (B9)

When the thread has Task nodes that belong to a plan (title, and metadata or
fields the agent already stores), render a card:

`PlanStep { title, note, risky, enabled, executor }`

`executor` is `graphirm` or `pi` when the metadata says so; otherwise `graphirm`.
A `pi` step is a label. The card does not spawn Pi.

**Run N steps** calls the existing session `resume`, with the enabled step titles
in the steer text. The director model decides whether that becomes `delegate_pi`.
No new run API.

---

## Approaches considered

1. **New client beside `web-app/`.** Clean slate, matches the v0 tree. Rejected:
   the brief locks one client, and steer, SSE, and the canvas would be copied.
2. **Redesign in place, unmount the canvas off the Graph tab.** Smaller DOM.
   Rejected: xyflow state and saved positions reset on every tab change, which
   breaks the "whiteboard still works" acceptance check.
3. **Redesign in place, keep the canvas mounted after first open, legacy layout
   behind `localStorage`.** Chosen. Reuses `useSession`, `GraphCanvas`, and the
   HITL handlers. Legacy is the escape hatch until the column has been used for
   real work.

Segment streaming: client prefix parse first (brief). Server segment events only
if B3 proves the prefix parse wrong.

Review data: walk the session graph (chosen) rather than extend `GET …/tasks`
inside B7. The brief says B7 is client only, and the graph payload already has
the Task nodes.

## What does not change

- `delegate_pi`, Pi process wrapper, graph writes, observe-only judge
- context selection, compaction, memory ranking, knowledge extraction
- routing strategy and `shadow`
- auth (`VITE_API_KEY`, Bearer, `client-config`)
- TUI layout

## Testing

- B1–B7, B9: `cd web-app && npm run build`. Manual pass at 430px and at a desktop
  width: send, stream blocks, chip, STEPS during a Pi delegation, approve a
  graphirm destructive call, pin a rule, open Graph and steer from a node, switch
  to legacy and back.
- B5's optional SSE field and B8: `cargo test` with the mock decisions transport.
  No network.
- `web-app/src/types/graph.ts` stays aligned with any new metadata fields.

## Acceptance

On a phone browser on the LAN at 430px: send a prompt, watch blocks stream, see
the Jev chip, watch STEPS fill with Pi's calls during a delegation, approve a
destructive graphirm call from the confirm card, pin a rule, open the Graph tab
and see the whiteboard. Desktop shows the column centred. Legacy layout is
selectable. Web build and `cargo test` are green.

## Non-goals (v1)

Hermes. Gating Pi's internal tool calls. Native packaging. Replacing the segment
envelope or GLiNER2. Changing routing behaviour. A non-coding domain. Fixing
`GET /api/graph/{id}/tasks` (backlog). Loading the full TUI config file (backlog).

## Decisions

| # | Choice | Why |
|---|---|---|
| 1 | Redesign in place; legacy behind `localStorage` `graphirm.layout` | Locked in the parent brief. Server must not need to know which chrome the browser uses. |
| 2 | Keep `GraphCanvas` mounted after the first Graph visit | Unmounting drops positions and selection. Hide, don't destroy. |
| 3 | STEPS from tool events, not a new segment label | Pi and graphirm tool calls already stream as `tool_start` / `tool_end`. A segment cannot list them live. |
| 4 | Add `caveat` and `action` labels only | Parent brief. Extraction and ranking stay put. |
| 5 | Client parses the segments envelope; no new SSE events in v1 | Parent brief. Server events are the fallback if the prefix parse fails in practice. |
| 6 | Confirm card keeps the tool-call id and the current action route | A4: the gate key is the tool call id. A new id would break approve. |
| 7 | Optional `hitl_judge` on `AwaitingApproval` | The score already exists before the pause. The card shows it when present and ignores it when not. |
| 8 | Review reads the session graph, not `GET …/tasks` | That endpoint misses `DelegatesTo` Tasks. B7 stays client-only. |
| 9 | Reply-judge thresholds are display-only | Observe-first, fail-soft, same rule as the tool judge. Baselines will move. |
| 10 | Plan "Run" is `resume` plus steer text | No new executor endpoint. A `pi` step does not call `delegate_pi` from the browser. |
| 11 | Chip describes the live router | `shadow = false` already shipped. Feedback in B8b is recorded, not applied. |
