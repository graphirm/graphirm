# Phone-first decision chat — implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn `web-app/` into a 430px chat-first client (Chat / Review / Rules / Graph) without changing routing, Pi, or the measured context/memory path.

**Architecture:** Redesign in place. `useSession`, `client.ts`, `sse.ts`, and `GraphCanvas` stay. A shell chooses `chat` vs `legacy` from `localStorage`. New UI is mostly presentational over data the session hook already has. Two server additions only: optional `hitl_judge` on the approval event (B5), and reply-judge metadata plus routing-feedback (B8).

**Tech Stack:** React 19, Vite, TypeScript (`web-app/`), Rust agent/server. No new test runner. Pure TS modules are checked with `node --experimental-strip-types --test`. Rust with `cargo test`. Web gate is `cd web-app && npm run build`.

**Key decisions:** See `docs/plans/2026-09-28-phone-first-decision-chat-design.md` (approved 2026-09-29). Do not reopen them.

**Risks:**
- `GraphCanvas` must stay mounted after the first Graph visit or positions reset.
- Confirm-card `node_id` is the tool call id. Do not look up a graph node for approve.
- `GET /api/graph/{id}/tasks` misses Pi Tasks. Review must use the loaded graph.
- Do not change context selection, compaction, memory ranking, knowledge extraction, the router, or `delegate_pi`.

**Done when:** `npm run build` and `cargo test` are green, and the design doc's acceptance list is possible by hand at 430px (send, stream blocks, chip, Pi STEPS, confirm card, pin a rule, Graph tab, legacy switch).

---

### Task B1: Layout shell and legacy switch

**Files:**
- Create: `web-app/src/layout/layoutMode.ts`
- Create: `web-app/src/layout/layoutMode.test.ts`
- Create: `web-app/src/components/DecisionShell.tsx`
- Create: `web-app/src/components/TabBar.tsx`
- Modify: `web-app/src/App.tsx`
- Modify: `web-app/src/App.module.css` (column, tab bar, hide-not-unmount)

**Step 1: Failing test**

`layoutMode.ts` exports `LayoutMode = 'chat' | 'legacy'`, `LAYOUT_STORAGE_KEY = 'graphirm.layout'`, `readLayoutMode(storage: { getItem(k: string): string | null }): LayoutMode` (anything other than `'legacy'` is `'chat'`), `writeLayoutMode(storage, mode)`.

`layoutMode.test.ts` uses `node:test` and `node:assert/strict`. Cover missing key → `chat`, `'legacy'` → `legacy`, `'chat'` and garbage → `chat`.

Run: `node --experimental-strip-types --test web-app/src/layout/layoutMode.test.ts`
Expected: FAIL (module missing).

**Step 2: Implement the module, re-run, expect PASS.**

**Step 3: Shell**

`DecisionShell` props: the same callbacks `App` already passes to `ChatPane` and `GraphCanvas`, plus `layoutMode` and `onLayoutMode`.

- Session bar stays on top (existing `SessionBar`). Add a text button "Layout: chat | legacy" that calls `onLayoutMode`.
- `layoutMode === 'legacy'` renders the current two-pane tree from `App.tsx` (move that JSX here, do not restyle it).
- `layoutMode === 'chat'` renders a centred column (`max-width: 430px; margin: 0 auto; min-height: 100%`) and a bottom `TabBar` with Chat, Review, Rules, Graph. Review and Rules can be empty placeholders (`<p>Review</p>`, `<p>Rules</p>`) until B6/B7.
- Chat tab shows existing `ChatPane`. Graph tab: mount `GraphCanvas` on first visit (`graphMounted` state) and thereafter set `hidden` and `inert` when the tab is not Graph. Do not conditionally return `null` after the first mount.
- Shortcuts `onFitView` / `onToggleLayout` no-op unless the Graph tab is the active chat-layout tab or the layout is legacy.

`App.tsx` reads/writes `layoutMode` via `localStorage` and passes it down. Default `chat`.

**Step 4: `cd web-app && npm run build`** — exit 0.

**Step 5: Commit** `feat(web): chat-first column with legacy layout switch`

- [x] B1 done

---

### Task B2: BlockView and STEPS

**Files:**
- Create: `web-app/src/chat/steps.ts`
- Create: `web-app/src/chat/steps.test.ts`
- Create: `web-app/src/components/BlockView.tsx`
- Create: `web-app/src/components/StepsRow.tsx`
- Modify: `web-app/src/components/ChatPane.tsx` (render blocks + steps)
- Modify: `web-app/src/components/SegmentCard.tsx` (delegate look to `BlockView`, or make `BlockView` the only renderer and keep `SegmentCard` as a thin wrapper so existing imports compile)
- Modify: `config/default.toml` `[agent.segments].labels` — append `"caveat"`, `"action"`
- Test: `crates/agent/src/knowledge/segments.rs` existing prompt test still passes; add one assertion that a labels slice containing `caveat` appears in `build_segment_prompt` output

**Step 1: Failing `steps.ts` test** (node:test)

```ts
export interface StepInput {
  toolName: string;
  callId: string;
  isError?: boolean;
  argsSummary?: string;
  piSummary?: string; // delegate_pi tool result text, if ended
}
export interface StepRow {
  callId: string;
  title: string;
  detail: string;
  nestedPi: boolean;
}

export function buildSteps(calls: StepInput[]): StepRow[]
```

- `bash` → title `bash`, detail = argsSummary, `nestedPi` false
- `delegate_pi` with no summary → title `delegated to Pi`, detail `judged, not gated`
- `delegate_pi` whose `piSummary` contains `Tool calls: 2` → detail includes `2 tool calls` and `judged, not gated`

**Step 2: Implement `buildSteps`. PASS the node test.**

**Step 3: UI**

`BlockView` props: `{ kicker: string; content: string; state: 'pending' | 'streaming' | 'done' }`. Mono uppercase kicker, a check when `done`, a cursor when `streaming`. `ChatPane` renders `message.segments` through `BlockView` when present, else the existing plain body.

`StepsRow` is collapsed by default (`<details>`). `ChatPane` builds steps from tool-role messages that share the turn (same pattern the pane already uses to group a turn). Pass `delegate_pi` result content as `piSummary`.

**Step 4: `npm run build`. `cargo test -p graphirm-agent segment` ** so the prompt test stays green.

**Step 5: Commit** `feat(web): typed blocks and collapsed STEPS rows`

- [x] B2 done

---

### Task B3: Streaming segment parser

**Files:**
- Create: `web-app/src/chat/segmentStream.ts`
- Create: `web-app/src/chat/segmentStream.test.ts`
- Modify: `web-app/src/hooks/useSession.ts` (`message_delta` handler)
- Modify: `web-app/src/types/graph.ts` if `streamingMessage.segments` needs a state field (`SegmentPart` plus `streamState`)

**Step 1: Failing tests** for `parseSegmentPrefix(buffer: string): { segments: { type: string; content: string; state: 'done' | 'streaming' }[]; showRecovery: boolean; plainText: string | null }`

- `''` → no segments, `plainText` null, `showRecovery` false
- `'Hello'` → `plainText` `'Hello'`, no recovery
- `'{"segments":[{"type":"reasoning","content":"Hi"}'` → one segment `state: 'streaming'`, content `Hi`, `showRecovery` false
- `'{"segments":[{"type":"reasoning","content":"Hi"},{"type":"answer","content":"Ok"}]}'` → two `done`
- `'{not json'` → `showRecovery` true, no segments
- Do not return the raw buffer as `plainText` when it starts with `{`

Implement by scanning for complete `{"type":...,"content":...}` objects with a small state machine or `JSON.parse` on the longest prefix that is valid JSON. Incomplete trailing object: take the `"content":"` value so far as `streaming`. Do not use `eval`.

**Step 2: PASS `node --experimental-strip-types --test web-app/src/chat/segmentStream.test.ts`.**

**Step 3: Wire `message_delta`.** Append text as today. Also set `streamingMessage.segments` from `parseSegmentPrefix`. `ChatPane` prefers those segments while `streamingMessage` is set. On `message_end`, clear the stream (existing) so persisted segments replace it.

**Step 4: `npm run build`.**

**Step 5: Commit** `feat(web): parse streaming segment JSON without showing the envelope`

- [x] B3 done

---

### Task B4: Jev chip and sheet

**Files:**
- Create: `web-app/src/chat/jevChip.ts`
- Create: `web-app/src/chat/jevChip.test.ts`
- Create: `web-app/src/components/JevChip.tsx`
- Modify: `web-app/src/components/ChatPane.tsx`
- Modify: `web-app/src/types/graph.ts` only if `model_tier` / `routing_strategy` / `routing_confidence` / `routing_reason` are not already on the message metadata type. They live on Interaction metadata today; thread them onto the chat message type if `ChatPane` cannot see them.

**Step 1: Failing test**

```ts
export function formatJevChip(meta: {
  model_tier?: string;
  routing_strategy?: string;
  routing_confidence?: number;
}): string | null
```

- all missing → `null`
- tier `smart`, strategy `jev_router`, confidence `0.99` → `smart · jev_router · p=0.99`
- only tier → `smart`

**Step 2: Implement. PASS.**

**Step 3: Chip button on each assistant message that has a label.** Click opens a sheet (a `<dialog>` or a panel inside the column) showing `routing_reason` or "No routing detail". Two buttons **Wrong pick** and **Keep it**, both `disabled`. No fetch.

**Step 4: `npm run build`.**

**Step 5: Commit** `feat(web): Jev chip and routing sheet (feedback disabled)`

- [x] B4 done

---

### Task B5: Confirm card and optional judge score

**Files:**
- Create: `web-app/src/chat/confirmBody.ts`
- Create: `web-app/src/chat/confirmBody.test.ts`
- Create: `web-app/src/components/ConfirmCard.tsx`
- Modify: `web-app/src/components/HitlOverlay.tsx` (re-export `ConfirmCard` or replace the body and keep the same props so `ChatPane` / `GraphCanvas` callers compile)
- Modify: `web-app/src/types/graph.ts` `PendingApproval` — optional `hitl_judge?: { p_irreversible?: number }`
- Modify: `crates/agent/src/event.rs` `AwaitingApproval` — add `hitl_judge: Option<serde_json::Value>`
- Modify: `crates/agent/src/workflow.rs` emit site (~line 1127 and the second emit ~1610) to pass the verdict metadata when `judge_outcome` is `Some`
- Modify: `crates/server/src/routes.rs` `agent_event_to_sse` to include `hitl_judge` when `Some`
- Test: existing `agent_event_awaiting_approval_maps_to_sse_awaiting_approval` plus one case with the field set

**Step 1: Failing `confirmBody` test**

```ts
export function confirmSections(toolName: string, args: Record<string, unknown> | string): { heading: string; body: string; note?: string }
```

- `bash` + `{ command: "ls" }` → body `ls`
- `write` + `{ path: "a.py", content: "print(1)" }` → body contains `a.py` and `print(1)`
- `delegate_pi` + `{ task: "add hello" }` → body contains `add hello` and note contains `not pause` (or "judged, not gated")
- `grep` + `{ pattern: "x" }` → body is pretty JSON

**Step 2: Implement. PASS the node test.**

**Step 3: `ConfirmCard` uses `confirmSections`. Approve / Reject / Modify stay wired to the existing callbacks and `approval.node_id`. Add **Cancel the rest** → `onAbort` (thread abort from `ChatPane`; the prop already exists as `onAbort`). Show `P(irreversible)` only when `hitl_judge.p_irreversible` is a number.

**Step 4: Rust.** Add the optional field. Every existing `AwaitingApproval { .. }` literal must compile (`hitl_judge: None` where there is no verdict). SSE JSON includes the key only when `Some`. `cargo test -p graphirm-agent workflow::tests::agent_event` and `cargo test -p graphirm-server awaiting_approval`. Fix the exact test names if the filter matches nothing (`cargo test -p graphirm-server awaiting`).

**Step 5: `npm run build`.**

**Step 6: Commit** `feat: confirm card per tool, judge score on the approval event`

- [x] B5 done

---

### Task B6: Rules tab

**Files:**
- Create: `web-app/src/components/RulesTab.tsx`
- Modify: `web-app/src/api/client.ts` — if pinned list / create / patch / delete are missing, add them:
  - `GET /api/knowledge/pinned`
  - `POST /api/knowledge` body `{ summary, entity, entity_type, pinned: true }` (match `create_knowledge` in `crates/server/src/routes.rs`; read the handler before inventing fields)
  - `PATCH /api/knowledge/{id}`
  - `DELETE /api/knowledge/{id}`
- Modify: `web-app/src/components/DecisionShell.tsx` to render `RulesTab` on the Rules tab

**Step 1:** Read `create_knowledge` and the PATCH body. Add client methods that match those structs. No new server route.

**Step 2:** `RulesTab` loads pinned items when the tab is selected. A form creates a pinned note. Each row can edit the summary (PATCH) and delete. Show errors from a non-OK response as text. Empty list copy: "No pinned rules."

**Step 3: `npm run build`.**

**Step 4: Commit** `feat(web): Rules tab for pinned knowledge`

- [x] B6 done

---

### Task B7: Review tab

**Files:**
- Create: `web-app/src/chat/reviewItems.ts`
- Create: `web-app/src/chat/reviewItems.test.ts`
- Create: `web-app/src/components/ReviewTab.tsx`
- Modify: `DecisionShell.tsx`

**Step 1: Failing test**

```ts
export interface ReviewItem { kind: 'approval' | 'failed' | 'paused' | 'pi'; id: string; label: string }
export function buildReviewItems(input: {
  pending: { node_id: string; tool_name: string } | null;
  sessions: { id: string; status?: string; name?: string }[];
  tasks: { id: string; status?: string; executor?: string; title?: string }[];
}): ReviewItem[]
```

- pending → one `approval`
- session status `failed` and `token_cap_exceeded` → `failed`; `paused` → `paused`; `completed` omitted
- task `executor: 'pi'` and status `running` → `pi`; status `completed` omitted

**Step 2: Implement. PASS.**

**Step 3: `ReviewTab` gets sessions, `pendingApproval`, and tasks derived in the shell from `graphData` nodes whose type is Task and `metadata.executor === 'pi'`. Do not call `GET /api/graph/{id}/tasks`. Clicking an approval item switches the tab to Chat (pass `onOpenChat`).

**Step 4: `npm run build`.**

**Step 5: Commit** `feat(web): Review tab from sessions and the loaded graph`

- [ ] B7 done

---

### Task B8a: Reply-judging seat

**Files:**
- Modify: `crates/agent/src/hitl_judge.rs` or a new `crates/agent/src/reply_judge.rs` (prefer a new module so the tool judge stays untouched)
- Modify: `crates/agent/src/workflow.rs` after the assistant message is recorded (`message_end` path)
- Modify: `crates/agent/src/lib.rs` to export the module if needed
- Test: offline transport, same style as `hitl_judge::test_support`

**Behaviour:**
- After the assistant Interaction is stored, if the decisions client is configured, call Jev with the five questions `move`, `claims_completion`, `cites_evidence`, `presents_decision`, `presents_as_options`. Copy the wording from `/home/krs/jev-hooks/jev_hooks/questions.py` (tagset v4). Do not paraphrase.
- `previous_message` is the text of up to the last three turns.
- Store `metadata.reply_judge = { version, scores: { name: number }, latency_ms }` on that Interaction via the existing node update path. Include the node in the turn `graph_update` (the normal post-turn update is enough if it re-reads the node).
- On transport error or timeout: `tracing::warn`, leave metadata unset, return `Ok` from the seat. The turn must not fail.
- Version constant `REPLY_JUDGE_VERSION` (start at `"v4"` to match the question tagset).

**Tests:**
- happy path stores five numeric scores and the version
- hanging transport does not fail the caller and does not write `reply_judge`
- a score is not clamped or thresholded in the writer

**Client (same commit or the next, still B8a):**
- `web-app/src/chat/replyHints.ts` + node test:
  - `claims_completion >= 0.8 && cites_evidence < 0.4` → caveat line (pick these cutoffs as the "high" / "low" pair; document them next to the function)
  - `presents_decision >= 0.6` → decision
  - `presents_as_options < 0.7` → options hint
- `ChatPane` renders the hint strings under the assistant message when `metadata.reply_judge` is present.
- Rules tab: if the latest human message metadata has `pin_candidate` above `0.6`, show "Suggested rule" with a button that calls the existing pin API. If human-side judging is not implemented in this task, skip the suggested-rule row and say so in the commit message. Do not block B8a on `pin_candidate`. Asking `pin_candidate` on the human message is in scope if the same client can be called once more without delaying the assistant response (spawn it, do not await it before the next turn starts; if that is awkward, leave a backlog line and skip).

**Verify:** `cargo test -p graphirm-agent reply_judge` and `npm run build` and the node test.

**Commit** `feat(agent): observe-only reply judge on assistant turns`

- [ ] B8a done

---

### Task B8b: Routing feedback

**Files:**
- Modify: `crates/server/src/routes.rs` — `POST /api/interactions/{id}/routing-feedback`
- Modify: `crates/server/src/types.rs` — request `{ verdict: "wrong" | "keep" }`
- Modify: the routing report aggregator so counts of `wrong` and `keep` are included (read `routing_report` first; add fields rather than changing existing ones)
- Modify: `web-app/src/api/client.ts`
- Modify: `web-app/src/components/JevChip.tsx` — enable the two buttons; on success show the verdict, on failure show the error and leave the buttons enabled

**Step 1: Rust test** in `crates/server/tests/` or the routes unit tests: POST `wrong` then the interaction metadata contains `routing_feedback: "wrong"`. Unknown verdict → 400. Missing node → 404.

**Step 2: Implement the handler.** Write metadata with `spawn_blocking` / the graph update helper already used by other handlers. Do not call the router.

**Step 3: Wire the sheet.** `npm run build`. `cargo test -p graphirm-server routing_feedback`.

**Step 4: Commit** `feat: record routing feedback without changing the router`

- [ ] B8b done

---

### Task B9: Plan card

**Files:**
- Create: `web-app/src/chat/planCard.ts`
- Create: `web-app/src/chat/planCard.test.ts`
- Create: `web-app/src/components/PlanCard.tsx`
- Modify: `ChatPane.tsx` or `DecisionShell.tsx` to show the card above the composer when there are plan tasks

**Step 1: Failing test**

```ts
export interface PlanStep { id: string; title: string; note: string; risky: boolean; enabled: boolean; executor: 'graphirm' | 'pi' }
export function planStepsFromTasks(tasks: { id: string; title?: string; status?: string; metadata?: Record<string, unknown> }[]): PlanStep[]
```

- metadata `executor: "pi"` → `executor: 'pi'`; missing → `'graphirm'`
- metadata `risky: true` → `risky`
- completed tasks are omitted
- `runSteerText(steps)` returns a string that names each enabled title and says to resume those steps. Disabled steps are absent.

**Step 2: Implement. PASS.**

**Step 3: UI.** Checkboxes toggle `enabled` in component state (default true). **Run N steps** calls the existing `resumeSession` and then `sendPrompt(runSteerText(enabledSteps))` only if resume alone does not carry text. Read `resumeSession` in `useSession.ts`. If resume takes no body, send the steer as the next user message via `sendPrompt` and do not add a server field. The button does not call `delegate_pi`.

**Step 4: `npm run build`.**

**Step 5: Commit** `feat(web): plan card resumes with the enabled step titles`

- [ ] B9 done

---

## Checkpoint

After B9:

- `cd web-app && npm run build`
- `node --experimental-strip-types --test web-app/src/layout/layoutMode.test.ts web-app/src/chat/*.test.ts`
- `cargo test --workspace`
- `cargo fmt --all` and `cargo clippy --workspace --all-targets -- -D warnings` for any Rust task

Governance in the B9 commit (or a docs commit immediately after): tick Track B in `docs/backlog.md` only when the acceptance pass has been done by hand. Until then leave Track B open and tick the phase checkboxes in this file as each task lands. `docs/completion-log.md` gets one entry when the track is accepted, not after every task.

## Out of scope

`GET /api/graph/{id}/tasks` fix, TUI config loading, eval baseline, gating Pi's inner tools, new auth, unmounting the canvas, changing the router.
