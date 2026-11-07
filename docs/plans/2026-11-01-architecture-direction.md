# Architecture direction: graph as record, simple loop as worker

> **For Claude:** This is the working design and the list of changes it needs. It is not approved code work. Each item below gets its own plan doc and worktree when it starts. Do not change context selection, compaction, memory ranking, or knowledge extraction without the eval named on the item.

**Goal:** An agent as good as a plain linear agent, close to its cost, where every turn can be rebuilt and every claim traces to its evidence. Replies are split into parts that later work can link to and retrieve.

**Architecture:** The graph stays the permanent record. The working loop becomes a linear transcript if the eval in item 3 passes. Past work reaches the model through explicit lookups, not by being packed into every request.

**Status:** Proposed 2026-11-01 from the architecture review and the experiments since. Nothing here has been built on this branch except where a branch is named.

---

## 1. The design today

| Layer | What it does | Where |
|---|---|---|
| Record | Every message, tool call, file, task and fact is a typed node in SQLite. 5 node types, 15 edge types. | `crates/graph/` |
| Context | Each turn scores nodes (recency 0.3, edge weight 0.2, BFS distance 0.3, PageRank 0.2) and packs them into a token budget. | `crates/agent/src/context.rs` |
| Reply structure | Native path: JSON segments with `src` and `rel`. Pi path: an HTML page cut into `reply_part` and `reply_line` nodes. | `knowledge/segments.rs`, `pi_delegate/` |
| Lookup | `graph_query` finds nodes. Its summaries omit `Content` text. | `crates/tools/src/graph_query.rs` |
| Safety | Approval gate on destructive tools; provider pairing rules enforced by Graphirm. | `hitl.rs`, `context.rs` |
| Surfaces | CLI, TUI, HTTP server with SSE, React web app, VS Code extension. | `src/`, `crates/tui`, `crates/server`, `web-app/`, `graphirm-vscode/` |

## 2. What the evidence says

| Finding | Evidence | Consequence |
|---|---|---|
| Graph scoring is not shown to beat recency | 21 vs 16 wins at 16k, p = 0.26. Underpowered, so not proof of no effect. | Test a linear loop before deleting anything. |
| Graphirm ties a plain agent at about 2× the cost | T1 + T4, 316 tasks: +2.2pp, CI −2.5 to +7.0. | Cost is the problem to fix, not quality. |
| Carrying earlier tasks in the context does worse than starting fresh | Budget rerun, 55 tasks: fresh 54, persist-base 36, persist-budget 34. Exploratory, not pre-registered. | Fresh is the safe default. |
| That persist-budget result is confounded | All 15 one-call quits come from the budget warning firing at 99% window fill on the first call of a task. | The memory question stays open until the warning is fixed. |
| With the warning off, persist ties fresh at about 6× the cost (2026-11-07) | `G-persist-nonote-budget`: 54/55, 0 one-call quits, 152k input per call against 21k for fresh. | Packing earlier tasks costs without helping. Memory goes through lookup. |
| A plain notes file matches the register on recall | Notes instruction: 14 of 14. At ten sessions and 140 changing facts, notes and register both recall 67 of 68. | Do not adopt the register for recall. |
| The register beat an agent with no notes instruction | 95.5% vs 9.8%, but the rival above removed this gap. | This is not a reason to adopt it. |
| Register may cost less | Half the calls in one run each. Not a result. | Needs its own pre-registration if pursued. |
| Structured claims cost no quality | 96% of citations point to a real node of the right type. | Keep them. They are the audit value. |
| Reply parts are written but the model cannot read them | Context skips `Contains`; `segmented` is unset; `graph_query` shows no text. | Unfinished, not dead. See item 5. |

## 3. Target design

- **Working loop:** an append-only transcript with capped tool outputs, counted with a real tokenizer. Compaction is the only rewrite. Graph scoring moves behind a flag, then goes if item 3 passes.
- **Record:** every node is still written. Each assistant node stores the ids, hash and token count of what was sent, so replay reads it instead of recomputing.
- **Reply structure:** label, then link, then retrieve. Citations and part references become real edges.
- **Memory:** fresh sessions by default. Past work is reached by lookup. Any memory feature must beat an agent told to keep a notes file, not only a plain agent.
- **Claim:** auditable, rebuildable, and close to a linear agent on cost. Not "codes better" and not "remembers better than notes", since neither is measured.

## 4. Changes needed

### Item 1: Land the finished fixes — P1 · S

- [ ] Merge `feat/context-budget-honesty`: o200k counter, 20 kB head and tail output cap, `context_kept_node_ids`, `context_sha256`, `context_tokens_est`.
- [ ] Merge `feat/reply-checker`, including the prose-lead fix. Resends stay off.
- [ ] Merge [PR #5](https://github.com/graphirm/graphirm/pull/5) (extraction fixes) once CI passes.
- [ ] Check that replay matches stored context on every turn. The rerun did not print this check.

**Gate:** tests pass on `main` after each merge. `feat/reply-checker` sits on the budget branch, so the order matters.

### Item 2: Fix the budget warning and settle the memory question — P1 · S

- [x] Make the warning measure session spend, not window fill (PR #9, 2026-11-07). It still measures fill on `main`, `feat/context-budget-honesty` and `plan/context-organization`.
- [x] Decide on counting tool-call arguments and schemas by default: yes on `main`, unchanged on eval binaries in use (backlog item, 2026-11-07). Arguments are counted only behind `GRAPHIRM_COUNT_TOOL_ARGS` on `plan/context-organization`. A 100k window was really 140–155k when sent.
- [x] Rerun `G-persist-budget` against the existing base and fresh results: `G-persist-nonote-budget`, 54/55, $5.91 (estimate was $1.20).

**Gate:** no one-call quits caused by the warning. If persist still trails fresh clearly, stop packing earlier tasks into context.

**Result (2026-11-07):** passed. 0 one-call quits, and persist no longer trails fresh (54/55 each). It costs about 6× for no gain on these tasks, so earlier tasks are not packed by default.

### Item 3: Linear working loop behind a flag — P1 · L

- [ ] Add an append-only loop that still writes every graph node.
- [ ] Run paired T1 and T4 tasks against mini-swe-agent and today's loop, with margins pre-registered.

**Gate:** cost within about 1.2× of mini, pass rate no more than about 3 points worse.

### Item 4: Remove graph scoring, only if item 3 passes — P2 · M

- [ ] Remove per-turn PageRank, distance and edge-weight scoring. Keep `recency_only` as the path.

**Gate:** item 3's result. Do not start this otherwise.

### Item 5: Make reply parts readable, linked and retrievable — P1 · M

The user's stated priority is splitting messages and forming a tree of choices. Today only the labelling step exists.

- [ ] **Read:** show `kind` and `heading` in `compact_node_summary`, and add a way to fetch one node's full text plus its lines.
- [ ] **Find:** look up a part by its number. Numbers repeat across replies, so the lookup also takes the reply.
- [ ] **Link:** add an edge when a later reply cites an earlier part. This changes the test in `docs/plans/2026-10-08-html-piece-index.md` that pins no edge, on purpose.
- [ ] **Tell the model:** describe all of this in the `graph_query` description.
- [ ] **Measure:** does the model use it, and do answers improve when an earlier part is needed after a session break? A notes-file arm is the baseline.

**Gate:** beats the notes arm on something measured, or at least matches it at lower cost. If not, remove the HTML page from `delegate_pi`, since every task pays for it and a failed cut reruns Pi.

#### Format test: JSON against HTML in the native loop

JSON and HTML have never been compared on the same tasks. This test decides the format for the native loop. It has not been run, and it needs its own pre-registration file, committed before any paid run. The 2026-10-30 journal entry left this choice to item 5, and `.cursor/rules/html-part-index.mdc` is not evidence either way.

**Before the test**

- [ ] Land item 1, so the JSON arm has the prose-lead fix.
- [ ] Store parts from both formats in the same node shape, so link and retrieve work on either. Today JSON uses the segment type as `content_type`, and HTML uses `reply_part` with the kind in metadata.
- [ ] Add an off-by-default HTML reply mode to the native loop. `cut_html_index` is only called from `pi_delegate/` today.
- [ ] Write one shared checker for both arms: at least one part, every part has a known kind, and every cited target exists. pi-play's stricter checker is not used, because it would fail HTML on rules JSON is never held to.

**Setup**

| Setting | Value |
|---|---|
| Arms | `J` (JSON segments), `H` (HTML page) |
| Tasks | The first 60 tasks of the T1 headline list in `eval/swebench/instances.lock.json`, skipping the three smoke tasks. Each task runs once per arm. |
| Model | DeepSeek V4.1 Flash, the same as the benchmarks |
| Binary | One build for both arms, with its hash recorded. Only the reply mode differs. |
| Kinds | One kind list for both arms. They differ today (7 segment labels, 11 HTML kinds), which would change the task as well as the format. |
| Prompt | The same wording apart from the format instructions. Record the token count of each. |
| Off in both | `check_retries = 0`, `gliner2_fallback = false`, Knowledge extraction, and every experiment flag |
| Spend | Expected about $1–2 per arm, going by `G-fresh-budget` ($0.95 for 56 tasks). Recheck on the pilot. Stop at $10 billed. |

**Pilot:** 5 tasks per arm, not from the 60, to check the harness and cost. Pilot results don't count, and the criteria can't change after it.

**Measures**

- **Clean reply (primary):** the share of replies that pass the shared checker on the first try, with no retry or fallback. Also report first replies on their own, since that's where JSON failed before.
- **Tasks solved:** the same grader as T1.
- **Cost:** median tokens and median primary cost per task.
- **Citation validity:** the share of citations pointing to a real node of the right type.

**Decision, set in advance.** Switch the native loop to HTML only if all of these hold:

1. `H` has at least 10 points more clean replies than `J`, and the lower bound of the paired 95% bootstrap interval is above 0.
2. `H` solves no more than 2 fewer tasks than `J`, out of 60. This is about the 3-point margin in item 3.
3. `H`'s median tokens and median cost per task are each no more than 1.10× `J`'s.
4. `H`'s citation validity is no more than 2 points below `J`'s.

In every other case, including a tie, the native loop keeps JSON, because it is the only format already tested at scale on coding tasks. Remove the HTML mode from the native loop if `H` fails.

**Reruns and stops:** harness errors are rerun once, on both arms for that task, as in `eval/swebench/prereg.md` §5. No tasks are added afterwards. If more than 10% of tasks still fail from harness errors, report the run as invalid and don't apply the decision.

**What this test can't show**

- **Tasks solved:** 60 tasks can't confirm a 3-point difference, so criterion 2 only guards against a clear loss.
- **Pi:** this doesn't settle Pi's format, because Pi can behave differently and only forced-schema JSON has been tried there. HTML stays on Pi until the gate above decides it.
- **Readability and later retrieval:** neither is measured here. Part retrieval is measured separately by the notes-arm test above.

### Item 6: Citations as real edges — P2 · M

- [ ] Turn `src` and `rel` into edges, checked when written. `segments.rs` creates only `Contains` today.
- [ ] Decide between a new edge type and existing ones. A new type changes the count in `AGENTS.md`.

**Gate:** validity stays at or above 96%, and any claim reaches its evidence in one query.

### Item 7: Check the remaining mechanisms — P2 · M

- [ ] Run the writer and reader audit over model routing, the approval judge, escalation, impact and knowledge extraction.
- [ ] Each one gets a reader and a measured effect, or goes off by default.

**Gate:** one ablation per mechanism on T1.

## 5. Not doing now

- **Register for recall:** stop testing it, per the 2026-11-01 journal entry on `plan/context-organization`. Flags stay off.
- **Taxonomy service:** parked. The noise came from extraction.
- **Automatic resends:** off until an empty resend cannot replace a reply.
- **Scoring weight tuning:** pointless until item 3 answers whether scoring stays in the loop.

## 6. Decision for the user

**Regulated use:** this would need tamper evidence, named users, human-confirmed reasons, retention rules and a W3C PROV export. None of it is measured or planned. It is a product decision, not a next step.

## 7. Risk

- The numbers come from one model (DeepSeek V4.1 Flash) and mostly exploratory runs.
- A rebuilt binary with the same settings swung one result 43 vs 21 of 55, so compare arms by the stored context hash.
- Item 3 is the largest change and may fail. Items 1, 2, 5 and 6 are worth doing either way.

**Refs:** architecture review session (2026-10-28); `docs/journal.md` on `plan/context-organization`; `/home/krs/graphirm-runs/budget`; `docs/plans/2026-10-28-context-budget-honesty.md`.
