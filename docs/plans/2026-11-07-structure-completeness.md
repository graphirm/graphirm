# Structure completeness check (step 0.4)

**Parent plan:** `docs/plans/2026-11-07-prove-the-structure.md`, check 0.4.
**Backlog:** "Prove the structure — P1 · L".

## Goal

A script that reads finished session databases and reports what is missing from the record. It does
not compare against a plain agent. It only checks Graphirm's own output against what Graphirm says it
records.

## Checks and pass thresholds

Thresholds are fixed here before the first measurement and are not changed after seeing results.

| Check | What counts | Pass |
|---|---|---|
| C1 Reply parts | Assistant replies with non-empty text: each has `Contains` children, or carries a recorded reason (`tool_call_leak`) | 100% |
| C2 Tool results | Each tool call in an assistant node's `tool_calls` has a `tool` Interaction with the same `tool_call_id` | 100% |
| C3 File changes | Each `write` / `edit` call whose result is not an error has a `Modifies` edge from its assistant node to a `file` Content node whose path ends with the call's `path` | 100% |
| C4 Approvals | Each `write` / `edit` / `bash` call has an `ApprovedBy` edge on its result, or a `tool_rejection` with `RejectedBy` | 100% |
| C5 Integrity | Edges whose ends are missing: 0. Interactions with a `Produces` edge from their session's Agent: 100% | as stated |

Reported without a threshold, because the record has no field for them yet:

- `bash` calls: file changes made through `bash` have no `Modifies` edge by design, so C3 cannot see them.
- How an approval was given: whether an `ApprovedBy` edge was a human or auto-approve. Nothing stores it.

A corpus passes 0.4 when every check meets its threshold.

## Corpus

The latest run of each track that has session databases under `/home/krs/graphirm-runs/`: T4 (`G`),
budget (four arms), replycheck, decide (scale pilot). These come from eval-branch binaries, not `main`,
so the result describes those binaries. A run on a `main` binary follows once the step 0 builds land.

## Tasks

- [x] Script `eval/structure/completeness.py` (stdlib only: `sqlite3`, `json`), one pooled JSON report
      per corpus with example misses
- [x] Tests on a small fixture built with the real schema (`eval/structure/test_completeness.py`, 6 pass)
- [x] Run on the corpus, record results in this plan and the parent plan

## Result (2026-11-07): fails

No corpus passes. Persist arms are counted on the largest database per repo, because each persist
database carries every earlier task and pooling all 56 would count early tasks many times.

| Corpus | DBs | C1 parts | C2 results | C3 file changes | C4 approvals | C5 produced | `bash` calls |
|---|---|---|---|---|---|---|---|
| T4 `G` | 106 | 76/286 | 7115/7119 | 333/333 | 0/6771 | 13327/13328 | 6438 |
| budget `G-fresh-budget` | 56 | 53/72 | 1968/1968 | 54/54 | 0/1833 | 3821/3821 | 1778 |
| budget `G-persist-base` | 2 | 12/58 | 668/668 | 24/24 | 0/636 | 1439/1439 | 611 |
| budget `G-persist-budget` | 2 | 13/59 | 698/698 | 8/8 | 0/689 | 1497/1497 | 681 |
| budget `G-persist-nonote-budget` | 2 | 15/56 | 1868/1868 | 36/36 | 0/1864 | 3838/3838 | 1827 |
| replycheck | 20 | 18/31 | 661/661 | 25/25 | 0/623 | 1293/1293 | 597 |
| decide scale pilot | 2 | 107/282 | 1002/1002 | 35/35 | 0/857 | 2298/2298 | 822 |

No dangling edges anywhere. No `ApprovedBy` edge records whether a human or auto-approve gave it.

What the misses are:

- **C2, C5 (T4, 5 misses):** sessions killed at the time limit in the middle of a call. The Agent node
  still says `active`, so the record does not say why the session ended.
- **C3:** every `write` / `edit` is recorded. But 82–98% of calls are `bash`, and file changes made
  through `bash` have no `Modifies` edge, so most file changes are outside the record.
- **C4:** these are headless runs with no approval gate (`session.hitl` is `None`), so destructive calls
  run with no approval record and nothing says that no gate was attached.
- **C1, final replies:** fresh keeps parts on 53/56, T4 73/82, replycheck 18/23. Persist keeps them on
  15/54: parts stop after the first two tasks in each repo. The packed context shows earlier replies as
  plain display text (`segment_display_text`), while the system prompt asks for JSON segments; from the
  third task on, the model answers in the plain style of its own earlier replies, the JSON parse fails,
  and no parts are written. The raw reply text is not stored, so the failed parse is inferred from the
  missing parts plus the plain-text earlier replies in `G-persist-nonote-budget.pack-trace.jsonl`. In
  the decide pilot the `An` arm keeps parts on 17/108 final replies, `Tc` on 90/106.
- **C1, replies with tool calls:** text written next to tool calls never gets parts and no reason is
  recorded. 3 of the 303 interim replies across all corpora have parts (all in T4).

Fixes are backlog items (see "Prove the structure"). The check is rerun after them, on a `main` binary.

Reports: `/tmp/claude-1000/-home-krs-graphirm-repo/1348251e-d93d-4da5-acec-04edf0cf337c/scratchpad/completeness/*.json`
(regenerate with the script; the run databases stay under `/home/krs/graphirm-runs/`).
