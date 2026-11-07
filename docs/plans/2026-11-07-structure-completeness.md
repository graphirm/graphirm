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

- [ ] Script `eval/structure/completeness.py` (stdlib only: `sqlite3`, `json`), one JSON report per
      database and a pooled summary per corpus
- [ ] Tests on a small fixture built with the real schema (`eval/structure/test_completeness.py`)
- [ ] Run on the corpus, record results in this plan and the parent plan
