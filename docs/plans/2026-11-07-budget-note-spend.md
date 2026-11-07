# Budget warning measures session spend

**Backlog:** "Budget warning measures session spend — P1 · S" in `docs/backlog.md`.

## Problem

The `[Budget]` note in `crates/agent/src/workflow.rs` divides the packed window's
tokens by `max_tokens`. That measures how full the window is, not how much the
session has spent. A session that starts with a full window (earlier tasks carried
in the graph) is told to wrap up on its first call. In `G-persist-budget` this caused
all 15 one-call quits. The rerun with the note off (`G-persist-nonote-budget`,
2026-11-07) matched fresh on the first 32 tasks.

## Change

- `budget_note_text(thresholds, spent, cap)`: a pure function. The ratio is the
  session's spent tokens (`Session::llm_tokens_used`, the counter the cap check
  already keeps) divided by `max_session_tokens`. No cap, a zero cap, or no
  thresholds gives no note.
- The note text and the 0.7 / 0.9 thresholds stay the same.
- The note stays on the system message, as on `main` today. Moving it to the last
  message (the eval branch does this to keep the cached prefix stable) is a
  separate change.
- Doc comments on `budget_warning_thresholds` and the `config/default.toml`
  comment say "fraction of `max_session_tokens`".

## Behaviour change

With no `max_session_tokens` set (the committed default), no note is shown at all.
Before, the note fired whenever the window passed 70% of `max_tokens`.

## Tests

- [x] A first call with a full window and no spend gets no note (no window input exists any more).
- [x] No cap or a zero cap gives no note at any spend.
- [x] Spend at 75% and 95% of the cap gives the "wrapping up" and "prioritize" texts.
- [x] An empty threshold list gives no note.
