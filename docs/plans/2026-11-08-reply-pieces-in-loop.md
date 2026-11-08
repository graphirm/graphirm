# Reply pieces in Graphirm's own loop (HTML index)

**Backlog:** "Keep the reply format when earlier replies are packed — P1 · M" and "Parts for text written
next to tool calls — P2 · S" in `docs/backlog.md`. Found by check 0.4
(`docs/plans/2026-11-07-structure-completeness.md`). Decided 2026-11-08: option B (split whatever the
model writes) with the HTML index kinds, replacing JSON segments.

## What exists on `main`

Built for Pi replies in `crates/agent/src/pi_delegate/`:

- `pieces::cut_html_index` cuts an HTML index reply (shape version 3, the kinds in
  `.cursor/rules/html-part-index.mdc`) into `Piece`s with kind, heading, `data-src` and `data-rel`.
- `graph::attach_reply_pieces` writes them as `reply_part` / `reply_line` Content nodes under the reply
  with `Contains` edges.
- `pieces::structure_segment(text, narration)` splits plain markdown into pieces with the deterministic
  baseline labeler; `narration = true` is text next to tool calls.
- `tool::HTML_PIECE_INDEX_CONTRACT` is the prompt text; a test keeps it in step with the rule file.

## Design

1. `[agent.reply_pieces] enabled` (default `false` until the eval below). When on, the system prompt
   gets the HTML index contract instead of the JSON segment prompt; `[agent.segments]` must be off
   (config check at load).
2. After every assistant reply with text, final or next to tool calls:
   - `cut_html_index` succeeds → `reply_part` / `reply_line` nodes; `pieces_source = "html"`.
   - otherwise `structure_segment(text, has_tool_calls)` → the same node shapes, kinds from the
     baseline; `pieces_source = "markdown"`.
   - over the character cap → one `statement` piece; `pieces_source = "over_cap"`.
   Every reply node gets `pieces_source`, so check 0.4 can tell why a reply has the pieces it has.
3. The reply node keeps the model's own text. No display-text rewrite on this path, so earlier replies
   go back into the window in the format the model was asked for.
4. Move `attach_reply_pieces`, `part_node`, `line_node` from `pi_delegate/graph.rs` to a shared
   `knowledge/reply_pieces.rs`; Pi and the loop call the same code.

## Measurement (before turning it on by default)

- Check 0.4 C1 on a fresh and a persist arm: every reply with text has pieces or a `pieces_source`.
- Paired run on the 55 budget tasks, `G-fresh-budget` against the same arm with `reply_pieces` on:
  tasks solved, output tokens per reply, cost. Estimate about $1–2 per fresh arm (fresh was $0.95);
  the persist arm about $6.
- Share of replies cut as `html` against `markdown`, overall and by task position in persist runs.
- Step 0.1 needs a hand-labeled set of 100 replies; labels by a model are not hand labels.
- Context selection changes with the switch on: `GraphStore::pagerank` is unweighted over the whole
  graph, so each reply's K `Contains` children dilute the rank it passes back along `RespondsTo`, and
  every reply adds K + lines nodes (more PageRank work per turn). The paired eval measures the switch
  as built; leaving `reply_part` / `reply_line` out of PageRank is a separate change, measured on its own.
- The HTML contract was written for Pi's one-shot task: with the switch on, every turn (greetings,
  narration) is asked for HTML, earlier HTML replies are packed verbatim, and the TUI and VS Code show raw
  HTML. Output and input tokens per reply are part of the eval.

## Tasks

- [x] Shared `knowledge/reply_pieces.rs`; Pi path uses it (no behaviour change, Pi tests pass)
- [x] Config `[agent.reply_pieces]` (off in `config/default.toml`), prompt swap; enabling it drops
      `[agent.segments]` at load, and the loop also skips segments when a session asks for them
- [x] Cut every reply with text in `workflow.rs`; `pieces_source` and `pieces_count` on the reply node;
      a markdown paragraph's text becomes its one `reply_line`; outline extraction skips cut replies
- [x] `cargo test --workspace` (1280 pass), clippy clean
- [x] Smoke run: `eval/structure/smoke_reply_pieces.py` (three prompts on a tiny repo, then check 0.4),
      deepseek-v4.1-flash, 2026-11-08. See "Smoke result" below.
- [ ] Paired eval on the 55 budget tasks (spend reported first)

A reply that hits `max_session_tokens` is cut before the loop stops. If writing pieces fails partway,
the reply records `pieces_source = "failed"`.

## Smoke result (2026-11-08)

Two runs of `eval/structure/smoke_reply_pieces.py`, one session, three prompts each. Not an eval: one
tiny repo, one run per prompt version.

| Run | Prompt | Replies with text | Cut as HTML | Check 0.4 | Tokens in / out |
|---|---|---|---|---|---|
| 1 | Pi contract only | 4 | 0 (all `markdown`) | passes | 1,006,775 / 4,043 (23 calls) |
| 2 | Pi contract + example page | 3 | 3 | passes | 57,662 / 2,695 (9 calls) |

- Run 1: the model wrote HTML index pages, but with no kind class on the sections and lines numbered
  `N.0.0`, so `cut_html_index` refused every page and the markdown fallback labelled them `statement`
  and `code` only. Adding a three-section example page (`reply_pieces::PAGE_EXAMPLE`) to the loop's
  prompt fixed it; Pi's prompt is unchanged.
- Run 2: all three final replies cut as HTML with the model's own kinds (`result`, `statement`).
- Both runs: every destructive call recorded `approval = auto_approve`; the session ended `completed`
  after SIGTERM (nothing was mid-run). The model wrote no text next to its tool calls, so the
  narration path is covered only by unit tests.
- C3 had no `write` / `edit` calls in run 2: the model changed files through `bash`.
