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

## Tasks

- [ ] Shared `knowledge/reply_pieces.rs`; Pi path uses it (no behaviour change, Pi tests pass)
- [ ] Config `[agent.reply_pieces]`, prompt swap, mutual exclusion with `[agent.segments]`
- [ ] Cut every reply with text in `workflow.rs`; `pieces_source` on the reply node; tests
- [ ] Rerun check 0.4 on a small local run; then the paired eval (spend reported first)
