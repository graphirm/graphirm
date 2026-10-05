# Reply pieces

> **For Claude:** Milestone 1 is implemented on `feat/reply-pieces`. Do not restart it. The next milestone is the hand-labeled set.

**Goal:** Split one finished Pi text segment into pieces by shape, without using the question.

**Architecture:** The parser owns the text. A labeler only names the shape. `structure_segment` in `crates/agent/src/pi_delegate/pieces.rs` counts characters first, then `pulldown-cmark` marks blocks. Offsets are UTF-8 bytes. If the blocks do not tile the segment, the whole segment is one statement.

**Tech stack:** Rust, `pulldown-cmark` 0.13.

**Key decisions:**

- Narration (`stopReason: toolUse`) is parsed and not judged. Every block is a statement. A fence is code.
- A final reply uses the baseline: trailing `?` is a question, imperative list items are steps, a paragraph opening with "This will", "Note", or "Be careful" is a caveat, a fence is code. Known misses stay: "This will print `hi`" is a caveat, and "Use X / Use Y" is steps.
- A heading lead-in is one line of at most 80 characters ending in `:`, including a bold line like `**Note:**`, directly before a list or a fence. A markdown heading attaches to the next block.
- A trailing question run is split by scanning back from the final `?` to punctuation followed by a space or a newline. Abbreviations are `e.g.`, `i.e.`, `etc.`, and `vs.`. `?` inside inline code or a URL does not count.
- Over 16,000 Unicode scalar values, the segment is one statement and the parser is not called.
- `order` numbers pieces. Item `position` numbers lines inside a piece.
- Cursor transcripts are read from `~/.cursor` by the tiling test and are not committed.
- Every production Pi run appends its stdout to `~/.graphirm/pi-runs`. `GRAPHIRM_PI_RUNS_DIR=off` disables it.

## Milestone 1 — done

Parser, splitter, baseline labeler, coverage check, Cursor tiling test, Pi JSONL recording.

## Label rows and scorer — done

A row in `piece-labels.jsonl` stores the run file, the segment index, a SHA-256 of the segment text, `parser_version`, `baseline_version`, and each piece's byte range. It does not store piece order. `graphirm label-pieces` skips a segment already in that file, so `quit` can resume. `graphirm score-pieces` matches on the byte range, reports single-piece replies apart from replies with several pieces, and prints a confusion matrix.

## Still open

1. Hand-label about 100 real Pi replies from `~/.graphirm/pi-runs`, using `docs/guides/reply-piece-labels.md` and `graphirm label-pieces` with no `--show-baseline`. Use varied tasks so lists, fences, and questions appear. Then run `graphirm score-pieces`.
2. A grammar-constrained llama.cpp labeler that only returns one kind per block and has to beat the baseline. The OpenRouter client cannot force a schema.
3. Cross-turn edges. Same-turn adjacency is not `applies_to`. Positional candidates start only after the labeled set holds up.
