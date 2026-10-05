# Piece-version snapshot subset

`cursor-subset.jsonl` is a hand-checked fixture of 35 replies. It is not a dump of Cursor transcripts. The replies are generic samples: lists, fences, headings, tables, a quote, trailing questions, and seven texts over the 16,000-character cap. Those seven are placeholder paragraphs, long enough that the cap path stores one statement and does not parse them.

`cursor_subset_snapshot_locks_parser_and_baseline_versions` hashes block boundaries and headings, then baseline kinds, over this file. Editing the wording moves the hashes. Update the snapshot when that happens. Bump `PIECE_PARSER_VERSION` or `PIECE_BASELINE_VERSION` only when the rules change. This file is not the labeled measurement set.
