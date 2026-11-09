# Provenance labels

Labels used on saved claim checks:

- `supported` — the cited source says what the claim says.
- `citation_does_not_support` — the claim may be true, but the citation does not show it.
- `rel_type_wrong` — a `data-rel` link names the wrong relation.
- `missing_src` — a statement or result line has no `data-src`.

Examples:

- `001-citation-mismatch.json` — `citation_does_not_support`
- `002-relation-mislabel.json` — `rel_type_wrong` (and the lines also lack `data-src`)
- `003-relation-mislabel.json` — `rel_type_wrong`; `contradicts:#counts` should be `qualifies:#counts`
