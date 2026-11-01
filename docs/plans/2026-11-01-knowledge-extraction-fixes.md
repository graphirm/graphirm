# Knowledge extraction: one node per entity, visible missing summaries, correct source link

**Status:** plan, 2026-11-01. No code yet. Backlog item: `docs/backlog.md` → Graph Intelligence →
"Knowledge extraction fixes". Comes from the taxonomy experiment
(`docs/plans/2026-10-31-taxonomy-mcp-server-design.md` §10.2).

Knowledge extraction is measured by `graphirm-eval` (AGENTS.md → Current State). None of the three
fixes lands without the eval comparison in §4.

---

## 1. Evidence

The only graph with Knowledge nodes is `~/.local/share/graphirm/graph.db`. It has 46 nodes. 34 of
them were extracted; the other 12 were written by hand and pinned. All 34 extracted nodes have an
empty `summary`. Their confidences look like GLiNER2 span scores, with values such as 0.7985 and
0.7521.

The repeats were created in the same extraction call, milliseconds apart, from the same message:

| Entity | Nodes | Created | Source message |
|---|---|---|---|
| `add` | 6 | 2026-03-15 21:15:49.67–.73 | 7eb92c09 |
| `semicolon` | 3 | 21:15:49.73–.75 | 7eb92c09 |
| `planning node` | 3 | 2026-03-20 19:18:14.22–.25 | 221e6763 |
| `Pin`, `Unpin`, `List` | 2 each | 2026-03-22 01:37:38.77–.83 | ef95a429 |
| `/api/health` | 2 | 2026-03-20 18:18:42.44–.46 | cf3a68fe |

One assistant message (`ef95a429`, the `graphirm knowledge` CLI summary) produced 9 nodes, every one
of them a bare code name. Only 2 repeats cross messages: `ImportCursor` and `graphirm_graph`.

Two nodes are not found in the message they link to. `grep-then-targeted-read` links to a message
about `/api/health`. `graphirm_graph` links to a message about `graph_query` tests.

## 2. Causes, all in `crates/agent/src/knowledge/`

1. **No merge by name within a call.** `persist_extracted_entities` (`extraction.rs`) creates one
   node per returned entity. GLiNER2 returns one span per mention, so a name mentioned three times
   becomes three nodes. `name_to_id.insert` then keeps only the last copy of each name, so
   `RelatesTo` edges attach to one copy and the others are left unconnected.
2. **Missing summaries are silent.**
   - The local backend sets `description: String::new()` (`local_extraction.rs`). That is by
     design, since GLiNER2 has no generator.
   - The hybrid enrichment prompt asks for "the ExtractionResponse format" but never shows the
     schema.
   - A reply that fails to parse falls back to the local result through `.unwrap_or(local_result)`,
     with no log line. The node is stored with `summary: ""`, and nothing records why.
3. **`derived_from` points at the response only.** `post_turn_extract` extracts from the parent
   user message plus the assistant response, but `persist_extracted_entities` links every entity to
   `response_node_id`. An entity found only in the user message, or in injected context, links to a
   message that never mentions it.

Out of scope here:
- Duplicates across messages (`ImportCursor`, `graphirm_graph`).
- The `entity_types` list in `config/default.toml`. Including `function` and `library` invites
  bare code names.
- `[knowledge] similarity_threshold`. It is set to 0.7 but no crate reads it.

These are listed as open questions in §6.

## 3. Changes

### 3.1 Merge entities by normalised name before persisting

In `persist_extracted_entities`, group `filtered` entities by name, normalised to lowercase letters
and digits (the same key the taxonomy experiment's B1 uses). For each group:
- `confidence` = the highest confidence in the group.
- `description` = the first non-empty description.
- `entity_type` = the type of the highest-confidence entity.
- `relationships` = the union, with targets also resolved through the normalised key.
- `metadata.mentions` = the group size, so the mention count is not lost.

This applies to every backend. Nothing stops the LLM backend from returning a name twice either.

### 3.2 Make a missing summary explicit

- Add `metadata.summary_source` to every node. It is one of `llm` or `none`, plus `enrich_failed`
  for a hybrid fallback.
- Put the `ExtractionResponse` JSON shape into the hybrid enrichment prompt, the same example block
  `build_extraction_prompt` uses.
- Replace `.unwrap_or(local_result)` with a match that logs `tracing::warn!` with the parse error
  and sets `enrich_failed`.
- No change to the local backend's empty description. The point is that it becomes visible and
  countable, not filled in by guessing.

### 3.3 Link each entity to the message that mentions it

Pass the message node ids alongside `messages`: the parent ids from `post_turn_extract` and
`response_node_id`. For each merged entity:
- Link `derived_from` to every message whose content contains the name (case-insensitive).
- If none does, link to the response as today and set `metadata.grounded = false`.

This works the same for all three backends, and needs no span offsets from GLiNER2.

`GET /api/graph/{session}/knowledge` walks interaction → incoming `derived_from`. A node linked to
two messages would therefore be listed twice, so this fix also dedupes that route by node id. The
eval verifiers already dedupe by id, so they score the same before and after.

`extract_knowledge` (the direct, non-hook entry point) keeps its single `source_node_id` and gets
the same `grounded` check against that one message.

## 4. How `graphirm-eval` measures it

Today the knowledge suite has two tasks (`entity-recall`, `multi-entity`). Both use
`KnowledgeNodeCount { min_count }`, which cannot detect a duplicate, an empty summary or a wrong
link. Three verifiers and three tasks are added in `graphirm-eval/src/`. They all use data the
harness already fetches: `get_knowledge` and `get_graph`, which return nodes and edges.

| Verifier | Passes when | Measures fix |
|---|---|---|
| `KnowledgeUniqueNames` | No two Knowledge nodes in the session share a normalised name | 3.1 |
| `KnowledgeSummaryRate { min_fraction }` | At least that fraction of Knowledge nodes have a non-empty `summary` | 3.2 |
| `KnowledgeGrounded { min_fraction }` | At least that fraction of Knowledge nodes have a `derived_from` target whose content contains the entity name | 3.3 |

| New task | Prompt | Verifier |
|---|---|---|
| `repeated-mentions` | "Acknowledge (no tools needed):" followed by the `ef95a429` CLI summary text. It names `Pin`, `Unpin` and `List` two or three times each | `All[KnowledgeNodeCount{3}, KnowledgeUniqueNames, KnowledgeGrounded{1.0}]` |
| `user-only-entity` | A user message that names an entity the assistant is told not to repeat ("acknowledge in one word") | `KnowledgeGrounded{1.0}`. The entity must link to the user message |
| `knowledge-summaries` | The `multi-entity` prompt | `All[KnowledgeNodeCount{1}, KnowledgeSummaryRate{0.9}]` |

`KnowledgeSummaryRate { 0.9 }` is a separate task, `knowledge-summaries`, with the same prompt as
`multi-entity`. `multi-entity` itself stays unchanged, so its history stays comparable. On the
`local` backend this task is expected to fail, because GLiNER2 writes no descriptions. For that
backend the result is recorded but not gated.

**Protocol:**
1. Build a baseline binary from the current `main`, and a candidate binary with the three fixes.
   Each binary gets its own `target/` directory, the same pattern as `target-r/`.
2. Run `cargo run -p graphirm-eval -- --filter knowledge --experiment KX-base` and
   `--experiment KX-fix`, three times each, because extraction is not deterministic. Repeat with
   `--filter memory`, since cross-session memory embeds `entity` and `summary`
   (`memory.rs`, `"[type] entity: summary"`).
3. Run each binary with the LLM backend. If a GLiNER2 model is present, also run with
   `--features local-extraction`.
4. Record per run:
   - Knowledge nodes per session
   - Duplicate-name rate
   - Summary rate
   - Grounded rate
   - Pass/fail per task
   - Extraction cost, from the session token totals

**Gate.** The fixes ship if all of these hold:
- The candidate passes the new tasks in 3 of 3 runs, and the baseline fails `repeated-mentions` at
  least once. That shows the task detects the bug.
- No existing knowledge or memory task goes from pass to fail in any run.
- Extraction cost per turn rises by no more than 10%.

**Offline check (before the eval, no API cost).** A unit test in `extraction.rs` uses
`MockProvider` to feed the exact entity list GLiNER2 returned for `ef95a429`: 9 entities, including
the repeats. It asserts:
- 6 nodes: `Pin`, `Unpin`, `List`, `ImportCursor`, `list_pinned_knowledge()`, `graphirm_graph`.
- `mentions` is 2 on `Pin`, `Unpin` and `List`.
- Every `RelatesTo` edge lands on a surviving node.

## 5. Order

1. Offline unit tests for 3.1 and 3.3. Each lands with its fix.
2. Eval verifiers and tasks in `graphirm-eval`. They land first, and the baseline is run with them
   before any extraction change.
3. Fix 3.1. Then 3.3. Then 3.2.
4. Candidate eval runs. Record the result in `docs/completion-log.md` and the backlog.

The eval code goes in before the extraction changes, so the baseline numbers come from the same
verifiers.

## 6. Open questions

- **Cross-message duplicates.** Should a new entity whose normalised name matches an existing
  Knowledge node in the same session reuse that node and add a `derived_from` edge? This is the
  taxonomy experiment's exact-name step moved into extraction. It changes memory ranking inputs, so
  it is a separate eval run.
- **Entity types.** Should `function` and `library` stay in `entity_types`? 19 of the 34 extracted
  nodes have type `function` or `library`. Removing them is a one-line config change, but it
  needs its own eval run.
- **`similarity_threshold`.** It is in `[knowledge]` but unused. Either wire it, or remove it from
  the config.
- **Hybrid usage.** Which backend produced the March nodes? The logs are gone. After 3.2,
  `summary_source` answers this for new nodes.
