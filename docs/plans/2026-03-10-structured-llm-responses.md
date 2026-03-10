# Structured LLM Response Discovery Plan

> **For Claude:** When implementing, use superpowers:executing-plans (worktree, batch tasks, verification). Phase 1 task breakdown below; Phases 2–6 can be expanded similarly.

**Goal:** Discover what structure exists inside LLM responses, validate it empirically using GLiNER2, and define a schema that Graphirm can request from models and persist in the graph.

**Status:** Phase 1 implemented (corpus export); Phases 2–6 not started.

---

## The Problem

An LLM response is a blob of text. Inside that blob, the model is doing different things — observing facts, reasoning about them, generating code, making decisions, stating a plan, giving a final answer. But nothing in the response tells us *which part is which*.

Today's APIs give us two levels of structure:

- **Provider-level**: Anthropic gives `thinking` vs `text` blocks. DeepSeek gives `reasoning_content` vs `content`. OpenAI gives nothing (flat text or JSON-mode).
- **Tool calls**: Structured separately from text (name, arguments, result).

That's it. Everything else — observation vs reasoning vs code vs plan vs answer — is mixed into a single string. We can't traverse it, weight it, filter it, or link specific segments to graph entities.

## What We Want

A **segment schema** — a set of named types (e.g. `observation`, `reasoning`, `code`, `plan`, `decision`, `answer`) — such that:

1. Each segment type captures a genuinely distinct kind of content.
2. The boundaries between segments are consistent and detectable.
3. The schema is small enough that models can reliably produce it.
4. The schema is useful — it enables better context building, graph traversal, and audit.

## Why We Don't Guess the Schema

We could sit down and define `["observation", "reasoning", "code", "plan", "answer"]` and call it done. But:

- We don't know if those labels match what models actually produce.
- We don't know if `observation` vs `reasoning` is a meaningful boundary in practice, or if they blur.
- We don't know if there are segment types we haven't thought of (e.g. `uncertainty`, `reference`, `self_correction`).
- We don't know what granularity works — sentence-level? paragraph-level? block-level?

So: **the schema is discovered from data, not assumed.**

---

## Pipeline Overview

### Phase 1: Collect Conversation Corpus

Gather real LLM responses from Graphirm usage (or generate them by running the agent on eval tasks).

**Sources:**
- Existing graph DB sessions (Interaction nodes with assistant role)
- graphirm-eval task runs (once the eval harness exists)
- Manual conversations through the TUI

**Output:** A corpus of `(session_id, turn, role, text)` tuples, focusing on assistant turns.

**Scale target:** 200–500 assistant turns across varied tasks (coding, debugging, explanation, multi-step, tool-heavy).

#### Task breakdown — Phase 1 (Corpus export)

| Task | Summary | Verification | Done |
|------|---------|---------------|------|
| 1.1 | Add `GraphStore::get_session_interactions(session_id)` returning Interaction nodes for that session, ordered by `created_at` | Unit test: add 2 sessions with interactions, call for each session, assert order and content | ✅ |
| 1.2 | Define corpus record type (e.g. `CorpusTurn`: session_id, turn_index, role, text) and JSONL serialisation | Unit test: round-trip serialise/deserialise | ✅ |
| 1.3 | Add corpus export: open graph at path, list session IDs (from Agent nodes), for each session call `get_session_interactions`, filter role=assistant, write JSONL to file or stdout | Integration test: in-memory graph with 1 session and 2 assistant turns, export to temp file, assert 2 lines and content | ✅ |
| 1.4 | Add CLI: `graphirm export-corpus --db <path> [--out <file>]` (default stdout), document in README or docs | Manual: run against a real graph DB, confirm JSONL lines | ✅ |

**Files (Phase 1):**
- Modify: `crates/graph/src/store.rs` (new method)
- Create or modify: `crates/graph/src/export.rs` or new `crates/graph/src/corpus.rs` (corpus types + export)
- Modify: `src/main.rs` or CLI crate (export-corpus subcommand)
- Test: `crates/graph/src/store.rs` (tests), `crates/graph/src/export.rs` or corpus tests

### Phase 2: Candidate Label Exploration

Run GLiNER2 over the corpus with a broad set of candidate labels to see what it finds.

**Candidate label sets to try (iteratively):**

Round 1 — coarse:
```
["observation", "reasoning", "code", "instruction", "answer"]
```

Round 2 — finer:
```
["observation", "analysis", "hypothesis", "plan", "code", "command",
 "decision", "caveat", "reference", "answer", "question"]
```

Round 3 — domain-specific:
```
["file_reference", "error_diagnosis", "fix_proposal", "code_generation",
 "tool_selection", "explanation", "summary"]
```

**For each run, capture:**
- Span count per label
- Average confidence per label
- Span length distribution per label
- Overlap rate between labels (do two labels frequently cover the same text?)
- Coverage (what % of the text is captured by any label?)

### Phase 3: Analyse and Converge

From the data collected in Phase 2, determine:

1. **Which labels are real** — consistently detected with high confidence across many turns.
2. **Which labels are redundant** — high overlap with another label (merge them).
3. **Which labels are noise** — low confidence, inconsistent boundaries (drop them).
4. **What granularity works** — are spans sentence-level? multi-sentence? Does GLiNER2 find paragraph-level blocks?
5. **What's missing** — are there recurring text patterns that no label captures?

**Decision criteria:**
- A label is "real" if: average confidence > 0.6, appears in > 30% of turns, and has < 40% overlap with any other label.
- A label is "redundant" if: > 60% of its spans overlap with another label's spans.
- Granularity is "right" if: spans align with natural boundaries (code fences, paragraph breaks, sentence ends) > 70% of the time.

**Output:** A final schema — a list of 4–8 segment types with definitions and boundary rules.

### Phase 4: Validate with Human Annotation

Take a sample (50–100 turns) and manually annotate segments using the discovered schema. Compare:

- GLiNER2's labels vs human labels (agreement rate)
- Inter-annotator agreement if possible (is the schema unambiguous?)

**Pass criterion:** > 75% agreement between GLiNER2 and human annotation on segment type and approximate boundary.

### Phase 5: Define the Structured Output Schema

With a validated schema in hand, define:

1. **The JSON format** that we ask LLMs to produce:
   ```json
   { "segments": [
     { "type": "<segment_type>", "content": "..." },
     ...
   ]}
   ```

2. **The prompt template** that instructs the model to use this format.

3. **The graph representation** — how segments map to Graphirm nodes/edges:
   - Option A: One `Interaction` node with a `segments` metadata field (array of typed spans).
   - Option B: Child `Content` nodes per segment, linked to the parent `Interaction` via `Contains` edges with `segment_type` on the edge or node.

4. **Fallback** — when the model doesn't produce valid structure (malformed JSON, missing segments), fall back to GLiNER2 post-processing on the raw text.

### Phase 6: Integration

Wire the schema into Graphirm's agent loop:

- **Request path**: System prompt includes segment format instructions → model responds with structured segments → parse and persist.
- **Fallback path**: Model responds with plain text → run GLiNER2 with the discovered labels → persist segments.
- **Context engine**: When building context, filter or weight by segment type (e.g. prioritise `code` and `decision`, downweight `reasoning`).

---

## How to Test if It's Working

### Quantitative

| Metric | Target | How to measure |
|--------|--------|----------------|
| Label detection rate | > 80% of turns have ≥ 2 segment types | Count turns with multi-type coverage |
| Confidence | Average > 0.65 across all labels | Mean of GLiNER2 span confidence |
| Coverage | > 85% of response text assigned to a segment | (chars in segments) / (total chars) |
| Human agreement | > 75% match on type + boundary | Manual annotation comparison |
| Schema compliance | > 90% of structured-output responses parse correctly | JSON validation on model output |
| Round-trip fidelity | Concatenating segments reproduces original text | Exact string match minus whitespace |

### Qualitative

- Can we show a "reasoning trace" for a turn by filtering to `reasoning` + `decision` segments? Does it make sense?
- Can we find "all code the agent generated in session X" by querying `code` segments? Is it complete?
- Does the context engine produce better results when it can filter by segment type vs using full text?

---

## Dependencies

| Dependency | Status | Notes |
|-----------|--------|-------|
| GLiNER2 ONNX pipeline | ✅ Implemented | `OnnxExtractor` in `crates/agent/src/knowledge/local_extraction.rs` |
| Conversation corpus | ❌ Needed | Requires either real sessions or eval harness runs |
| graphirm-eval | 🔧 In progress | Plan exists at `2026-03-09-graphirm-eval.md` |
| Structured output support in LLM crate | ❌ Needed | Anthropic and OpenAI support it; need to wire `response_format` / `output_config` in providers |

## Risks

- **GLiNER2 may not generalise to discourse** — it's trained for NER (named entities), not discourse segmentation. If zero-shot performance on labels like "reasoning" is poor, we may need a different model or fine-tuning.
- **Schema too fine-grained** — too many segment types leads to noisy, unreliable labelling. Start coarse, refine.
- **Structured output hurts quality** — forcing the model to emit JSON segments may degrade the quality of its reasoning or code. Need to compare structured vs unstructured outputs on the same tasks.
- **Latency** — running GLiNER2 as a post-processing step adds ~150–200ms per turn. Acceptable for async/background processing, but not for streaming display.

## Open Questions

1. Should segments be **nested** (e.g. a `plan` segment contains `code` sub-segments) or strictly **flat** (sequential, non-overlapping)?
2. Should we store **both** the raw text and the segmented version, or only one?
3. Is there value in segment-level **embeddings** (embed each segment separately for HNSW search)?
4. How does this interact with **thinking blocks** from Anthropic/DeepSeek — is `thinking` just another segment type, or a separate axis?
