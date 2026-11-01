# Taxonomy MCP server — design

> **For Claude:** Design only. Nothing here is implemented. Step 0 is a read-only measurement on
> saved graphs and must report before any code under `crates/` or `src/` is written. Use
> `superpowers:writing-plans` to turn the chosen phase into an implementation plan.

**Status:** design, 2026-10-31. Step 0 measured, the step 1 prototype built, and steps 3 and 4 run the same day in pi-play (§10). Noise and dedup pass the gate, categories do not (§10.2). Backlog item: `docs/backlog.md` → Graph Intelligence →
"Taxonomy MCP server".

## 1. Problem

On a new project nobody builds a taxonomy; it accretes. Every label is chosen at the moment
something needs one — a folder, a tag, a Knowledge `entity_type`, a Knowledge node name — by
whoever is writing, with no view of the earlier choices. The result is the usual set of defects:

- **Synonym drift.** `auth`, `authentication`, `login-flow` as three nodes.
- **Granularity mismatch.** `backend` beside `retry-logic-for-s3-uploads` as siblings.
- **Lock-in.** Week-1 labels shape everything after, and renaming breaks references, so nobody
  consolidates.

In Graphirm the label set for extraction is a fixed list in config
(`[agent.extraction].entity_types`, six types), and Knowledge node names are whatever the
extractor emits per session. `search_knowledge` can find similar nodes, but nothing reconciles
them. `schema-suggest` (`crates/agent/src/knowledge/schema_suggest.rs`) applies after-the-fact
real / redundant / noise verdicts to segment labels from a report — the same need, offline and for
one label family.

**Measured premise (2026-10-31, §10).** The only Graphirm graph with Knowledge nodes is
`~/.local/share/graphirm/graph.db` (46 nodes). `~/.graphirm/graph.db` is empty, and the 211 saved T1
graphs have none, because extraction was off in those runs. Of the 46, 17 are noise (`add` ×6,
`semicolon` ×3, `List`, `cargo`, …) and 8 are duplicates. 7 of those duplicates are exact name
repeats. Only one is a synonym (`grep-then-targeted-read` = `read-tool-efficiency`). Type labels
overlap: 5 of the 9 types in use are not in the configured `entity_types`. Synonym drift, the case
this design leads with, is nearly absent at this size. Noise and inconsistent typing are the real
defects.

## 2. Goal

One service that is **the single authority every agent and tool asks before labelling**. The
taxonomy is grown and maintained online as content arrives, not built once. Batch induction over
a corpus (embed → cluster → name → arrange) is one operation of the service, not the whole of it.

Non-goals: a general ontology editor; OWL/SKOS reasoning; replacing `entity_types` in the
extraction prompt before Step 0 and an eval say it helps.

## 3. Lifecycle

### 3.1 Cold start

`taxonomy_init(description, seed_docs?)` — a strong model drafts 5–10 top-level categories from
the project description and any seed documents. Every node is created `provisional`. When a corpus
already exists, `taxonomy_induce(corpus)` runs the batch pipeline instead (§5) and its output is
also provisional.

### 3.2 Steady state

- `classify(text)` → `{node_id, confidence}` or `no_fit`. Callers never invent labels.
- `no_fit` items go to a **holding pool**, not to a new category. This one rule is what prevents
  proliferation.
- A category is proposed only when a cluster in the pool reaches minimum support (start: ≥5 items
  within an embedding radius, confirmed by an LLM check that it is not an existing node).

### 3.3 Consolidation

`consolidate()` — every N classifications or on a schedule:

- merge siblings whose definitions and members overlap (embedding distance, then LLM verdict);
- split categories above a size share;
- promote provisional nodes with sustained use; retire unused ones;
- re-check that siblings sit at comparable granularity.

Every change is a versioned operation (`merge(a, b → c)`, `split(a → b, c)`, `rename`, `move`,
`retire`) with an old-ID → new-ID map, so existing references keep resolving. Safe cleanup is the
part manual taxonomies lack.

### 3.4 Health signals

`taxonomy_health()` reports: holding-pool rate (high = missing categories), label agreement on
repeated classification (low = overlapping or vague categories), per-node usage, depth and branch
balance. These are also the acceptance metrics for the batch pipeline.

## 4. MCP surface

| Tool | Kind | Notes |
|---|---|---|
| `taxonomy_init` | write | Seed from description / docs. |
| `taxonomy_induce` | write, long-running | Batch pipeline over a corpus path. Returns a job id. |
| `classify` | read (+ pool write on `no_fit`) | Hot path; must be cheap. |
| `classify_batch` | read | Same, many items. |
| `taxonomy_get` | read | Tree, or a subtree; includes definitions and include/exclude notes. |
| `resolve` | read | Old node id → current node id through the op log. |
| `propose` / `consolidate` | write | Return a diff; applied only after approval unless `auto_apply`. |
| `apply_ops` | write | Apply a reviewed diff. |
| `taxonomy_health` | read | §3.4. |

Writes that restructure (`merge`, `split`, `retire`) are approval-gated by default, matching how
Graphirm treats destructive tools under HITL.

## 5. Batch induction

Used by `taxonomy_induce` and by the first `consolidate` on a large pool.

1. Chunk (500–2,000 tokens on structural boundaries); keep id + source.
2. Cheap model per chunk → JSON gist + 3–5 key concepts.
3. Embed the gists; hierarchical clustering (HDBSCAN or agglomerative) to 30–200 leaves.
4. Mid-tier model names each leaf from ~20 samples: name, one-line definition, includes, excludes.
5. Strong model arranges the leaves: dedupe, 5–15 top-level groups, consistent depth, flag orphans.
6. Validate on 200–500 held-out chunks with a `no_fit` option; feed §3.4 metrics back to step 5;
   2–3 rounds.

Models through OpenRouter (cheap tier for 2 and 6, strong tier for 5), matching the existing
`[agent.routing]` cheap/smart split. Every LLM response is cached to disk by content hash so a
rerun pays only for changed steps. Approach follows TnT-LLM (Microsoft, 2024) and Clio (Anthropic,
2024).

## 6. Storage and Graphirm fit

Taxonomy nodes are `Knowledge` nodes with `metadata.taxonomy = true`, a `status`
(`provisional` / `active` / `retired`), definition, includes, excludes. Hierarchy is `RelatesTo`
with `metadata.role = "parent"`. Members link to their category with `RelatesTo`
(`role = "member"`); the evidence that justified a category is `DerivedFrom`. The op log is a
table (or `Knowledge` nodes of kind `taxonomy_op`) so `resolve` and audit both read from it. No new
node or edge types.

## 7. Options considered

| Option | For | Against |
|---|---|---|
| **A. Standalone MCP server, own SQLite** (Python, FastMCP — as Nodestradamus MCP) | Fast to prototype; usable from Claude Code, Cursor, Pi, anything; no freeze conflict. | Second store; Graphirm Knowledge nodes not reconciled unless synced. |
| **B. Inside Graphirm** (new module, exposed as a built-in tool and over MCP) | One graph; extraction can call `classify` directly; op log is graph-native. | Graphirm has no MCP server today; touches `crates/agent` knowledge path, which is eval-measured. |
| **C. Offline only** (extend `schema-suggest`) | Smallest. | Keeps the after-the-fact model; no single authority at label time. |

**Recommendation:** A for the prototype, with the storage schema of §6 so a later move to B is a
port, not a redesign. B only if Step 0 shows real duplication and Step 2 shows the eval gain.

## 8. Phases

**Step 0 — measure (read-only, S). Done 2026-10-31; see §1 and §10.** The T1 graphs have no Knowledge
nodes, so the source is `~/.local/share/graphirm/graph.db`. The gate is now: **does the taxonomy beat
exact name dedup (B1)?** Name dedup alone finds 8 of the 9 duplicate pairs. Any gain has to come from
noise and grouping.

**Step 1 — prototype in pi-play (M). Built 2026-10-31; see §10.** Core, tests, Pi extension, labelled
corpus, scorer and baselines. No MCP yet, and no `taxonomy_induce`.

**Step 2 — eval (M).** Run `graphirm-eval` with extraction routed through `classify` vs. current
extraction. Metrics: duplicate Knowledge share, cross-session memory retrieval hit rate, task
resolve rate (must not drop). Per `AGENTS.md`, knowledge extraction is not changed on intuition.

**Step 3 — consolidation + op log (M).** Merge / split / retire with `resolve`.

**Step 4 — Graphirm integration (B, L).** Only if Step 2 passes.

## 9. Open questions

- Tree or DAG? A strict tree is easier to validate; real content often has two parents. Start with
  a tree plus `RelatesTo` cross-links.
- One taxonomy per project, or a global one with project subtrees? Pinned Knowledge is global
  today.
- Who approves restructures on an unattended server — auto-apply under a confidence bar, or queue
  for the phone-first Review tab?
- Embeddings: OpenRouter endpoint vs. a local model; the HNSW index in `graphirm-graph` already
  holds embeddings for Knowledge nodes and may be reusable.

## 10. pi-play test (2026-10-31)

The prototype lives in `/home/krs/pi-play/experiment/taxonomy/` and has its own `README.md`. It is
tried as a Pi extension that calls the core directly. The MCP wrapper comes later, as interface code
over the same core. That way the first test checks the behaviour, not MCP plumbing.

- `taxonomy.mjs` is deterministic and has no LLM. `classify` only retrieves candidates (idf-weighted
  word overlap); the model decides and calls `assign` or `hold`. `applyOps` is all or nothing, logs
  every op, and keeps merged ids resolvable. An `add` can set a `ref`, so later ops in the same batch
  can name it. The first smoke run showed this is needed: without it, the model cannot place a
  child under a parent it creates in the same call. 18 tests.
- `corpus.json` holds the 46 Knowledge nodes. `labels.json` holds hand labels: 21 keep,
  8 duplicate, 17 noise, plus reference categories. Claude labelled them; a person has not reviewed
  them yet.
- Baselines (pair metrics only over real items): B1, exact name dedup, gets noise recall 0 and
  same-entity pairs P 1.00 / R 0.89. B2, B1 plus "empty summary is noise", gets noise P 0.50 / R 1.00
  but loses every real extracted node, since extracted nodes never have a summary.
- Smoke run on `~deepseek/deepseek-v4-flash-latest`: the tree was built, items were assigned and
  held, and health was reported, for $0.002. The `classify` threshold (0.3) scored a clear match below
  threshold, so it is uncalibrated.

**Next:** a driver feeds the corpus to Pi item by item and turns the store into a prediction
(assigned → keep / duplicate, discarded → noise). It scores the prediction against B1 and repeats
the run to check that labels agree. Step 2's `graphirm-eval` comparison is still the gate for any
change inside Graphirm.

### 10.1 Step 3 result: item-by-item runs (2026-10-31)

Two runs on `~deepseek/deepseek-v4-flash-latest`. Each run fed the 46 items to Pi one fresh process
at a time, with a consolidation turn after every 10 items and one at the end. Each run cost about
$0.056 for 51 calls. Details: `/home/krs/pi-play/experiment/taxonomy/README.md` → Step 3.

| | Noise P / R | Same-entity pairs P / R | Category pairs P / R |
|---|---|---|---|
| B1 exact name dedup | – / 0 | 1.00 / 0.89 | – |
| Run a | 0.75 / 0.88 | 1.00 / 0.67 | 0.79 / 0.45 |
| Run b | 0.58 / 0.82 | 1.00 / 0.33 | 0.33 / 0.26 |

Agreement between the runs: verdict kappa 0.63, same-category pairs Jaccard 0.28.

**Gate: not passed.** The model loses exact duplicates that B1 gets for free, and its categories
change from run to run. Noise filtering is the one gain: it catches 14–15 of the 17 noise items,
where B1 catches none. Most of the model's false "noise" calls are bare code symbols with no summary
(Pin, Unpin, ImportCursor, link_session). Whether those count as knowledge is a labelling rule a
person has to set. Identical text got different verdicts in separate calls, so earlier decisions
have to come from the store, not from the model.

**Next version (not run):**
- Exact-name lookup against members, the pool and the discarded list runs first, as a
  deterministic step.
- `same_as` accepts pool items.
- The bare-symbol rule is settled in `labels.json`.
- Categories are seeded once with `taxonomy_init` instead of built from scratch in each
  consolidation.

Simulating the name-first step on the stored verdicts raises same-entity recall to 0.78 / 0.56. It
does not fix the categories.

### 10.2 Step 4 result: v2 runs (2026-10-31)

The four changes from §10.1 were built:

- Labels v2: a bare code identifier with no summary is noise. 8 items changed (journal, 2026-10-31).
- `same_as` can point at held items.
- An exact-name step runs before the model.
- One seed turn builds the categories from all item texts.

Two runs, c and d, together cost $0.078. The exact-name step settled 15 of the 46 items in each run without a model call. Details: `/home/krs/pi-play/experiment/taxonomy/README.md` → Step 4.

All rows are scored against labels v2:

| | Noise P / R | Same-entity pairs P / R | Category pairs P / R |
|---|---|---|---|
| B1 exact name dedup | – / 0 | 1.00 / 0.83 | – |
| Run a (step 3) | 0.95 / 0.76 | 1.00 / 0.83 | 0.74 / 0.65 |
| Run b (step 3) | 0.88 / 0.84 | 1.00 / 0.50 | 0.35 / 0.48 |
| Run c | 1.00 / 1.00 | 1.00 / 1.00 | 0.40 / 0.61 |
| Run d | 0.93 / 1.00 | 1.00 / 0.83 | 0.79 / 0.71 |

Agreement between c and d:
- verdict kappa 0.92 (a vs b: 0.63)
- same-entity pairs Jaccard 0.83 (0.25)
- same-category pairs Jaccard 0.36 (0.28)

**Gate: noise and dedup pass; categories fail.** Both runs catch all 25 noise items. Both find the synonym pair that B1 misses. Run d wrongly rejects `graphirm_graph`, and the name step carries that call to its repeat. The noise rule is written into both the labels and the prompt, and Claude wrote both, so this pass still needs a person to review `labels.json`.

Categories stay unstable, and the cause is now isolated. Consolidation made no structural change in either run, so each tree is exactly what its seed turn built. One seed made a single "Coding conventions" node for 9 items. The other split them three ways. Both are reasonable; they differ in granularity.

**Next options:**

1. Ship only noise filtering plus exact-name dedup, as a post-extraction step. The parts that passed are this step. It is a knowledge-extraction change, so it needs a `graphirm-eval` comparison first.
2. Test categories against a fixed seed tree written by a person. That measures whether items land in the same leaf, instead of whether the model invents the same tree.
3. Both. Option 2 can run in pi-play for about $0.04 a run.
