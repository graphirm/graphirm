# 100-repo meta-analysis (Nodestradamus corpus)

**Purpose:** After a **clean full-corpus batch run** (`run_full_graph_pipeline` across the intended ~41–100 repos, regenerated `batch_summary.json` and per-repo `graph.json` / `insights.json`), aggregate the outputs and answer whether the **MCP v1 tools** (`list_repos`, `get_hotspots`, `get_cycles`, `get_duplicates`, `get_dead_code`, `get_coupling`, `search_repos`, `batch_summary`) expose the right signals—or what to change.

**Scope:** Nodestradamus pipeline + batch cache (`batch_output/` on the ndstrms repo, or `NDSTRMS_BATCH_DIR`). **Not** Graphirm’s session graph (`~/.graphirm/graph.db`).

**Prerequisite:** [`docs/backlog.md`](../backlog.md) item *Clean 100-repo batch run* — prior Phase 3 studies used smaller or partially buggy corpora (see *Pre-corpus signals* below).

**Companion deliverable:** Turn conclusions into concrete MCP changes → backlog item *MCP v1 enhancement recommendations*.

---

## 1. Data sources and layout

| Artifact | Typical location | Use in analysis |
|----------|------------------|-----------------|
| Per-repo `graph.json` | `batch_output/<repo>/graph.json` | Edge counts by type, node counts, structural vs semantic volume |
| Per-repo `insights.json` | same tree | Hotspots, cycles, duplicates, dead_code, coupling (as extracted) |
| Per-repo `metrics.json` | same tree | Layer timings, `layer_metrics`, parse stats by language |
| Aggregate index | `batch_summary.json` (regenerated after full run) | Cross-repo rollups, pass/fail, timing |
| Git metadata (optional) | Clone under cache path | Churn, LOC, commit recency for hotspot validation |

**Loader in code:** `InsightsLoader` in ndstrms (`src/nodestradamus/mcp/`, `NDSTRMS_BATCH_DIR`) — same JSON the MCP tools read; use it or equivalent scripts for batch statistics.

### Numbered experiment runs

Use a stable **experiment id** when you want comparable, repeatable artifacts (harness smoke, partial batch, full corpus).

| ID | Stack | How to run | Artifacts |
|----|-------|------------|-----------|
| **E1** (example) | **Graphirm** | From workspace root: `cargo run -p graphirm-eval -- --skip-memory --experiment E1 --filter basic` | `./results/experiments/E1/eval.json` and `eval.md` (gitignored) |
| **E-NDS-\*** | **Nodestradamus** (spoke `91.98.94.217`) | `ssh root@91.98.94.217` → `/root/project/scripts/run_experiment.sh E-NDS-001 [flags...]` (sources env, runs `insights batch-cache --experiment …`). Or: `PYTHONPATH=/root/project python -m src.nodestradamus.cli insights batch-cache --output-base /root/project/batch_output --config config/downloader.yaml --experiment E-NDS-001 …` | `batch_output/_experiments/<ID>/manifest.json` (git SHA, pipeline version, CLI args), `batch_summary.json`, per-repo dirs with `graph.json` / `insights.json` / `metrics.json` |

**Nodestradamus:** `--experiment` + manifest shipped in ndstrms (`src/nodestradamus/insights/batch.py`), wrapper `scripts/run_experiment.sh` (commit on spoke as of 2026-04-02). **MCP `InsightsLoader`:** still defaults to `batch_output/` root — for tools against an experiment run, set `NDSTRMS_BATCH_DIR` to `…/batch_output/_experiments/<ID>` (or extend loader later).

**Graphirm:** `--experiment` is in **graphirm-eval** (`graphirm-eval/src/main.rs`).

---

## 2. Pre-corpus signals (known before the full 100-repo run)

These come from Phase 3 (14-repo batch, March 2026), dogfood notes, and backlog text. They **inform** the questions below but do **not** replace a full-corpus pass.

- **Structural vs semantic volume:** Semantic / similarity edges can be orders of magnitude larger than CALLS / REFERENCE / INHERITS; interpretability and threshold tuning matter.
- **Phase 3 sample:** 10/14 repos had structural edges; 141 cycles across those 10; PageRank hotspots concentrated in a subset (e.g. Python alembic); Go frameworks hit older cycle **display** caps (since raised to 100).
- **Coupling blind spot:** `get_coupling` is weak or zero when import resolution fails (Go ~6% effective historically; Rust/Ruby often ~0%). Meta-analysis should quantify **by language** and recommend **honest metadata** on tool responses.
- **Dead code:** Most useful where symbol graph + references are reliable (historically Python-heavy); other languages need false-positive taxonomy.
- **MCP surface:** Eight tools were designed before large real corpora existed; gap analysis is expected.

*Replace this section with dated summary statistics after the clean batch completes.*

---

## 3. Research questions → methods → MCP actions

### Q1 — Hotspot quality

**Question:** Do PageRank (or similar) hotspots align with what maintainers care about (churn, ownership, issue references)?

| Step | Method | Output metric |
|------|--------|---------------|
| A | List top-N hotspots per repo from `insights.json` | Ranked file/symbol keys |
| B | Join with `git log --numstat` / churn per path (optional script) | Spearman or overlap@k vs hotspot rank |
| C | Manual spot-check stratified sample (10 repos × 5 files) | Qualitative misrank labels |

**MCP implications (TBD):** e.g. optional `get_hotspots` mode blending structural PageRank with churn; expose confidence / graph type used.

**Findings:** *Pending full batch.*

**Recommendation:** *Pending.*

---

### Q2 — Cycle signal

**Question:** Are symbol-level cycles mostly noise, or do recurring patterns (length, language, subgraph) correlate with known pain (large PRs, bug-heavy files)?

| Step | Method | Output metric |
|------|--------|---------------|
| A | Distribution of cycle lengths / SCC sizes per repo | Histograms by language |
| B | Compare cycle endpoints to hotspot / churn lists | Overlap rate |
| C | Sample manual review of “large SCC” repos | Taxonomy (e.g. test doubles, DI, false import cycles) |

**MCP implications (TBD):** `get_cycles` grouping by length/severity, deduplication across runs, `total_found` vs capped return (already improved in ndstrms; verify at scale).

**Findings:** *Pending full batch.*

**Recommendation:** *Pending.*

---

### Q3 — Semantic vs structural

**Question:** When do semantic similarity edges add signal vs. noise? Does a single similarity threshold generalize across languages?

| Step | Method | Output metric |
|------|--------|---------------|
| A | Per repo: count SEMANTIC_SIMILAR / TFIDF_SIMILAR vs CALLS/REFERENCE | Ratio distributions |
| B | Bucket by primary language (from `metrics` / file histogram) | Threshold vs precision proxy (e.g. same-module vs cross-module) |
| C | Ablate: high-sim edges only inside same package/dir | Change in duplicate_cluster quality (if labeled) |

**MCP implications (TBD):** Semantic centrality option for `get_hotspots` when structural graph is sparse; document threshold sensitivity in tool description.

**Findings:** *Pending full batch.*

**Recommendation:** *Pending.*

---

### Q4 — Coupling blind spot

**Question:** For Go/Rust/Ruby (and any language with low resolution), what would *useful* coupling metrics look like after better import resolution—or what should tools report in the meantime?

| Step | Method | Output metric |
|------|--------|---------------|
| A | `get_coupling` non-zero rate by primary language | Table |
| B | Correlate with Layer 3 `parse_errors` / `by_language` stats | Identify systematic failure modes |
| C | Draft “ideal” coupling definition for one language (e.g. Go) | Spec for engineering follow-up |

**MCP implications (TBD):** Required **`language_coverage`** (or similar) field on `get_coupling` / `batch_summary`; avoid silent zero interpreted as “no coupling”.

**Findings:** *Pending full batch.*

**Recommendation:** *Pending.*

---

### Q5 — Dead code false positives

**Question:** Which patterns cause false positives per language (decorators, reflection, entrypoints, generated code)?

| Step | Method | Output metric |
|------|--------|---------------|
| A | Sample `dead_code_candidates` from N repos per language | Label FP / TP / unknown |
| B | Cross-check with test file paths, `*_pb.go`, common framework dirs | FP rate by pattern |
| C | Align with FileTreeWalker / gitignore improvements | Before/after if re-run |

**MCP implications (TBD):** Filters or tags on candidates; stricter defaults for dynamic languages.

**Findings:** *Pending full batch.*

**Recommendation:** *Pending.*

---

### Q6 — MCP tool gaps

**Question:** Which fields exist in `batch_output/` but are not reachable via the eight MCP tools? What do Graphirm/Cursor-style agents need most?

| Step | Method | Output metric |
|------|--------|---------------|
| A | Schema-diff `insights.json` + `metrics.json` keys vs tool return shapes | Gap list |
| B | Replay typical Graphirm questions (“where is risk highest?”, “what changed structurally?”) against tools only | Failure modes |
| C | Prioritize by frequency × user value | Ranked backlog |

**MCP implications (TBD):** New tools or parameters (see also backlog *MCP v1 enhancement recommendations*).

**Findings:** *Pending full batch.*

**Recommendation:** *Pending.*

---

## 4. Synthesis table (fill after analysis)

| Theme | Primary finding | MCP change | Owner / issue |
|-------|-----------------|------------|----------------|
| Hotspots | TBD | TBD | |
| Cycles | TBD | TBD | |
| Semantic vs structural | TBD | TBD | |
| Coupling | TBD | TBD | |
| Dead code | TBD | TBD | |
| Coverage / gaps | TBD | TBD | |

---

## 5. Related documents

- [`docs/backlog.md`](../backlog.md) — *Clean 100-repo batch run*, *Import resolution*, *Language gap closures*, *MCP v1 enhancement recommendations*
- [`docs/dogfood-findings.md`](../dogfood-findings.md) — Phase 2/3 batch notes, Layer 3 fixes, extractor milestones
- Nodestradamus plan / design docs on the ndstrms repository (batch scripts, `generate_phase3_report.py` if present)
- `docs/plans/2026-04-01-public-readiness-p1-design.md` — unrelated to batch corpus but useful for “production insight surfaces” mindset

---

## Document history

| Date | Change |
|------|--------|
| 2026-04-02 | Initial framework: questions, methods, pre-corpus signals, placeholders for post-batch findings |
| 2026-04-02 | Numbered experiment runs: graphirm-eval `--experiment`; registry table for ndstrms convention |
