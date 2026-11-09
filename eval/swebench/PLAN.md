# Graphirm SWE-bench benchmark plan (v2, 2026-10-11)

## 1. Goal
Compare Graphirm against a strong minimal agent on real GitHub issues, using the same model and the same tasks.

Claims to test:
- **Parity:** Graphirm resolves about as many issues as mini-swe-agent.
- **Cost:** what Graphirm costs per task, cache-aware, compared with mini-swe-agent.
- **Format:** whether segment-structured replies help or hurt compared with plain markdown.
- **Provenance:** how often Graphirm's cited sources exist and match the claim.
- **Memory (secondary tracks):** whether a graph carried across tasks beats a fresh graph, and whether it beats a notes file, on later tasks in the same repo.

Graph scoring is tested on the headline set by G-b16 against G-b16-recency. At the 100k budget a single task drops nothing, so the 16k arms are the test.

## 2. Model
- **Primary:** `deepseek/deepseek-v4.1-flash`, reasoning high, through OpenRouter. Pin Morph (`morph/fp8`) with `provider.only` and fallbacks off. The official DeepSeek endpoint is not used.
- It's Graphirm's default model family, and it's about 10–50× cheaper than Grok 4.7. Mid-range resolve rates also leave room for differences between agents to show.
- Grok 4.7 smoke results are excluded from all analysis. A Grok confirmation run is future work, not a registered track.

## 3. Arms

| Arm | Description |
|---|---|
| **M** | mini-swe-agent v2 (bash only), the baseline |
| **G** | Graphirm, `graphirm run`, product defaults: segments, dedup and prefix-stable packing |
| **G-md** | Graphirm with `--reply-format md`, the format ablation |

Rules for all arms:
- Same model, provider, budgets, issue template, container environment and patch-filter rule.
- Pi handoff off. Auto-approve on (the container is the sandbox).

## 4. Datasets

| Set | Source | Size | Role |
|---|---|---|---|
| Calibration | SWE-bench Verified, seed 20261011 | 30 | Tuning and harness checks only |
| **Headline** | SWE-rebench HF, Jan–May 2026 | 212 (215 minus 3 smoke tasks) | Main paired comparison |
| Robustness | Harbor `swe-rebench-07-2026` | 109 (111 minus 2 smoke tasks) | Recent multi-language check; "post-cutoff" only if V4.1 Flash's cutoff is before July 2026 |
| **SWE-Bench-CL** | `thomasjoshi/agents-never-forget` curriculum JSON (arXiv 2507.00014). Built from SWE-bench Verified. | 273 tasks, 8 repo sequences | Secondary memory track, after the headline set |
| **CL-Bench codebase** | `pgasawa/continual-learning-bench` codebase-adaptation track (arXiv 2606.05661). | 19 tasks: 9 tablib, then 10 tenacity | Secondary memory track, after the headline set |

- Tasks touched during smoke runs are excluded from the headline and robustness sets: vyper-4801, keras-22316, tox-3931, nanobot-4048 and perry-3982.
- The tox sequential track is removed. Memory is measured on SWE-Bench-CL and the CL-Bench codebase track.
- Memory arms, same model, budgets and issue template as the headline:

| Arm | Memory |
|---|---|
| **M** | mini-swe-agent, no memory between tasks |
| **M-notes** | mini-swe-agent plus `/memory/NOTES.md`, carried within a repo, capped at 4k tokens, oldest content truncated first |
| **G-fresh** | Graphirm, new graph per task |
| **G-persist** | Graphirm, one graph per repo sequence |

- Each task runs in a fresh container at its own base commit. Hidden tests run only in the harness afterwards.
- Within a repo, tasks run in the published sequence order. Parallelism is across repos only.
- Memory resets between repos, for G-persist and for M-notes.
- Calibration tasks that also appear in SWE-Bench-CL stay in the sequence. Removing them would break the order. Calibration informed configuration, not memory behaviour. The overlap is listed in `instances.lock.json` and will be noted in `prereg.md`.

## 5. Status

**Done:**
- `graphirm run` (headless, budgets, summary JSON, `run_metadata` node, strict config, routing refused).
- rustls instead of OpenSSL, and a musl static binary that works in `ubuntu:22.04`.
- `serve` model handling fixed.
- `--reply-format segments|html|md`, with segments as the default.
- Segment `src`/`rel` fields, the provenance checker and node-ID tags in context.
- DB lock fixed (WAL once, busy_timeout).
- Cache-aware three-rate cost formula, plus a pre-call cost cap.
- Prefix-stable and excerpt packing flags.
- SWE-bench and SWE-rebench harness paths, and Harbor adapters for both agents.
- Sampling script and lock files (not yet committed).

**Known problem:** Graphirm cost 3–5× more than M in the Grok smoke runs. Causes found:
- duplicate packing (tool result plus derived content node);
- content-first ordering that breaks prompt caching (23–40% cache hits vs M's 80–93%).

## 6. Phases
Each phase has a gate. Don't start the next phase until the gate passes.

### Phase A: packing fixes (code only, $0)

#### A1. Dedup
- **Problem:** each file read is stored as a `read` tool-result node and as a `content:file` node. Each bash run is stored as a tool-result node and as a `content:command_output` node. The packer sends both, so the largest items appear twice in every call.
- **Rule:** when a tool-result node is in the selected set, skip its derived content node.
  - Pack the derived node only when its source tool result isn't selected (for example, carried-over memory from a past session).
- Keep storing both nodes in the graph. Only packing changes.
- **Tests:**
  - a file read appears exactly once in the packed context;
  - a bash output appears exactly once;
  - when the tool result is excluded (simulate a past-session node), the derived content node is still packed.

#### A2. Prefix-stable packing as the default
- **Problem:** the packer emits content and knowledge nodes first, then the conversation. Each new file lands near the front of the next prompt, which changes the start of every request and defeats provider prompt caching.
- **Rule:** emit packed items in a stable order that only grows at the end:
  1. system prompt and tool schemas, unchanged across calls;
  2. selected nodes in creation order;
  3. a content node placed right after the tool result it belongs to, never splitting a tool call from its result.
- If a node is dropped later (only possible when the budget binds), the cache breaks from that point onward, which is acceptable.
- Make this the default for `run`, `serve` and `chat`. Keep the old order available with `--packing legacy`, for ablation only.
- Record the packing mode in the `run_metadata` node and in the summary JSON.
- **Tests:**
  - for a scripted 5-turn session, each call's packed text starts with the previous call's packed text;
  - tool calls and their results stay paired;
  - `--packing legacy` reproduces the old order.

#### A3. Relation vocabulary
- Add `qualifies` and `limits` to the allowed relation types, for both `data-rel` and segment `rel`.
- In the prompt: a caveat that narrows a claim uses `qualifies`; use `contradicts` only when two claims can't both be true.
- Log the three earlier mislabels as examples, as `001`–`003` in `provenance/`.

#### Gate
- All tests above pass, and existing tests still pass.
- Rebuild the static binary with `cargo zigbuild`, and confirm it still runs the fixture in `ubuntu:22.04`.
- Then go straight to Phase B (the offline replay) and report the replay table before spending anything.

### Phase B: offline replay ($0)
- Add `graphirm replay-pack --graph <db> --config <toml> --out <csv>`, which re-packs existing graphs turn by turn without calling the model.
- Run it over the 5 SWE and 2 Harbor Graphirm smoke graphs with these configs: legacy, dedup, dedup+stable, dedup+stable+excerpt(K=3), and dedup+stable at a 16k budget.
- Report per config: input tokens per call, prefix share, duplicates and projected cost at V4.1 Flash prices.

Gate: dedup+stable projects at least a 2× cost reduction over legacy. If it doesn't, stop and report.

The replay dropped the excerpt variant from later candidate configs. Shortening old files changed the start of the next prompt and raised projected cost. Dedup plus stable at a 16k budget stays as the Phase D arm **G-b16**.

### Phase C: model switch and real check (about $5–10)
1. Pin V4.1 Flash and its provider, and confirm `cached_tokens` comes back on one call.
2. Re-freeze `prices.lock.json` from the pinned provider's endpoint entry.
3. Find V4.1 Flash's knowledge cutoff, and label the robustness set accordingly.
4. Run M and G on the first 5 calibration tasks. Cap each task at $0.50 and stop the batch at $10.

Gate:
- both arms complete with no harness errors;
- G's median real cost is within 2× of M's;
- measured per-task costs are recorded for budgeting.

### Phase D: calibration tuning (about $5–15)
- Run M, G, G-md, **G-b16**, and **G-b16-recency** on all 30 calibration tasks. G-b16 is dedup plus stable packing with a 16k context budget. G-b16-recency is that budget with `--scoring recency`.
- Two extra repetitions of all five arms on the first 10 calibration tasks.
- The excerpt variant is not a candidate.
- Do not freeze the config in this phase. The headline arms are chosen after the calibration report, and written into `prereg.md` before any headline run.
- Do not change the running batch. When it finishes, the report also includes, for G-b16 and G-b16-recency across all runs: the share of calls that dropped at least one node; the median and 90th percentile of nodes and tokens dropped per call; and how often the two arms dropped different nodes on the same task at the same turn (an approximation is enough).
- If fewer than about 30% of those calls drop anything, or the drops are mostly 1–2 small nodes, say that plainly and propose G-b8 and G-b8-recency (8k budget) on the 30 calibration tasks, run before pre-registration.

**Tuning rule: only calibration results may inform configuration choices.** Headline and robustness results never do.

Gate: config frozen and written into `prereg.md`.

### G config frozen 2026-10-12

The last cache change puts the budget note on the latest user or tool message and leaves `budget_warning_thresholds` at the product default. One G run on the first 10 calibration tasks then measured median cache hit 0.828, token-weighted cache hit 0.816, median primary cost $0.0217, and 9 of 10 resolved. M's Phase D median cache hit was 0.941. That gap is a known limitation and future work. No further cache tuning before pre-registration.

### Phase E: pre-registration (Claude writes it; $0)
`prereg.md` contains the hypotheses, margins, metrics, budgets, sample sizes, exclusions, frozen config hash and analysis code path.

Then commit `prereg.md`, `instances.lock.json`, `prices.lock.json` and the frozen eval config **in one commit**, before any headline run. Record the commit SHA.

### Phase F: main runs
Registered in `prereg.md`. Order is T1, then T3, then T2, then T4. A track stops at its spend limit.

1. **T1 Headline:** M, G, G-md, G-b16 and G-b16-recency on 212 tasks. Limit $35.
2. **T3 Memory B:** M, M-notes, G-fresh and G-persist on the CL-Bench codebase track (19 tasks). Limit $5.
3. **T2 Memory A:** those four arms on SWE-Bench-CL (273 tasks, 8 sequences). Limit $40.
4. **T4 Robustness, if budget allows:** M and G on 109 Harbor tasks. Limit $25.

Run them in that order, checking spend after each. Workers start at 4 and rise to 6 only if disk stays under 75% and there are no 429s. Harness `--cache_level env --clean True`.

### Phase G: analysis and write-up
`analyze.py` produces everything in section 7, and the results go into a write-up.

## 7. Metrics and statistics

**Primary:** % resolved under the official definition (all FAIL_TO_PASS and PASS_TO_PASS tests pass).

**Cost:**
- **Primary:** uncached × p_in + cached × p_cache + output × p_out, using provider-reported tokens and frozen prices.
- **Secondary:** list price on all tokens.
- **Cross-check:** OpenRouter billed cost, flagging differences above 5%.

**Other:** turns, wall time, cache-hit ratio, exit reason and error taxonomy (no patch, apply failure, wrong fix, budget, provider error).

**Provenance (G arms):**
- Automatic: exists and type-match rates for every `src`.
- Hand-labelled: a stratified sample of 100 claims marked supported or unsupported.
- Labels: `supported`, `citation_does_not_support`, `rel_type_wrong`, `missing_src`.

**Reconstruction check:** for 10 random turns, rebuild the packed context from the graph and diff it byte for byte against the logged request.

**Statistics:**
- Tasks are the unit of analysis, compared pairwise across arms.
- Report win/tie/loss counts, McNemar's test and a paired bootstrap 95% CI (10k resamples).
- Run 3 repetitions on a 30-task headline subset to measure run-to-run variance.

**Hypotheses:** confirmatory, secondary and descriptive hypotheses are in `prereg.md`. That file replaces the numbering in earlier drafts of this plan.

**Memory-track metrics, in addition to resolved, cost, calls and tokens:**
- Resolve rate overall and by position in the sequence.
- Forward transfer: G-persist minus G-fresh, and M-notes minus M, on resolve rate, calls, tokens and cost, by position.
- Efficiency trend: calls and tokens per task against position.
- CL-Bench also reports its own reward and gain. Reward is `1 - (steps_to_solve - 1) / 40` when solved, else 0. Gain is stateful reward minus stateless reward on the same instance.

## 8. Budget summary (rough, at V4.1 Flash prices)

| Phase | Estimate |
|---|---|
| A, B, E, G | $0 |
| C | $5–10 |
| D | $5–15 |
| F headline | $20–60 |
| F robustness | $10–30 |
| F SWE-Bench-CL (1,092 runs, $0.01–$0.03) | $11–$33 |
| F CL-Bench codebase (76 runs, $0.01–$0.03) | $1–$3 |

SWE-Bench-CL wall time at about 6 minutes per run and 4 workers is about 27 hours. The longest sequence is 50 tasks, so more than 4 workers shortens the clock only until disk use reaches 85%. CL-Bench is one sequence of 19 per arm. Four arms in parallel is about 2 hours at the same pace.

On 2026-10-12 the OpenRouter credits endpoint left $7.36. **Top up before T1.** The registered spend limits are T1 $35, T3 $5, T2 $40 and T4 $25.

## 9. Known limitations (state these in the write-up)
- The headline and robustness sets are single-session. Memory is measured only on the two secondary tracks, which run after the headline set.
- The context budget never filled during smoke runs (max about 41k of 100k, nothing dropped), so graph scoring doesn't affect these results.
- Headline tasks predate the model's cutoff. Contamination inflates absolute scores, but it affects both arms in a paired comparison. Robustness results are labelled by cutoff.
- One model and one provider. A Grok confirmation run is future work, not part of this pre-registration.
- Provenance citations are model-reported; the checker verifies existence and type, and "supported" relies on hand labels.

## 10. Next action for Cursor
- Finish Phase D. Do not start a paid memory-track run until that batch has finished and the calibration report is in.
- Memory-track prep (loaders, persistence, mock dry runs) is separate from Phase D and does not spend.

## 11. Future work, not in this plan
- AMA-Bench or Memoria-Bench: graph scoring against long-context and summary baselines. Needs a question-answering adapter.
- T5 Fable is not registered. SWE-Marathon remains the comparison for Fable.
- A Grok 4.7 confirmation on headline tasks, M and G only.
- Store a per-turn pack manifest in the graph: the ordered node ids that were sent, plus the budget note or the inputs that regenerate it. Exact reconstruction then holds from the graph. Re-running selection does not, because the live packer keeps the already-sent id list in session memory and the graph does not store it. `context_dropped_node_ids` lists thread nodes left out of the pack. It is not the ordered sent list, and it does not list derived content nodes omitted by dedup.
- Store the full system text in the graph, for example on `run_metadata`, so sequential replay does not need the external pack trace for that text.
- Investigate the one stable sequential-replay divergence: `chaoss__augur-3512` G-b16 from call 15, where the budget fit dropped a derived command output differently from the live pack.
- The packing budget is Graphirm's estimate, words divided by 0.75, not provider tokens. One T1 call reached 202552 provider prompt tokens under the nominal 100k budget. At the end of T1, report for each Graphirm arm the median and 90th-percentile provider prompt tokens per call next to the nominal budget (100k or 16k). Do not change the frozen config to chase the provider count.
- At the end of T1, compare the median per-task cache-hit ratio of each arm on the tasks finished before the 2026-10-15 pause with the tasks finished after it. Report the two medians. Do not retune caching from that comparison.
