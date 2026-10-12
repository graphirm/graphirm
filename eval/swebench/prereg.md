# Graphirm benchmark — pre-registration

Date: 2026-10-12. Commit: 4bae8c993af85ffa91524f45ed1ca82f49ee21d0. Nothing below is changed after the first
headline run; deviations go to docs/journal.md with a reason, logged before results are seen.

## 1. Frozen system
- Graphirm: commit 4bae8c993af85ffa91524f45ed1ca82f49ee21d0, binary sha256 a4e75be8392e7a6f520618984f9ca17c4c3448ee21d06d19620cbbfa3ae3495c, eval config hash f791cc2803a1219088716944750698551dd6bd9bb38bbacc76a975e567086487.
  Defaults: segments replies, dedup + prefix-stable packing (already-sent content kept in place),
  budget note on the final message only, product warning thresholds [0.7, 0.9],
  task node always packed, Pi handoff off, auto-approve on.
- Baseline: mini-swe-agent 2.4.6, config `adapters/mini_config.yaml` sha256 294ef69c734a99b56c68ceb2ec19620b16e38ed0c0e95d1e7aa55dc80f7c0f3b.
- Model: deepseek/deepseek-v4.1-flash, reasoning high, OpenRouter provider morph/fp8 only,
  fallbacks off. The official DeepSeek endpoint is not used (account privacy setting).
- Prices (prices.lock.json): input 0.000000042, cache read 0.000000009, output 0.0000006 USD per token.
- Budgets (all arms): step limit 100, cost cap $0.50 per task (Graphirm checks before each call,
  mini-swe-agent after each call), wall clock 30 min.
- Issue template: adapters/task_template.md (shared). Patch filter: one function in run_arm.py,
  applied to every arm.

## 2. Tracks, datasets, exclusions
| Track | Data | n | Arms |
|---|---|---|---|
| T1 Headline | SWE-rebench HF, created 2026-01-01..2026-05-12 | 212 (215 minus smoke tasks vyper-4801, keras-22316, tox-3931) | M, G, G-md, G-b16, G-b16-recency |
| T2 Memory A | SWE-Bench-CL (thomasjoshi/agents-never-forget), official curriculum order, 8 repo sequences | 273 | M, M-notes, G-fresh, G-persist |
| T3 Memory B | Continual Learning Bench codebase track (pgasawa/continual-learning-bench) | 19 (9 tablib, then 10 tenacity) | M, M-notes, G-fresh, G-persist |
| T4 Robustness (if budget allows) | Harbor swe-rebench-07-2026, labelled "recent" (model cutoff unpublished) | 109 (111 minus nanobot-4048, perry-3982) | M, G |

- G-b16: G with a 16k packing budget. G-b16-recency: G-b16 with recency-only scoring.
- T2 includes 14 calibration tasks (listed in journal); they stay, to keep sequence order.
  Calibration informed configuration only.
- T2/T3 memory: G-persist keeps one graph per repo; M-notes keeps /memory/NOTES.md (cap 4k tokens,
  oldest truncated). Both reset at every repo switch. Tasks within a repo run strictly in order.

## 3. Hypotheses (confirmatory = C, secondary = S, descriptive = D)
- **H1 (C) Parity, T1:** resolve rate G − M. Non-inferior if the lower bound of the paired bootstrap
  95% CI is above −5pp.
- **H2 (C) Graph scoring, T1:** G-b16 ≥ G-b16-recency. One-sided exact McNemar on discordant
  pairs, α = 0.05.
- Multiplicity: Holm correction across H1 and H2.
- **H3 (S) Format, T1:** G vs G-md resolve rate, two-sided, paired bootstrap CI.
- **H4 (S) Cost, T1:** median primary cost ratio G/M and cost per resolved task, bootstrap CI.
  The 16k arms are reported too (cache breaks expected to raise their cost).
- **H5 (S) Memory, T2 and T3:** G-persist vs G-fresh, and G-persist vs M-notes, on resolve rate,
  calls, tokens and cost — overall and by sequence position (first, middle, last third).
  CIs by repo-cluster bootstrap (sequences are not independent); task-level CIs shown alongside.
- **H6 (D) Provenance:** share of segment items with src; exists and type-match rate under
  (a) strict rule: read/write/edit node, or a cat/sed/head/less of that exact file, and
  (b) lenient rule: any bash command or output naming the path. Plus a hand-labelled supported
  rate on a stratified sample of 100 claims.

## 4. Metrics
- Resolved: official harness definition (all FAIL_TO_PASS and PASS_TO_PASS pass).
- Primary cost: uncached × p_in + cached × p_cache + output × p_out, provider-reported tokens.
  Secondary: list price. Cross-check: OpenRouter billed cost (flag differences above 5%).
- Also: token-weighted cache hit, calls, wall time, exit reason, error taxonomy,
  nodes and tokens dropped per call (16k arms).

## 5. Run protocol
- One repetition per task per arm (Phase D showed 0–2 flips per arm across 3 repetitions).
- Interleave arms per task. Workers start at 4; raise to 6 only if disk stays under 75% and no 429s.
- Harness `--cache_level env --clean True`.
- Provider errors: retry; retries don't count toward budgets.
- Harness error: rerun once; if it fails again, count as unresolved and log it.
- Order: T1 → T3 → T2 → T4. Check spend after each track.

## 6. Spend limits
T1 $35, T3 $5, T2 $40, T4 $25. A track stops at its limit; the remaining tasks are reported
as not run.

## 7. Pre-specified reporting
- Report the number of discordant pairs for H1 and H2. If fewer than 10 discordant pairs exist for
  H2, it is reported as underpowered, not as evidence either way. No extra tasks are added afterwards.
- If both arms resolve more than 90% on T1, ceiling effects are stated in the write-up.
- All arms and tracks are reported, including null and negative results.

## 8. Known limitations (stated in the write-up)
- T1 tasks predate the model's cutoff. Contamination inflates absolute scores but affects every arm.
- With the 100k budget nothing is dropped on single tasks; scoring is tested only by H2 (16k).
- Graphirm's cache hit (~0.83) trails mini-swe-agent's (~0.94), raising its cost.
- The provenance check rule was widened after Phase D (bash views count); both rules are reported.
- One model and one provider; results may not transfer to frontier models (future work).
