# Graphirm Decision Journal

Why things are the way they are. `docs/completion-log.md` records **what** shipped;
this file records **why** — decisions, rejected alternatives, and reversals.
Newest entry first. One entry per decision, not per session.

Add an entry whenever a choice is made that a future reader could not reconstruct from
the code or the plan alone: architecture, dependency, convention, measured trade-off,
something tried and abandoned. Skip routine work.

Entry template:

```markdown
## YYYY-MM-DD — <decision in one line>

**Context:** what forced the choice.
**Decision:** what was chosen.
**Alternatives:** what was considered and why it lost.
**Consequences:** what this commits us to, what to watch.
**Refs:** plan / commit / eval / transcript.
```

---

## 2026-10-14 — The wiped T1 runs are discarded

**Context:** A reboot cleared `/tmp`. The only copies of the first 94 T1 patches, trajectories, graphs, and ledger were there. A search under the home directory, `/var`, and `/opt` found no `model_patch.diff` and no T1 ledger. Those 94 tasks had been graded with the broken grader.
**Decision:** Those 94 tasks are discarded entirely. They are not regraded and they are not carried into the rerun. The per-arm counts seen at 93 tasks came from that broken grader and carried no information about the arms.
**Alternatives:** Treating the broken-grader counts as a partial result was rejected. The gold check showed the official harness resolves tasks the broken grader marked unresolved.
**Consequences:** The write-up does not use those per-arm counts. Phase D's cache summary remains the journal figure from 2026-10-12 (G median cache hit 0.828, M Phase D median 0.941). The fuller cache-confirm figures, including 9 of 10 resolved and median primary cost $0.0217, are in `eval/swebench/PLAN.md`. The raw Phase D graphs under `/tmp/phase-d` are gone.
**Refs:** `eval/swebench/PLAN.md`, `/home/krs/graphirm-runs/t1-validity/validity.jsonl`.

## 2026-10-14 — Rerun T1 fresh on the 210 valid tasks

**Context:** The headline gold check finished. Two tasks drop out: `sigma67__ytmusicapi-909_interface` (official gold did not resolve) and `huggingface__trl-5135` (an empty patch resolves). `pydap__pydap-624` gold resolved on this run, so it stays. The saved agent patches for the first 94 tasks are gone.
**Decision:** The T1 rerun starts fresh on the 210 valid tasks, with the corrected SWE-rebench fork grader and durable storage. Hypotheses, arms, configs, and the agent environment are unchanged.
**Alternatives:** Resuming the wiped ledger is impossible. Regrading the 94 patches is impossible.
**Consequences:** The rerun has not started. It waits until the free validity checks for T3, T2, and T4 are reported.
**Refs:** `eval/swebench/prereg.md`, `eval/swebench/grade_fork.py`, `/home/krs/graphirm-runs/t1-validity/validity.jsonl`.

## 2026-10-14 — The lost $12.88 is outside the rerun cap

**Context:** The wiped T1 run had spent about $12.88 of calculated primary cost before it was discarded.
**Decision:** That $12.88 is reported separately. The pre-registered $35 T1 limit applies to the rerun, not to the sum of the lost run and the rerun.
**Alternatives:** Subtracting $12.88 from the $35 cap would shrink the rerun because of a grader bug and a wiped disk, which is not what the cap was for.
**Consequences:** The rerun may spend up to $35 of new calculated primary cost. The lost $12.88 stays in the write-up as spent and discarded.
**Refs:** `eval/swebench/prereg.md`.

## 2026-10-14 — Eval outputs stay out of /tmp

**Context:** The only copies of the T1 patches were under `/tmp`, and a reboot deleted them. This machine has one disk.
**Decision:** Run outputs go under `/home/krs/graphirm-runs/<track>/`. The driver refuses a path under `/tmp`. Ledger lines are flushed and fsynced. After each completed task, the record is committed in `/home/krs/graphirm-runs-backup`, a separate git repo on the same disk. `graph.db` files stay under `graphirm-runs` and are tarred when a track finishes.
**Alternatives:** A second physical disk is not available. A branch of the product repo would mix run artifacts with source.
**Consequences:** The backup survives a reboot and a `/tmp` wipe. It does not survive loss of this disk.
**Refs:** `eval/swebench/durable.py`.

## 2026-10-13 — Rerun the gold check outside /tmp

**Context:** A power cut rebooted the machine at 18:16. `/tmp` was cleared. The 94 tasks of agent patches, the partial gold-check log, the T1 ledger, and the SWE-rebench fork clone were only stored there. A search under the home directory, `/var`, and `/opt` found no copies.
**Decision:** Rerun the gold and empty checks for all 212 headline tasks onto `/home/krs/graphirm-runs`, which survives a reboot. Do not call the model for this check. Do not rerun agent tasks until this check's exclusion list exists. The exclusion rule and the fork grader decision already recorded still stand.
**Alternatives:** Waiting for a backup was checked, and none exists. Spending on the 94 agent tasks again before the exclusion list would repeat work on tasks that may carry no signal.
**Consequences:** Those 94 runs cannot be regraded. A later T1 is a new run, not a resume from the ledger. The earlier commit sample already showed the agent and the grading image on the same commit, including `pydap__pydap-624`. `pandas-dev__pandas-63945` has a dataset `base_commit` field that differs from the image both sides used.
**Refs:** `docs/journal.md` entries from earlier today, `/home/krs/graphirm-runs`.

## 2026-10-13 — Grade T1 with the SWE-rebench fork

**Context:** The T1 grader ran `test_cmd` in a login shell and did not activate conda or apply `test_patch`. Official gold patches passed 4 of 30 that way, and 29 of 30 through the SWE-rebench fork.
**Decision:** Deviation: T1 grading now uses the SWE-rebench fork's `run_evaluation` path, including conda activation and `test_patch`. Images are removed after each task. This corrects a grading bug. Hypotheses, arms, configs, and the agent environment are unchanged.
**Alternatives:** Rerunning failed agent runs was rejected. Missing the testbed Python is the agent's own behaviour, and it happened in both arms. A second try for only those runs would bias the comparison.
**Consequences:** Saved patches are regraded. Remaining T1 tasks use the corrected grader and the same agent container as the tasks already run. Agents still have to find the testbed Python themselves. That limit stays in the write-up.
**Refs:** `/tmp/t1/drive.py`, `/tmp/t1/grade_fork.py`, `/tmp/SWE-bench-fork`.

## 2026-10-13 — Exclude headline tasks that carry no signal

**Context:** The T1 grader was paused after a validity check. Some tasks cannot separate a correct fix from no fix.
**Decision:** Deviation: tasks are excluded from all arms if (a) the official gold patch fails under the official SWE-rebench harness, or (b) an empty patch resolves them. Reason: such tasks carry no signal. The rule was set before regraded results were seen.
**Alternatives:** Keeping every task would mix grader and task failures into the arm comparison. Dropping tasks after seeing which arm solved them would bias the comparison.
**Consequences:** The same exclusion list applies to every arm. Excluded tasks are omitted from scoring. The list is fixed from gold and empty runs only.
**Refs:** `eval/swebench/prereg.md`, `/tmp/t1-validity/`.

## 2026-10-13 — Pause T1 to check grading validity

**Context:** At 93 fully graded tasks, interim per-arm resolve counts were reported. The plan said to hold those counts until T1 finished. Every arm resolved about 7 of 93, and paired comparisons were almost all ties, which can mean the grader rather than the agents is deciding the outcome.
**Decision:** Stop launching new T1 tasks after the task already in progress finished grading. Keep every saved patch, trajectory, and graph. Grade official gold patches through the existing grader before any further spend. Do not change hypotheses, arms, or configs.
**Alternatives:** Continuing the run would spend more on scores that may not mean the agents failed.
**Consequences:** T1 stays paused until that validity report is reviewed. Restarting `/tmp/t1/drive.py` resumes from the ledger and does not repeat finished arms.
**Refs:** `/tmp/t1/ledger.jsonl`, `eval/swebench/prereg.md`.

## 2026-10-12 — Pre-register the SWE-bench tracks without Fable

**Context:** Calibration is finished. The headline comparison needs a frozen document before any T1 task runs.
**Decision:** `eval/swebench/prereg.md` registers T1 (M, G, G-md, G-b16, G-b16-recency), T2 SWE-Bench-CL, T3 CL-Bench codebase, and T4 Harbor if the budget allows. T5 Fable is not registered. A Grok confirmation run is future work. Commit 4bae8c993af85ffa91524f45ed1ca82f49ee21d0.
**Alternatives:** Holding pre-registration until Graphirm's cache hit reached about 90% was rejected. Caching changes cost, not resolve rate.
**Consequences:** Deviations after the first headline run go in this journal before results are seen. T2 keeps these 14 calibration tasks so the sequence order stays intact: django__django-10999, django__django-13023, sympy__sympy-12096, sympy__sympy-15345, sympy__sympy-14976, sympy__sympy-18698, sphinx-doc__sphinx-8475, sphinx-doc__sphinx-9258, sphinx-doc__sphinx-9591, matplotlib__matplotlib-25775, scikit-learn__scikit-learn-14983, scikit-learn__scikit-learn-25102, astropy__astropy-13977, pytest-dev__pytest-6197.
**Refs:** `eval/swebench/prereg.md`, `eval/swebench/PLAN.md`.

## 2026-10-12 — Freeze the G cache setup after one budget-note measurement

**Context:** A changing usage percentage near the start of the prompt breaks the cache. Disabling the warning would hide a product feature, and another round of tuning on the same 30 tasks would fit the setup to them.
**Decision:** Append the budget note to the latest user or tool message, keep the product thresholds, measure G once on the first 10 calibration tasks, and freeze. That run's median cache hit was 0.828, against M's Phase D median of 0.941.
**Alternatives:** An empty threshold list removes the line and was rejected. Waiting for about a 90% hit before pre-registration was rejected, because caching changes cost and not resolve rate.
**Consequences:** The gap versus M stays a known limitation. The note is not stored, so the next call rebuilds that one message without it and the cache stops at that message.
**Refs:** `eval/swebench/PLAN.md`, `/tmp/cache-confirm-3/summary.json`.

## 2026-10-12 — The segment reminder stays on the system message

**Context:** Phase D's pack trace showed the segment reminder moving onto the latest human message. Call 1 wrote it onto the task. Call 2 wrote it onto a later content node and the task text shrank, so the prompt cache missed from the first message. A budget warning on the system message would do the same once usage crossed 70%.

**Decision:** The reminder is appended to the system message, which is identical on every call. Nodes sent on the previous call keep that order, and newly selected nodes append. The budget warning, whose text changes with usage, is a trailing message. Each assistant turn stores `context_dropped_node_ids`.

**Alternatives:** Leave the reminder on the latest human message so it stays the nearest instruction. That is the cache break. Freeze the whole prefix by never dropping a node that was already sent. That would ignore the context budget.

**Consequences:** A node the token budget drops still breaks the cache at that point. A derived content node that was already sent stays in the pack, so dedup does not punch a hole in the prefix. The budget warning is appended to the latest user or tool message, not stored, and the eval config keeps the product default thresholds.

**Refs:** `crates/agent/src/knowledge/segments.rs` `append_segment_json_request`, `crates/agent/src/context.rs` `keep_previous_pack_order`.

## 2026-10-12 — A carried graph pins the current task, and reuses the existing agent

**Context:** G-persist keeps one graph for every task in a repository. The pin from 2026-10-11 keeps the oldest user message, which would be the first task forever. `graphirm run` also called `Session::new`, so a second run on the same file started a new agent and the new task did not chain onto the old conversation.

**Decision:** `graphirm run` calls `Session::open`. An existing graph binds the oldest agent and appends the new task to that conversation. The task message is marked `task`. The packer pins the latest marked message. A thread with no mark still pins the oldest user message, so a single-task graph behaves as before. A later retry nudge is not marked and does not take the pin.

**Alternatives:** Pin the oldest message and start a new agent per task, then search the whole graph for earlier tasks. That leaves the first issue in every pack and does not put later tasks on the conversation thread the packer walks. Pin every marked task. A long sequence would spend the budget on old issues.

**Consequences:** Single-task runs that set the mark pin the same node as before. A multi-task graph pins the issue for the run that is in progress. The Phase D musl binary was not rebuilt, so the running calibration batch still uses the oldest-message pin.

**Refs:** `crates/agent/src/session.rs` `Session::open`, `crates/agent/src/context.rs` `current_task_user_id`. Tests `open_reuses_the_agent_and_chains_the_next_task` and `recency_16k_pins_the_current_task_not_the_first`.

## 2026-10-11 — The first user message stays packed at any budget

**Context:** G-b16-recency scores by recency inside a 16k pack. The oldest node is the issue text. Recency drops the oldest nodes first, so the task could vanish mid-run and make scoring look stronger than it is.

**Decision:** `build_context_with_stats` always keeps the oldest user message. Its tokens still reduce the budget. It is not part of the droppable set. Each assistant turn records `task_node_present`.

**Alternatives:** Leave it to the four-turn tail. That tail is the newest turns, not the issue. Give the issue a score bonus under recency. That still lets a tight budget skip it when the bonus is not enough.

**Consequences:** A long issue can crowd out other nodes at 16k. The issue text itself stays. 16k runs from before this pin are discarded.

**Refs:** `crates/agent/src/context.rs` `first_task_user_id`. Test `recency_16k_keeps_the_first_user_message`.

## 2026-10-11 — Replay token estimate uses characters/3.8, and a pinned model retries itself

**Context:** Phase C measured Morph input tokens at about 0.53 of the replay estimate, which divided characters by 2. The official DeepSeek endpoint is blocked by the account privacy setting, so the eval pins one OpenRouter provider and has no fallback model. A 429 on that single model ended the turn.

**Decision:** `replay-pack` estimates input tokens as characters/3.8 and writes that factor on every CSV row. A retryable provider error is tried again on the same model, up to five times, and a failed attempt is not written into usage or the cost total.

**Alternatives:** Leave the characters/2 estimate and apply 0.53 only in the report. That keeps the tool wrong for the next replay. Treat 429 as fatal so a rate limit cannot hide inside the budget. That throws away the run instead of the failed attempt.

**Consequences:** Replay dollar figures still use the old blended Flash rates, not the Morph lock. The cache-prefix projection is unchanged. Same-model retries can add provider-billed cost that the frozen ledger does not include, because the failed response is not stored.

**Refs:** `eval/swebench/PLAN.md` Phase C and Phase D. `src/commands/replay_pack.rs`. `crates/agent/src/workflow.rs`.

## 2026-10-09 — Compaction uses the session model when none is set

**Context:** Auto-compaction sent `model: ""`. OpenRouter answered `No models provided`. The failure is logged and the chat call still runs, so a long session never gets summarized.
**Decision:** Use the session model, then the first cheap-tier model. Refuse the provider call when both are empty. Eval preflight fails when health reports compaction `unconfigured`.
**Alternatives:** A new `compaction_model` config key. Nothing in the toml sets one today, and the session model is already a model the provider accepts.
**Consequences:** `app.graphirm.ai` has compaction on, so this has to be deployed before long sessions cross the threshold. The summary text from the 4,000-token selection run was deleted with the session and was not logged.
**Refs:** `resolve_compaction_model`, `docs/plans/2026-10-08-tool-pair-atomicity.md`.

## 2026-10-09 — A graph_query loop stops on a repeat or after four calls

**Context:** `graph-query-invalid-mode` repeated one result (the same search, or `depth is required`) until the prompt timed out. `graph-query-bfs` changed the query each time and also timed out. Line-count misses were the digits `3537` written as `3,537`. `write-fibonacci` once timed out after the file and its tests already existed.
**Decision:** The same `graph_query` arguments return the previous result instead of running again. A fifth distinct `graph_query` in one agent loop is refused. Four is the cap because the BFS prompt needs two calls and a retry still fits. The tool schema requires `node_id` and `depth` when mode is `bfs`. The grader strips thousands separators before a command-output compare. A timeout whose verifier already passes is recorded as `finished but didn't stop`.
**Alternatives:** One cap for every graph_query loop. That would hide the identical-call case, which already has a result. Raising the prompt time limit. The passes finished inside the limit; the failures were still calling the tool.
**Consequences:** The Phase 2 comparison is 3 runs of this harness on a main binary and 3 on `feat/tool-pair-units`. Both binaries get the stop rule, the LLM extraction fallback, the health fields, and the eval rate-limit burst. A remaining score gap is the unit change. The main checkout itself is untouched; that binary is built from a detached worktree.
**Refs:** `crates/agent/src/graph_query_guard.rs`, `crates/tools/src/graph_query.rs`, `graphirm-eval/src/task.rs`.

## 2026-10-08 — Eval tasks get a fresh workspace copy

**Context:** `GRAPHIRM_DISABLE_WORKSPACES=1` made session tools use the checkout. Counting then passed, and `git status` showed no task-written files, because the write prompts use absolute `/tmp` paths. A relative write would still land in the repo, and a shared directory makes the next run see leftovers.
**Decision:** Workspaces stay on. Each task gets a new directory. The harness copies `crates/` and `src/` into it and deletes the shared `/tmp/eval_*` files first. The in-flight repo-cwd suite is a counting check, not the Phase 2 baseline. Cross-session ranking stays described as available: it is off on app.graphirm.ai until `EMBEDDING_BACKEND` is set, which would send user content to the embedding provider.
**Alternatives:** Leave tools pointed at the checkout for the three runs. That checks counting and risks the tree.
**Consequences:** Run 1 of `results/fix-*.json` still used the checkout. Later runs in that loop pick up the workspace copy when the harness recompiles.
**Refs:** `graphirm-eval/src/workspace.rs`, `README.md`.

## 2026-10-08 — Eval tools stay in the repo, and a missing extractor falls back once

**Context:** Auto-select chose the local GLiNER2 backend because the model files were on disk. This binary was built without `local-extraction`, so every turn logged that error and no knowledge node was written. The same toml sets `workspaces_root` to `/data/workspaces`, so eval sessions ran in `/data/workspaces/session`, which has no `crates/` tree. `app.graphirm.ai` does not have the GLiNER2 files, so it already uses the LLM extractor (`Knowledge extraction complete backend=Llm`). Its log also says `EMBEDDING_BACKEND not set`, so cross-session ranking is off there. The dogfood host did not accept SSH from this machine.
**Decision:** `select_compiled_backend` uses local ONNX only when that feature is compiled. Otherwise it uses the LLM backend and logs that once at startup. Eval preflight refuses a run whose health reports extraction disabled or memory off. The eval child keeps `EMBEDDING_BACKEND` and sets `GRAPHIRM_DISABLE_WORKSPACES=1`. A task that does not pass stores the final answer and tool outputs before the session is deleted.
**Alternatives:** Rebuild eval with `local-extraction` and keep selecting ONNX. That would measure a path production is not on. Leaving `workspaces_root` and reading the bash error from the next artifacts. The empty workspace is already visible on disk.
**Consequences:** Knowledge tasks now depend on the LLM extractor. Counting tasks run against the repo. `cascading-pipeline` keeps its 120s prompt limit: the pass finished all three prompts in 130s, and both failures were still inside the first prompt, so finishing within the limit is part of that task. `graph_query` already lists the valid modes; the invalid-mode timeout was a loop of successful calls.
**Refs:** `crates/agent/src/knowledge/extraction.rs`, `src/commands/serve.rs`, `graphirm-eval/src/harness.rs`.

## 2026-10-08 — A timeout counts as infrastructure only before the model answers

**Context:** The GLM variance suite printed several tasks as `0 turns` plus `session timed out`. That counter is the harness prompt count, and `TaskResult::fail` writes 0. The server log for every one of those tasks already had assistant replies and tool calls. `cascading-pipeline` was cancelled mid-loop on both failures (turns 6 and 12).
**Decision:** A prompt wait with no assistant message is retried, then excluded from the score, the same way a 429 is. A timeout after at least one assistant message stays a failure, and the result stores that message count.
**Alternatives:** Treat every printed `0 turns` timeout as infrastructure. That would have dropped the graph-query loop and both pipeline timeouts, which were the agent still working when the prompt clock ran out.
**Consequences:** This suite's 58–68% band is not an artifact of empty waits. The next suite can still exclude a true no-response timeout.
**Refs:** `graphirm-eval/src/harness.rs`, `results/v4-1.json` through `results/v4-3.json`.

## 2026-10-08 — Smart tier follows GLM Latest

**Context:** Both tiers had just been set to pinned DeepSeek V4 Flash. The ask was for smart to be GLM 5.3 or newer. OpenRouter `~z-ai/glm-latest` currently resolves to the flagship `z-ai/glm-5.3` and moves when a newer GLM flagship ships. GLM 5.3 Prime and Flash are separate models.
**Decision:** Smart is `openrouter/~z-ai/glm-latest` on this branch and on the main checkout. Cheap on this branch stays the pinned `openrouter/deepseek/deepseek-v4-flash`. The Flash-only suite was stopped so the next runs use the split.
**Alternatives:** Pin `openrouter/z-ai/glm-5.3`, which is what main had and what the 2026-10-06 entry chose. That stays on 5.3 when a newer flagship appears.
**Consequences:** A smart turn follows whatever GLM Latest points at. A server already running keeps the tiers it loaded until restart.
**Refs:** `config/default.toml` `[agent.routing]`.

## 2026-10-08 — Eval turns use pinned DeepSeek V4 Flash

**Context:** The variance suite logged `deepseek/deepseek-v3.2` on every turn. `EVAL_MODEL` is qwen, and `GRAPHIRM_MODEL` is v4 Flash, but the Jev router reads `[agent.routing]` in `config/default.toml`. On this branch both tiers, `[model].name`, and `[agent].model` were still v3.2.
**Decision:** Those four strings are the pinned `deepseek/deepseek-v4-flash` (OpenRouter-prefixed where the old string was). Both tiers stay the same model. The v3.2 suite was stopped and the full suite started again.
**Alternatives:** The moving `~deepseek/deepseek-v4-flash-latest` alias, or the main checkout's split (that alias for cheap, GLM 5.3 for smart). The request was to replace v3.2 with V4 Flash, so both tiers use the pinned id.
**Consequences:** A server already running keeps the model it loaded until restart. Dated journal lines above still name v3.2 as the setting they were written about.
**Refs:** `config/default.toml` `[model]`, `[agent].model`, `[agent.routing]`.

## 2026-10-08 — Eval 429s stay out of the score

**Context:** The coding smoke test recorded two task failures because the harness polled `GET /api/sessions/{id}` twice a second, drained the burst-60 limiter, and decoded the 429 body as a harness error. A slower poll then passed 2/2. tower_governor reports a sub-second wait as `Retry-After: 0`.
**Decision:** A 429 is retried, then recorded as an infrastructure error and left out of the pass rate. The server wait is at least one second. The harness waits on the session SSE stream, which it opens before the prompt. Eval traffic stays on the loopback process this harness spawns, with `GRAPHIRM_RATE_LIMIT_BURST=1000` on that process only. `app.graphirm.ai` is refused.
**Alternatives:** Keep polling, only slower. That still shares a bucket with whatever else is on the host, and a 429 would still lower the score. Point the harness at the dogfood host. This binary does not open a remote base URL, so that host is not a target here.
**Consequences:** Production burst stays 60. A rate-limit blip no longer changes the percentage, so runs can be compared. A timeout or a failed loop is still a failure. The 2/2 coding pass was a smoke test; the full suite still has to be repeated before it can judge the context-unit change.
**Refs:** `graphirm-eval` client and harness, `governor_error_response`.

## 2026-10-08 — The approval placeholder is payload-only

**Context:** Writing `awaiting approval` into the graph left a second result with the same `tool_call_id` once the real result was recorded.
**Decision:** The placeholder is appended to the messages for that build and is not stored. A placeholder already in the graph is still dropped from the payload when the real result is present.
**Alternatives:** Keep the stored placeholder and delete it on approval. That needs a write on the approve path and still races a context build in between.
**Consequences:** A still-pending call shows `awaiting approval` on every build until a real result exists. The graph has no `pending_approval` node from this path.
**Refs:** `append_pending_approval_results`, `docs/plans/2026-10-08-tool-pair-atomicity.md`.

## 2026-10-08 — Two indexes is a failed cut

**Context:** A chat reply closed every tag and still wrote a `div` index and a `nav` index, both `id="index"`. The cutter kept the first. The rule file on `8ec1d18` already asked only for the nav. This chat's prompt still held the older div wording, so the reply wrote both.
**Decision:** `cut_html_index` fails when the page has any count of `id="index"` other than one. The rule and the Pi contract say there is one index and it is that nav. A test fails if either text contains a div index or "each part is a div". The doubled page is a fixture.
**Alternatives:** Reject every div index, including a page that has only the old shape. That page still cuts when it has one index.
**Consequences:** Cursor chat replies are not passed through the cutter. The sentence in the rule is what stops the next chat reply. Pi replies with two indexes no longer become piece nodes.
**Refs:** `cut_html_index`, `.cursor/rules/html-part-index.mdc`.

## 2026-10-08 — Drop the approval placeholder once the real result exists

**Context:** A pending tool call is given an `awaiting approval` result so the model still sees the call. That node stays on the `RespondsTo` chain. When approval lands, the real result is recorded with the same `tool_call_id`, and both results were sent.
**Decision:** If a non-placeholder result with that id is in the assembled nodes, drop the `pending_approval` node from the payload and log its id. The graph node stays.
**Alternatives:** Delete the placeholder from the graph when the real result is written. That loses the record that approval was open. Keeping both and hoping providers dedupe lost because a repeated id is a rejected request.
**Consequences:** A still-pending call keeps the placeholder. A call that has since returned keeps the real result only.
**Refs:** `omit_stale_pending_placeholders`, `docs/plans/2026-10-08-tool-pair-atomicity.md`.

## 2026-10-08 — HTML index shape 2 is nav, section, and h2

**Context:** A `div` index read as a wall of tags. `ol` would paint a second number on top of `1.0.0`. A number inside `pre` is copied with the code. An `aside` is announced as complementary, which listeners skip, and a caveat is often the part that matters. The Cursor rule and the Pi contract had already drifted once.
**Decision:** Shape 2 asks for `nav` and `ul`, a `section` per part, and an `h2` that repeats the link's number and heading. The link is the heading. A missing or different `h2` logs a warning and the cut keeps the link. Lists are `ul` because the number is already in the text. A code block stores the number in `data-n`. A caveat is a `section` with class `caveat`, not an `aside`. An example wraps its lines in `figure`. `HTML_INDEX_SHAPE_VERSION` is `"2"`, stamped on each `reply_part`. A test fails when a contract line is absent from `.cursor/rules/html-part-index.mdc`. The hand-labeled markdown set stays on `structure_segment` and does not take HTML rows.
**Alternatives:** `list-style: none` on `ol` lost because this page has no stylesheet. `role="note"` lost because the class is already the kind. Generating the rule from the Rust string lost to a containment test, which allows the rule a chat-only paragraph and an example.
**Consequences:** Shape 1 pages still cut when their lines are numbered `p` or `li` text. A `pre` without `data-n` does not. A set that later mixes both shapes has to store the shape version on the row.
**Refs:** `.cursor/rules/html-part-index.mdc`, `HTML_PIECE_INDEX_CONTRACT`, `cut_html_index`.

## 2026-10-08 — Piece numbers are N.0.0

**Context:** The chat rule numbers every index entry and every line. Pi was still asked for `[kind] Heading` with no numbers, so a page that followed the rule failed the cut.
**Decision:** The delegate contract asks for `N.0.0 [kind] Heading` and `N.0.M` on each line. The cutter requires that shape, checks `N` against part order and `M` against line position, and stores the words after the number. A page without numbers is a failed cut.
**Alternatives:** Accept both shapes. That keeps old pages working and lets Pi skip the numbers.
**Consequences:** Recorded runs and tests that embed an HTML index need the numbers. `order` and `position` stay the stored identity.
**Refs:** `.cursor/rules/html-part-index.mdc`, `cut_html_index`.

## 2026-10-08 — async-trait 0.1.92 so clippy 1.99 can build

**Context:** CI runs `cargo clippy --all-targets --all-features -- -D warnings` on stable. Clippy 1.99 flags `double_must_use` on every `#[async_trait]` method because 0.1.89 stamps a bare `#[must_use]` on a future that is already must-use. Local clippy 1.93 did not have that lint, so the branch went green here and red on the pull request.

**Decision:** Bump `async-trait` from 0.1.89 to 0.1.92. That release stops emitting the attribute. The `Cargo.toml` constraint stays `0.1`.

**Alternatives:** `#[allow(clippy::double_must_use)]` on each trait. That papers over the macro in every crate that uses it, and the next trait added would fail CI again.

**Consequences:** The lockfile pulls `syn` 3 for the macro. MSRV stays 1.88 (`async-trait` 0.1.92 asks for 1.71).

**Refs:** https://github.com/graphirm/graphirm/pull/1

## 2026-10-08 — Piece records are nodes under the reply

**Context:** The HTML-index plan stored the cut as a `pieces` array on the assistant Interaction. A list inside one node cannot take an edge. The split exists so a person can accept, drop, or connect one part.
**Decision:** A clean cut creates Content nodes. `content_type` is `reply_part` or `reply_line`. The assistant message `Contains` each part. Each part `Contains` its lines. No new node type and no new edge type. A failed cut creates none of these nodes.
**Alternatives:** Metadata on the assistant message lost, because the graph cannot point at one entry. A sixth node type lost for now. Content already carries a `content_type` and a body.
**Consequences:** The plan `docs/plans/2026-10-08-html-piece-index.md` Task 4 writes the subgraph, not metadata. Same-turn neighbors still do not get `applies_to`. A later reply that names an earlier part gets its own nodes. The mention stays in the text.
**Refs:** `docs/plans/2026-10-08-html-piece-index.md`.

## 2026-10-08 — An HTML index is how a Pi reply becomes pieces

**Context:** Asking the model again for an outline, then once per heading, copied or dropped the reply and cost a call per heading. A single HTML page with an index of links cut cleanly: every link found its part, and the kind was the class on that part.
**Decision:** Graphirm asks Pi for that page, cuts it locally, and stores the records beside the reply. The cutter accepts a part only when the link, the id, and one of the eight kind words match. A failed cut is retried once. The reply text stays either way. Markdown replies keep `structure_segment`.
**Alternatives:** Outline-then-expand lost. Forcing the body with a provider JSON schema lost earlier (empty arrays, wrong kinds). A separate kind labeler stays optional and is not on the cut path.
**Consequences:** The kind on the record is the kind Pi wrote. Context selection, compaction, memory ranking, and knowledge extraction do not change. Plan: `docs/plans/2026-10-08-html-piece-index.md`.
**Refs:** `pi_groceries.py` cutter, `docs/plans/2026-10-05-reply-pieces.md`.

## 2026-10-07 — A Pi reply is the source, piece records are the target

**Context:** A plain grocery request to Pi returned normal markdown. Forcing that reply into `{name, quantity}` JSON was rejected.
**Decision:** The source is the reply as Pi wrote it. The target is one record per block: `order`, `kind`, `heading`, `items`. The words stay in the source. The grocery lists are `statement`.
**Alternatives:** Ask Pi to emit the target. Put a grocery schema in the prompt and reject anything else.
**Consequences:** The next check is source in, target out, on that grocery reply. The target is not a rewrite of the shopping list.
**Refs:** `pi_groceries.py`.

## 2026-10-07 — Qwen 7B does not beat the piece baseline

**Context:** The same 27 replies and the same eight-word grammar were run through `Qwen2.5-7B-Instruct` Q4_K_M on local llama.cpp. The 1.5B model had matched 46 of 192 kinds on several-piece replies.
**Decision:** This model is not the labeler either. It matched 50 of 192. The baseline matches 119 of 192. No output was rejected. It never said `options`.
**Alternatives:** Treat the move from 46 to 50 as progress and try a larger model next.
**Consequences:** Size from 1.5B to 7B did not fix kind naming. The parser stays.
**Refs:** `feat/piece-kind-labeler`, `~/.graphirm/piece-labels.jsonl`.

## 2026-10-07 — Qwen 1.5B does not beat the piece baseline

**Context:** llama.cpp b11461 and `Qwen2.5-1.5B-Instruct` Q4_K_M labeled the 27 Pi replies through a grammar that allows only the eight kind words.
**Decision:** This model is not the labeler. On several-piece replies it matched 46 of 192 kinds. The baseline matches 119 of 192. No output was rejected.
**Alternatives:** Count the five `options` hits as success. The baseline scores zero there.
**Consequences:** The next measurement is a larger model under the same grammar. The parser stays.
**Refs:** `feat/piece-kind-labeler`, `~/.graphirm/piece-labels.jsonl`.

## 2026-10-07 — The kind labeler returns one grammar word

**Context:** The parser already cuts a reply. The baseline names kinds with a few string rules and misses options, examples, and most warnings. A Pi agent asked for a kind would explain and edit files. OpenRouter cannot force the reply to be one of the eight words.
**Decision:** On `feat/piece-kind-labeler`, `kind_label` builds a prompt from the piece alone and a llama.cpp grammar whose only outputs are the eight kind words. `parse_kind_label` rejects a sentence, a capital letter, or a second word. `score_predicted` compares those words with the labeled ranges. A live `llama-cli` run has not been made.
**Alternatives:** Ask Pi for the kind. Send the user's question in the prompt. Fall back to the baseline when the model writes a sentence.
**Consequences:** The labeler has not yet been shown to beat the baseline.
**Refs:** `crates/agent/src/pi_delegate/kind_label.rs`.

## 2026-10-07 — One piece has one kind, cut by function

**Context:** The kind list is eight words. A labeler that may return two kinds for one block has no clear failure. Sentence boundaries are the easy wrong cut: an options list can sit inside one sentence, and a description of a process ("Spark reads the file, shuffles, then writes") looks like steps.
**Decision:** One piece, one kind. If a block seems to be two kinds, split it. Cut by function, not by sentence. How something works stays a statement. What the reader does is steps. The Spark sentence is a labeler test case, not a new kind. Caveat severity (`warning`, `caution`, `note`) and one `applies_to` link wait until a caveat is used to allow or block an action. An expected result waits until a verifier exists, and it is a sub-part of steps.
**Alternatives:** A second kind on the same piece. A process kind. Three caveat kinds. An expected-result kind.
**Consequences:** The labeler returns one of the eight words. A piece it cannot name is a bad cut. The guide's steps near-miss includes the Spark sentence.
**Refs:** `docs/guides/reply-piece-labels.md`, `docs/plans/2026-10-05-reply-pieces.md`.

## 2026-10-07 — Evidence rules wait with the decision message

**Context:** A proposal put one speech act, STE wording, a glossary, and evidence tags on every reply. The piece score had already shown that one label on a whole reply collapses a statement, steps, a caveat, and a question into one blob, and that an imperative rewrite turns an options list into a fake procedure.
**Decision:** Piece kinds stay a closed list in code. The glossary holds project terms only. STE defines what a step, a statement, and a caveat look like, and does not rewrite the model's words. The next build is the parser's kind labeler. A decision-message envelope waits until such a message exists. When it does, these rules apply: one act on a piece or on that message; evidence tagged per claim; certainty taken from that tag; `assumed` cannot authorize an irreversible action; `refuse` and `failure` carry a reason; only a result no model produced can write `observed` (a test, a file read, a shell command), and a model-produced result is `reported` with the tool named as the source; model confidence may only add caution, and the check runs regardless. `observed` is what the tool returned. "The test passed" is `observed`. "The bug is fixed" is `inferred`. A named source on `reported` returns with the envelope, or `reported` is `assumed` under another name.
**Alternatives:** One act and STE imperatives on the whole reply. Treat any shell or test result as `observed`, including a command that calls a model. Let a classifier probability skip a check when it is high. Build the envelope before the kind labeler.
**Consequences:** Jev keeps the two seats it already has, cheap-or-smart and an added pause on a destructive command. It does not take context selection, compaction, knowledge extraction, or requirement verification.
**Refs:** `docs/guides/reply-piece-labels.md`, `docs/plans/2026-10-05-reply-pieces.md`.

## 2026-10-06 — Smart tier is GLM 5.3

**Context:** The smart tier was pinned to `z-ai/glm-5.2`. Z.ai has since released GLM 5.3, which OpenRouter describes as the newer model in that line. `~z-ai/glm-latest` currently redirects to GLM 5.3. GLM 5.3 Flash and GLM 5.3 Prime are separate variants.
**Decision:** Smart is the pinned slug `openrouter/z-ai/glm-5.3` in both the main config and the server checkout. Cheap stays on DeepSeek V4 Flash.
**Alternatives:** Stay on 5.2. Use `~z-ai/glm-latest`, which would move again without a config edit. Use 5.3 Prime, the faster variant, or 5.3 Flash.
**Consequences:** A smart turn costs GLM 5.3 rates. The running server keeps its old smart model until restart.
**Refs:** `config/default.toml` `[agent.routing]`.

## 2026-10-06 — Default model is DeepSeek V4 Flash

**Context:** Pi replies in the piece experiment were `deepseek/deepseek-v4-flash`. The director still loaded `deepseek/deepseek-v3.2` from `[model]`, `[agent].model`, `GRAPHIRM_MODEL`, and the server checkout's routing tiers.
**Decision:** Those settings use `deepseek/deepseek-v4-flash`. The cheap tier that already tracks `~deepseek/deepseek-v4-flash-latest` stays. Smart was still GLM 5.2 here; the next entry moves it to 5.3. This reverses the 2026-09-29 choice to leave the default on v3.2.
**Alternatives:** Leave the director on v3.2 and only pin Pi. Use the moving `~deepseek/deepseek-v4-flash-latest` alias for the default too.
**Consequences:** A running server keeps v3.2 until it is restarted. Routed turns on the main checkout still follow the cheap and smart lists.
**Refs:** `config/default.toml`, `GRAPHIRM_MODEL`.

## 2026-10-05 — The Cursor tiling check is ignored unless asked for

**Context:** The tiling stress test reads live transcripts from `~/.cursor/projects`. A silent pass looks like the check ran. An empty directory used to fail, which is a reason to commit transcripts so CI can see them.
**Decision:** Mark `cursor_transcripts_tile_when_present` with `#[ignore]`. A missing home directory, a missing projects directory, or no assistant text returns without failing. Untiled text still fails when the test is run with `--ignored` and transcripts are present.
**Alternatives:** Keep the early return and let cargo report a pass. Gate it on an environment variable inside the default test run.
**Consequences:** CI does not need Cursor transcripts. The local command is `cargo test -p graphirm-agent --lib cursor_transcripts_tile_when_present -- --ignored --nocapture`.
**Refs:** `crates/agent/src/pi_delegate/pieces.rs`.

## 2026-10-05 — The piece-version snapshot is a hand-checked fixture

**Context:** A hash-prefix sample of local Cursor transcripts was committed as `cursor-subset.jsonl`. That is a random sample of session text, not a scrub. Gitleaks found no keys. The replies still contained private code, paths, and project detail. The commit had not been pushed.
**Decision:** Replace the file with 35 hand-written replies: lists, fences, headings, tables, a quote, trailing questions, and seven texts over the 16,000-character cap. Regenerate the snapshot hashes. Leave `PIECE_PARSER_VERSION` and `PIECE_BASELINE_VERSION` at `"1"`, because the rules did not change. Drop the unpushed commit that contained the raw sample so it is not on the branch.
**Alternatives:** Scrub all 135 in place. Keep the raw file and rely on the scanner.
**Consequences:** A rule change still fails the snapshot test until the matching version and the snapshot hashes are updated together. A wording edit of the fixture moves the hashes and does not by itself require a version bump.
**Refs:** `crates/agent/tests/fixtures/pieces/README.md`.

## 2026-10-05 — Labels are keyed by byte range, not piece number

**Context:** A later splitter change renumbers pieces. A label that says "piece 3 is a caveat" would then point at a different block. Labeling 100 replies also will not finish in one sitting.
**Decision:** Each label row stores the run file, the segment index, a SHA-256 of the segment text, `parser_version`, `baseline_version`, and each piece's start and end. `graphirm label-pieces` skips a row already in `piece-labels.jsonl`. `graphirm score-pieces` matches on the byte range, counts a moved range separately from a kind error, and reports single-piece replies apart from replies with several pieces.
**Alternatives:** Key the file by piece order. Pre-fill the baseline and correct it.
**Consequences:** Bump `PIECE_PARSER_VERSION` when boundaries change and `PIECE_BASELINE_VERSION` when the kind rules change. A segment whose text hash no longer matches is not scored as a kind error.
**Refs:** `docs/guides/reply-piece-labels.md`.

## 2026-10-05 — Pi recordings are capped, and the first labels are blind

**Context:** Raw Pi stdout keeps thinking, tool arguments, and file contents. It is written on every production run. The baseline labeler will be scored against hand labels, and a suggested kind is easy to accept.
**Decision:** `~/.graphirm/pi-runs` stays at or below 256 MiB by deleting the oldest `*.jsonl` files. One run stops copying at 32 MiB. `graphirm label-pieces` does not show the baseline kind unless `--show-baseline` is passed. The kind definitions live in `docs/guides/reply-piece-labels.md`.
**Alternatives:** Leave the directory uncapped until the labeled set exists. Pre-fill the baseline kind and ask the person to correct it.
**Consequences:** A long run's tail is missing from the recording after 32 MiB. The first score is only as good as the blind labels.
**Refs:** `docs/plans/2026-10-05-reply-pieces.md`, `docs/guides/reply-piece-labels.md`.

## 2026-10-05 — The parser owns the reply text

**Context:** A finished Pi reply should become ordered pieces (statement, options, steps, instructions, example, caveat, code, question) without using the question. A 0.6B structure model was tried as a schema filler and copied schema words instead of the reply.
**Decision:** `pulldown-cmark` cuts the flattened assistant text and stores UTF-8 byte offsets. The character cap is checked first and counts Unicode scalar values, the same unit as the 16,000-character interaction cap. Narration (`stopReason: toolUse`) is parsed, and every block stays a statement except fences, which stay code. Final replies get a deterministic baseline labeler, including its known misses. Pi stdout is copied to `~/.graphirm/pi-runs` (override `GRAPHIRM_PI_RUNS_DIR`, `off` disables). Cursor transcripts may stress-test tiling from `~/.cursor` and are not committed.
**Alternatives:** Asking the answering model to emit the pieces while it writes. Asking Osmosis to fill a nested pieces schema. Parsing inside a Pi TypeScript extension. Storing same-turn adjacency as `applies_to`.
**Consequences:** Cross-turn edges (`answered_by`, `executed_by`, `implements`) wait until about 100 real Pi replies are hand-labeled and the baseline has a score. The grammar-constrained labeler has to beat that score. `adjacent_to` is not stored yet.
**Refs:** `docs/plans/2026-10-05-reply-pieces.md`.

## 2026-09-30 — Numbered blocks are the segment contract

**Context:** Marker splits only cut where the model wrote `<<<type>>>`. A grocery list came back as one answer plus one caveat, with the headings still inside the answer.
**Decision:** The prompt asks for `{"segments":[{"n","type","title","items"}]}` and nothing else. The parser stores one block per element. The chat prints `n` and numbers `items`. If the reply is markdown headings and lists instead, the same blocks are built in code. Markers stay a fallback when neither path yields blocks.
**Alternatives:** An API JSON schema would force the shape and still allow one block. A second model can miss the same cut.
**Consequences:** A list with no headings and no `items` array is still one block. The OpenRouter client still does not send `response_format`.
**Refs:** `parse_structured_segments` and `parse_markdown_sections` in `crates/agent/src/knowledge/segments.rs`.

## 2026-09-30 — Segment markers can split a reply in code

**Context:** Valid segment JSON with one `answer` still shows as one block. An API schema would force the shape and still allow that one block.
**Decision:** The question now asks for `<<<type>>>` markers, including a new `<<<answer>>>` before each heading or list section. Two or more known markers are split in code and stored as segments, ahead of a single JSON segment. One marker, or none, leaves the old JSON path in place. The system prompt still describes JSON and mentions markers as an alternative.
**Alternatives:** Retrying the model does not change the schema. A trained splitter needs labeled cuts we do not have.
**Consequences:** The model still has to write the markers. A grocery list with no markers stays one block.
**Refs:** `parse_marked_segments` in `crates/agent/src/knowledge/segments.rs`.

## 2026-09-30 — Segment JSON is also requested on the user question

**Context:** Structured segments are requested only in the system prompt. DeepSeek V4 Flash answered "list of groceries" with markdown. The parser stored that text. GLiNER2 labels spans; it does not emit the segment JSON, and this debug server does not run it.
**Decision:** When `structured_output` is on, append a one-line JSON request to the latest user message in the LLM context, after the tool gate has read the original text. The graph message is unchanged.
**Alternatives:** API `response_format` would force JSON, but the OpenRouter client does not send one. A second model rewrite has the same failure mode.
**Consequences:** A turn can still ignore the request. The chat will not show the suffix.
**Refs:** `crates/agent/src/knowledge/segments.rs` `append_segment_json_request`.

## 2026-09-29 — No automatic "continue the implementation" nudge

**Context:** After any `write` or `edit`, a text-only reply was treated as unfinished work. The loop stored "Continue with the implementation. What is the next step?" as a user message, up to twice, then the verification checklist. The system prompt said the same thing: never stop with text while work remains.
**Decision:** Delete that nudge and the `max_continuations` knob. Delete the Continuity section of the system prompt. A text-only reply after a write ends the loop, unless `pre_completion_verify` still injects its checklist once.
**Alternatives:** Hiding the nudge in the chat would still spend a model turn. Gating it on a real "unfinished" check needs a signal we do not have.
**Consequences:** The director can stop after editing. The verification checklist is unchanged.
**Refs:** `crates/agent/src/workflow.rs`, `config/default.toml`.

## 2026-09-29 — Cheap and smart are different OpenRouter models

**Context:** Both routing tiers were `deepseek/deepseek-v3.2`, so the chip's tier did not change which model answered. A greeting on turn 1 is forced to smart.
**Decision:** Cheap is OpenRouter `~deepseek/deepseek-v4-flash-latest` (the alias that tracks the newest DeepSeek V4 Flash). Smart is `z-ai/glm-5.2`. The default `[model]` / `[agent].model` stay on v3.2; routed turns use the tier list. Pi stays on the pinned `deepseek/deepseek-v4-flash`.
**Alternatives:** Pinning `deepseek/deepseek-v4-flash` would freeze the cheap tier on one checkpoint. `~deepseek/deepseek-flash-latest` is a different alias, not the V4 Flash family.
**Consequences:** A smart turn now costs GLM 5.2 rates and can behave differently from cheap. The running server must be restarted to load the toml.
**Refs:** `config/default.toml` `[agent.routing]`.

## 2026-09-29 — PageRank sums dangling rank once per pass

**Context:** A one-word chat turn timed out at 300s with no assistant node. Context build calls `GraphStore::pagerank` on the whole graph. 1,853 of 5,695 nodes have no outgoing edge, and each of those walked every node on each of 100 iterations (`HashMap` updates, debug binary). The model was never called. The turn timeout logged the failure and left that thread running.
**Decision:** Keep the same scores. Sum the dangling mass once per iteration and add `damping * sum / n` to every node. Neighbor lists are built once into dense slots.
**Alternatives:** Fewer iterations or skipping PageRank would change which context nodes are selected. Caching scores across turns would go stale on every write. Cancelling the blocking thread does not fix the next turn.
**Consequences:** A debug `serve` can build context on this graph in well under a second. The formula matches the old one, including parallel edges (each edge stays a separate outgoing slot).
**Refs:** `crates/graph/src/store.rs` `pagerank`, test `pagerank_dangling_nodes_stay_fast`.

## 2026-09-29 — Public lock-down is an env var, not the committed toml

**Context:** `app.graphirm.ai` loads `config/default.toml` from the image. That file now has `[agent.pi] enabled = true`, and `disable_bash` stays commented so local dev still has a shell. A Coolify rebuild would otherwise offer `bash` and `delegate_pi` on the public server, and the image has no `pi` binary.
**Decision:** `GRAPHIRM_DISABLE_BASH=true` (also `1` or `yes`) forces `disable_bash` after the toml is read. Unset leaves the file alone. The spoke sets the variable in Coolify; the committed flag stays off.
**Alternatives:** Uncommenting `disable_bash` in `default.toml` would lock local dev too. A second toml copied only in the image would drift from the file Coolify builds.
**Consequences:** A redeploy is safe only when the Coolify env is set before the new container starts. The variable does not change routing, context, or Pi itself.
**Refs:** `src/commands/mod.rs`, `config/default.toml`.

## 2026-09-29 — Phone-first chat design approved as written

**Context:** Track B B0 draft. The parent brief already locked the column, the four tabs, and the legacy switch.
**Decision:** Approve `docs/plans/2026-09-28-phone-first-decision-chat-design.md` with no row changed. Implementation plan is `docs/plans/2026-09-28-phone-first-decision-chat.md`.
**Alternatives:** None raised at approval.
**Consequences:** B1–B9 can start. Graph canvas stays mounted after first open. Review does not use `GET …/tasks`. Plan "Run" only resumes.
**Refs:** design doc Decisions table.

## 2026-09-28 — Session restore skips spawned Agents; `enabled = true` stays the product default

**Context:** Final review of Track A. `restore_sessions_from_graph` loaded every Agent node. A Pi Agent (`status = "running"`, no workspace) would become a promptable session after a spoke restart. Separately, `config/default.toml` is both the local default and the spoke file (`workspaces_root = /data/workspaces`), and A4.2 set `[agent.pi] enabled = true` while `disable_bash` stays commented out.

**Decision:** Skip any Agent with an incoming `SpawnedBy` edge (Pi and in-process subagents share that edge). Keep `enabled = true` — the live check passed and A4.2 required the flip. A public deploy of this file must set `disable_bash = true`, which already hides and refuses `delegate_pi`. Do not turn `disable_bash` on in the committed file; that would also disable `bash` on every local and spoke session.

**Alternatives:** Filter only `metadata.executor == "pi"` (misses in-process subagents). Revert `enabled = false` (undoes the approved A4.2). Set `disable_bash = true` in `default.toml` (breaks the coding agent wherever this file is used).

**Consequences:** Spoke restart no longer lists `"pi"` sessions. Deploying this branch to `app.graphirm.ai` without `disable_bash` leaves `delegate_pi` registered; it fails closed only while the image has no `pi` binary.

**Refs:** `crates/server/src/session.rs`, `config/default.toml`, design doc security row on `disable_bash`.

## 2026-09-28 — `delegate_pi`: only a turn-ending message is Pi's result; judge observe-only reuses `JudgeOutcome`; TUI lifts only `[agent.pi]`

**Context:** Assembling `PiDelegateTool` (plan A2.7) over the reviewed `process.rs` / `graph.rs`.
Three choices were not determined by the design doc.
**Decision:** (1) `last_assistant_text` is set only by an assistant `message_end` whose
`stopReason != "toolUse"`. Pi narrates before tool calls ("Let me check the directory…") as a
`toolUse` message; the plan's "nonzero exit *without a result* → failure" rule only works if that
narration does not count. Every assistant message is still recorded as a graph node. (2) The
observe-only judge verdict is written through the existing `hitl::JudgeOutcome::to_metadata()` with
`pause: false, action: JUDGE_ACTION_OBSERVED` ("observed"), so Pi tool nodes carry exactly the keys
the in-process gate writes (`version`, `p_irreversible`, `threshold`, `action`, `latency_ms`) and
read-outs can group by `action`. The judge is spawned at `tool_execution_start` and awaited at
`tool_execution_end` under `DestructiveJudge::timeout()`; one warning per run on failure; handles
without an end event are aborted. (3) `graphirm chat` has never read `config/default.toml`;
`chat.rs` now lifts only `pi` from the file so `[agent.pi].enabled` reaches the TUI without
changing its prompt/judge/routing. `serve.rs` loads config before tools via
`commands::load_agent_config()`. (4) `register_pi_delegate` is `async` (both callers are async);
a missing binary is a startup warning and the tool is still registered so the failure is visible
to the model. (5) `ToolError::Timeout(u64)` has no message slot; the timeout's partial summary
goes to `Task.metadata.failure_detail`.
**Alternatives:** Count any assistant text as the result — made `FAKE_PI_NO_END` + exit 2 a
success. A separate metadata builder for Pi verdicts — drifts from the gate's keys. Load the whole
TOML in the TUI — right long-term (backlog S·P2) but a behaviour change the task did not ask for.
`block_in_place` for a sync `register_pi_delegate` — panics on a current-thread runtime for no gain.
`ExecutionFailed("timed out …")` for the timeout — loses the typed variant the loop can match on.
**Consequences:** A Pi run whose final message is a `toolUse` turn (Pi died mid-turn) reports
"(Pi produced no final message)" on exit 0 and fails on nonzero exit. `JUDGE_ACTION_OBSERVED`
is part of the metadata contract A3.1 tests against. The TUI still ignores every other TOML
section until the backlog item lands.
**Refs:** `docs/plans/2026-09-28-pi-delegate-executor.md` A2.7 implementation notes;
`crates/agent/src/pi_delegate/tool.rs`; `crates/agent/tests/fixtures/pi/hello-run.jsonl`
(line 170 is the `stop` message; earlier assistant messages are `toolUse`).

## 2026-09-28 — Pi subprocess wrapper: drain to EOF, receiver-drop discards, handle+JoinHandle shape

**Context:** `delegate_pi` (plan A2.5) runs `pi --mode json` as a child and turns its JSONL stdout
into `PiEvent`s. Four things about Pi 0.85.1 shaped the wrapper: it auto-retries transient provider
errors (`agent_end {willRetry:true}` → `auto_retry_start` → `agent_start` …), so `agent_end` is not
a terminator; it parses a leading `@` in a positional argument as a file reference (`Error: File not
found`, exit 1); its `bash` tool can leave a backgrounded grandchild holding the stdout pipe after Pi
itself has exited; and a consumer that stops reading events would, through a bounded channel and a
full 64 KiB pipe, stall Pi until the deadline and misreport `Timeout`.
**Decision:** (1) Read stdout to EOF and then wait for exit; `AgentEnd` is forwarded like any other
event and never ends the read loop. (2) A dropped or closed event receiver flips the reader into
*discard* mode instead of stopping it, so Pi can never block on a full pipe; `PiRunHandle::wait()`
closes the receiver first — "wait" means "stop consuming". (3) The public shape is
`spawn_pi(spec, cancel) -> PiRunHandle { events: mpsc::Receiver<PiEvent>, done: JoinHandle<..> }`
rather than an async `on_event` callback; the tool layer (A2.7) awaits graph writes between
`recv()`s, and `process.rs` stays free of graph/judge concerns. Dropping the handle aborts the
driver; a `GroupKillGuard` armed in `spawn_pi` right after `child.id()` SIGKILLs the whole process
group when the driver future is dropped — armed at spawn, not in the driver, because a handle dropped
before the driver's first poll never runs driver code. (4) Tasks longer than 64 KiB *or* starting
with `@` are written to a `0600` `create_new` temp file and passed as
`-- @<path> "Carry out the task described in the attached file."`; the file is removed when the run
ends. (5) After exit, the pipes get `POST_EXIT_GRACE` (3 s, capped by the remaining deadline) to
close; on expiry the group is killed and the run still returns `Ok` with the exit code and
`pipes_lingered: true` — Pi *did* finish; a straggler is a warning, not a failure. (6) Pi's env is
the inherited env plus `PI_SKIP_VERSION_CHECK=1` minus `GRAPHIRM_API_KEY`; the provider key is Pi's
own business (`~/.pi/agent/auth.json` / provider env vars) and is never read or logged.
**Alternatives:** Stop reading at `agent_end` — wrong under auto-retry and races the exit. Stop the
reader when the receiver drops — deadlocks Pi on a full pipe; the "caller must cancel" contract was
too easy to violate (the reviewer hit it by calling `wait()` without draining). Async callback API —
forces graph I/O into the process layer and makes cancel/timeout tests awkward. Returning `Timeout`
when a straggler holds the pipe after a clean exit — loses the exit code for a condition Pi is not
responsible for. Passing `--fake-knob` env through `spawn_pi` — rejected; the fake reads knobs from
argv (`PiConfig.extra_args`) so `spawn_pi` never takes arbitrary env and tests stay parallel-safe.
**Consequences:** `PiRunHandle` has a `Drop` impl, so it cannot be destructured; consumers use
`handle.events.recv()` + `handle.wait()`. Under `pipes_lingered` the last stdout lines may be lost
(the reader is aborted); counters and the stderr tail survive via shared atomics/mutex. A pgid is
never signalled after the direct child has been reaped except through the explicit
`kill_group_and_reap` path (same pid-reuse caveat as `bash`). The `@` rule means a task that
legitimately starts with an `@mention` still works — via the file.
**Refs:** `docs/plans/2026-09-28-pi-delegate-executor-design.md` D1;
`docs/plans/2026-09-28-pi-delegate-executor.md` A2.5; `crates/agent/src/pi_delegate/process.rs`;
`crates/agent/tests/fixtures/pi/fake_pi.sh`; commits `266a892` and its review follow-up.

## 2026-09-28 — `bash` children run in their own process group and are SIGKILLed as a group on cancel/timeout

**Context:** `BashTool` cancelled by `task.abort()` on a `tokio::spawn`ed `wait_with_output`.
That drops the future but never signals the OS process: the `bash -c` shell and everything it
forked (`sleep 30`, `npm run dev &`, …) kept running after the agent gave up. Task A1.4 fixed
this now because `delegate_pi` (A2.5) needs the identical kill-on-cancel pattern.
**Decision:** Spawn the shell with `process_group(0)` + `kill_on_drop(true)`, `stdin(null)`;
capture `child.id()` immediately after `spawn()` as the pgid; on timeout, cancel, or an I/O error
from the reader, call `graphirm_tools::process::kill_group_and_reap(&mut child, pgid)` — an
explicit `libc::kill(-pgid, SIGKILL)`, then `start_kill()` as portable fallback, then `wait()`
under a 5 s reap timeout. `libc` is a new `[target.'cfg(unix)'.dependencies]` of
`graphirm-tools` (already in the lock file via tokio). The pgid is captured at spawn because
`child.id()` is `None` once the shell has been reaped — exactly the "shell exited, grandchild
still holds the pipe" case the group kill exists for.
**Alternatives:** (a) `start_kill()` alone — kills the shell but orphans grandchildren; proven
by `cancel_kills_the_shells_descendants`, which fails without the group kill. (b) Shelling out
to `kill -9 -- -<pgid>` (no `libc` dep) — first implementation; rejected because `kill(1)` is
absent from the `debian:bookworm-slim` runtime image (no `procps`), so the spawn failed with
ENOENT, was swallowed, and descendants still leaked in production. (c) Inherited stdin — with
the shell in a background process group a child reading the tty would get SIGTTIN and hang
until timeout; null stdin gives a deterministic EOF.
**Consequences:** bash children no longer receive the terminal's SIGINT/SIGHUP — acceptable
because the TUI is raw-mode (Ctrl-C is an event, not a signal) and `serve` cancels sessions on
SIGINT. `serve` still ignores SIGTERM, so a hard `pkill -f 'graphirm serve'` leaves live
sessions' children detached (backlog, S·P2). `process.rs` is the reference implementation for
`delegate_pi`'s kill path; A2.5 reuses it instead of adding `libc` to `graphirm-agent`.
**Refs:** `docs/plans/2026-09-28-pi-delegate-executor.md` Task A1.4; `crates/tools/src/process.rs`,
`crates/tools/src/bash.rs`; commits `136c7af` (kill(1) version) and its follow-up.

## 2026-09-28 — Pi becomes the delegated coding executor; graphirm stays the director

**Context:** Graphirm has the control plane (Jev routing in shadow, HITL judge, graph memory,
pinned rules) but its in-process `delegate` is unreachable from the server and TUI, and the
agent loop's own coding ability is what `graphirm-eval` measures. Codeporate already drives Pi
(`@earendil-works/pi-coding-agent`) via `--mode json` with a Python event mapper. A director
without a strong executor and an executor without a control plane are complementary.

**Decision:** Pi is the only external executor, run as one subprocess per delegation in
`--mode json`, observe-only (its tool calls are recorded and judged, not paused). Its work is
graph-native: Task + Pi Agent + `Interaction{role:"tool"}` nodes in the same shape in-process
tools produce. Live visibility comes from a new optional `ToolEventSink` on `ToolContext`.
Design with 15 sub-decisions: `docs/plans/2026-09-28-pi-delegate-executor-design.md`.

**Alternatives:** Hermes as executor — rejected: it is a second director with its own memory
and routing, leaving Jev no seat. Gating Pi's calls in v1 via `--mode rpc` or a Pi extension —
deferred: observe first, keep the parser process-agnostic so the switch is additive.
`GraphUpdate`-on-insert instead of a sink — rejected: the `EventBus` is per prompt and the
graph store has no channel, so there is nothing to emit from.

**Consequences:** `ToolContext` gains a field (21 mechanical edits). A new `[agent.pi]` config
block, disabled until the A4 live check passes. Track B (phone-first chat) is designed only
after A4. Nothing in context selection, compaction, memory ranking, or knowledge extraction
changes.

**Refs:** design doc above; backlog "Director / Pi executor"; Jev seat scoring
`~/codeporate-connect/docs/evaluations/2026-09-27-jev-where-in-graphirm.md`.

## 2026-09-28 — Governance docs must be updated in the same commit as the work

**Context:** Audit found the scaffolding (root + 7 scoped `AGENTS.md`, 3 Cursor rules, hooks,
1,500-line backlog, completion log) was mature but drifting: `backlog.md` header said
"Phases 0–35" while `AGENTS.md` said 0–55; `docs/README.md` was stuck at Phase 12;
`CHANGELOG.md` frozen since 2026-03-08; ~60% of `backlog.md` was completed items despite the
file's own convention to move them out. Root cause: no rule, hook, or skill obliged anyone to
update these files. The only doc-update obligation (`000-skills-first.mdc` → update
`00-execution-strategy.md`) pointed at a document abandoned in March. No decision journal existed.

**Decision:** Extend `.cursor/rules/000-skills-first.mdc` with a "Governance docs" block —
plan link on start, ✅ + completion-log entry on ship, journal entry on non-obvious decisions,
backlog entry for discovered-but-unstarted work — all in the same commit as the work. Drop the
`00-execution-strategy.md` line. Create this file. Add a pointer block to `AGENTS.md` because
`.cursor/` is gitignored and only `AGENTS.md` reaches the spoke and other clones.

**Alternatives:** (a) A separate `200-governance.mdc` rule — rejected; the existing rule already
owns "commit and progress rules", and a fourth always-apply rule adds context cost for no
isolation benefit. (b) A pre-commit hook that fails when `crates/` changes without a
`docs/` change — deferred; too blunt for refactors and hotfixes, revisit if drift recurs.
(c) ADR directory (`docs/decisions/NNNN-*.md`) instead of a single journal — rejected for now;
one file is easier to grep and append to at the current decision rate.

**Consequences:** Every agent session that ships anything now touches at least two docs. The
known drift (backlog header, `docs/README.md`, `CHANGELOG.md`, stale `src/main.rs` pointers in
`AGENTS.md`) is **not** fixed by this entry — it is separate cleanup work.

**Refs:** `.cursor/rules/000-skills-first.mdc`, `AGENTS.md` → Governance docs, commit `85ffc1a`
(same day: phase table moved out of `AGENTS.md` into `completion-log.md`).
