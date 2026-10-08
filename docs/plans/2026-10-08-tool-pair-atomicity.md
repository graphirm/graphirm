# Tool-pair atomicity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** A context window never contains a tool result without its call, or a call without its result, and a request that cannot fit the selected model's window is cut or refused before the provider sees it.

**Architecture:** Phase 1 is a pairing pass during assembly in `build_context`, while each graph node is still in hand. It drops orphans and logs the node id. Phase 2 groups a tool exchange into one budget unit, counts the reserved tail in units, and compacts whole exchanges. The provider-window check is separate and later: it runs in `stream_and_record` after the router picks a model, and again on every fallback switch. Phase 1 does not wait on any of that.

**Tech Stack:** Rust. Tests are `#[cfg(test)]` in `crates/agent/src/context.rs`, `crates/agent/src/compact.rs`, `crates/agent/src/workflow.rs`, and `crates/llm`. Token estimates stay the existing word heuristic (`words / 0.75`).

**Key decisions:**

- A tool exchange is one unit. Score is the max of its members. Cost is the sum of their tokens.
- `guaranteed_recent_turns` keeps its name and, from Phase 2, counts units. `tail_max_fraction` defaults to 0.5.
- The newest unit is sent whole when it exceeds the knapsack budget. It is not sent whole when it exceeds the provider window.
- `context_window(model) -> Option<usize>` lives on `LlmProvider`. An optional config map overrides per model. `None` from both means no pre-send cut.
- `LlmError::ContextLength` is not `is_retryable`. One recut to 70% on the same model handles it. Model fallback is a different path and re-runs the size check.
- Truncation rewrites the payload copy only. The graph node stays as stored.
- Out of scope: PageRank caching, moving the four top-level weights into config, and the gateless-startup line.

**Success:** `assert_tool_pairing` holds on every window `build_context` returns. T1–T7 pass. After the window work, T8–T11 pass. `cargo test -p graphirm-agent` and `cargo test -p graphirm-llm` pass. `graphirm-eval` task outcomes do not regress.

**Risk:** The word heuristic undercounts code, JSON, and paths. The 85% target and the one 70% recut are the guards. A provider that reports context length as a plain string, and is not mapped to `ContextLength`, still falls through to model fallback with the same payload.

---

## Problem

`build_context` places Content and Knowledge first. It then reserves the newest 4 interaction nodes on the `RespondsTo` chain, and fills the rest one node at a time by score. Three things follow:

1. If the tail starts on a tool result, that result always ships, even when its call falls outside the tail and loses on budget.
2. The older fill can keep a tool call and drop its result, or the reverse.
3. A result with an empty `tool_call_id` is logged in `node_to_message` and then sent anyway. Anthropic rejects an empty id.

Compaction has the same flaw. `select_nodes_for_compaction` takes older nodes one at a time, and the next `build_context` omits anything marked compacted, so a surviving half becomes an orphan.

## Pairing rule

A contiguous run of tool-result messages belongs to the single assistant message just before the run. The ids in that assistant message's tool-call parts must equal the ids in the run.

`assert_tool_pairing` uses this run rule. The old rule ("the assistant message directly before this result") fails every later result of a parallel call. Do not implement that.

`tool_call_id` decides which result belongs to which call. `RespondsTo` only checks contiguity: the first result points at the assistant, and each later result points at the result before it. Membership is not read off `RespondsTo`, because only the first result's edge target is the assistant.

## Phase 1 — stopgap

Ship this before any window work. It depends only on the pairing rule.

Run the check during assembly in `crates/agent/src/context.rs`, while each graph node is still in hand. `LlmMessage` has no node id, so the pass cannot log a useful drop after conversion.

Drop:

- a result whose `tool_call_id` is empty
- a result whose id is not in the assistant message that owns the run
- a tool-call part with no result in the run
- an `awaiting approval` placeholder is payload-only. It is not stored. When a real result with the same id is already in the assembled nodes, the placeholder is left out of the payload too.

If an assistant message then has no text and no tool-call parts, drop the message.

Log each drop once at `warn`, with the node id. Delete the empty-id warning in `node_to_message` (the `tracing::warn!` that emits an empty id). After Phase 2 this pass stays as a safety net. A warning from it means the grouping is wrong.

## Phase 2 — units and tail

A context unit is either one interaction node or one tool exchange. A tool exchange is an assistant node that has tool calls, plus every result node whose id matches, including parallel calls.

`fit_to_budget` keeps or drops whole units, then emits them in chronological order. It never splits a unit.

`select_nodes_for_compaction` uses the same units. An exchange is compacted entirely or not at all. Otherwise the next turn recreates the orphan and Phase 1 drops the surviving half.

`guaranteed_recent_turns` counts units, so the tail cannot start mid-exchange. Document that change on the field. Do not rename it.

Add `tail_max_fraction` to `ContextConfig`, default `0.5`. Walk back from the newest unit and add units while the tail fits under that fraction of the context budget (`AgentConfig.max_tokens`, the knapsack budget, default 8192).

The newest unit always goes in whole when it exceeds that budget, including when it exceeds the whole budget. Splitting it is the failure this plan prevents. Compaction does not produce a payload for the current turn. The provider-window section below is the only case where that unit is cut or the turn is failed.

Knowledge and content stay capped at `max_content_nodes` (default 20) and compete with older units for the budget the tail did not take.

## Phase 2 — provider window

This is not part of the Phase 1 pull request.

### Where the window lives

Add `fn context_window(&self, model: &str) -> Option<usize>` to `LlmProvider` in `crates/llm/src/provider.rs`. Implement it on Anthropic, OpenAI, DeepSeek, OpenRouter, Ollama, and `MockProvider`.

The window depends on the model. Routing in `stream_and_record` can switch models after `build_context` returns, so a single `AgentConfig` field would be stale. Read the window after `select`, with the model just chosen.

Optional override: a map from model id to window, not one number. Look up the selected model in the map first. If the map has no entry, use the trait method.

`None` from both is a real answer. Ollama's window depends on the server's `num_ctx`, not only the model name, and some OpenRouter models report the window unreliably. When both return `None`, skip pre-send truncation. Rely on `LlmError::ContextLength` plus one recut. A guessed number either cuts for no reason or lets an oversized request through.

### What the check compares

Compare the whole request, not the unit alone:

- system prompt
- tool schemas
- selected messages
- reserved output tokens (`max_output_tokens`, falling back as `stream_and_record` already does)

The provider window covers input and output together. A payload that "fits" without the output reserve is still rejected, and the reply comes back truncated.

Target about 85% of the window, not 100%. Words divided by 0.75 undercounts code, JSON, and paths, which is what tool output is.

The check lives in `stream_and_record`, after the router picks the model, on the assembled request. It does not live inside `build_context`.

On every model switch, including a rate-limit fallback onto a smaller window, re-run the size check against the new model's window, recut if needed, then send. Do not reuse the previous model's cut.

### ContextLength

Add `LlmError::ContextLength` in `crates/llm/src/error.rs`. It is a prerequisite. Without it the recut cannot be told apart from any other provider error.

Each provider maps its own error into that variant:

- Anthropic: HTTP 400 whose message says the prompt is too long
- OpenAI and OpenRouter: `context_length_exceeded`
- Ollama and DeepSeek: the context-length error those APIs actually return, confirmed against a live or recorded response before the mapping is committed

`ContextLength` is not `is_retryable`. The existing fallback loop sends the same payload to the next model. Only the recut path handles `ContextLength`.

If the provider still returns `ContextLength` after the 85% cut, retry once against the same model with a tighter cut, about 70% of the window, then fail the turn. That retry is not a fallback attempt.

### How to cut

Truncate the payload copy only. Leave the stored graph node unchanged.

Shrink the largest tool result first, then the next, until the full-request estimate is under the target. Keep roughly 20% of the kept text at the head and 80% at the tail, with the marker at the cut. Tail-only is right for test and build output and wrong for `read`, `grep`, and `ls`. The marker says the count is a word estimate, not a provider token count. Log each truncation once at `warn`, with the node id.

If the assistant text alone exceeds the window, do not send the unit. Fail the turn with a named error that includes the node id, and do not call the provider. "Send whole" applies to the knapsack budget, not to the hard window.

If the fixed overhead alone (system prompt, tool schemas, and reserved output) exceeds 85% of the window, cutting a result cannot make the request fit. Fail the turn with a named error and do not call the provider.

## Acceptance checks

`assert_tool_pairing(messages)` applies the run rule.

| Test | Setup | Expected |
|------|--------|----------|
| T1 | The 4-node tail starts on a result, and the budget cannot buy back its call | Pairing holds. Phase 1 drops the result. |
| T2 | A result with an empty id | Dropped, logged once. The old `node_to_message` warning is gone. |
| T3 | Older fill keeps a call and drops its result | No unmatched tool call remains. |
| T4 | T1's setup under Phase 2 | The exchange is fully in or fully out. The tail does not begin mid-exchange. |
| T5 | Parallel calls, three results | All results stay with their call. |
| T6 | Four huge units in the tail | At least `(1 − 0.5) × budget` remains for the rest, unless the newest unit alone exceeds the budget. |
| T7 | Compaction over a tool exchange | The whole exchange is compacted, or none of it is. |
| T8 | The full request exceeds 85% of the selected model's window | The result keeps its id. The marker is present and says the count is a word estimate. The full-request estimate is under 85% of that window. The graph node is unchanged. |
| T9 | The assistant text alone exceeds the window | A named error includes the node id. `complete` and `stream` are not called. |
| T10 | System prompt, tool schemas, and `max_output_tokens` alone exceed 85% of the window | A named error. No provider call. |
| T11 | A fallback switches to a model with a smaller window | The request is re-checked and recut for that model before send. The original model's cut is not reused. |
| Eval | `graphirm-eval` | No regression in task outcomes. |

## Shipping order

Phase 1 (Task 1) is this session. Phase 2 is a fresh session against this plan, one checkpoint at a time, with a review between checkpoints. Checkpoints 3 and 4 do not depend on 1 and 2. Run them one at a time anyway. After checkpoint 2, run `graphirm-eval` and confirm task outcomes do not regress before any window work.

### Phase 2 checkpoints

1. Units, the unit-based tail, and `tail_max_fraction` (T4–T6).
2. Compaction on the same units (T7). Then `graphirm-eval`.
3. `LlmError::ContextLength` and one mapping per provider. It is not `is_retryable`.
4. `context_window`, the per-model override map, and the size check before every send, including after a model switch (T10, T11).
5. Payload-copy truncation, and the named error when assistant text alone exceeds the window (T8, T9).

### Task 1: Phase 1 pairing pass

**Files:**

- Modify: `crates/agent/src/context.rs` (`node_to_message`, `build_context_with_stats`)
- Test: `crates/agent/src/context.rs`

**Steps:**

1. Write failing tests for T1, T2, T3, and for an intact parallel run (three results already adjacent to their assistant). The parallel test must pass the run rule and must fail if the check uses "the message directly before."
2. Run `cargo test -p graphirm-agent context::` and confirm the new tests fail.
3. Implement the assembly pass and delete the empty-id warning in `node_to_message`.
4. Run the same tests and confirm they pass.
5. Commit the Phase 1 change on its own. Do not include the window work.

### Task 2: Units, tail, compaction

**Files:**

- Modify: `crates/agent/src/context.rs` (`fit_to_budget`, `ContextConfig`, `build_context_with_stats`)
- Modify: `crates/agent/src/compact.rs` (`select_nodes_for_compaction`)
- Test: `crates/agent/src/context.rs`, `crates/agent/src/compact.rs`

**Steps:**

1. Write failing tests for T4, T5, T6, and T7.
2. Run them and confirm they fail.
3. Group exchanges, score by max, cost by sum, count the tail in units, add `tail_max_fraction = 0.5`, and select compaction candidates as units.
4. Run `cargo test -p graphirm-agent context:: compact::` and confirm they pass.
5. Commit.

### Task 3: Provider window

**Files:**

- Modify: `crates/llm/src/error.rs`
- Modify: `crates/llm/src/provider.rs`
- Modify: `crates/llm/src/anthropic.rs`, `openai.rs`, `openrouter.rs`, `deepseek.rs`, `ollama.rs`, `mock.rs`
- Modify: `crates/agent/src/workflow.rs` (`stream_and_record`)
- Modify: `crates/agent/src/config.rs` and `config/default.toml` for the per-model override map
- Test: `crates/llm/src/error.rs`, `crates/agent/src/workflow.rs`

**Steps:**

1. Write failing tests for T8, T9, T10, and T11, plus `ContextLength` excluded from `is_retryable`.
2. Run them and confirm they fail.
3. Add the error variant and the provider mappings. Add `context_window`. Run the size check after `select` and again on each fallback model, then the same-model 70% recut.
4. Run `cargo test -p graphirm-llm` and `cargo test -p graphirm-agent workflow::`.
5. Commit.

### Task 4: Eval

Run `graphirm-eval` on the coding suite against a server built from this branch. Task outcomes do not regress. Record the result in the pull request.
