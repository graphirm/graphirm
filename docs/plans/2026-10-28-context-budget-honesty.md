# Context budget honesty Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** The context budget is counted in real tokens, a single tool output cannot flood the window, and every LLM call records exactly which nodes it was sent, so the T2 memory result can be rerun without the measurement bugs that may have caused it.

**Architecture:** Three independent changes, one commit each. (1) `estimate_tokens_str` counts with a BPE tokenizer instead of `words / 0.75`. (2) Tool results and Content bodies are cut to a head and a tail when they become LLM messages; the graph keeps the full text, and the token estimate counts the capped text. (3) The assistant node of each call stores the ordered ids that were packed, a hash of the full request, and the estimated token count; `replay-pack` uses the stored ids as the prefix chain when they exist.

**Tech Stack:** Rust. New dependency `tiktoken-rs` in `graphirm-agent`. Tests are `#[cfg(test)]` in `crates/agent/src/context.rs`, `crates/agent/src/workflow.rs`, and `src/commands/replay_pack.rs`.

**Key decisions:**

- Tokenizer: `tiktoken-rs` `o200k_base`, process-wide singleton, counts for texts of 256 bytes or more cached by content hash. It is a stand-in, not DeepSeek's tokenizer. See the 2026-10-28 journal entry.
- Cap: `WINDOW_OUTPUT_MAX_BYTES = 20_000` (about 5k tokens). First and last 10k bytes, cut at line breaks, with a notice that names the omitted bytes and lines. Applies to `tool` Interaction nodes and Content nodes only. User, assistant and Knowledge text is sent as stored.
- The cap is applied when a node becomes a message, not in the tools. The graph keeps the full output for audit and for `graph_query`.
- Stored per call on the assistant node: `context_kept_node_ids` (ordered), `context_sha256` (system, tool schemas and messages, hashed the same way as `GRAPHIRM_PACK_TRACE`), `context_tokens_est`. `usage_input` is already stored, so the estimate can be checked against the provider's count with one query.
- Out of scope: rebuilding a window from stored ids alone (the `recorded` mode in `reconstruct_check.py`), provider-window truncation (open in `2026-10-08-tool-pair-atomicity.md`), any change to scoring, compaction thresholds or memory.

**Success:**

- `cargo test -p graphirm-agent` and `cargo test` pass; `cargo clippy` and `cargo fmt --check` are clean.
- On a new run, the median of `context_tokens_est / usage_input` over assistant nodes is within 0.85–1.15. The old estimate gives roughly 0.5 on code.
- No message sent to the provider contains a tool result or Content body longer than the cap plus the notice.
- Sequential `replay-pack` on a graph written by this build reproduces `context_kept_node_ids` for every call.

**Risk:**

- With a real count, the same `max_tokens` selects about half as much context as before. That is the intended fix, but every earlier result that depended on the budget was measured under the old count. Reruns must say which build they used.
- Tokenizing every candidate on every turn costs CPU. The cache keyed by content hash covers repeated nodes. If turn latency rises noticeably, measure before optimising further.
- A model that needs a line from the middle of a long output has to rerun the command narrower or use `read` with `offset`/`limit`. The notice says so.

---

## Task 1 — Real token counting

**Files:** `crates/agent/Cargo.toml`, `crates/agent/src/context.rs`.

- [ ] Add `tiktoken-rs` to `graphirm-agent`.
- [ ] Replace the body of `estimate_tokens_str` with an `o200k_base` count. Cache counts for long texts by hash; clear the cache past 100k entries.
- [ ] Update the tests that assert word-based numbers. New tests: empty string is 0; a known short string has the o200k count; a code sample counts at least 1.5× the old word estimate.
- [ ] `cargo test -p graphirm-agent`; fix every test that relied on the old numbers by asserting relationships rather than new magic numbers where possible.
- [ ] Commit.

## Task 2 — Cap tool output in the window

**Files:** `crates/agent/src/context.rs`.

- [ ] Add `WINDOW_OUTPUT_MAX_BYTES`, `cap_window_text(&str) -> Cow<str>` and `window_text(&GraphNode) -> Cow<str>`.
- [ ] `node_to_message` uses the capped text for `tool` results and Content bodies. `estimate_tokens` counts `window_text`.
- [ ] Tests: text at the limit is unchanged; a 1 MB output becomes head + notice + tail under the limit plus the notice; the cut never splits a UTF-8 character; the notice counts the omitted bytes and lines; the graph node is unchanged after `build_context`; `estimate_tokens` of a huge tool node is bounded.
- [ ] Commit.

## Task 3 — Persist the sent context per call

**Files:** `crates/agent/src/workflow.rs`, `src/commands/replay_pack.rs`.

- [ ] Keep `window.kept_node_ids` and `window.total_tokens` until the response node is written.
- [ ] Hash the final request (system, tool schemas, messages) right where `append_pack_trace` runs, after the budget note and segment reminder are added.
- [ ] Write `context_kept_node_ids`, `context_sha256`, `context_tokens_est` into the assistant node's metadata, next to `context_dropped_node_ids`.
- [ ] `replay-pack` sequential mode: when the previous call has `context_kept_node_ids`, use it as `pack_prefix_ids`; otherwise keep the current replayed chain.
- [ ] Tests: a mock-provider turn writes all three keys; the stored ids equal the window's kept ids; two identical requests give the same hash and a changed budget note changes it.
- [ ] Commit.

## Task 4 — Review and close

- [ ] Spec review subagent: does the code match this plan?
- [ ] Code-quality review subagent, after the spec review passes.
- [ ] Fix findings, commit.
- [ ] Governance: `### ✅` on the backlog item, entry at the top of `docs/completion-log.md`, update `crates/agent/AGENTS.md` (token counting, output cap, stored pack keys).
- [ ] Rebuild the musl binary and write the T2 rerun steps (below) into this plan.

## T2 rerun

Filled in at Task 4.
