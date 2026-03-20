# Dogfood Findings

Cursor evaluation log for graphirm agent sessions. One row per completed task.

| Date | Session ID | Task | Grade | Finding |
|------|-----------|------|-------|---------|
| 2026-03-20 | 6cb219fe | message-count-field | partial | Struct field correct; session failed mid-run — agent used `cd` in bash (state not persisted) and was misled by Cursor worktree paths in ls output |
| 2026-03-20 | 94ce01fd | tool-call-count-field | partial | All code correct and compiles; session failed at context overflow (273k > 262k) — agent read full routes.rs (2000+ lines) multiple times instead of grep → targeted read |
| 2026-03-20 | 0b1fde7d | health-session-count | partial | All code correct and compiles (incl. test update); context overflow again — Qwen ignores grep-first system prompt instruction; fix is to provide exact line numbers in task prompt |
| 2026-03-20 | db0590a9 | health-session-count-v2 | pass | First full pass — agent used grep→read(offset+limit) pattern, 11 tool calls, no context overflow, compiled + tests passed. Key fix: documented offset/limit in system prompt tool description |
| 2026-03-20 | 5b997cfb | planning-layer-project-mode | partial | All 4 actions implemented correctly (create/list/link_session/update), graph model understood well. Failed on Rust dep management (added petgraph 0.6 when graph crate re-exports 0.7) and ownership in spawn_blocking closures. No tests written — context exhausted on compile error loop. Fix: document re-exports in system prompt + add spawn_blocking clone pattern |
| 2026-03-20 | f7b269d3 | project-mode-tests | pass | All 5 tests written correctly (create, create_with_parent, list, link_session, update). Used graphirm_graph re-exports — no petgraph/chrono added. Proper Arc cloning in spawn_blocking. Fixed String borrow issues independently. 18 tool calls. System prompt crate dep + async pattern sections validated |
| 2026-03-20 | 8de099f4 | cargo-check-tool | partial | Tool structure excellent — correct trait impl, registration, 9 tests, clippy/fmt clean, used re-exports. Two critical bugs: (1) early return on non-zero exit code meant errors never parsed; (2) JSON deserialization expected top-level fields but cargo nests diagnostics inside `message` object. Tests only covered clean path. 43 tool calls. Agent self-fixed clippy/fmt issues. Fix: need to teach cargo JSON envelope format |
