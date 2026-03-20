# Dogfood Findings

Cursor evaluation log for graphirm agent sessions. One row per completed task.

| Date | Session ID | Task | Grade | Finding |
|------|-----------|------|-------|---------|
| 2026-03-20 | 6cb219fe | message-count-field | partial | Struct field correct; session failed mid-run — agent used `cd` in bash (state not persisted) and was misled by Cursor worktree paths in ls output |
| 2026-03-20 | 94ce01fd | tool-call-count-field | partial | All code correct and compiles; session failed at context overflow (273k > 262k) — agent read full routes.rs (2000+ lines) multiple times instead of grep → targeted read |
| 2026-03-20 | 0b1fde7d | health-session-count | partial | All code correct and compiles (incl. test update); context overflow again — Qwen ignores grep-first system prompt instruction; fix is to provide exact line numbers in task prompt |
| 2026-03-20 | db0590a9 | health-session-count-v2 | pass | First full pass — agent used grep→read(offset+limit) pattern, 11 tool calls, no context overflow, compiled + tests passed. Key fix: documented offset/limit in system prompt tool description |
