# Record approval source and session end reason

**Backlog:** "Record how each destructive call was approved — P1 · S" and "Record why a session ended — P2 · S"
in `docs/backlog.md`, both found by check 0.4 (`docs/plans/2026-11-07-structure-completeness.md`).

## Fix 1: approval source

Every `write` / `edit` / `bash` result (and any registry-destructive tool) records `approval`:

| Value | When |
|---|---|
| `human` | A human approved or rejected the call at the HITL gate |
| `human_modified` | A human approved it with changed arguments |
| `auto_approve` | Per-session auto-approve ran it without asking |
| `gate_dropped` | The gate closed without a decision; the call was rejected |
| `no_gate` | The session had no HITL gate (headless runs) |

Written on the `tool` Interaction node, on its `ApprovedBy` edge, and on the `tool_rejection` node and its
`RejectedBy` edge. Code: `ApprovalSource` in `crates/agent/src/workflow.rs`.

## Fix 2: session end reason

The Agent node status already recorded `completed`, `cancelled`, `error`, `limit_reached` and
`token_cap_exceeded`. Missing: runs that end because the process dies.

- `run_agent_loop` sets the status to `active` at the start of every run, so a later prompt on a finished
  session is also visible while it runs.
- `graphirm serve` marks every user session still `active` as `interrupted` (with `interrupted_at`) at
  startup and after shutdown. Shutdown now also listens for SIGTERM (`docker stop`, Coolify).
- Restore never reports a session as `Running`: nothing runs at startup, and a `Running` handle refused new
  prompts, so killed, cancelled or errored sessions came back unpromptable. Unknown statuses restore `Idle`.

Limit: a process killed with SIGKILL cannot mark itself; its sessions are marked at the next `serve` start.
Eval databases that are never reopened by `serve` keep `active`.

## Tasks

- [x] `ApprovalSource` and `approval` on results, edges and rejections; tests for human, auto-approve,
      rejection, no gate, and the mapping
- [x] `mark_interrupted_sessions`, startup and shutdown calls, SIGTERM, restore mapping; tests
- [x] `cargo fmt`, `cargo clippy --workspace --all-targets` (clean), `cargo test --workspace` (1273 pass)
- [ ] Rerun 0.4 on sessions from a binary with these fixes
