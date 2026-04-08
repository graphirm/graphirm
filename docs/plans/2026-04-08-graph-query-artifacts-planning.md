# graph_query / project: Link Artifacts to Planning (P1) — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend planning ↔ **artifact** traceability beyond **file** `Content` nodes so **delegated work (Task)** and optionally other durable artifacts can be linked to planning Knowledge using the same **`relates_to` + `artifact_link`** conventions as `link_content`, with **`graph_query` `project`** support and web-app visibility in plan-focused views.

**Architecture:** Reuse the model proven in `docs/plans/2026-04-06-planning-code-linkage.md`: **outgoing `RelatesTo`** from **planning Knowledge → artifact**, **`metadata.artifact_link`**: `implements` | `documents`. This matches `crates/tools/src/planning_link.rs` (`link_planning_content_edge`) and the web-app’s `computePlanGraphAllowedIds` + `LabelledEdge` (which key off **`relates_to`** + **`artifact_link`**). Add a parallel **`link_planning_task_edge`** (or generalized helper) for **`NodeType::Task`**, with **session/delegation-scoped validation** so arbitrary tasks cannot be linked across unrelated sessions. Optional follow-up: configurable **auto-link** for Task creation when the parent session is already **`link_session`**’d (mirror `auto_link_write_to_planning`).

**Tech Stack:** Rust (`graphirm-tools` / `planning_link`, `graph_query`), optional `graphirm-agent` (`multi.rs` / delegate path), TypeScript (`web-app` plan-graph helpers).

**Key decisions:**

- **Edge shape:** Prefer **`RelatesTo` planning → Task** + **`artifact_link`** for parity with file artifacts and **zero new edge-type branches** in `LabelledEdge.tsx`. The planning-layer design doc (`docs/plans/2026-03-20-planning-layer-design.md`) mentioned **`DerivedFrom` Task → planning**; that conflicts with the shipped Content model. **Choose one canonical story** in Task 1: either update the older doc to match `RelatesTo` + `artifact_link`, or extend the web-app to style **`derived_from`** for planning↔Task (more work). **Recommendation:** `RelatesTo` planning → Task + `artifact_link`, same as Content.

- **Validation:** `Content` links require `metadata.session_id == agent_id` (see `link_planning_content_edge`). **Task** nodes today may not carry `session_id` in metadata (`crates/agent/src/multi.rs`). **Either** (a) stamp `metadata.session_id` on Task nodes at creation (parent or child agent id — document which), **or** (b) validate via graph structure: e.g. **`DelegatesTo`(parent_agent → task)** when the caller is the parent, or **`SpawnedBy`(task → child_agent)** when the caller is the subagent. (b) avoids redundant metadata if edge checks are reliable.

- **Scope v1:** **Task + `graph_query` `link_task`** is the core P1 deliverable. Non-file **Content** types (`impact_brief`, segment `Content`, etc.) can keep using manual **`link_content`** today; optional **`auto_link_content_types`** config is a stretch goal, not required for v1.

---

## Relationship to existing work

| Piece | Status |
|-------|--------|
| `graph_query` `project` `link_content` | Shipped |
| `auto_link_write_to_planning` for **`file`** writes/edits | Shipped (`planning_link::try_auto_link_written_file_content`) |
| Web: dashed **`relates_to`** + **`artifact_link`** label | Shipped |
| **Task** / other artifacts ↔ planning | **This plan** |

---

## Success criteria

- [x] `graph_query` `project` exposes a **`link_task`** action (name bikeshed OK: `link_task` vs `link_artifact` with `target_kind`) with JSON schema documented in the tool description.
- [x] Shared Rust helper validates planning node + Task + scoping; idempotent like `link_content`.
- [x] Unit tests in `planning_link` + `graph_query` cover happy path, idempotency, and invalid cases.
- [x] Web-app **plan graph filter** (`planGraphOnly`) includes **Task** nodes reachable via planning **`relates_to`** with **`artifact_link`** (extend `computePlanGraphAllowedIds` in `web-app/src/hooks/useGraphData.ts` if Task targets are not already included — today only Content targets are pulled in via `relates_to`).
- [x] `AGENTS.md` (Planning ↔ files section) updated to mention Task linkage; `docs/backlog.md` P1 item closed or narrowed to “optional auto-link / extra content types”.
- [x] Optional **Task 7:** auto-link delegated Task when `auto_link_write_to_planning` + `link_session` (`multi.rs`).

---

## Risks

- **Ambiguous session ownership** for Task (parent vs subagent): validation must be explicit and tested; wrong rules could link foreign tasks to a plan.
- **Metadata vs edges:** Prefer graph-native validation (edges) where possible so restored DBs behave consistently.

---

## Dependency order

```text
Task 1 (design + validation rules) → Task 2 (planning_link helper)
  → Task 3 (graph_query link_task) → Task 4 (tests)
  → Task 5 (web-app allowed-ids) → Task 6 (docs/backlog)
```

Optional **Task 7:** auto-link Task on `spawn_subagent` / delegate when `resolve_session_planning_node(parent_agent_id)` is `Some`.

---

### Task 1: Lock validation rules and edge direction

**Files:**

- Read: `docs/plans/2026-03-20-planning-layer-design.md` (Task ↔ planning prose)
- Read: `crates/tools/src/planning_link.rs`, `crates/agent/src/multi.rs` (Task creation)

**Deliverable:** Short subsection **in this plan or in `planning_link.rs` module docs** stating:

1. Exact edge: **`RelatesTo` (planning_id → task_id)** with **`artifact_link`** JSON on the edge.
2. Who may call: primary session and/or subagent; how we prove Task “belongs” to the session (edge predicates + optional `session_id` on Task).

**Step 1:** Write the rules (no code).

**Step 2:** Commit: `docs: planning↔task link validation rules` (if you add a small `docs/` note; otherwise skip empty commit).

---

### Task 2: `link_planning_task_edge` in `planning_link.rs`

**Files:**

- Modify: `crates/tools/src/planning_link.rs`
- Modify: `crates/tools/src/lib.rs` (only if new exports needed)

**Behavior (mirror `link_planning_content_edge`):**

- Input: `graph`, `agent_id` (current session agent), `planning_id`, `task_id`, `relationship` (`implements` | `documents`).
- Load Task node; must be `NodeType::Task`.
- Load planning node; Knowledge + `metadata.planning == true`.
- **Scoping:** Implement the rules from Task 1 (e.g. `DelegatesTo` / `SpawnedBy` checks using `graph.edges_for_node` / `graphirm_graph` APIs — avoid holding locks across await).
- Idempotent: existing **`RelatesTo`** planning → task with same endpoints → `AlreadyLinked`.
- Insert edge with same **`artifact_link`** metadata shape as Content.

**Step 1:** Unit tests: memory graph fixtures for allow/deny scoping.

**Step 2:** Run: `cargo test -p graphirm-tools planning_link`

**Step 3:** Commit: `feat(tools): link planning Knowledge to Task via relates_to`

---

### Task 3: `graph_query` `project` — `link_task` action

**Files:**

- Modify: `crates/tools/src/graph_query.rs` — `execute_project`, `parameters()`, `description()`
- Optionally refactor: `link_content` and `link_task` share argument parsing (YAGNI duplicate OK for small delta)

**Behavior:**

- `action: "link_task"` (add to JSON enum for `action`).
- Args: `planning_node_id`, `task_id`, optional `relationship` (default `implements`).
- Call `link_planning_task_edge` inside `spawn_blocking`; map `PlanningContentLink`-style enum to tool errors / success strings (consider renaming shared enum to `PlanningArtifactLink` in a follow-up commit if it bothers you).

**Step 1:** Run: `cargo test -p graphirm-tools`

**Step 2:** Commit: `feat(tools): graph_query project link_task`

---

### Task 4: Integration / regression tests

**Files:**

- Modify: `crates/tools/tests/integration.rs` (or `graph_query` tests in the same crate)

**Minimum:**

- One integration-style test: seed Agent + planning + Task with valid delegation edges → `link_task` → assert `RelatesTo` exists with `artifact_link`.

**Step 1:** `cargo test -p graphirm-tools`

**Step 2:** Commit: `test(tools): link_task integration`

---

### Task 5: Web-app — plan graph includes Task artifacts

**Files:**

- Modify: `web-app/src/hooks/useGraphData.ts` — `computePlanGraphAllowedIds`

**Behavior:**

- When expanding allowed ids from planning nodes, treat **`relates_to`** targets as allowed not only for Content but also **Task** node types (so `planGraphOnly` does not hide linked tasks).

**Step 1:** `cd web-app && npm run build`

**Step 2:** Commit: `feat(web-app): plan filter includes planning-linked tasks`

---

### Task 6: Documentation

**Files:**

- Modify: `AGENTS.md` — planning ↔ artifacts paragraph
- Modify: `docs/backlog.md` — P1 “link artifacts” line: point to this plan + mark done when shipped

**Step 1:** Commit: `docs: planning↔task artifact links`

---

### Task 7 (optional): Auto-link Task to planning on delegate

**Files:**

- Modify: `crates/agent/src/multi.rs` (after Task node persisted / edge `DelegatesTo` added)
- Reuse: `resolve_session_planning_node`, new `link_planning_task_edge`

**Behavior:**

- If parent **`AgentConfig.auto_link_write_to_planning`** (or a new flag **`auto_link_task_to_planning`**) is true and parent has a resolved planning node, add **`RelatesTo`** planning → task with `implements`.
- Never fail delegate if linking fails (same spirit as file auto-link).

**Step 1:** Config + default + TOML comment.

**Step 2:** Tests in agent or tools as appropriate.

**Step 3:** Commit: `feat(agent): auto-link delegated Task to session planning`

---

## Verification (before claiming done)

```bash
cargo fmt
cargo clippy --workspace -- -D warnings
cargo test -p graphirm-tools
cargo test --workspace
cd web-app && npm run build
```

---

## Execution handoff

**Plan saved to:** `docs/plans/2026-04-08-graph-query-artifacts-planning.md`

**Execution options:**

1. **Subagent-driven (this session)** — Use superpowers:subagent-driven-development: one subagent per task, reviews between tasks.
2. **Parallel session** — Use superpowers:executing-plans in a dedicated worktree (see superpowers:using-git-worktrees).

**Which approach?**
