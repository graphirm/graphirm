# Graph-Diff Tool (Session-Aware Blast Radius) — Design

**Goal:** Give the agent an explicit tool to query the blast radius of code changes —
listing dependent files, surfacing stale Knowledge from prior sessions, and computing
risk — so it can reason about impact before or after modifying code.

**Approach:** Standalone tool in `graphirm-tools` (Approach A from brainstorming).
Self-contained: runs `rg` for dependents, queries `GraphStore.search_knowledge()` for
cross-session Knowledge notes. No dependency on `ImpactProvider` trait.

---

## Architecture

Three files touched (one new, two modified):

| File | Role |
|------|------|
| `crates/tools/src/graph_diff.rs` | **Create** — `GraphDiffTool` impl, `rg` dependent listing, knowledge query, output formatting |
| `crates/tools/src/lib.rs` | **Modify** — add `pub mod graph_diff;` |
| `src/main.rs` | **Modify** — register `GraphDiffTool` in `build_tool_registry()` |

**Nodestradamus validation:** `get_impact` on `crates/tools/src/lib.rs` confirms zero upstream
dependencies and only `registry.rs` downstream. Adding a module declaration is safe — same
pattern as `bash_paths`, `impact`, `diff`, `read_many`.

## Data Flow

```
1. Agent calls graph_diff in one of two modes:

   git mode:
     {mode: "git", ref: "HEAD~3", path: "src/auth/"}
     → runs `git diff --name-only [ref] [-- path]` in working_dir
     → produces Vec<PathBuf> of changed files

   paths mode:
     {mode: "paths", paths: ["src/auth/tokens.rs", "src/main.rs"]}
     → uses paths directly as Vec<PathBuf>

2. For each changed file:
   a. Run `rg --files-with-matches --no-messages <file_stem>` in working_dir
      → collect up to `limit` (default 20) dependent file paths
      → exclude .git, target, node_modules, and the file itself

   b. Query ctx.graph.search_knowledge(file_stem, None, None, 50)
      → filter out Knowledge nodes from the CURRENT session (ctx.agent_id)
      → collect cross-session notes (entity, summary, session_id)

   c. Compute risk via impact::compute_risk(dep_count, has_notes)

3. Format structured output:
   ## Changed Files (N)
   ### path — Risk: LEVEL
   Dependents (count):
     file1.rs
     file2.rs
   Stale Knowledge (count):
     ⚠ [session X] "entity — summary" — may be invalidated
   ...

4. Return ToolOutput::success(formatted_output)
```

## Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `mode` | `"git" \| "paths"` | yes | — | How to resolve changed files |
| `paths` | `string[]` | paths mode | — | Explicit list of changed file paths |
| `ref` | `string` | no | working tree vs index | Git ref to diff against |
| `path` | `string` | no | all files | Restrict git diff scope |
| `cached` | `bool` | no | `false` | Show staged changes (git mode) |
| `limit` | `int` | no | `20` | Max dependent files listed per changed file |

## Output Format

```
## Changed Files (3)

### src/auth/tokens.rs — Risk: HIGH
Dependents (7, showing 7):
  src/auth/middleware.rs
  src/api/routes.rs
  src/api/handlers.rs
  src/api/session.rs
  tests/auth_test.rs
  tests/integration.rs
  benches/auth_bench.rs
Stale Knowledge (2):
  ⚠ [session abc123] "token_refresh — race condition on concurrent refresh" — may be invalidated
  ⚠ [session def456] "tokens.rs — handles JWT auth" — may be invalidated

### src/utils/format.rs — Risk: LOW
No dependents found.
No prior knowledge references.

### src/main.rs — Risk: MEDIUM
Dependents (3, showing 3):
  tests/integration.rs
  benches/startup.rs
  docs/examples/basic.rs
No prior knowledge references.
```

## Error Handling

Non-fatal throughout, matching Phase 22 convention.

| Failure | Behavior |
|---------|----------|
| `git diff` fails (not a git repo, bad ref) | Return `ToolError::ExecutionFailed` with stderr |
| `rg` not found or errors | Skip dependents for that file, show "dependents unknown" |
| `search_knowledge` fails | Skip knowledge section for that file |
| No changed files found | Return success: "No changed files." |
| Empty `paths` array | Return `ToolError::InvalidArguments` |

## What It Reuses

- `impact::compute_risk()` from `crates/tools/src/impact.rs` — risk scoring logic
- `GraphStore::search_knowledge()` — cross-session Knowledge query
- `rg --files-with-matches` pattern from `GraphImpactProvider` (but collects names, not just count)
- `ToolContext.graph` and `ToolContext.working_dir` — already available

## Non-Goals

- No automatic Knowledge node invalidation or cleanup (just surfaces warnings)
- No full dependency graph edge enumeration (would require a code parser per language)
- No integration with the pre-edit hook (Phase 22 handles that separately)
- No eval task (can be added later as a follow-up)
