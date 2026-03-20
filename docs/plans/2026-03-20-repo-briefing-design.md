# Repo Briefing on Session Start — Design

**Goal:** When a session starts in a workspace, automatically inject a compact structural
fingerprint into the system prompt (Tier 1), and provide a detailed `repo_briefing` tool
for on-demand deep analysis (Tier 2). The agent arrives warm — knowing the codebase
structure and its own history — without burning excessive tokens every turn.

**Approach:** Tiered. Lightweight auto-injection (~200 tokens, always present) + full
`repo_briefing` non-destructive tool (on-demand). No Nodestradamus dependency — uses `rg`
mention counting and existing `GraphStore.search_knowledge()`.

---

## Architecture

Two independent components:

### Tier 1: Compact auto-injection

| File | Role |
|------|------|
| `crates/agent/src/briefing.rs` | **Create** — `build_repo_briefing(workspace, graph)` → compact string |
| `crates/agent/src/lib.rs` | **Modify** — `pub mod briefing;` |
| `crates/agent/src/config.rs` | **Modify** — `repo_briefing: bool` (default true) |
| `crates/server/src/routes.rs` | **Modify** — call `build_repo_briefing` in `create_session`, append to system prompt |

### Tier 2: On-demand tool

| File | Role |
|------|------|
| `crates/tools/src/repo_briefing.rs` | **Create** — `RepoBriefingTool` with `Tool` trait |
| `crates/tools/src/lib.rs` | **Modify** — `pub mod repo_briefing;` |
| `src/main.rs` | **Modify** — register in `build_tool_registry()` |

## Data Flow

### Tier 1 (session creation)

```
1. POST /api/sessions — create_session handler

2. After workspace resolution + build_workspace_context (existing):
   if config.repo_briefing:
     briefing = build_repo_briefing(&config.working_dir, &state.graph).await
     config.system_prompt.push_str(&briefing)

3. build_repo_briefing does:
   a. Count files by extension — walk workspace dir (depth 3, skip .git/target/node_modules)
      → "Rust (142 files), TypeScript (38 files), 12 other"

   b. Top 5 files by rg mention count:
      - Gather candidate files from workspace (e.g. all .rs/.ts/.py in src/)
      - For each candidate: `rg --count --fixed-strings --no-messages <stem> .` → sum
      - Sort by count, take top 5
      - Cap at 5 rg invocations to keep latency under 1s
      Optimization: instead of N rg calls, scan all files once via
      `rg --count --fixed-strings "stem1|stem2|..." .` — single invocation

   c. Knowledge summary — query graph for Knowledge nodes with session_id != current:
      - Count total Knowledge nodes in graph
      - Get 3 most recent Knowledge entity names
      - Count distinct session_ids

   d. Format as ~200 token block:
      ## Repo Briefing
      Language: Rust (142 files), TypeScript (38 files), 12 other
      Key files (by reference count):
        store.rs (74 refs), workflow.rs (42), context.rs (38), ...
      Prior sessions: 4 sessions, 23 Knowledge nodes
      Recent knowledge: "GraphImpactProvider", "token_refresh race", "ScriptTool"
      Use `repo_briefing` tool for full analysis.

4. System prompt now contains workspace context + repo briefing
   → agent's first turn already has structural awareness
```

### Tier 2 (tool call)

```
1. Agent calls: graph_diff { mode: "all" }   (or section-specific)

2. RepoBriefingTool.execute:
   a. "files" section: top 10 files by rg mention count, WITH dependent file names
      (reuses the find_dependents pattern from graph_diff)

   b. "knowledge" section: all Knowledge nodes from prior sessions,
      grouped by session_id, entity + summary, capped at limit

   c. "git" section:
      - `git log --oneline -5` → recent commits
      - `git branch --show-current` → active branch
      - `git status --porcelain | wc -l` → dirty file count

   d. Format structured Markdown output

3. Return ToolOutput::success(formatted)
```

## Top-File Discovery Strategy

The key design question for Tier 1 is how to find "important" files without a full dependency
parser. The approach: `rg`-based mention counting.

**Strategy:** Collect candidate file stems from the workspace's first-level `src/` or root
directory (up to 30 candidates). For each, count how many other files mention that stem using
`rg --count --fixed-strings --no-messages`. Sort by count, take top 5.

**Why this works:** Files that are mentioned by many other files (via imports, module
declarations, use statements) are structurally important. This is a quick proxy for PageRank
without needing a full AST parse.

**Performance bound:** At most 30 `rg` invocations, each fast (milliseconds on warm filesystem).
Total: <1s even on large repos. Alternative: batch into a single regex alternation
`rg -c "stem1|stem2|stem3" .` — trades precision for speed.

**Graceful fallback:** If rg is not available or the workspace has no recognizable source files,
the briefing omits the "Key files" section entirely (non-fatal).

## Knowledge Summary

Uses `GraphStore.search_knowledge()` and `list_nodes_by_type("knowledge", ...)` to build:

- **Total count:** All Knowledge nodes in the graph (any session)
- **Session count:** Distinct `metadata["session_id"]` values
- **Recent entities:** The 3 most recently created Knowledge nodes' `entity` field
- **Workspace filter:** Only include Knowledge nodes whose `entity` or `summary` matches
  a file in the current workspace (by stem). This prevents cross-workspace noise.

## Error Handling

Non-fatal throughout, matching Phases 22-23 convention.

| Failure | Behavior |
|---------|----------|
| rg not found | Omit "Key files" section, rest of briefing still generated |
| Workspace dir unreadable | Omit language breakdown, rest continues |
| Graph query fails | Omit knowledge section, rest continues |
| No knowledge nodes exist | Show "No prior sessions" instead of knowledge section |
| `repo_briefing = false` | Skip entirely, zero overhead |
| `repo_briefing` tool: git not available | Omit git section |

## Config

```toml
[agent]
repo_briefing = true  # default; set false to disable auto-injection
```

The `repo_briefing` tool is always available regardless of this flag — the flag only
controls the automatic system prompt injection.

## Tier 2 Tool Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `section` | `"all" \| "files" \| "knowledge" \| "git"` | `"all"` | Which sections to include |
| `limit` | `int` | `10` | Max items per section |

## Non-Goals

- No Nodestradamus dependency (deferred to "Any-Repo Instant Analysis")
- No PageRank or community detection (uses rg mention counting as proxy)
- No caching of briefing (computed fresh — cheap enough)
- No web UI endpoint (can be added later as `GET /api/sessions/{id}/briefing`)
- No briefing persistence as graph nodes (just a string in the system prompt)
