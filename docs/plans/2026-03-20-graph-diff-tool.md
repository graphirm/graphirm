# Graph-Diff Tool Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `graph_diff` non-destructive tool that the agent can call to understand the blast radius of code changes — listing dependent files, surfacing stale Knowledge from prior sessions, and computing risk.

**Architecture:** Standalone tool in `crates/tools/src/graph_diff.rs`. Resolves changed files via git or explicit paths, finds dependents via `rg`, queries `GraphStore.search_knowledge()` for cross-session Knowledge, computes risk via existing `compute_risk()`. Registered in `build_tool_registry()`.

**Tech Stack:** Rust, `tokio::process::Command` (git, rg), `graphirm_graph::GraphStore`, `graphirm_tools::impact::compute_risk`

---

### Task 1: Add module declaration and tool registration

**Files:**
- Modify: `crates/tools/src/lib.rs:1-18` (add module declaration)
- Create: `crates/tools/src/graph_diff.rs` (empty placeholder)
- Modify: `src/main.rs:780-782` (register tool)

**Step 1: Create empty module file**

Create `crates/tools/src/graph_diff.rs`:

```rust
//! Non-destructive graph_diff tool: blast radius analysis for code changes.
```

**Step 2: Add module declaration**

In `crates/tools/src/lib.rs`, add after `pub mod graph_query;` (line 8):

```rust
pub mod graph_diff;
```

**Step 3: Register in build_tool_registry**

In `src/main.rs`, after the `read_many` registration (line 782), add:

```rust
registry.register(Arc::new(graphirm_tools::graph_diff::GraphDiffTool::new()));
```

**Step 4: Verify it compiles**

This won't compile yet (no `GraphDiffTool` struct), but confirms the wiring is ready. Skip compile check — Task 2 will make it compile.

**Step 5: Commit**

```bash
git add crates/tools/src/graph_diff.rs crates/tools/src/lib.rs src/main.rs
git commit -m "feat(tools): scaffold graph_diff module and registration"
```

---

### Task 2: Implement GraphDiffTool struct, parameters, and mode dispatch

**Files:**
- Modify: `crates/tools/src/graph_diff.rs`

**Step 1: Write the failing test**

Add to bottom of `graph_diff.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::make_test_context;
    use serde_json::json;

    #[tokio::test]
    async fn invalid_mode_returns_error() {
        let tool = GraphDiffTool::new();
        let ctx = make_test_context();
        let result = tool.execute(json!({"mode": "teleport"}), &ctx).await;
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[tokio::test]
    async fn missing_mode_returns_error() {
        let tool = GraphDiffTool::new();
        let ctx = make_test_context();
        let result = tool.execute(json!({}), &ctx).await;
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[tokio::test]
    async fn paths_mode_empty_array_returns_error() {
        let tool = GraphDiffTool::new();
        let ctx = make_test_context();
        let result = tool.execute(json!({"mode": "paths", "paths": []}), &ctx).await;
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[tokio::test]
    async fn paths_mode_missing_paths_returns_error() {
        let tool = GraphDiffTool::new();
        let ctx = make_test_context();
        let result = tool.execute(json!({"mode": "paths"}), &ctx).await;
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p graphirm-tools graph_diff`
Expected: compile error (no `GraphDiffTool`)

**Step 3: Implement the struct and Tool trait**

Add to `graph_diff.rs` before the `#[cfg(test)]`:

```rust
use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::json;
use tokio::process::Command;

use crate::{Tool, ToolContext, ToolError, ToolOutput};
use crate::impact::compute_risk;

pub struct GraphDiffTool;

impl GraphDiffTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GraphDiffTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for GraphDiffTool {
    fn name(&self) -> &str {
        "graph_diff"
    }

    fn description(&self) -> &str {
        "Analyze the blast radius of code changes. Two modes:\n\
         \n\
         • git — resolve changed files from git diff, then analyze each file's dependents \
         and cross-session Knowledge references.\n\
         • paths — analyze an explicit list of file paths.\n\
         \n\
         For each changed file, reports: dependent files (via ripgrep), stale Knowledge \
         notes from prior sessions that may be invalidated, and a risk score (Low/Medium/High).\n\
         \n\
         Read-only — never mutates files or the graph."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["git", "paths"],
                    "description": "How to resolve changed files"
                },
                "paths": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Explicit file paths to analyze (required for paths mode)"
                },
                "ref": {
                    "type": "string",
                    "description": "Git ref to diff against (git mode, e.g. 'HEAD~3', 'main'). Default: working tree vs index."
                },
                "path": {
                    "type": "string",
                    "description": "Restrict git diff to this path (git mode, optional)"
                },
                "cached": {
                    "type": "boolean",
                    "description": "Show staged changes (git mode, default: false)"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Max dependent files listed per changed file (default: 20)"
                }
            },
            "required": ["mode"]
        })
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let mode = args["mode"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("'mode' is required".into()))?;

        let limit = args["limit"].as_u64().unwrap_or(20) as usize;

        let changed_files = match mode {
            "git" => resolve_git_changed_files(&args, ctx).await?,
            "paths" => resolve_explicit_paths(&args)?,
            other => {
                return Err(ToolError::InvalidArguments(format!(
                    "unknown mode '{other}'; must be 'git' or 'paths'"
                )));
            }
        };

        if changed_files.is_empty() {
            return Ok(ToolOutput::success("No changed files."));
        }

        let output = analyze_changed_files(&changed_files, ctx, limit).await;
        Ok(ToolOutput::success(output))
    }
}

async fn resolve_git_changed_files(
    args: &serde_json::Value,
    ctx: &ToolContext,
) -> Result<Vec<PathBuf>, ToolError> {
    let mut cmd_args = vec!["diff".to_string(), "--name-only".to_string()];

    if args["cached"].as_bool().unwrap_or(false) {
        cmd_args.push("--cached".to_string());
    }

    if let Some(git_ref) = args["ref"].as_str() {
        cmd_args.push(git_ref.to_string());
    }

    if let Some(path) = args["path"].as_str() {
        cmd_args.push("--".to_string());
        cmd_args.push(path.to_string());
    }

    let output = Command::new("git")
        .args(&cmd_args)
        .current_dir(&ctx.working_dir)
        .output()
        .await
        .map_err(|e| ToolError::ExecutionFailed(format!("failed to run git diff: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ToolError::ExecutionFailed(format!(
            "git diff failed: {}",
            stderr.trim()
        )));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let paths: Vec<PathBuf> = stdout
        .lines()
        .filter(|l| !l.is_empty())
        .map(PathBuf::from)
        .collect();

    Ok(paths)
}

fn resolve_explicit_paths(args: &serde_json::Value) -> Result<Vec<PathBuf>, ToolError> {
    let arr = args["paths"]
        .as_array()
        .ok_or_else(|| ToolError::InvalidArguments("'paths' array is required for paths mode".into()))?;

    if arr.is_empty() {
        return Err(ToolError::InvalidArguments("'paths' must not be empty".into()));
    }

    let mut paths = Vec::with_capacity(arr.len());
    for v in arr {
        let s = v.as_str().ok_or_else(|| {
            ToolError::InvalidArguments("each path must be a string".into())
        })?;
        paths.push(PathBuf::from(s));
    }
    Ok(paths)
}

/// Placeholder — implemented in Task 3
async fn analyze_changed_files(
    changed_files: &[PathBuf],
    _ctx: &ToolContext,
    _limit: usize,
) -> String {
    let mut lines = vec![format!("## Changed Files ({})", changed_files.len())];
    for path in changed_files {
        lines.push(format!("\n### {} — Risk: Low", path.display()));
        lines.push("Analysis not yet implemented.".to_string());
    }
    lines.join("\n")
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p graphirm-tools graph_diff`
Expected: 4 tests pass

**Step 5: Run fmt and clippy**

```bash
cargo fmt -p graphirm-tools
cargo clippy -p graphirm-tools -- -D warnings
```

**Step 6: Commit**

```bash
git add crates/tools/src/graph_diff.rs crates/tools/src/lib.rs src/main.rs
git commit -m "feat(tools): implement GraphDiffTool struct with mode dispatch and validation"
```

---

### Task 3: Implement dependent file listing via rg

**Files:**
- Modify: `crates/tools/src/graph_diff.rs`

**Step 1: Write the failing test**

Add to `tests` module:

```rust
#[tokio::test]
async fn find_dependents_returns_file_names() {
    let dir = tempfile::TempDir::new().unwrap();
    // Create files that reference each other
    std::fs::write(dir.path().join("lib.rs"), "pub mod utils;").unwrap();
    std::fs::write(dir.path().join("utils.rs"), "use crate::lib;").unwrap();
    std::fs::write(dir.path().join("main.rs"), "mod lib;").unwrap();

    let result = find_dependents(
        &PathBuf::from("lib.rs"),
        dir.path(),
        20,
    ).await;

    // rg should find utils.rs and main.rs referencing "lib"
    assert!(result.len() >= 1, "should find at least one dependent: {result:?}");
}

#[tokio::test]
async fn find_dependents_excludes_self() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::write(dir.path().join("store.rs"), "fn store() {}").unwrap();

    let result = find_dependents(
        &PathBuf::from("store.rs"),
        dir.path(),
        20,
    ).await;

    // store.rs should not list itself
    for dep in &result {
        assert_ne!(dep.file_name().unwrap().to_str().unwrap(), "store.rs");
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p graphirm-tools graph_diff::tests::find_dependents`
Expected: compile error (no `find_dependents` function)

**Step 3: Implement find_dependents**

Add before `analyze_changed_files`:

```rust
/// Find files that reference the given file's stem via ripgrep.
/// Returns up to `limit` file paths, excluding the file itself.
async fn find_dependents(
    path: &PathBuf,
    working_dir: &std::path::Path,
    limit: usize,
) -> Vec<PathBuf> {
    let file_stem = match path.file_stem() {
        Some(s) => s.to_string_lossy().to_string(),
        None => return vec![],
    };

    let output = match Command::new("rg")
        .args([
            "--files-with-matches",
            "--no-messages",
            "--glob", "!.git",
            "--glob", "!target",
            "--glob", "!node_modules",
            &file_stem,
        ])
        .current_dir(working_dir)
        .output()
        .await
    {
        Ok(o) => o,
        Err(_) => return vec![],
    };

    if !output.status.success() {
        return vec![];
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let path_str = path.to_string_lossy();

    stdout
        .lines()
        .filter(|l| !l.is_empty() && *l != path_str)
        .take(limit)
        .map(PathBuf::from)
        .collect()
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p graphirm-tools graph_diff`
Expected: all tests pass

**Step 5: Commit**

```bash
cargo fmt -p graphirm-tools && cargo clippy -p graphirm-tools -- -D warnings
git add crates/tools/src/graph_diff.rs
git commit -m "feat(tools): add find_dependents via rg for graph_diff"
```

---

### Task 4: Implement cross-session Knowledge query and stale warnings

**Files:**
- Modify: `crates/tools/src/graph_diff.rs`

**Step 1: Write the failing test**

Add to `tests` module:

```rust
#[tokio::test]
async fn find_stale_knowledge_returns_cross_session_notes() {
    let ctx = make_test_context();

    // Add a Knowledge node from a different session
    let mut k = graphirm_graph::nodes::GraphNode::new(
        graphirm_graph::nodes::NodeType::Knowledge(graphirm_graph::nodes::KnowledgeData {
            entity: "store.rs".to_string(),
            entity_type: "file".to_string(),
            summary: "Has a race condition on concurrent access".to_string(),
            confidence: 0.9,
        }),
    );
    k.metadata["session_id"] = serde_json::json!("other-session");
    ctx.graph.add_node(k).unwrap();

    let notes = find_stale_knowledge(
        &PathBuf::from("src/store.rs"),
        &ctx.graph,
        &ctx.agent_id.to_string(),
    );

    assert_eq!(notes.len(), 1);
    assert!(notes[0].summary.contains("race condition"));
}

#[tokio::test]
async fn find_stale_knowledge_skips_current_session() {
    let ctx = make_test_context();
    let session_id = ctx.agent_id.to_string();

    // Add a Knowledge node from the current session
    let mut k = graphirm_graph::nodes::GraphNode::new(
        graphirm_graph::nodes::NodeType::Knowledge(graphirm_graph::nodes::KnowledgeData {
            entity: "store.rs".to_string(),
            entity_type: "file".to_string(),
            summary: "Session-local note".to_string(),
            confidence: 0.9,
        }),
    );
    k.metadata["session_id"] = serde_json::json!(session_id);
    ctx.graph.add_node(k).unwrap();

    let notes = find_stale_knowledge(
        &PathBuf::from("src/store.rs"),
        &ctx.graph,
        &session_id,
    );

    assert!(notes.is_empty());
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p graphirm-tools graph_diff::tests::find_stale_knowledge`
Expected: compile error (no `find_stale_knowledge`)

**Step 3: Implement find_stale_knowledge**

Add a struct and function before `analyze_changed_files`:

```rust
use graphirm_graph::GraphStore;

struct StaleNote {
    session_id: String,
    entity: String,
    summary: String,
}

/// Find Knowledge nodes from other sessions that mention this file.
fn find_stale_knowledge(
    path: &PathBuf,
    graph: &GraphStore,
    current_session_id: &str,
) -> Vec<StaleNote> {
    let file_stem = match path.file_stem() {
        Some(s) => s.to_string_lossy().to_string().to_lowercase(),
        None => return vec![],
    };

    let nodes = match graph.search_knowledge(&file_stem, None, None, 50) {
        Ok(n) => n,
        Err(_) => return vec![],
    };

    let mut notes = Vec::new();
    for node in nodes {
        let node_session = node
            .metadata
            .get("session_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if node_session == current_session_id {
            continue;
        }

        if let graphirm_graph::nodes::NodeType::Knowledge(kd) = &node.node_type {
            notes.push(StaleNote {
                session_id: node_session.to_string(),
                entity: kd.entity.clone(),
                summary: kd.summary.clone(),
            });
        }

        if notes.len() >= 10 {
            break;
        }
    }

    notes
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p graphirm-tools graph_diff`
Expected: all tests pass

**Step 5: Commit**

```bash
cargo fmt -p graphirm-tools && cargo clippy -p graphirm-tools -- -D warnings
git add crates/tools/src/graph_diff.rs
git commit -m "feat(tools): add find_stale_knowledge for cross-session blast radius"
```

---

### Task 5: Wire analyze_changed_files to produce full output

**Files:**
- Modify: `crates/tools/src/graph_diff.rs`

**Step 1: Write the failing test**

Add to `tests` module:

```rust
#[tokio::test]
async fn paths_mode_produces_formatted_output() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::write(dir.path().join("lib.rs"), "pub fn hello() {}").unwrap();
    std::fs::write(dir.path().join("main.rs"), "use lib::hello;").unwrap();

    let mut ctx = make_test_context();
    ctx.working_dir = dir.path().to_path_buf();

    let tool = GraphDiffTool::new();
    let out = tool
        .execute(
            json!({"mode": "paths", "paths": ["lib.rs"]}),
            &ctx,
        )
        .await
        .unwrap();

    assert!(!out.is_error);
    assert!(out.content.contains("Changed Files"));
    assert!(out.content.contains("lib.rs"));
    assert!(out.content.contains("Risk:"));
}

#[tokio::test]
async fn not_destructive() {
    let tool = GraphDiffTool::new();
    assert!(!tool.is_destructive());
}
```

**Step 2: Run to verify first test fails (placeholder output)**

Run: `cargo test -p graphirm-tools graph_diff::tests::paths_mode`
Expected: may pass with placeholder — the key test is formatting quality

**Step 3: Replace analyze_changed_files placeholder with full implementation**

Replace the placeholder `analyze_changed_files` function:

```rust
async fn analyze_changed_files(
    changed_files: &[PathBuf],
    ctx: &ToolContext,
    limit: usize,
) -> String {
    let mut lines = vec![format!("## Changed Files ({})", changed_files.len())];
    let session_id = ctx.agent_id.to_string();

    for path in changed_files {
        let dependents = find_dependents(path, &ctx.working_dir, limit).await;
        let stale_notes = find_stale_knowledge(path, &ctx.graph, &session_id);

        let dep_count = dependents.len();
        let has_notes = !stale_notes.is_empty();
        let risk = compute_risk(Some(dep_count), has_notes);

        lines.push(format!("\n### {} — Risk: {}", path.display(), risk));

        if dependents.is_empty() {
            lines.push("No dependents found.".to_string());
        } else {
            lines.push(format!("Dependents ({dep_count}):"));
            for dep in &dependents {
                lines.push(format!("  {}", dep.display()));
            }
        }

        if stale_notes.is_empty() {
            lines.push("No prior knowledge references.".to_string());
        } else {
            lines.push(format!("Stale Knowledge ({}):", stale_notes.len()));
            for note in &stale_notes {
                let summary = truncate(&note.summary, 80);
                let session_short = truncate(&note.session_id, 12);
                lines.push(format!(
                    "  ⚠ [session {session_short}] \"{entity} — {summary}\" — may be invalidated",
                    entity = note.entity,
                ));
            }
        }
    }

    lines.join("\n")
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}…", &s[..max])
    } else {
        s.to_string()
    }
}
```

**Step 4: Run all tests**

Run: `cargo test -p graphirm-tools graph_diff`
Expected: all tests pass

**Step 5: Run fmt and clippy**

```bash
cargo fmt -p graphirm-tools && cargo clippy -p graphirm-tools -- -D warnings
```

**Step 6: Commit**

```bash
git add crates/tools/src/graph_diff.rs
git commit -m "feat(tools): complete graph_diff output with dependents, knowledge, and risk"
```

---

### Task 6: Add git mode integration test

**Files:**
- Modify: `crates/tools/src/graph_diff.rs`

**Step 1: Write the test**

Add to `tests` module:

```rust
#[tokio::test]
async fn git_mode_finds_changed_files() {
    let dir = tempfile::TempDir::new().unwrap();

    // Init git repo
    let _ = std::process::Command::new("git")
        .args(["init"])
        .current_dir(dir.path())
        .output();
    let _ = std::process::Command::new("git")
        .args(["config", "user.email", "test@test.com"])
        .current_dir(dir.path())
        .output();
    let _ = std::process::Command::new("git")
        .args(["config", "user.name", "Test"])
        .current_dir(dir.path())
        .output();

    std::fs::write(dir.path().join("a.rs"), "fn a() {}").unwrap();
    let _ = std::process::Command::new("git")
        .args(["add", "."])
        .current_dir(dir.path())
        .output();
    let _ = std::process::Command::new("git")
        .args(["commit", "-m", "init"])
        .current_dir(dir.path())
        .output();

    // Modify the file
    std::fs::write(dir.path().join("a.rs"), "fn a() { changed }").unwrap();

    let mut ctx = make_test_context();
    ctx.working_dir = dir.path().to_path_buf();

    let tool = GraphDiffTool::new();
    let out = tool
        .execute(json!({"mode": "git"}), &ctx)
        .await
        .unwrap();

    assert!(!out.is_error);
    assert!(
        out.content.contains("a.rs"),
        "should detect changed file a.rs: {}",
        out.content
    );
}

#[tokio::test]
async fn git_mode_no_changes_returns_no_files() {
    let dir = tempfile::TempDir::new().unwrap();

    let _ = std::process::Command::new("git")
        .args(["init"])
        .current_dir(dir.path())
        .output();
    let _ = std::process::Command::new("git")
        .args(["config", "user.email", "test@test.com"])
        .current_dir(dir.path())
        .output();
    let _ = std::process::Command::new("git")
        .args(["config", "user.name", "Test"])
        .current_dir(dir.path())
        .output();

    std::fs::write(dir.path().join("a.rs"), "fn a() {}").unwrap();
    let _ = std::process::Command::new("git")
        .args(["add", "."])
        .current_dir(dir.path())
        .output();
    let _ = std::process::Command::new("git")
        .args(["commit", "-m", "init"])
        .current_dir(dir.path())
        .output();

    let mut ctx = make_test_context();
    ctx.working_dir = dir.path().to_path_buf();

    let tool = GraphDiffTool::new();
    let out = tool
        .execute(json!({"mode": "git"}), &ctx)
        .await
        .unwrap();

    assert!(!out.is_error);
    assert!(
        out.content.contains("No changed files"),
        "should report no changes: {}",
        out.content
    );
}
```

**Step 2: Run tests**

Run: `cargo test -p graphirm-tools graph_diff`
Expected: all tests pass

**Step 3: Run full test suite**

Run: `cargo test`
Expected: all pass

**Step 4: Commit**

```bash
cargo fmt -p graphirm-tools && cargo clippy -p graphirm-tools -- -D warnings
git add crates/tools/src/graph_diff.rs
git commit -m "test(tools): add git mode integration tests for graph_diff"
```

---

### Task 7: Update docs and backlog

**Files:**
- Modify: `docs/backlog.md` — mark Graph-Diff Tool as done
- Modify: `AGENTS.md` — add Phase 23 to Current State table
- Modify: `crates/tools/AGENTS.md` — add `graph_diff.rs` to key components

**Step 1: Update backlog**

Replace the Graph-Diff Tool entry with:

```markdown
### ✅ Graph-Diff Tool (Session-Aware Blast Radius) — P1 · S
Done 2026-03-20. `graph_diff` non-destructive tool: git or explicit paths → per-file dependent listing (rg), cross-session Knowledge query with stale warnings, risk scoring (Low/Medium/High). Reuses `compute_risk` from Phase 22. Capped at 20 dependents per file.
Plan: `docs/plans/2026-03-20-graph-diff-tool.md`
Design: `docs/plans/2026-03-20-graph-diff-tool-design.md`
```

**Step 2: Update AGENTS.md**

Add to the Current State table after Phase 22:

```markdown
| 23 | `graph_diff` tool — session-aware blast radius analysis (git/paths → dependents + stale knowledge + risk) | ✅ done |
```

Add a summary paragraph after the Phase 22 summary:

```markdown
**Graph-diff tool (Phase 23):**
- `graph_diff` non-destructive tool in `crates/tools/src/graph_diff.rs` — two modes: `git` (resolve changed files via `git diff --name-only`) and `paths` (explicit file list)
- For each changed file: lists up to 20 dependent files via `rg --files-with-matches`, queries `GraphStore.search_knowledge()` for cross-session Knowledge notes, computes risk via `compute_risk`
- Output: structured markdown with dependents, stale knowledge warnings ("may be invalidated"), and per-file risk level
- Registered in `build_tool_registry()` alongside other non-destructive tools
```

**Step 3: Update crates/tools/AGENTS.md**

Add `graph_diff.rs` to the key components table:

```markdown
| `graph_diff.rs` | `GraphDiffTool` — blast radius analysis: dependents via rg, cross-session stale knowledge, risk scoring |
```

**Step 4: Commit**

```bash
git add docs/backlog.md AGENTS.md crates/tools/AGENTS.md
git commit -m "docs: mark graph_diff tool complete (Phase 23)"
```
