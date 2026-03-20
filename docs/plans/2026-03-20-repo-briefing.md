# Repo Briefing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Auto-inject a compact repo briefing (~200 tokens) into the system prompt at session start (Tier 1), and provide a `repo_briefing` non-destructive tool for on-demand deep analysis (Tier 2).

**Architecture:** `briefing.rs` in `graphirm-agent` for auto-injection; `repo_briefing.rs` in `graphirm-tools` for the tool. Uses `rg` mention counting for file importance, `GraphStore.search_knowledge()` for prior session knowledge. No Nodestradamus dependency.

**Tech Stack:** Rust, `tokio::process::Command` (rg, git), `tokio::fs`, `graphirm_graph::GraphStore`

---

### Task 1: Add `repo_briefing` config flag to AgentConfig

**Files:**
- Modify: `crates/agent/src/config.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn repo_briefing_defaults_to_true() {
    let config = AgentConfig::default();
    assert!(config.repo_briefing);
}

#[test]
fn repo_briefing_can_be_disabled() {
    let toml = r#"
        [agent]
        name = "test"
        model = "test"
        system_prompt = "test"
        max_turns = 5
        repo_briefing = false
    "#;
    let config = AgentConfig::from_toml(toml).unwrap();
    assert!(!config.repo_briefing);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p graphirm-agent config::tests::repo_briefing`
Expected: compile error (no field)

**Step 3: Implement**

In `AgentConfig` struct, add after `pre_edit_impact`:

```rust
#[serde(default = "default_repo_briefing")]
pub repo_briefing: bool,
```

Add the default function:

```rust
fn default_repo_briefing() -> bool {
    true
}
```

Update `Default` impl to include `repo_briefing: true`.

Update `AgentConfigSection` struct (the TOML deserialization struct) with the same field.

Update `from_toml` to pass `repo_briefing: file.agent.repo_briefing`.

**Step 4: Run tests**

Run: `cargo test -p graphirm-agent config`
Expected: all pass

**Step 5: fmt, clippy, commit**

```bash
cargo fmt -p graphirm-agent && cargo clippy -p graphirm-agent -- -D warnings
git add crates/agent/src/config.rs
git commit -m "feat(agent): add repo_briefing config flag (default true)"
```

---

### Task 2: Implement `build_repo_briefing` in briefing.rs — language breakdown

**Files:**
- Create: `crates/agent/src/briefing.rs`
- Modify: `crates/agent/src/lib.rs` (add `pub mod briefing;`)

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn count_files_by_extension_finds_rust_files() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.path().join("lib.rs"), "pub mod foo;").unwrap();
        std::fs::write(dir.path().join("readme.md"), "# Hello").unwrap();

        let counts = count_files_by_extension(dir.path()).await;
        assert_eq!(*counts.get("rs").unwrap_or(&0), 2);
        assert_eq!(*counts.get("md").unwrap_or(&0), 1);
    }

    #[tokio::test]
    async fn count_files_skips_git_and_target() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join(".git/config"), "gitconfig").unwrap();
        std::fs::create_dir_all(dir.path().join("target/debug")).unwrap();
        std::fs::write(dir.path().join("target/debug/build.rs"), "").unwrap();
        std::fs::write(dir.path().join("src.rs"), "fn main() {}").unwrap();

        let counts = count_files_by_extension(dir.path()).await;
        assert_eq!(*counts.get("rs").unwrap_or(&0), 1);
    }

    #[tokio::test]
    async fn format_language_breakdown_sorts_by_count() {
        let mut counts = std::collections::HashMap::new();
        counts.insert("rs".to_string(), 50);
        counts.insert("ts".to_string(), 20);
        counts.insert("md".to_string(), 3);
        counts.insert("toml".to_string(), 2);

        let formatted = format_language_breakdown(&counts);
        assert!(formatted.starts_with("Rust"));
        assert!(formatted.contains("TypeScript"));
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p graphirm-agent briefing`
Expected: compile error

**Step 3: Implement**

Create `crates/agent/src/briefing.rs`:

```rust
//! Repo briefing: compact structural fingerprint for system prompt injection.

use std::collections::HashMap;
use std::path::Path;

/// Walk workspace directory (depth 3, skip .git/target/node_modules) and count
/// files by extension.
pub async fn count_files_by_extension(workspace: &Path) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    let workspace = workspace.to_path_buf();

    tokio::task::spawn_blocking(move || {
        walk_dir_counting(&workspace, &mut counts, 0, 3);
        counts
    })
    .await
    .unwrap_or_default()
}

fn walk_dir_counting(dir: &Path, counts: &mut HashMap<String, usize>, depth: usize, max_depth: usize) {
    if depth > max_depth {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        // Skip hidden dirs, target, node_modules
        if name.starts_with('.') || name == "target" || name == "node_modules" {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            walk_dir_counting(&path, counts, depth + 1, max_depth);
        } else if let Some(ext) = path.extension() {
            let ext = ext.to_string_lossy().to_lowercase();
            *counts.entry(ext).or_insert(0) += 1;
        }
    }
}

/// Format extension counts as a human-readable language summary.
/// Maps common extensions to language names. Shows top 3 by count, rest as "N other".
pub fn format_language_breakdown(counts: &HashMap<String, usize>) -> String {
    let ext_to_lang: HashMap<&str, &str> = [
        ("rs", "Rust"), ("ts", "TypeScript"), ("tsx", "TypeScript"),
        ("js", "JavaScript"), ("jsx", "JavaScript"),
        ("py", "Python"), ("go", "Go"), ("java", "Java"),
        ("rb", "Ruby"), ("c", "C"), ("cpp", "C++"), ("h", "C/C++"),
        ("cs", "C#"), ("swift", "Swift"), ("kt", "Kotlin"),
    ].into_iter().collect();

    // Aggregate by language name
    let mut lang_counts: HashMap<&str, usize> = HashMap::new();
    let mut other_count = 0usize;
    for (ext, count) in counts {
        if let Some(lang) = ext_to_lang.get(ext.as_str()) {
            *lang_counts.entry(lang).or_insert(0) += count;
        } else {
            other_count += count;
        }
    }

    let mut sorted: Vec<(&&str, &usize)> = lang_counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));

    let mut parts: Vec<String> = sorted
        .iter()
        .take(3)
        .map(|(lang, count)| format!("{} ({} files)", lang, count))
        .collect();

    if other_count > 0 {
        parts.push(format!("{other_count} other"));
    }

    if parts.is_empty() {
        "(no source files found)".to_string()
    } else {
        parts.join(", ")
    }
}
```

Add `pub mod briefing;` to `crates/agent/src/lib.rs` (alphabetically, between `compact` and `config`).

**Step 4: Run tests**

Run: `cargo test -p graphirm-agent briefing`
Expected: 3 tests pass

**Step 5: fmt, clippy, commit**

```bash
cargo fmt -p graphirm-agent && cargo clippy -p graphirm-agent -- -D warnings
git add crates/agent/src/briefing.rs crates/agent/src/lib.rs
git commit -m "feat(agent): add briefing module with language breakdown"
```

---

### Task 3: Add top-file discovery to briefing.rs

**Files:**
- Modify: `crates/agent/src/briefing.rs`

**Step 1: Write the failing test**

```rust
#[tokio::test]
async fn find_top_files_returns_sorted_by_mention_count() {
    let dir = tempfile::TempDir::new().unwrap();
    // "lib" is mentioned in main.rs and utils.rs
    std::fs::write(dir.path().join("lib.rs"), "pub fn hello() {}").unwrap();
    std::fs::write(dir.path().join("main.rs"), "use lib::hello;").unwrap();
    std::fs::write(dir.path().join("utils.rs"), "use lib::helper;").unwrap();
    // "main" is mentioned in nobody (self only)
    // "utils" is mentioned in nobody

    let top = find_top_files(dir.path(), 5).await;
    // lib should be first (mentioned by 2 other files)
    assert!(!top.is_empty(), "should find at least one file: {top:?}");
    assert_eq!(top[0].0, "lib");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p graphirm-agent briefing::tests::find_top_files`
Expected: compile error

**Step 3: Implement**

```rust
use tokio::process::Command;

/// Discover the top-N most-referenced file stems in the workspace.
/// Uses `rg --count --fixed-strings` per candidate stem.
/// Returns Vec of (stem, mention_count) sorted descending.
pub async fn find_top_files(workspace: &Path, top_n: usize) -> Vec<(String, usize)> {
    // Gather candidate stems from the workspace root + src/
    let candidates = gather_candidate_stems(workspace).await;
    if candidates.is_empty() {
        return vec![];
    }

    let mut results = Vec::new();
    for stem in candidates.iter().take(30) {
        let count = count_mentions(stem, workspace).await;
        // Subtract 1 for self-reference (the file that defines the stem)
        let adjusted = count.saturating_sub(1);
        if adjusted > 0 {
            results.push((stem.clone(), adjusted));
        }
    }

    results.sort_by(|a, b| b.1.cmp(&a.1));
    results.truncate(top_n);
    results
}

/// Gather file stems from workspace root and src/ directory.
async fn gather_candidate_stems(workspace: &Path) -> Vec<String> {
    let workspace = workspace.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let mut stems = Vec::new();
        // Check root and src/ for source files
        for dir in [workspace.as_path(), workspace.join("src").as_path()] {
            let entries = match std::fs::read_dir(dir) {
                Ok(e) => e,
                Err(_) => continue,
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        let ext = ext.to_string_lossy().to_lowercase();
                        if matches!(ext.as_str(), "rs" | "ts" | "tsx" | "js" | "py" | "go" | "java" | "rb") {
                            if let Some(stem) = path.file_stem() {
                                stems.push(stem.to_string_lossy().to_string());
                            }
                        }
                    }
                }
            }
        }
        stems.sort();
        stems.dedup();
        stems
    })
    .await
    .unwrap_or_default()
}

/// Count how many files mention this stem via rg.
async fn count_mentions(stem: &str, workspace: &Path) -> usize {
    let output = match Command::new("rg")
        .args([
            "--count",
            "--fixed-strings",
            "--no-messages",
            "--glob", "!.git",
            "--glob", "!target",
            "--glob", "!node_modules",
            stem,
            ".",
        ])
        .current_dir(workspace)
        .output()
        .await
    {
        Ok(o) => o,
        Err(_) => return 0,
    };

    if !output.status.success() {
        return 0;
    }

    // rg --count outputs "file:count" per line; we want number of files
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|l| !l.is_empty())
        .count()
}
```

**Step 4: Run tests**

Run: `cargo test -p graphirm-agent briefing`
Expected: 4 tests pass

**Step 5: fmt, clippy, commit**

```bash
cargo fmt -p graphirm-agent && cargo clippy -p graphirm-agent -- -D warnings
git add crates/agent/src/briefing.rs
git commit -m "feat(agent): add top-file discovery via rg mention counting"
```

---

### Task 4: Add knowledge summary to briefing.rs

**Files:**
- Modify: `crates/agent/src/briefing.rs`

**Step 1: Write the failing test**

```rust
#[tokio::test]
async fn build_knowledge_summary_counts_cross_session_nodes() {
    let graph = std::sync::Arc::new(graphirm_graph::GraphStore::open_memory().unwrap());

    // Add Knowledge nodes from two different sessions
    for i in 0..5 {
        let mut k = graphirm_graph::nodes::GraphNode::new(
            graphirm_graph::nodes::NodeType::Knowledge(graphirm_graph::nodes::KnowledgeData {
                entity: format!("entity_{i}"),
                entity_type: "concept".to_string(),
                summary: format!("Summary {i}"),
                confidence: 0.9,
            }),
        );
        k.metadata["session_id"] = serde_json::json!(if i < 3 { "session-a" } else { "session-b" });
        graph.add_node(k).unwrap();
    }

    let summary = build_knowledge_summary(&graph);
    assert!(summary.contains("5"), "should show total count: {summary}");
    assert!(summary.contains("2"), "should show 2 sessions: {summary}");
}

#[tokio::test]
async fn build_knowledge_summary_empty_graph() {
    let graph = std::sync::Arc::new(graphirm_graph::GraphStore::open_memory().unwrap());
    let summary = build_knowledge_summary(&graph);
    assert!(summary.contains("No prior sessions") || summary.is_empty());
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p graphirm-agent briefing::tests::build_knowledge_summary`

**Step 3: Implement**

```rust
use graphirm_graph::GraphStore;

/// Build a compact knowledge summary: session count, total Knowledge nodes,
/// and 3 most recent entity names.
pub fn build_knowledge_summary(graph: &GraphStore) -> String {
    let nodes = match graph.list_nodes_by_type("knowledge", None, None, 500) {
        Ok(n) => n,
        Err(_) => return String::new(),
    };

    if nodes.is_empty() {
        return "No prior sessions with extracted knowledge.".to_string();
    }

    // Count distinct sessions
    let mut session_ids = std::collections::HashSet::new();
    let mut recent_entities = Vec::new();

    for node in &nodes {
        if let Some(sid) = node.metadata.get("session_id").and_then(|v| v.as_str()) {
            session_ids.insert(sid.to_string());
        }
        if recent_entities.len() < 3 {
            if let graphirm_graph::nodes::NodeType::Knowledge(kd) = &node.node_type {
                if !kd.entity.is_empty() {
                    recent_entities.push(kd.entity.clone());
                }
            }
        }
    }

    let mut parts = vec![format!(
        "Prior sessions: {} session{}, {} Knowledge node{}",
        session_ids.len(),
        if session_ids.len() == 1 { "" } else { "s" },
        nodes.len(),
        if nodes.len() == 1 { "" } else { "s" },
    )];

    if !recent_entities.is_empty() {
        let quoted: Vec<String> = recent_entities.iter().map(|e| format!("\"{e}\"")).collect();
        parts.push(format!("Recent knowledge: {}", quoted.join(", ")));
    }

    parts.join("\n")
}
```

**Step 4: Run tests**

Run: `cargo test -p graphirm-agent briefing`
Expected: 6 tests pass

**Step 5: fmt, clippy, commit**

```bash
cargo fmt -p graphirm-agent && cargo clippy -p graphirm-agent -- -D warnings
git add crates/agent/src/briefing.rs
git commit -m "feat(agent): add knowledge summary to repo briefing"
```

---

### Task 5: Assemble `build_repo_briefing` and wire into session creation

**Files:**
- Modify: `crates/agent/src/briefing.rs` (assemble function)
- Modify: `crates/server/src/routes.rs` (call it in `create_session`)

**Step 1: Write the failing test**

```rust
#[tokio::test]
async fn build_repo_briefing_produces_formatted_output() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::write(dir.path().join("lib.rs"), "pub mod utils;").unwrap();
    std::fs::write(dir.path().join("utils.rs"), "use lib;").unwrap();
    std::fs::write(dir.path().join("main.rs"), "mod lib;").unwrap();

    let graph = std::sync::Arc::new(graphirm_graph::GraphStore::open_memory().unwrap());

    let briefing = build_repo_briefing(dir.path(), &graph).await;
    assert!(briefing.contains("Repo Briefing"), "should have header: {briefing}");
    assert!(briefing.contains("Rust") || briefing.contains("rs"), "should mention language: {briefing}");
}
```

**Step 2: Run test to verify it fails**

**Step 3: Implement the assembler**

```rust
/// Build a compact repo briefing for system prompt injection.
/// Returns a ~200 token string with language breakdown, top files, and knowledge summary.
pub async fn build_repo_briefing(workspace: &Path, graph: &GraphStore) -> String {
    let mut sections = vec!["\n\n## Repo Briefing".to_string()];

    // Language breakdown
    let ext_counts = count_files_by_extension(workspace).await;
    if !ext_counts.is_empty() {
        let total: usize = ext_counts.values().sum();
        sections.push(format!(
            "Language: {} ({total} files total)",
            format_language_breakdown(&ext_counts)
        ));
    }

    // Top files
    let top_files = find_top_files(workspace, 5).await;
    if !top_files.is_empty() {
        let file_strs: Vec<String> = top_files
            .iter()
            .map(|(stem, count)| format!("{stem} ({count} refs)"))
            .collect();
        sections.push(format!("Key files: {}", file_strs.join(", ")));
    }

    // Knowledge summary
    let knowledge = build_knowledge_summary(graph);
    if !knowledge.is_empty() {
        sections.push(knowledge);
    }

    sections.push("Use `repo_briefing` tool for detailed analysis.".to_string());

    sections.join("\n")
}
```

**Step 4: Wire into routes.rs**

In `crates/server/src/routes.rs`, in `create_session`, after the `build_workspace_context` block (around line 193), add:

```rust
// Inject repo briefing if enabled
if config.repo_briefing {
    let briefing = graphirm_agent::briefing::build_repo_briefing(
        &config.working_dir,
        &state.graph,
    ).await;
    config.system_prompt.push_str(&briefing);
}
```

**Step 5: Run tests**

Run: `cargo test -p graphirm-agent briefing`
Then: `cargo test -p graphirm-server`
Expected: all pass

**Step 6: fmt, clippy, commit**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings
git add crates/agent/src/briefing.rs crates/server/src/routes.rs
git commit -m "feat(agent): assemble build_repo_briefing and wire into session creation"
```

---

### Task 6: Implement `RepoBriefingTool` (Tier 2) — scaffold and files section

**Files:**
- Create: `crates/tools/src/repo_briefing.rs`
- Modify: `crates/tools/src/lib.rs` (add `pub mod repo_briefing;`)
- Modify: `src/main.rs` (register in `build_tool_registry()`)

**Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::make_test_context;
    use serde_json::json;

    #[tokio::test]
    async fn invalid_section_returns_error() {
        let tool = RepoBriefingTool::new();
        let ctx = make_test_context();
        let result = tool.execute(json!({"section": "weather"}), &ctx).await;
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[tokio::test]
    async fn files_section_returns_formatted_output() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("lib.rs"), "pub fn hello() {}").unwrap();
        std::fs::write(dir.path().join("main.rs"), "use lib::hello;").unwrap();

        let mut ctx = make_test_context();
        ctx.working_dir = dir.path().to_path_buf();

        let tool = RepoBriefingTool::new();
        let out = tool
            .execute(json!({"section": "files"}), &ctx)
            .await
            .unwrap();

        assert!(!out.is_error);
        assert!(out.content.contains("Top Files") || out.content.contains("lib"));
    }

    #[tokio::test]
    async fn not_destructive() {
        let tool = RepoBriefingTool::new();
        assert!(!tool.is_destructive());
    }
}
```

**Step 2: Implement the tool**

Create `crates/tools/src/repo_briefing.rs` with:
- `RepoBriefingTool` struct + `Tool` trait
- Parameters: `section` (enum, default "all"), `limit` (default 10)
- `files` section: uses `find_dependents` pattern from `graph_diff.rs` (rg --files-with-matches per top stem)
- `knowledge` section: lists Knowledge nodes from graph grouped by session
- `git` section: recent commits, branch, dirty count
- `all` section: combines all three

Register in `lib.rs` and `main.rs`.

**Step 3: Run tests**

Run: `cargo test -p graphirm-tools repo_briefing`
Expected: 3 tests pass

**Step 4: fmt, clippy, commit**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings
git add crates/tools/src/repo_briefing.rs crates/tools/src/lib.rs src/main.rs
git commit -m "feat(tools): implement RepoBriefingTool with files, knowledge, and git sections"
```

---

### Task 7: Add knowledge and git sections to RepoBriefingTool

**Files:**
- Modify: `crates/tools/src/repo_briefing.rs`

**Step 1: Write the tests**

```rust
#[tokio::test]
async fn knowledge_section_shows_cross_session_notes() {
    let ctx = make_test_context();

    let mut k = graphirm_graph::nodes::GraphNode::new(
        graphirm_graph::nodes::NodeType::Knowledge(graphirm_graph::nodes::KnowledgeData {
            entity: "auth_handler".to_string(),
            entity_type: "function".to_string(),
            summary: "Handles JWT authentication".to_string(),
            confidence: 0.9,
        }),
    );
    k.metadata["session_id"] = serde_json::json!("other-session");
    ctx.graph.add_node(k).unwrap();

    let tool = RepoBriefingTool::new();
    let out = tool
        .execute(json!({"section": "knowledge"}), &ctx)
        .await
        .unwrap();

    assert!(!out.is_error);
    assert!(out.content.contains("auth_handler") || out.content.contains("Knowledge"));
}

#[tokio::test]
async fn git_section_shows_branch_info() {
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
    std::fs::write(dir.path().join("a.rs"), "fn main() {}").unwrap();
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

    let tool = RepoBriefingTool::new();
    let out = tool
        .execute(json!({"section": "git"}), &ctx)
        .await
        .unwrap();

    assert!(!out.is_error);
    assert!(out.content.contains("init") || out.content.contains("Branch"));
}

#[tokio::test]
async fn all_section_includes_everything() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::write(dir.path().join("lib.rs"), "pub fn hello() {}").unwrap();

    let mut ctx = make_test_context();
    ctx.working_dir = dir.path().to_path_buf();

    let tool = RepoBriefingTool::new();
    let out = tool
        .execute(json!({}), &ctx)   // default section = "all"
        .await
        .unwrap();

    assert!(!out.is_error);
    // Should contain at least the files section
    assert!(!out.content.is_empty());
}
```

**Step 2: Implement knowledge and git sections**

These may already be stubbed from Task 6. Fill in:
- Knowledge: `ctx.graph.list_nodes_by_type("knowledge", ...)` grouped by session
- Git: `git log --oneline -N`, `git branch --show-current`, `git status --porcelain | wc -l`

**Step 3: Run tests**

Run: `cargo test -p graphirm-tools repo_briefing`
Expected: 6+ tests pass

**Step 4: fmt, clippy, commit**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings
git add crates/tools/src/repo_briefing.rs
git commit -m "feat(tools): add knowledge and git sections to RepoBriefingTool"
```

---

### Task 8: Update docs and backlog

**Files:**
- Modify: `docs/backlog.md` — mark Repo Briefing as done
- Modify: `AGENTS.md` — add Phase 24 to Current State
- Modify: `crates/agent/AGENTS.md` — add `briefing.rs`
- Modify: `crates/tools/AGENTS.md` — add `repo_briefing.rs`

**Step 1: Update backlog**

Replace the Repo Briefing entry with:

```markdown
### ✅ Repo Briefing on Session Start (Structural + Memory Onboarding) — P1 · M
Done 2026-03-20. Tiered approach: compact ~200-token auto-injection into system prompt (language breakdown, top-5 files by rg mention count, prior Knowledge summary) + `repo_briefing` non-destructive tool for detailed on-demand analysis (files, knowledge, git sections). `repo_briefing: bool` config flag (default true). No Nodestradamus dependency.
Plan: `docs/plans/2026-03-20-repo-briefing.md`
Design: `docs/plans/2026-03-20-repo-briefing-design.md`
```

**Step 2: Update AGENTS.md**

Add Phase 24 row and summary paragraph.

**Step 3: Update crate AGENTS.md files**

Add `briefing.rs` to `crates/agent/AGENTS.md` and `repo_briefing.rs` to `crates/tools/AGENTS.md`.

**Step 4: Commit**

```bash
git add -f docs/backlog.md AGENTS.md crates/agent/AGENTS.md crates/tools/AGENTS.md
git commit -m "docs: mark repo briefing complete (Phase 24)"
```
