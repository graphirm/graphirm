//! Non-destructive graph_diff tool: blast radius analysis for code changes.

use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::json;
use tokio::process::Command;

use crate::{Tool, ToolContext, ToolError, ToolOutput};

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
    let arr = args["paths"].as_array().ok_or_else(|| {
        ToolError::InvalidArguments("'paths' array is required for paths mode".into())
    })?;

    if arr.is_empty() {
        return Err(ToolError::InvalidArguments(
            "'paths' must not be empty".into(),
        ));
    }

    let mut paths = Vec::with_capacity(arr.len());
    for v in arr {
        let s = v
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("each path must be a string".into()))?;
        paths.push(PathBuf::from(s));
    }
    Ok(paths)
}

/// Find files that reference the given file's stem via ripgrep.
/// Returns up to `limit` file paths, excluding the file itself.
#[allow(dead_code)]
async fn find_dependents(
    path: &std::path::Path,
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
            "--glob",
            "!.git",
            "--glob",
            "!target",
            "--glob",
            "!node_modules",
            &file_stem,
            ".",
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
        .filter(|l| !l.is_empty())
        .map(|l| l.strip_prefix("./").unwrap_or(l))
        .filter(|l| *l != path_str)
        .take(limit)
        .map(PathBuf::from)
        .collect()
}

/// Placeholder — will be replaced in Task 5.
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
        let result = tool
            .execute(json!({"mode": "paths", "paths": []}), &ctx)
            .await;
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[tokio::test]
    async fn paths_mode_missing_paths_returns_error() {
        let tool = GraphDiffTool::new();
        let ctx = make_test_context();
        let result = tool.execute(json!({"mode": "paths"}), &ctx).await;
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[tokio::test]
    async fn find_dependents_returns_file_names() {
        let dir = tempfile::TempDir::new().unwrap();
        // Create files that reference each other
        std::fs::write(dir.path().join("lib.rs"), "pub mod utils;").unwrap();
        std::fs::write(dir.path().join("utils.rs"), "use crate::lib;").unwrap();
        std::fs::write(dir.path().join("main.rs"), "mod lib;").unwrap();

        let result = find_dependents(&PathBuf::from("lib.rs"), dir.path(), 20).await;

        // rg should find utils.rs and main.rs referencing "lib"
        assert!(
            result.len() >= 1,
            "should find at least one dependent: {result:?}"
        );
    }

    #[tokio::test]
    async fn find_dependents_excludes_self() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("store.rs"), "fn store() {}").unwrap();

        let result = find_dependents(&PathBuf::from("store.rs"), dir.path(), 20).await;

        // store.rs should not list itself
        for dep in &result {
            assert_ne!(dep.file_name().unwrap().to_str().unwrap(), "store.rs");
        }
    }
}
