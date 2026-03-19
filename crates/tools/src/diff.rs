//! Non-destructive diff tool: file compare and git diff.

use async_trait::async_trait;
use serde_json::json;
use tokio::process::Command;

use crate::{Tool, ToolContext, ToolError, ToolOutput};

pub struct DiffTool;

impl DiffTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DiffTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for DiffTool {
    fn name(&self) -> &str {
        "diff"
    }

    fn description(&self) -> &str {
        "Show differences between files or git changes. Two modes:\n\
         - file mode: compare two files (file_a, file_b)\n\
         - git mode: run git diff with optional ref/path arguments"
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["file", "git"],
                    "description": "Comparison mode. 'file' compares two files, 'git' runs git diff. Default: 'file' if file_a/file_b given, 'git' otherwise."
                },
                "file_a": {
                    "type": "string",
                    "description": "First file path (file mode)"
                },
                "file_b": {
                    "type": "string",
                    "description": "Second file path (file mode)"
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
                }
            }
        })
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let mode = args["mode"].as_str().unwrap_or_else(|| {
            if args["file_a"].is_string() {
                "file"
            } else {
                "git"
            }
        });

        match mode {
            "file" => self.execute_file_diff(&args, ctx).await,
            "git" => self.execute_git_diff(&args, ctx).await,
            other => Err(ToolError::InvalidArguments(format!(
                "unknown mode '{}', expected 'file' or 'git'",
                other
            ))),
        }
    }
}

impl DiffTool {
    async fn execute_file_diff(
        &self,
        args: &serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let file_a = args["file_a"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'file_a'".into()))?;
        let file_b = args["file_b"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'file_b'".into()))?;

        let path_a = if std::path::Path::new(file_a).is_absolute() {
            std::path::PathBuf::from(file_a)
        } else {
            ctx.working_dir.join(file_a)
        };
        let path_b = if std::path::Path::new(file_b).is_absolute() {
            std::path::PathBuf::from(file_b)
        } else {
            ctx.working_dir.join(file_b)
        };

        let output = Command::new("diff")
            .args(["-u", &path_a.to_string_lossy(), &path_b.to_string_lossy()])
            .current_dir(&ctx.working_dir)
            .output()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("failed to run diff: {}", e)))?;

        // diff exits 1 when files differ — that's normal, not an error
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if output.status.code() == Some(2) {
            return Err(ToolError::ExecutionFailed(format!(
                "diff error: {}",
                stderr.trim()
            )));
        }

        let result = if stdout.is_empty() {
            "Files are identical.".to_string()
        } else {
            stdout.to_string()
        };

        Ok(ToolOutput::success(result))
    }

    async fn execute_git_diff(
        &self,
        args: &serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let mut cmd_args = vec!["diff".to_string()];

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
            .map_err(|e| ToolError::ExecutionFailed(format!("failed to run git diff: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ToolError::ExecutionFailed(format!(
                "git diff failed: {}",
                stderr.trim()
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let result = if stdout.is_empty() {
            "No differences.".to_string()
        } else {
            stdout.to_string()
        };

        Ok(ToolOutput::success(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::make_test_context;
    use serde_json::json;
    use tempfile::TempDir;

    fn make_ctx_with_dir(dir: &TempDir) -> ToolContext {
        let mut ctx = make_test_context();
        ctx.working_dir = dir.path().to_path_buf();
        ctx
    }

    #[tokio::test]
    async fn diff_two_files() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), "line 1\nline 2\n").unwrap();
        std::fs::write(dir.path().join("b.txt"), "line 1\nline 3\n").unwrap();

        let tool = DiffTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool
            .execute(json!({"file_a": "a.txt", "file_b": "b.txt"}), &ctx)
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("line 2") || out.content.contains("line 3"));
    }

    #[tokio::test]
    async fn diff_not_destructive() {
        let tool = DiffTool::new();
        assert!(!tool.is_destructive());
    }

    #[tokio::test]
    async fn diff_git_mode() {
        let dir = TempDir::new().unwrap();
        let _ = std::process::Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output();
        std::fs::write(dir.path().join("f.txt"), "original\n").unwrap();
        let _ = std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(dir.path())
            .output();
        let _ = std::process::Command::new("git")
            .args(["commit", "-m", "init"])
            .current_dir(dir.path())
            .output();
        std::fs::write(dir.path().join("f.txt"), "changed\n").unwrap();

        let tool = DiffTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool.execute(json!({"mode": "git"}), &ctx).await.unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("original") || out.content.contains("changed"));
    }

    #[tokio::test]
    async fn diff_missing_files_returns_error() {
        let dir = TempDir::new().unwrap();
        let tool = DiffTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let result = tool
            .execute(json!({"file_a": "nope.txt", "file_b": "nada.txt"}), &ctx)
            .await;
        assert!(result.is_err() || result.unwrap().is_error);
    }
}
