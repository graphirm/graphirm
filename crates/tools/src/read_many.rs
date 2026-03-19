//! Read multiple files in a single tool call.

use async_trait::async_trait;
use serde_json::json;

use crate::{Tool, ToolContext, ToolError, ToolOutput};

const MAX_FILES: usize = 20;
const DEFAULT_MAX_LINES: usize = 500;

pub struct ReadManyTool;

impl ReadManyTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ReadManyTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for ReadManyTool {
    fn name(&self) -> &str {
        "read_many"
    }

    fn description(&self) -> &str {
        "Read multiple files in a single call. Returns all file contents concatenated \
         with clear path headers. More efficient than calling read N times."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of file paths to read (absolute or relative to working_dir). Maximum 20."
                },
                "max_lines_per_file": {
                    "type": "integer",
                    "description": "Maximum lines to include per file (default: 500). Truncates with a notice."
                }
            },
            "required": ["paths"]
        })
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let paths: Vec<String> = args["paths"]
            .as_array()
            .ok_or_else(|| ToolError::InvalidArguments("'paths' must be an array".into()))?
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();

        if paths.is_empty() {
            return Err(ToolError::InvalidArguments(
                "'paths' must not be empty".into(),
            ));
        }
        if paths.len() > MAX_FILES {
            return Err(ToolError::InvalidArguments(format!(
                "too many files ({}), maximum is {}",
                paths.len(),
                MAX_FILES
            )));
        }

        let max_lines = args["max_lines_per_file"]
            .as_u64()
            .map(|n| n as usize)
            .unwrap_or(DEFAULT_MAX_LINES);

        let mut sections = Vec::with_capacity(paths.len());

        for path_str in &paths {
            let full_path = if std::path::Path::new(path_str).is_absolute() {
                std::path::PathBuf::from(path_str)
            } else {
                ctx.working_dir.join(path_str)
            };

            match tokio::fs::read_to_string(&full_path).await {
                Ok(content) => {
                    let lines: Vec<&str> = content.lines().collect();
                    let total = lines.len();
                    let display_lines = if total > max_lines {
                        &lines[..max_lines]
                    } else {
                        &lines
                    };

                    let width = total.to_string().len().max(4);
                    let numbered: String = display_lines
                        .iter()
                        .enumerate()
                        .map(|(i, line)| format!("{:>width$}|{}", i + 1, line, width = width))
                        .collect::<Vec<_>>()
                        .join("\n");

                    let truncation = if total > max_lines {
                        format!("\n... ({} more lines truncated)", total - max_lines)
                    } else {
                        String::new()
                    };

                    sections.push(format!(
                        "=== {} ({} lines) ===\n{}{}",
                        path_str, total, numbered, truncation
                    ));
                }
                Err(e) => {
                    sections.push(format!(
                        "=== {} (error) ===\nFailed to read: {}",
                        path_str, e
                    ));
                }
            }
        }

        Ok(ToolOutput::success(sections.join("\n\n")))
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
    async fn read_many_basic() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.rs"), "fn a() {}").unwrap();
        std::fs::write(dir.path().join("b.rs"), "fn b() {}").unwrap();

        let tool = ReadManyTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool
            .execute(json!({"paths": ["a.rs", "b.rs"]}), &ctx)
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("fn a()"));
        assert!(out.content.contains("fn b()"));
        assert!(out.content.contains("a.rs"));
        assert!(out.content.contains("b.rs"));
    }

    #[tokio::test]
    async fn read_many_partial_failure() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("exists.txt"), "hello").unwrap();

        let tool = ReadManyTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool
            .execute(
                json!({"paths": ["exists.txt", "missing.txt"]}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("hello"));
        assert!(out.content.contains("missing.txt") && out.content.contains("error"));
    }

    #[tokio::test]
    async fn read_many_empty_paths() {
        let tool = ReadManyTool::new();
        let ctx = make_test_context();
        let result = tool.execute(json!({"paths": []}), &ctx).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn read_many_not_destructive() {
        let tool = ReadManyTool::new();
        assert!(!tool.is_destructive());
    }

    #[tokio::test]
    async fn read_many_with_limit() {
        let dir = TempDir::new().unwrap();
        let mut content = String::new();
        for i in 1..=100 {
            content.push_str(&format!("line {}\n", i));
        }
        std::fs::write(dir.path().join("big.txt"), &content).unwrap();

        let tool = ReadManyTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool
            .execute(
                json!({"paths": ["big.txt"], "max_lines_per_file": 10}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("line 1"));
        assert!(out.content.contains("line 10"));
        assert!(!out.content.contains("line 11"));
    }
}
