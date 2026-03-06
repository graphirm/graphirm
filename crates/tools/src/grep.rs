use async_trait::async_trait;
use graphirm_graph::edges::{EdgeType, GraphEdge};
use graphirm_graph::nodes::{ContentData, GraphNode, NodeType};
use serde_json::json;
use tokio::process::Command;

use crate::{Tool, ToolContext, ToolError, ToolOutput};

pub struct GrepTool;

impl GrepTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GrepTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search for a pattern in files using ripgrep (rg). Returns matching lines with line numbers."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in (optional, defaults to working_dir)"
                },
                "include": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g. '*.rs')"
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Whether to search case-insensitively (default: false)"
                }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let pattern = args["pattern"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'pattern' field".into()))?;

        let search_path = if let Some(p) = args["path"].as_str() {
            if std::path::Path::new(p).is_absolute() {
                std::path::PathBuf::from(p)
            } else {
                ctx.working_dir.join(p)
            }
        } else {
            ctx.working_dir.clone()
        };

        let mut cmd = Command::new("rg");
        cmd.arg("--line-number")
            .arg("--no-heading")
            .arg("--color=never");

        if args["case_insensitive"].as_bool().unwrap_or(false) {
            cmd.arg("--ignore-case");
        }

        if let Some(include) = args["include"].as_str() {
            cmd.arg("--glob").arg(include);
        }

        cmd.arg(pattern).arg(&search_path);

        let output = cmd
            .output()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("failed to run rg: {e}")))?;

        let exit_code = output.status.code().unwrap_or(-1);

        let result_text = match exit_code {
            0 => String::from_utf8_lossy(&output.stdout).to_string(),
            1 => "No matches found.".to_string(),
            _ => {
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                return Err(ToolError::ExecutionFailed(format!(
                    "rg failed (exit {exit_code}): {stderr}"
                )));
            }
        };

        let content_node = ctx
            .graph
            .add_node(GraphNode::new(NodeType::Content(ContentData {
                content_type: "search_results".to_string(),
                path: Some(search_path.to_string_lossy().to_string()),
                body: result_text.clone(),
                language: None,
            })))?;

        ctx.graph.add_edge(GraphEdge::new(
            EdgeType::Reads,
            ctx.interaction_id.clone(),
            content_node.clone(),
        ))?;

        Ok(ToolOutput::success_with_node(result_text, content_node))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::make_test_context;
    use serde_json::json;
    use std::io::Write;
    use tempfile::TempDir;

    fn make_ctx_with_dir(dir: &TempDir) -> ToolContext {
        let mut ctx = make_test_context();
        ctx.working_dir = dir.path().to_path_buf();
        ctx
    }

    #[tokio::test]
    async fn grep_finds_matches() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("sample.txt");
        std::fs::write(&file, "hello world\ngoodbye world\nhello again").unwrap();

        let tool = GrepTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool
            .execute(json!({"pattern": "hello"}), &ctx)
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("hello world"));
        assert!(out.content.contains("hello again"));
    }

    #[tokio::test]
    async fn grep_no_matches() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("empty.txt"), "nothing here").unwrap();

        let tool = GrepTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool
            .execute(json!({"pattern": "XXXXNOTFOUND"}), &ctx)
            .await
            .unwrap();
        assert!(!out.is_error);
        assert_eq!(out.content.trim(), "No matches found.");
    }

    #[tokio::test]
    async fn grep_with_include_filter() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.path().join("readme.md"), "fn not code").unwrap();

        let tool = GrepTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool
            .execute(json!({"pattern": "fn", "include": "*.rs"}), &ctx)
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("main.rs"));
        assert!(!out.content.contains("readme.md"));
    }

    #[tokio::test]
    async fn grep_specific_file() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("target.txt");
        let mut f = std::fs::File::create(&file).unwrap();
        writeln!(f, "foo bar").unwrap();
        writeln!(f, "baz qux").unwrap();

        let tool = GrepTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool
            .execute(
                json!({"pattern": "foo", "path": file.to_str().unwrap()}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("foo"));
    }

    #[tokio::test]
    async fn grep_missing_pattern() {
        let tool = GrepTool::new();
        let ctx = make_test_context();
        let result = tool.execute(json!({}), &ctx).await;
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }
}
