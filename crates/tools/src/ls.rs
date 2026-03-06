use async_trait::async_trait;
use graphirm_graph::edges::{EdgeType, GraphEdge};
use graphirm_graph::nodes::{ContentData, GraphNode, NodeType};
use serde_json::json;

use crate::{Tool, ToolContext, ToolError, ToolOutput};

pub struct LsTool;

impl LsTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LsTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for LsTool {
    fn name(&self) -> &str {
        "ls"
    }

    fn description(&self) -> &str {
        "List the contents of a directory. Directories are suffixed with '/'. Hidden files are excluded by default."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to list (optional, defaults to working_dir)"
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files and directories starting with '.' (default: false)"
                }
            }
        })
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let dir_path = if let Some(p) = args["path"].as_str() {
            if std::path::Path::new(p).is_absolute() {
                std::path::PathBuf::from(p)
            } else {
                ctx.working_dir.join(p)
            }
        } else {
            ctx.working_dir.clone()
        };

        let show_hidden = args["show_hidden"].as_bool().unwrap_or(false);

        if !dir_path.is_dir() {
            return Err(ToolError::ExecutionFailed(format!(
                "'{}' is not a directory",
                dir_path.display()
            )));
        }

        let mut reader = tokio::fs::read_dir(&dir_path).await.map_err(|e| {
            ToolError::ExecutionFailed(format!(
                "failed to read directory '{}': {e}",
                dir_path.display()
            ))
        })?;

        let mut entries: Vec<String> = Vec::new();

        while let Some(entry) = reader.next_entry().await? {
            let name = entry.file_name().to_string_lossy().to_string();

            if !show_hidden && name.starts_with('.') {
                continue;
            }

            let entry_path = entry.path();
            let is_dir = entry_path.is_dir();

            if is_dir {
                entries.push(format!("{}/", name));
            } else {
                entries.push(name);
            }
        }

        entries.sort();

        let result_text = if entries.is_empty() {
            "(empty directory)".to_string()
        } else {
            entries.join("\n")
        };

        let content_node = ctx
            .graph
            .add_node(GraphNode::new(NodeType::Content(ContentData {
                content_type: "directory_listing".to_string(),
                path: Some(dir_path.to_string_lossy().to_string()),
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
    use tempfile::TempDir;

    fn make_ctx_with_dir(dir: &TempDir) -> ToolContext {
        let mut ctx = make_test_context();
        ctx.working_dir = dir.path().to_path_buf();
        ctx
    }

    #[tokio::test]
    async fn ls_basic() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("file.txt"), "").unwrap();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();

        let tool = LsTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool.execute(json!({}), &ctx).await.unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("file.txt"));
        assert!(out.content.contains("subdir/"));
    }

    #[tokio::test]
    async fn ls_hides_dotfiles_by_default() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("visible.txt"), "").unwrap();
        std::fs::write(dir.path().join(".hidden"), "").unwrap();

        let tool = LsTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool.execute(json!({}), &ctx).await.unwrap();
        assert!(out.content.contains("visible.txt"));
        assert!(!out.content.contains(".hidden"));
    }

    #[tokio::test]
    async fn ls_show_hidden() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("visible.txt"), "").unwrap();
        std::fs::write(dir.path().join(".hidden"), "").unwrap();

        let tool = LsTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool
            .execute(json!({"show_hidden": true}), &ctx)
            .await
            .unwrap();
        assert!(out.content.contains(".hidden"));
        assert!(out.content.contains("visible.txt"));
    }

    #[tokio::test]
    async fn ls_subdir() {
        let dir = TempDir::new().unwrap();
        let sub = dir.path().join("inner");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(sub.join("inner.txt"), "").unwrap();

        let tool = LsTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool.execute(json!({"path": "inner"}), &ctx).await.unwrap();
        assert!(out.content.contains("inner.txt"));
    }

    #[tokio::test]
    async fn ls_empty_dir() {
        let dir = TempDir::new().unwrap();
        let empty = dir.path().join("empty");
        std::fs::create_dir(&empty).unwrap();

        let tool = LsTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool.execute(json!({"path": "empty"}), &ctx).await.unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("(empty directory)"));
    }

    #[tokio::test]
    async fn ls_not_a_directory() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("afile.txt"), "").unwrap();

        let tool = LsTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let result = tool.execute(json!({"path": "afile.txt"}), &ctx).await;
        assert!(matches!(result, Err(ToolError::ExecutionFailed(_))));
    }

    #[tokio::test]
    async fn ls_sorted_output() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("z.txt"), "").unwrap();
        std::fs::write(dir.path().join("a.txt"), "").unwrap();
        std::fs::write(dir.path().join("m.txt"), "").unwrap();

        let tool = LsTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool.execute(json!({}), &ctx).await.unwrap();
        let lines: Vec<&str> = out.content.lines().collect();
        assert_eq!(lines[0], "a.txt");
        assert_eq!(lines[1], "m.txt");
        assert_eq!(lines[2], "z.txt");
    }
}
