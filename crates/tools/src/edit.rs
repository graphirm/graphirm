use async_trait::async_trait;
use graphirm_graph::edges::{EdgeType, GraphEdge};
use graphirm_graph::nodes::{ContentData, GraphNode, NodeType};
use serde_json::json;

use crate::{Tool, ToolContext, ToolError, ToolOutput};

pub struct EditTool;

impl EditTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for EditTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for EditTool {
    fn name(&self) -> &str {
        "edit"
    }

    fn description(&self) -> &str {
        "Replace an exact string in a file. The old_string must match exactly once."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit"
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact string to replace (must occur exactly once)"
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement string"
                }
            },
            "required": ["path", "old_string", "new_string"]
        })
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let path_str = args["path"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'path' field".into()))?;

        let old_string = args["old_string"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'old_string' field".into()))?;

        let new_string = args["new_string"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'new_string' field".into()))?;

        let full_path = if std::path::Path::new(path_str).is_absolute() {
            std::path::PathBuf::from(path_str)
        } else {
            ctx.working_dir.join(path_str)
        };

        let original = tokio::fs::read_to_string(&full_path).await.map_err(|e| {
            ToolError::ExecutionFailed(format!("failed to read '{}': {}", full_path.display(), e))
        })?;

        let count = original.matches(old_string).count();
        if count == 0 {
            return Err(ToolError::ExecutionFailed(format!(
                "'old_string' not found in '{}'",
                full_path.display()
            )));
        }
        if count > 1 {
            return Err(ToolError::ExecutionFailed(format!(
                "'old_string' found {} times in '{}' — must match exactly once",
                count,
                full_path.display()
            )));
        }

        let updated = original.replacen(old_string, new_string, 1);
        tokio::fs::write(&full_path, &updated).await?;

        let content_node = ctx
            .graph
            .add_node(GraphNode::new(NodeType::Content(ContentData {
                content_type: "file".to_string(),
                path: Some(full_path.to_string_lossy().to_string()),
                body: updated,
                language: None,
            })))?;

        ctx.graph.add_edge(
            GraphEdge::new(
                EdgeType::Modifies,
                ctx.interaction_id.clone(),
                content_node.clone(),
            )
            .with_metadata(json!({
                "old_string": old_string,
                "new_string": new_string
            })),
        )?;

        Ok(ToolOutput::success_with_node(
            format!("Edited '{}': replaced 1 occurrence", full_path.display()),
            content_node,
        ))
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
    async fn edit_replaces_string() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("code.rs");
        std::fs::write(&file, "fn greet() { println!(\"hello\"); }").unwrap();

        let tool = EditTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool
            .execute(
                json!({
                    "path": "code.rs",
                    "old_string": "hello",
                    "new_string": "world"
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        let content = std::fs::read_to_string(&file).unwrap();
        assert!(content.contains("world"));
        assert!(!content.contains("hello"));
    }

    #[tokio::test]
    async fn edit_not_found() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("text.txt");
        std::fs::write(&file, "some content here").unwrap();

        let tool = EditTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let result = tool
            .execute(
                json!({"path": "text.txt", "old_string": "MISSING", "new_string": "x"}),
                &ctx,
            )
            .await;
        assert!(matches!(result, Err(ToolError::ExecutionFailed(_))));
    }

    #[tokio::test]
    async fn edit_ambiguous_match() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("dup.txt");
        std::fs::write(&file, "foo foo foo").unwrap();

        let tool = EditTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let result = tool
            .execute(
                json!({"path": "dup.txt", "old_string": "foo", "new_string": "bar"}),
                &ctx,
            )
            .await;
        assert!(matches!(result, Err(ToolError::ExecutionFailed(_))));
        if let Err(ToolError::ExecutionFailed(msg)) = result {
            assert!(msg.contains("3"), "expected count 3 in error: {msg}");
        }
    }

    #[tokio::test]
    async fn edit_file_not_exists() {
        let dir = TempDir::new().unwrap();
        let tool = EditTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let result = tool
            .execute(
                json!({"path": "missing.txt", "old_string": "x", "new_string": "y"}),
                &ctx,
            )
            .await;
        assert!(matches!(result, Err(ToolError::ExecutionFailed(_))));
    }

    #[tokio::test]
    async fn edit_stores_diff_in_graph() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("src.txt");
        std::fs::write(&file, "replace me once").unwrap();

        let tool = EditTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool
            .execute(
                json!({"path": "src.txt", "old_string": "replace me once", "new_string": "done"}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(out.node_id.is_some(), "should have graph node");
    }
}
