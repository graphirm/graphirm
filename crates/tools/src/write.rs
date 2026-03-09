use async_trait::async_trait;
use graphirm_graph::edges::{EdgeType, GraphEdge};
use graphirm_graph::nodes::{ContentData, GraphNode, NodeType};
use serde_json::json;

use crate::{Tool, ToolContext, ToolError, ToolOutput};

pub struct WriteTool;

impl WriteTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for WriteTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for WriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn description(&self) -> &str {
        "Write content to a file, creating parent directories as needed."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to write (absolute or relative to working_dir)"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
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

        let content = args["content"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'content' field".into()))?;

        let full_path = if std::path::Path::new(path_str).is_absolute() {
            std::path::PathBuf::from(path_str)
        } else {
            ctx.working_dir.join(path_str)
        };

        let existed = full_path.exists();

        if let Some(parent) = full_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        tokio::fs::write(&full_path, content).await?;

        let action = if existed { "updated" } else { "created" };

        let mut node = GraphNode::new(NodeType::Content(ContentData {
            content_type: "file".to_string(),
            path: Some(full_path.to_string_lossy().to_string()),
            body: content.to_string(),
            language: None,
        }));
        ctx.label_content_node(&mut node)?;
        let content_node = ctx.graph.add_node(node)?;

        ctx.graph.add_edge(GraphEdge::new(
            EdgeType::Modifies,
            ctx.interaction_id.clone(),
            content_node.clone(),
        ))?;

        Ok(ToolOutput::success_with_node(
            format!("File {} '{}'", action, full_path.display()),
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
    async fn write_creates_file() {
        let dir = TempDir::new().unwrap();
        let tool = WriteTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool
            .execute(json!({"path": "hello.txt", "content": "hello world"}), &ctx)
            .await
            .unwrap();
        assert!(!out.is_error);
        let written = std::fs::read_to_string(dir.path().join("hello.txt")).unwrap();
        assert_eq!(written, "hello world");
        assert!(out.content.contains("created"));
    }

    #[tokio::test]
    async fn write_overwrites_existing() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("existing.txt"), "old content").unwrap();

        let tool = WriteTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool
            .execute(
                json!({"path": "existing.txt", "content": "new content"}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("updated"));
        let written = std::fs::read_to_string(dir.path().join("existing.txt")).unwrap();
        assert_eq!(written, "new content");
    }

    #[tokio::test]
    async fn write_creates_parent_dirs() {
        let dir = TempDir::new().unwrap();
        let tool = WriteTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool
            .execute(json!({"path": "a/b/c/file.txt", "content": "nested"}), &ctx)
            .await
            .unwrap();
        assert!(!out.is_error);
        let written = std::fs::read_to_string(dir.path().join("a/b/c/file.txt")).unwrap();
        assert_eq!(written, "nested");
    }

    #[tokio::test]
    async fn write_missing_args() {
        let tool = WriteTool::new();
        let ctx = make_test_context();

        let result = tool.execute(json!({"content": "data"}), &ctx).await;
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));

        let result = tool.execute(json!({"path": "foo.txt"}), &ctx).await;
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[tokio::test]
    async fn write_creates_graph_edges() {
        let dir = TempDir::new().unwrap();
        let tool = WriteTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool
            .execute(json!({"path": "graph.txt", "content": "tracked"}), &ctx)
            .await
            .unwrap();
        let node_id = out.node_id.expect("should have created a graph node");
        let node = ctx.graph.get_node(&node_id).unwrap();
        assert_eq!(node.label(), Some("content_1_1_1"));
        assert_eq!(
            node.metadata.get("session_id"),
            Some(&serde_json::json!(ctx.agent_id.to_string()))
        );
    }
}
