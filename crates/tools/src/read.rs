use async_trait::async_trait;
use graphirm_graph::edges::{EdgeType, GraphEdge};
use graphirm_graph::nodes::{ContentData, GraphNode, NodeType};
use serde_json::json;

use crate::{Tool, ToolContext, ToolError, ToolOutput};

pub struct ReadTool;

impl ReadTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ReadTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn description(&self) -> &str {
        "Read a file from the filesystem, returning numbered lines."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (absolute or relative to working_dir)"
                },
                "offset": {
                    "type": "integer",
                    "description": "1-based line number to start reading from (optional)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to return (optional)"
                }
            },
            "required": ["path"]
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

        let full_path = if std::path::Path::new(path_str).is_absolute() {
            std::path::PathBuf::from(path_str)
        } else {
            ctx.working_dir.join(path_str)
        };

        let content = tokio::fs::read_to_string(&full_path).await.map_err(|e| {
            ToolError::ExecutionFailed(format!("failed to read '{}': {}", full_path.display(), e))
        })?;

        let offset = args["offset"].as_u64().unwrap_or(1).max(1) as usize;
        let limit = args["limit"].as_u64().map(|l| l as usize);

        let lines: Vec<&str> = content.lines().collect();
        let start = offset.saturating_sub(1);
        let end = match limit {
            Some(l) => (start + l).min(lines.len()),
            None => lines.len(),
        };

        let selected: Vec<&str> = lines.get(start..end).unwrap_or(&[]).to_vec();

        let width = end.to_string().len().max(4);
        let formatted = selected
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{:>width$}|{}", start + i + 1, line, width = width))
            .collect::<Vec<_>>()
            .join("\n");

        let mut node = GraphNode::new(NodeType::Content(ContentData {
            content_type: "file".to_string(),
            path: Some(full_path.to_string_lossy().to_string()),
            body: content.clone(),
            language: None,
        }));
        ctx.label_content_node(&mut node)?;
        let content_node = ctx.graph.add_node(node)?;

        ctx.graph.add_edge(GraphEdge::new(
            EdgeType::Reads,
            ctx.interaction_id.clone(),
            content_node.clone(),
        ))?;

        Ok(ToolOutput::success_with_node(formatted, content_node))
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
    async fn read_file_full() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "line one\nline two\nline three\n").unwrap();

        let tool = ReadTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool
            .execute(json!({"path": "test.txt"}), &ctx)
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("line one"));
        assert!(out.content.contains("line two"));
        assert!(out.content.contains("line three"));
    }

    #[tokio::test]
    async fn read_file_with_offset_and_limit() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("nums.txt");
        let mut f = std::fs::File::create(&file_path).unwrap();
        for i in 1..=10 {
            writeln!(f, "line {i}").unwrap();
        }

        let tool = ReadTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool
            .execute(json!({"path": "nums.txt", "offset": 3, "limit": 3}), &ctx)
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("line 3"));
        assert!(out.content.contains("line 4"));
        assert!(out.content.contains("line 5"));
        assert!(!out.content.contains("line 6"));
    }

    #[tokio::test]
    async fn read_file_not_found() {
        let dir = TempDir::new().unwrap();
        let tool = ReadTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let result = tool.execute(json!({"path": "nonexistent.txt"}), &ctx).await;
        assert!(matches!(result, Err(ToolError::ExecutionFailed(_))));
    }

    #[tokio::test]
    async fn read_missing_path_arg() {
        let tool = ReadTool::new();
        let ctx = make_test_context();
        let result = tool.execute(json!({}), &ctx).await;
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[tokio::test]
    async fn read_creates_graph_edges() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("tracked.txt");
        std::fs::write(&file_path, "hello world").unwrap();

        let tool = ReadTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool
            .execute(json!({"path": "tracked.txt"}), &ctx)
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

    #[tokio::test]
    async fn read_absolute_path() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("abs.txt");
        std::fs::write(&file_path, "absolute content").unwrap();

        let tool = ReadTool::new();
        let ctx = make_test_context();
        let out = tool
            .execute(json!({"path": file_path.to_str().unwrap()}), &ctx)
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("absolute content"));
    }
}
