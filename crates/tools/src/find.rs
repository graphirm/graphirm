use async_trait::async_trait;
use graphirm_graph::edges::{EdgeType, GraphEdge};
use graphirm_graph::nodes::{ContentData, GraphNode, NodeType};
use serde_json::json;

use crate::{Tool, ToolContext, ToolError, ToolOutput};

const MAX_RESULTS: usize = 1000;

pub struct FindTool;

impl FindTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for FindTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for FindTool {
    fn name(&self) -> &str {
        "find"
    }

    fn description(&self) -> &str {
        "Find files matching a glob pattern. Returns relative paths sorted alphabetically."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match (e.g. '**/*.rs')"
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (optional, defaults to working_dir)"
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
        let pattern_str = args["pattern"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'pattern' field".into()))?;

        let search_dir = if let Some(p) = args["path"].as_str() {
            if std::path::Path::new(p).is_absolute() {
                std::path::PathBuf::from(p)
            } else {
                ctx.working_dir.join(p)
            }
        } else {
            ctx.working_dir.clone()
        };

        let full_pattern = format!("{}/{}", search_dir.display(), pattern_str);

        let mut matches: Vec<String> = glob::glob(&full_pattern)
            .map_err(|e| ToolError::InvalidArguments(format!("invalid glob pattern: {e}")))?
            .take(MAX_RESULTS)
            .filter_map(|entry| entry.ok())
            .filter_map(|path| {
                path.strip_prefix(&search_dir)
                    .ok()
                    .map(|rel| rel.to_string_lossy().to_string())
            })
            .collect();

        matches.sort();

        let result_text = if matches.is_empty() {
            "No files found.".to_string()
        } else {
            matches.join("\n")
        };

        let mut node = GraphNode::new(NodeType::Content(ContentData {
            content_type: "file_listing".to_string(),
            path: Some(search_dir.to_string_lossy().to_string()),
            body: result_text.clone(),
            language: None,
        }));
        ctx.label_content_node(&mut node)?;
        let content_node = ctx.graph.add_node(node)?;

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
    async fn find_rs_files() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.path().join("lib.rs"), "pub fn lib() {}").unwrap();
        std::fs::write(dir.path().join("readme.md"), "# docs").unwrap();

        let tool = FindTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool
            .execute(json!({"pattern": "**/*.rs"}), &ctx)
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("main.rs"));
        assert!(out.content.contains("lib.rs"));
        assert!(!out.content.contains("readme.md"));
    }

    #[tokio::test]
    async fn find_no_matches() {
        let dir = TempDir::new().unwrap();
        let tool = FindTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool
            .execute(json!({"pattern": "**/*.nonexistent"}), &ctx)
            .await
            .unwrap();
        assert!(!out.is_error);
        assert_eq!(out.content.trim(), "No files found.");
    }

    #[tokio::test]
    async fn find_specific_subdir() {
        let dir = TempDir::new().unwrap();
        let subdir = dir.path().join("src");
        std::fs::create_dir_all(&subdir).unwrap();
        std::fs::write(subdir.join("main.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.path().join("other.rs"), "// other").unwrap();

        let tool = FindTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool
            .execute(json!({"pattern": "*.rs", "path": "src"}), &ctx)
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("main.rs"));
        assert!(!out.content.contains("other.rs"));
    }

    #[tokio::test]
    async fn find_invalid_glob() {
        let dir = TempDir::new().unwrap();
        let tool = FindTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let result = tool.execute(json!({"pattern": "[invalid"}), &ctx).await;
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[tokio::test]
    async fn find_creates_graph_node() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), "a").unwrap();

        let tool = FindTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool
            .execute(json!({"pattern": "*.txt"}), &ctx)
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
