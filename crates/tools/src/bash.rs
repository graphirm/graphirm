use async_trait::async_trait;
use graphirm_graph::edges::{EdgeType, GraphEdge};
use graphirm_graph::nodes::{ContentData, GraphNode, NodeType};
use serde_json::json;
use std::time::Duration;
use tokio::process::Command;

use crate::{Tool, ToolContext, ToolError, ToolOutput};

pub struct BashTool;

impl BashTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for BashTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a bash command. Captures stdout and stderr. Non-zero exit codes are reported as errors."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                },
                "working_directory": {
                    "type": "string",
                    "description": "Working directory for the command (optional)"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 120)"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let command = args["command"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'command' field".into()))?;

        let timeout_secs = args["timeout"].as_u64().unwrap_or(120);

        let working_dir = if let Some(wd) = args["working_directory"].as_str() {
            std::path::PathBuf::from(wd)
        } else {
            ctx.working_dir.clone()
        };

        let mut cmd = Command::new("bash");
        cmd.arg("-c")
            .arg(command)
            .current_dir(&working_dir)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let child = cmd
            .spawn()
            .map_err(|e| ToolError::ExecutionFailed(format!("failed to spawn bash: {e}")))?;

        let signal = ctx.signal.clone();

        // Wrap the child in an abort-capable task so we can cancel it
        let mut task = tokio::spawn(async move { child.wait_with_output().await });

        let output = tokio::select! {
            result = &mut task => {
                result
                    .map_err(|e| ToolError::ExecutionFailed(format!("task join error: {e}")))?
                    .map_err(|e| ToolError::ExecutionFailed(format!("command error: {e}")))?
            }
            _ = tokio::time::sleep(Duration::from_secs(timeout_secs)) => {
                task.abort();
                return Err(ToolError::Timeout(timeout_secs));
            }
            _ = signal.cancelled() => {
                task.abort();
                return Err(ToolError::Cancelled);
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);

        let combined = if stderr.is_empty() {
            stdout.clone()
        } else if stdout.is_empty() {
            format!("stderr:\n{stderr}")
        } else {
            format!("{stdout}\nstderr:\n{stderr}")
        };

        let content_node = ctx
            .graph
            .add_node(GraphNode::new(NodeType::Content(ContentData {
                content_type: "command_output".to_string(),
                path: None,
                body: combined.clone(),
                language: None,
            })))?;

        ctx.graph.add_edge(GraphEdge::new(
            EdgeType::Produces,
            ctx.interaction_id.clone(),
            content_node.clone(),
        ))?;

        let is_error = !output.status.success();

        let output_text = if is_error {
            format!("Exit {exit_code}\n{combined}")
        } else {
            combined
        };

        if is_error {
            Ok(ToolOutput {
                content: output_text,
                is_error: true,
                node_id: Some(content_node),
            })
        } else {
            Ok(ToolOutput::success_with_node(output_text, content_node))
        }
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
    async fn bash_echo() {
        let tool = BashTool::new();
        let ctx = make_test_context();
        let out = tool
            .execute(json!({"command": "echo hello"}), &ctx)
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("hello"));
    }

    #[tokio::test]
    async fn bash_captures_stderr() {
        let tool = BashTool::new();
        let ctx = make_test_context();
        let out = tool
            .execute(json!({"command": "echo error >&2"}), &ctx)
            .await
            .unwrap();
        assert!(out.content.contains("error"));
    }

    #[tokio::test]
    async fn bash_exit_code_nonzero() {
        let tool = BashTool::new();
        let ctx = make_test_context();
        let out = tool
            .execute(json!({"command": "exit 1"}), &ctx)
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("Exit 1"));
    }

    #[tokio::test]
    async fn bash_working_directory() {
        let dir = TempDir::new().unwrap();
        let tool = BashTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool
            .execute(
                json!({"command": "pwd", "working_directory": dir.path().to_str().unwrap()}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.trim().contains(dir.path().to_str().unwrap()));
    }

    #[tokio::test]
    async fn bash_missing_command() {
        let tool = BashTool::new();
        let ctx = make_test_context();
        let result = tool.execute(json!({}), &ctx).await;
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[tokio::test]
    async fn bash_creates_graph_node() {
        let tool = BashTool::new();
        let ctx = make_test_context();
        let out = tool
            .execute(json!({"command": "echo tracked"}), &ctx)
            .await
            .unwrap();
        assert!(out.node_id.is_some(), "should create a graph node");
    }

    #[tokio::test]
    async fn bash_cancellation() {
        use tokio_util::sync::CancellationToken;

        let mut ctx = make_test_context();
        let token = CancellationToken::new();
        ctx.signal = token.clone();

        let tool = BashTool::new();

        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            token.cancel();
        });

        let result = tool.execute(json!({"command": "sleep 10"}), &ctx).await;
        assert!(matches!(result, Err(ToolError::Cancelled)));
    }
}
