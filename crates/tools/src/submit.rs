//! No-op `submit` tool for compatibility with models (e.g. Qwen) that call it to
//! signal task completion. Without this, `Tool not found: submit` causes sessions
//! to end with status "failed" even when the work completed successfully.

use async_trait::async_trait;
use serde_json::json;

use crate::{Tool, ToolContext, ToolError, ToolOutput};

pub struct SubmitTool;

impl SubmitTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SubmitTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for SubmitTool {
    fn name(&self) -> &str {
        "submit"
    }

    fn description(&self) -> &str {
        "No-op. Some models call this to signal task completion. Returns success."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Optional summary of completed work"
                }
            }
        })
    }

    async fn execute(
        &self,
        _args: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::success("Task marked complete."))
    }
}
