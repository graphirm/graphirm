//! context_report tool — summarises graph context utilisation for a session.
//!
//! Reads `context_stats` metadata from every assistant `Interaction` node in the
//! session and aggregates averages + counts into a markdown table.

use async_trait::async_trait;
use serde_json::json;

use crate::{Tool, ToolContext, ToolError, ToolOutput};

pub struct ContextReportTool;

impl ContextReportTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ContextReportTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract a u64 from a context_stats JSON sub-object.
fn stat_u64(stats: &serde_json::Value, key: &str) -> u64 {
    stats.get(key).and_then(|v| v.as_u64()).unwrap_or(0)
}

/// Extract a f64 from a context_stats JSON sub-object.
fn stat_f64(stats: &serde_json::Value, key: &str) -> f64 {
    stats.get(key).and_then(|v| v.as_f64()).unwrap_or(0.0)
}

/// Extract a bool from a context_stats JSON sub-object.
fn stat_bool(stats: &serde_json::Value, key: &str) -> bool {
    stats.get(key).and_then(|v| v.as_bool()).unwrap_or(false)
}

#[async_trait]
impl Tool for ContextReportTool {
    fn name(&self) -> &str {
        "context_report"
    }

    fn description(&self) -> &str {
        "Summarise graph context utilisation statistics for a session. \
         Shows averages for knowledge nodes, graph token percentage, pinned conventions, \
         cross-session links, and counts for compaction and repo-briefing injections."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "ID of the session to report on"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of turns to scan (default: 50)",
                    "default": 50
                }
            },
            "required": ["session_id"]
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let session_id = params
            .get("session_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArguments("session_id is required".into()))?
            .to_string();

        let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;

        let graph = ctx.graph.clone();

        let result = tokio::task::spawn_blocking(move || {
            let nodes = graph
                .get_session_interactions(&session_id)
                .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

            let mut total_knowledge = 0u64;
            let mut total_cross = 0u64;
            let mut total_pinned = 0u64;
            let mut total_token_pct = 0f64;
            let mut compaction_count = 0usize;
            let mut briefing_count = 0usize;
            let mut turns_with_stats = 0usize;

            for node in nodes.iter().take(limit) {
                if let Some(v) = node.metadata.get("context_stats")
                    && v.is_object()
                {
                    total_knowledge += stat_u64(v, "knowledge_count");
                    total_cross += stat_u64(v, "cross_session_links_count");
                    total_pinned += stat_u64(v, "pinned_conventions_count");
                    total_token_pct += stat_f64(v, "graph_token_percentage");
                    if stat_bool(v, "compaction_triggered") {
                        compaction_count += 1;
                    }
                    if stat_bool(v, "repo_briefing_included") {
                        briefing_count += 1;
                    }
                    turns_with_stats += 1;
                }
            }

            let n = turns_with_stats.max(1) as f64;
            let avg_knowledge = total_knowledge as f64 / n;
            let avg_cross = total_cross as f64 / n;
            let avg_pinned = total_pinned as f64 / n;
            let avg_token_pct = total_token_pct / n;

            let output = format!(
                "## Context Report — session `{session_id}`\n\n\
                 | Metric | Average |\n\
                 |--------|--------|\n\
                 | Knowledge nodes / turn | {avg_knowledge:.1} |\n\
                 | Graph token % / turn | {avg_token_pct:.1}% |\n\
                 | Pinned conventions / turn | {avg_pinned:.1} |\n\
                 | Cross-session links / turn | {avg_cross:.1} |\n\n\
                 **Turns with stats:** {turns_with_stats}  \n\
                 **Compaction triggered:** {compaction_count}  \n\
                 **Repo briefing included:** {briefing_count}"
            );

            Ok::<_, ToolError>(output)
        })
        .await
        .map_err(|e| ToolError::ExecutionFailed(e.to_string()))??;

        Ok(ToolOutput::success(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_name_and_description() {
        let t = ContextReportTool::new();
        assert_eq!(t.name(), "context_report");
        assert!(!t.description().is_empty());
    }

    #[test]
    fn parameters_require_session_id() {
        let t = ContextReportTool::new();
        let params = t.parameters();
        let required = params["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v.as_str() == Some("session_id")));
    }

    #[test]
    fn stat_helpers_handle_missing_keys() {
        let v = serde_json::json!({"a": 5, "b": 3.14, "c": true});
        assert_eq!(stat_u64(&v, "a"), 5);
        assert_eq!(stat_u64(&v, "missing"), 0);
        assert!((stat_f64(&v, "b") - 3.14).abs() < f64::EPSILON);
        assert_eq!(stat_f64(&v, "missing"), 0.0);
        assert!(stat_bool(&v, "c"));
        assert!(!stat_bool(&v, "missing"));
    }
}
