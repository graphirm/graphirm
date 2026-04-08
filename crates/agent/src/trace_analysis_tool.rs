//! `trace_analysis` tool — cross-session failure-pattern report (read-only).
//!
//! Wraps [`crate::trace_analysis::build_trace_report`] so the agent can self-diagnose
//! harness issues without using the CLI or HTTP API.

use async_trait::async_trait;
use graphirm_tools::{Tool, ToolContext, ToolError, ToolOutput};
use serde_json::json;

/// Non-destructive tool that runs [`crate::trace_analysis::build_trace_report`] on the graph.
#[derive(Debug, Default, Clone, Copy)]
pub struct TraceAnalysisTool;

impl TraceAnalysisTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for TraceAnalysisTool {
    fn name(&self) -> &str {
        "trace_analysis"
    }

    fn description(&self) -> &str {
        "Read-only cross-session analysis: scans recent sessions for heuristic failure patterns \
         (over-tooling, doom loops, token waste, tool errors without recovery, premature completion) \
         and returns aggregated counts plus suggested config tweaks. Does not modify the graph."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "max_sessions": {
                    "type": "integer",
                    "description": "Maximum number of most recent sessions to analyze (default: 50)",
                    "default": 50
                }
            },
            "required": []
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let max_sessions = params
            .get("max_sessions")
            .and_then(|v| v.as_u64())
            .unwrap_or(50) as usize;

        let graph = ctx.graph.clone();
        let report = tokio::task::spawn_blocking(move || {
            crate::trace_analysis::build_trace_report(&graph, max_sessions)
        })
        .await
        .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        let json = serde_json::to_string_pretty(&report)
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        let content = format!("# Trace analysis report\n\n```json\n{json}\n```");
        Ok(ToolOutput::success(content))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphirm_graph::GraphStore;
    use graphirm_graph::nodes::{AgentData, GraphNode, InteractionData, NodeType};
    use graphirm_tools::ToolContext;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::AtomicU32;
    use tokio_util::sync::CancellationToken;

    fn make_ctx() -> ToolContext {
        let graph = Arc::new(GraphStore::open_memory().expect("memory graph"));
        let agent_node = GraphNode::new(NodeType::Agent(AgentData {
            name: "test-agent".to_string(),
            model: "test".to_string(),
            system_prompt: None,
            status: "active".to_string(),
        }));
        let interaction_node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "test".to_string(),
            token_count: None,
        }));
        let agent_id = graph.add_node(agent_node).expect("agent node");
        let interaction_id = graph.add_node(interaction_node).expect("interaction node");
        ToolContext {
            graph,
            agent_id,
            interaction_id,
            working_dir: PathBuf::from("/tmp"),
            signal: CancellationToken::new(),
            turn: 1,
            turn_pos_counter: Arc::new(AtomicU32::new(0)),
            knowledge_retriever: None,
            impact_provider: None,
            disable_bash: false,
            auto_link_write_to_planning: true,
        }
    }

    #[test]
    fn tool_name_and_description() {
        let t = TraceAnalysisTool::new();
        assert_eq!(t.name(), "trace_analysis");
        assert!(!t.description().is_empty());
    }

    #[test]
    fn parameters_allow_empty_object() {
        let t = TraceAnalysisTool::new();
        let params = t.parameters();
        let required = params["required"].as_array().unwrap();
        assert!(required.is_empty());
    }

    #[tokio::test]
    async fn execute_empty_graph_returns_zero_sessions() {
        let t = TraceAnalysisTool::new();
        let ctx = make_ctx();
        let out = t.execute(json!({}), &ctx).await.expect("execute");
        assert!(!out.is_error);
        assert!(out.content.contains("\"sessions_analyzed\": 0"));
    }

    #[tokio::test]
    async fn execute_respects_max_sessions_param() {
        let t = TraceAnalysisTool::new();
        let ctx = make_ctx();
        let out = t
            .execute(json!({ "max_sessions": 3 }), &ctx)
            .await
            .expect("execute");
        assert!(!out.is_error);
        assert!(out.content.contains("\"sessions_analyzed\""));
    }
}
