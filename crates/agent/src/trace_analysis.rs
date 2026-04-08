use std::collections::{HashMap, HashSet};

use graphirm_graph::GraphStore;
use graphirm_graph::nodes::{GraphNode, NodeType};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionDigest {
    pub session_id: String,
    pub agent_name: String,
    pub status: String,
    pub turn_count: u32,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub tool_call_count: u32,
    pub tool_error_count: u32,
    pub unique_tools_used: Vec<String>,
    pub tools_gated_count: u32,
    pub fallback_count: u32,
    pub model_tiers: Vec<String>,
}

pub fn build_session_digest(graph: &GraphStore, session_id: &str) -> Option<SessionDigest> {
    let chain = graph.get_session_chain(session_id).ok()?;
    if chain.is_empty() {
        return None;
    }

    let mut total_input_tokens: u64 = 0;
    let mut total_output_tokens: u64 = 0;
    let mut tool_call_count: u32 = 0;
    let mut tool_error_count: u32 = 0;
    let mut tools_gated_count: u32 = 0;
    let mut fallback_count: u32 = 0;
    let mut turn_count: u32 = 0;
    let mut unique_tools: HashSet<String> = HashSet::new();
    let mut model_tiers: Vec<String> = Vec::new();

    for node in &chain {
        if let NodeType::Interaction(data) = &node.node_type {
            let meta = &node.metadata;

            match data.role.as_str() {
                "assistant" => {
                    turn_count += 1;

                    if let Some(v) = meta.get("usage_input").and_then(|v| v.as_u64()) {
                        total_input_tokens += v;
                    }
                    if let Some(v) = meta.get("usage_output").and_then(|v| v.as_u64()) {
                        total_output_tokens += v;
                    }
                    if let Some(tier) = meta.get("model_tier").and_then(|v| v.as_str()) {
                        model_tiers.push(tier.to_string());
                    }
                    if meta.get("tools_gated").and_then(|v| v.as_bool()) == Some(true) {
                        tools_gated_count += 1;
                    }
                    if let Some(arr) = meta.get("fallback_chain").and_then(|v| v.as_array())
                        && !arr.is_empty()
                    {
                        fallback_count += 1;
                    }
                    if let Some(calls) = meta.get("tool_calls").and_then(|v| v.as_array()) {
                        tool_call_count += calls.len() as u32;
                        for call in calls {
                            if let Some(name) = call.get("name").and_then(|v| v.as_str()) {
                                unique_tools.insert(name.to_string());
                            }
                        }
                    }
                }
                "tool" => {
                    if meta.get("is_error").and_then(|v| v.as_bool()) == Some(true) {
                        tool_error_count += 1;
                    }
                    if let Some(name) = meta.get("tool_name").and_then(|v| v.as_str()) {
                        unique_tools.insert(name.to_string());
                    }
                }
                _ => {}
            }
        }
    }

    let (agent_name, status) = find_agent_info(graph, session_id);

    let mut tools_sorted: Vec<String> = unique_tools.into_iter().collect();
    tools_sorted.sort();

    Some(SessionDigest {
        session_id: session_id.to_string(),
        agent_name,
        status,
        turn_count,
        total_input_tokens,
        total_output_tokens,
        tool_call_count,
        tool_error_count,
        unique_tools_used: tools_sorted,
        tools_gated_count,
        fallback_count,
        model_tiers,
    })
}

fn find_agent_info(graph: &GraphStore, session_id: &str) -> (String, String) {
    let agents = match graph.get_agent_nodes() {
        Ok(a) => a,
        Err(_) => return ("unknown".into(), "unknown".into()),
    };

    for (node, agent_data) in &agents {
        if node.metadata.get("session_id").and_then(|v| v.as_str()) == Some(session_id) {
            return (agent_data.name.clone(), agent_data.status.clone());
        }
    }

    ("unknown".into(), "unknown".into())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatch {
    pub pattern: String,
    pub severity: Severity,
    pub description: String,
    pub suggestion: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Info,
    Warning,
    Critical,
}

pub fn detect_over_tooling(digest: &SessionDigest, threshold: f64) -> Option<PatternMatch> {
    if digest.turn_count == 0 {
        return None;
    }
    let ratio = digest.tool_call_count as f64 / digest.turn_count as f64;
    if ratio > threshold {
        Some(PatternMatch {
            pattern: "over_tooling".into(),
            severity: Severity::Warning,
            description: format!(
                "Tool call ratio {ratio:.1} exceeds threshold {threshold:.1} ({} calls / {} turns)",
                digest.tool_call_count, digest.turn_count
            ),
            suggestion: "Agent may be thrashing — consider breaking the task into smaller steps."
                .into(),
        })
    } else {
        None
    }
}

/// Flag when >=3 consecutive tool-result nodes are all errors.
pub fn detect_doom_loops(chain: &[GraphNode]) -> Option<PatternMatch> {
    let mut consecutive_errors: u32 = 0;
    for node in chain {
        if let NodeType::Interaction(data) = &node.node_type
            && data.role == "tool"
        {
            let is_error = node
                .metadata
                .get("is_error")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if is_error {
                consecutive_errors += 1;
                if consecutive_errors >= 3 {
                    return Some(PatternMatch {
                        pattern: "doom_loops".into(),
                        severity: Severity::Critical,
                        description: format!(
                            "{consecutive_errors} consecutive tool errors detected"
                        ),
                        suggestion: "Agent is stuck in an error loop — intervene or abort session."
                            .into(),
                    });
                }
            } else {
                consecutive_errors = 0;
            }
        }
    }
    None
}

pub fn detect_token_waste(digest: &SessionDigest, threshold: u64) -> Option<PatternMatch> {
    if digest.turn_count == 0 {
        return None;
    }
    let per_turn = digest.total_output_tokens / digest.turn_count as u64;
    if per_turn > threshold && digest.status != "completed" {
        Some(PatternMatch {
            pattern: "token_waste".into(),
            severity: Severity::Warning,
            description: format!(
                "Output tokens per turn ({per_turn}) exceeds threshold ({threshold}) \
                 with status '{}'",
                digest.status
            ),
            suggestion: "High token spend without completion — check if the task is well-scoped."
                .into(),
        })
    } else {
        None
    }
}

/// Flag any tool that errored and was never successfully used again.
pub fn detect_tool_errors_without_recovery(chain: &[GraphNode]) -> Option<PatternMatch> {
    // tool_name → true means "errored and not yet recovered"
    let mut unrecovered: HashMap<String, bool> = HashMap::new();

    for node in chain {
        if let NodeType::Interaction(data) = &node.node_type {
            if data.role != "tool" {
                continue;
            }
            let tool_name = match node.metadata.get("tool_name").and_then(|v| v.as_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };
            let is_error = node
                .metadata
                .get("is_error")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            if is_error {
                unrecovered.entry(tool_name).or_insert(true);
            } else if unrecovered.get(&tool_name) == Some(&true) {
                unrecovered.insert(tool_name, false);
            }
        }
    }

    let failed: Vec<&String> = unrecovered
        .iter()
        .filter(|(_, still_bad)| **still_bad)
        .map(|(name, _)| name)
        .collect();

    if failed.is_empty() {
        None
    } else {
        let mut names: Vec<&str> = failed.iter().map(|s| s.as_str()).collect();
        names.sort();
        Some(PatternMatch {
            pattern: "tool_errors_without_recovery".into(),
            severity: Severity::Warning,
            description: format!(
                "Tool(s) errored and never succeeded afterward: {}",
                names.join(", ")
            ),
            suggestion: "Agent abandoned failed tool(s) without retrying — may indicate a \
                          misunderstanding of the error."
                .into(),
        })
    }
}

pub fn detect_premature_completion(digest: &SessionDigest) -> Option<PatternMatch> {
    if digest.status == "completed" && digest.turn_count <= 2 && digest.tool_call_count == 0 {
        Some(PatternMatch {
            pattern: "premature_completion".into(),
            severity: Severity::Info,
            description: format!(
                "Session completed in {} turn(s) with 0 tool calls",
                digest.turn_count
            ),
            suggestion: "Agent declared completion without doing any work — verify the outcome."
                .into(),
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphirm_graph::nodes::{AgentData, GraphNode, InteractionData};
    use std::sync::Arc;

    #[test]
    fn digest_empty_session_returns_none() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let digest = build_session_digest(&graph, "nonexistent");
        assert!(digest.is_none());
    }

    #[test]
    fn digest_with_single_turn() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let session_id = "sess-1";

        let mut agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "coder".into(),
            model: "deepseek".into(),
            system_prompt: None,
            status: "completed".into(),
        }));
        agent.metadata = serde_json::json!({ "session_id": session_id });
        graph.add_node(agent).unwrap();

        let mut user_msg = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".into(),
            content: "hello".into(),
            token_count: None,
        }));
        user_msg.metadata = serde_json::json!({ "session_id": session_id });
        graph.add_node(user_msg).unwrap();

        let mut asst = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".into(),
            content: "hi there".into(),
            token_count: None,
        }));
        asst.metadata = serde_json::json!({
            "session_id": session_id,
            "usage_input": 100,
            "usage_output": 50,
            "model_tier": "cheap",
            "tool_calls": [{ "name": "read" }, { "name": "write" }],
        });
        graph.add_node(asst).unwrap();

        let digest = build_session_digest(&graph, session_id).unwrap();
        assert_eq!(digest.session_id, session_id);
        assert_eq!(digest.agent_name, "coder");
        assert_eq!(digest.status, "completed");
        assert_eq!(digest.turn_count, 1);
        assert_eq!(digest.total_input_tokens, 100);
        assert_eq!(digest.total_output_tokens, 50);
        assert_eq!(digest.tool_call_count, 2);
        assert_eq!(digest.tool_error_count, 0);
        assert_eq!(digest.unique_tools_used, vec!["read", "write"]);
        assert_eq!(digest.tools_gated_count, 0);
        assert_eq!(digest.fallback_count, 0);
        assert_eq!(digest.model_tiers, vec!["cheap"]);
    }

    #[test]
    fn digest_counts_tool_errors_and_gated() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let session_id = "sess-err";

        let mut asst = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".into(),
            content: "calling tools".into(),
            token_count: None,
        }));
        asst.metadata = serde_json::json!({
            "session_id": session_id,
            "usage_input": 200,
            "usage_output": 80,
            "model_tier": "smart",
            "tools_gated": true,
            "fallback_chain": [{ "model": "m1", "error": "rate limit" }],
        });
        graph.add_node(asst).unwrap();

        let mut tool_ok = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "tool".into(),
            content: "ok".into(),
            token_count: None,
        }));
        tool_ok.metadata = serde_json::json!({
            "session_id": session_id,
            "tool_name": "bash",
            "is_error": false,
        });
        graph.add_node(tool_ok).unwrap();

        let mut tool_err = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "tool".into(),
            content: "error".into(),
            token_count: None,
        }));
        tool_err.metadata = serde_json::json!({
            "session_id": session_id,
            "tool_name": "write",
            "is_error": true,
        });
        graph.add_node(tool_err).unwrap();

        let digest = build_session_digest(&graph, session_id).unwrap();
        assert_eq!(digest.turn_count, 1);
        assert_eq!(digest.total_input_tokens, 200);
        assert_eq!(digest.total_output_tokens, 80);
        assert_eq!(digest.tool_error_count, 1);
        assert_eq!(digest.tools_gated_count, 1);
        assert_eq!(digest.fallback_count, 1);
        assert_eq!(digest.unique_tools_used, vec!["bash", "write"]);
        assert_eq!(digest.model_tiers, vec!["smart"]);
        assert_eq!(digest.agent_name, "unknown");
    }

    #[test]
    fn digest_multiple_turns_accumulates() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let session_id = "sess-multi";

        for i in 0..3 {
            let mut asst = GraphNode::new(NodeType::Interaction(InteractionData {
                role: "assistant".into(),
                content: format!("turn {i}"),
                token_count: None,
            }));
            asst.metadata = serde_json::json!({
                "session_id": session_id,
                "usage_input": 100,
                "usage_output": 50,
                "model_tier": if i < 2 { "cheap" } else { "smart" },
            });
            graph.add_node(asst).unwrap();
        }

        let digest = build_session_digest(&graph, session_id).unwrap();
        assert_eq!(digest.turn_count, 3);
        assert_eq!(digest.total_input_tokens, 300);
        assert_eq!(digest.total_output_tokens, 150);
        assert_eq!(digest.model_tiers, vec!["cheap", "cheap", "smart"]);
    }

    fn default_digest() -> SessionDigest {
        SessionDigest {
            session_id: "test".into(),
            agent_name: "test".into(),
            status: "completed".into(),
            turn_count: 5,
            total_input_tokens: 500,
            total_output_tokens: 250,
            tool_call_count: 8,
            tool_error_count: 0,
            unique_tools_used: vec![],
            tools_gated_count: 0,
            fallback_count: 0,
            model_tiers: vec![],
        }
    }

    fn make_tool_node(tool_name: &str, is_error: bool) -> GraphNode {
        let mut node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "tool".into(),
            content: if is_error { "error" } else { "ok" }.into(),
            token_count: None,
        }));
        node.metadata = serde_json::json!({
            "tool_name": tool_name,
            "is_error": is_error,
        });
        node
    }

    // --- Over-tooling ---

    #[test]
    fn detect_over_tooling_flags_high_ratio() {
        let digest = SessionDigest {
            tool_call_count: 30,
            turn_count: 5,
            ..default_digest()
        };
        let m = detect_over_tooling(&digest, 3.0).unwrap();
        assert_eq!(m.pattern, "over_tooling");
        assert_eq!(m.severity, Severity::Warning);
    }

    #[test]
    fn detect_over_tooling_passes_normal_ratio() {
        let digest = SessionDigest {
            tool_call_count: 8,
            turn_count: 5,
            ..default_digest()
        };
        assert!(detect_over_tooling(&digest, 3.0).is_none());
    }

    #[test]
    fn detect_over_tooling_zero_turns_returns_none() {
        let digest = SessionDigest {
            turn_count: 0,
            ..default_digest()
        };
        assert!(detect_over_tooling(&digest, 3.0).is_none());
    }

    // --- Doom loops ---

    #[test]
    fn detect_doom_loops_flags_three_consecutive_errors() {
        let chain = vec![
            make_tool_node("write", true),
            make_tool_node("write", true),
            make_tool_node("write", true),
        ];
        let m = detect_doom_loops(&chain).unwrap();
        assert_eq!(m.pattern, "doom_loops");
        assert_eq!(m.severity, Severity::Critical);
    }

    #[test]
    fn detect_doom_loops_passes_when_success_breaks_streak() {
        let chain = vec![
            make_tool_node("write", true),
            make_tool_node("write", true),
            make_tool_node("write", false),
            make_tool_node("write", true),
        ];
        assert!(detect_doom_loops(&chain).is_none());
    }

    // --- Token waste ---

    #[test]
    fn detect_token_waste_flags_high_output_with_failure() {
        let digest = SessionDigest {
            total_output_tokens: 15000,
            turn_count: 5,
            status: "failed".into(),
            ..default_digest()
        };
        let m = detect_token_waste(&digest, 2000).unwrap();
        assert_eq!(m.pattern, "token_waste");
        assert_eq!(m.severity, Severity::Warning);
    }

    #[test]
    fn detect_token_waste_passes_when_completed() {
        let digest = SessionDigest {
            total_output_tokens: 15000,
            turn_count: 5,
            status: "completed".into(),
            ..default_digest()
        };
        assert!(detect_token_waste(&digest, 2000).is_none());
    }

    #[test]
    fn detect_token_waste_passes_when_below_threshold() {
        let digest = SessionDigest {
            total_output_tokens: 5000,
            turn_count: 5,
            status: "failed".into(),
            ..default_digest()
        };
        assert!(detect_token_waste(&digest, 2000).is_none());
    }

    // --- Tool errors without recovery ---

    #[test]
    fn detect_tool_errors_without_recovery_flags_unrecovered() {
        let chain = vec![make_tool_node("bash", true), make_tool_node("read", false)];
        let m = detect_tool_errors_without_recovery(&chain).unwrap();
        assert_eq!(m.pattern, "tool_errors_without_recovery");
        assert!(m.description.contains("bash"));
    }

    #[test]
    fn detect_tool_errors_without_recovery_passes_when_recovered() {
        let chain = vec![make_tool_node("bash", true), make_tool_node("bash", false)];
        assert!(detect_tool_errors_without_recovery(&chain).is_none());
    }

    // --- Premature completion ---

    #[test]
    fn detect_premature_completion_flags_zero_tool_short_session() {
        let digest = SessionDigest {
            status: "completed".into(),
            turn_count: 1,
            tool_call_count: 0,
            ..default_digest()
        };
        let m = detect_premature_completion(&digest).unwrap();
        assert_eq!(m.pattern, "premature_completion");
        assert_eq!(m.severity, Severity::Info);
    }

    #[test]
    fn detect_premature_completion_passes_with_tool_calls() {
        let digest = SessionDigest {
            status: "completed".into(),
            turn_count: 1,
            tool_call_count: 3,
            ..default_digest()
        };
        assert!(detect_premature_completion(&digest).is_none());
    }

    #[test]
    fn digest_serde_roundtrip() {
        let digest = SessionDigest {
            session_id: "s1".into(),
            agent_name: "test".into(),
            status: "completed".into(),
            turn_count: 5,
            total_input_tokens: 1000,
            total_output_tokens: 500,
            tool_call_count: 10,
            tool_error_count: 1,
            unique_tools_used: vec!["read".into(), "write".into()],
            tools_gated_count: 2,
            fallback_count: 0,
            model_tiers: vec!["cheap".into(), "smart".into()],
        };
        let json = serde_json::to_string(&digest).unwrap();
        let back: SessionDigest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.session_id, "s1");
        assert_eq!(back.turn_count, 5);
        assert_eq!(back.total_input_tokens, 1000);
        assert_eq!(back.unique_tools_used, vec!["read", "write"]);
    }
}
