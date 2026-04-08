use std::collections::HashSet;

use graphirm_graph::GraphStore;
use graphirm_graph::nodes::NodeType;
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
