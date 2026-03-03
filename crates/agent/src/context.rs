// Context engine: graph traversal for relevant context, relevance scoring

use graphirm_graph::edges::EdgeType;
use graphirm_graph::nodes::NodeType;
use graphirm_graph::Direction;
use graphirm_llm::{ContentPart, LlmMessage, Role};

use crate::error::AgentError;
use crate::session::Session;

/// Build the LLM message context from the session's conversation graph.
///
/// MVP strategy: get all Interaction nodes linked to the Agent node via
/// Produces edges, sort by created_at, and convert to LlmMessage format.
/// Prepend the system prompt.
pub fn build_context(session: &Session) -> Result<Vec<LlmMessage>, AgentError> {
    let mut messages = Vec::new();

    // System prompt
    messages.push(LlmMessage::system(session.agent_config.system_prompt.clone()));

    // Get all conversation nodes linked to this agent
    let mut interactions = session
        .graph
        .neighbors(&session.id, Some(EdgeType::Produces), Direction::Outgoing)
        .map_err(|e| AgentError::Context(e.to_string()))?;

    // Sort by created_at timestamp
    interactions.sort_by(|a, b| a.created_at.cmp(&b.created_at));

    // Enforce max_context_messages budget: keep the most recent N messages.
    // The system prompt is always prepended regardless of this limit.
    if let Some(limit) = session.agent_config.max_context_messages {
        let len = interactions.len();
        if len > limit {
            interactions.drain(..len - limit);
        }
    }

    // Convert each node to an LlmMessage
    for node in &interactions {
        let NodeType::Interaction(data) = &node.node_type else {
            continue;
        };

        let role = match data.role.as_str() {
            "system" => Role::System,
            "user" => Role::Human,
            "assistant" => Role::Assistant,
            "tool" => Role::ToolResult,
            other => {
                return Err(AgentError::Context(format!(
                    "unknown interaction role '{other}' on node {}",
                    node.id
                )))
            }
        };

        match role {
            Role::Human => {
                messages.push(LlmMessage::human(data.content.clone()));
            }
            Role::Assistant => {
                // Check for tool calls in metadata (stored when we record assistant response)
                let tool_calls: Vec<(String, String, serde_json::Value)> = node
                    .metadata
                    .get("tool_calls")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| {
                                let id = v.get("id")?.as_str()?.to_string();
                                let name = v.get("name")?.as_str()?.to_string();
                                let arguments = v.get("arguments").cloned().unwrap_or(serde_json::Value::Null);
                                Some((id, name, arguments))
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                if tool_calls.is_empty() {
                    messages.push(LlmMessage::assistant(data.content.clone()));
                } else {
                    let mut content = Vec::new();
                    if !data.content.is_empty() {
                        content.push(ContentPart::text(data.content.clone()));
                    }
                    for (id, name, arguments) in tool_calls {
                        content.push(ContentPart::tool_call(id, name, arguments));
                    }
                    messages.push(LlmMessage::new(Role::Assistant, content));
                }
            }
            Role::ToolResult => {
                let tool_call_id = node
                    .metadata
                    .get("tool_call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let is_error = node.metadata.get("is_error").and_then(|v| v.as_bool()).unwrap_or(false);
                messages.push(LlmMessage::tool_result(
                    tool_call_id,
                    data.content.clone(),
                    is_error,
                ));
            }
            Role::System => {
                messages.push(LlmMessage::system(data.content.clone()));
            }
        }
    }

    Ok(messages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AgentConfig;
    use graphirm_graph::edges::GraphEdge;
    use graphirm_graph::nodes::{GraphNode, InteractionData};
    use graphirm_graph::GraphStore;
    use std::sync::Arc;

    #[test]
    fn test_build_context_empty_conversation() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig {
            system_prompt: "You are helpful.".to_string(),
            ..AgentConfig::default()
        };
        let session = Session::new(graph, config).unwrap();

        let context = build_context(&session).unwrap();

        assert_eq!(context.len(), 1);
        assert_eq!(context[0].role, Role::System);
        assert_eq!(
            context[0].content.first().map(|c| match c {
                ContentPart::Text { text } => text.as_str(),
                _ => "",
            }),
            Some("You are helpful.")
        );
    }

    #[test]
    fn test_build_context_with_messages() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig {
            system_prompt: "Be concise.".to_string(),
            ..AgentConfig::default()
        };
        let session = Session::new(graph.clone(), config).unwrap();

        let roles = ["user", "assistant", "user", "assistant", "user"];
        for (i, role) in roles.iter().enumerate() {
            let node = GraphNode::new(NodeType::Interaction(InteractionData {
                role: (*role).to_string(),
                content: format!("Message {}", i),
                token_count: None,
            }));
            let msg_id = graph.add_node(node).unwrap();
            graph
                .add_edge(GraphEdge::new(
                    EdgeType::Produces,
                    session.id.clone(),
                    msg_id.clone(),
                ))
                .unwrap();
        }

        let context = build_context(&session).unwrap();

        assert_eq!(context.len(), 6);
        assert_eq!(context[0].role, Role::System);
        assert_eq!(
            context[1].content.first().map(|c| match c {
                ContentPart::Text { text } => text.as_str(),
                _ => "",
            }),
            Some("Message 0")
        );
        assert_eq!(
            context[5].content.first().map(|c| match c {
                ContentPart::Text { text } => text.as_str(),
                _ => "",
            }),
            Some("Message 4")
        );
    }

    #[test]
    fn test_build_context_includes_tool_results() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig::default();
        let session = Session::new(graph.clone(), config).unwrap();

        // User message
        let user_node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "Run ls".to_string(),
            token_count: None,
        }));
        let user_id = graph.add_node(user_node).unwrap();
        graph
            .add_edge(GraphEdge::new(EdgeType::Produces, session.id.clone(), user_id))
            .unwrap();

        // Assistant with tool call (stored in metadata)
        let mut asst_metadata = serde_json::Map::new();
        asst_metadata.insert(
            "tool_calls".to_string(),
            serde_json::json!([{"id": "call_1", "name": "bash", "arguments": {"command": "ls"}}]),
        );
        let mut asst_node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".to_string(),
            content: "".to_string(),
            token_count: None,
        }));
        asst_node.metadata = serde_json::Value::Object(asst_metadata);
        let asst_id = graph.add_node(asst_node).unwrap();
        graph
            .add_edge(GraphEdge::new(EdgeType::Produces, session.id.clone(), asst_id))
            .unwrap();

        // Tool result (tool_call_id and is_error in metadata)
        let mut tool_metadata = serde_json::Map::new();
        tool_metadata.insert("tool_call_id".to_string(), serde_json::json!("call_1"));
        tool_metadata.insert("is_error".to_string(), serde_json::json!(false));
        let mut tool_node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "tool".to_string(),
            content: "file1.rs\nfile2.rs".to_string(),
            token_count: None,
        }));
        tool_node.metadata = serde_json::Value::Object(tool_metadata);
        let tool_id = graph.add_node(tool_node).unwrap();
        graph
            .add_edge(GraphEdge::new(EdgeType::Produces, session.id.clone(), tool_id))
            .unwrap();

        let context = build_context(&session).unwrap();

        assert_eq!(context.len(), 4);
        assert_eq!(context[3].role, Role::ToolResult);
        match &context[3].content[0] {
            ContentPart::ToolResult { id, content, .. } => {
                assert_eq!(id, "call_1");
                assert_eq!(content, "file1.rs\nfile2.rs");
            }
            _ => panic!("expected ToolResult content part"),
        }
    }
}
