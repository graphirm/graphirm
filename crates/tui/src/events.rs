use chrono::Utc;

use graphirm_agent::AgentEvent;
use graphirm_graph::nodes::{GraphNode, NodeType};
use graphirm_llm::{Role, StreamEvent};

use crate::app::{App, AppState, ChatMessage, GraphNodeEntry};

pub fn handle_agent_event(app: &mut App, event: AgentEvent) {
    match event {
        AgentEvent::AgentStart { .. } => {
            app.state = AppState::WaitingForAgent;
            app.status_bar.agent_state = "Working".to_string();
        }
        AgentEvent::AgentEnd { .. } => {
            app.state = AppState::Idle;
            app.status_bar.agent_state = "Idle".to_string();
        }
        AgentEvent::TurnStart { .. } => {
            app.status_bar.agent_state = "Thinking...".to_string();
        }
        AgentEvent::TurnEnd { .. } => {
            app.status_bar.agent_state = "Working".to_string();
        }
        AgentEvent::MessageStart { node_id } => {
            app.state = AppState::Streaming;
            app.chat.add_message(ChatMessage {
                role: Role::Assistant,
                content: String::new(),
                timestamp: Utc::now(),
                node_id: Some(node_id),
                is_tool_call: false,
                tool_name: None,
            });
            app.chat.scroll_to_bottom();
        }
        AgentEvent::MessageDelta { delta, .. } => {
            if let StreamEvent::TextDelta(text) = delta {
                app.chat.append_delta(&text);
                app.chat.scroll_to_bottom();
            }
        }
        AgentEvent::MessageEnd { .. } => {
            app.state = AppState::WaitingForAgent;
            app.chat.scroll_to_bottom();
        }
        AgentEvent::ToolStart {
            response_node_id,
            tool_name,
            ..
        } => {
            app.status_bar.agent_state = format!("Running {}...", tool_name);
            app.chat.add_message(ChatMessage {
                role: Role::ToolResult,
                content: format!("Executing {}...", tool_name),
                timestamp: Utc::now(),
                node_id: Some(response_node_id),
                is_tool_call: true,
                tool_name: Some(tool_name),
            });
            app.chat.scroll_to_bottom();
        }
        AgentEvent::ToolEnd { is_error, .. } => {
            if is_error {
                app.status_bar.agent_state = "Tool Error".to_string();
            } else {
                app.status_bar.agent_state = "Working".to_string();
            }
        }
        AgentEvent::GraphUpdate { recent_nodes, .. } => {
            app.graph_explorer.nodes = recent_nodes
                .into_iter()
                .map(|n| graph_node_to_entry(n))
                .collect();
        }
    }
}

/// Build a short display label for a graph node.
fn node_label(node: &GraphNode) -> String {
    match &node.node_type {
        NodeType::Interaction(d) => {
            let preview: String = d.content.chars().take(48).collect();
            let ellipsis = if d.content.len() > 48 { "…" } else { "" };
            format!("[{}] {}{}", d.role, preview, ellipsis)
        }
        NodeType::Agent(d) => format!("[agent] {}", d.name),
        NodeType::Content(d) => {
            let name = d.path.as_deref().unwrap_or(&d.content_type);
            format!("[content] {}", name)
        }
        NodeType::Task(d) => format!("[task] {}", d.title),
        NodeType::Knowledge(d) => format!("[knowledge] {}", d.entity),
    }
}

fn graph_node_to_entry(node: GraphNode) -> GraphNodeEntry {
    let label = node_label(&node);
    let node_type = node.node_type.type_name().to_string();
    GraphNodeEntry {
        id: node.id.to_string(),
        label,
        node_type,
        depth: 0,
        has_children: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::App;
    use graphirm_agent::AgentEvent;
    use graphirm_graph::nodes::NodeId;
    use graphirm_llm::StreamEvent;
    use tokio::sync::mpsc;

    fn make_app() -> App {
        let (_tx, rx) = mpsc::channel(16);
        App::new(rx, "test-model".to_string())
    }

    #[test]
    fn test_agent_start_sets_waiting_state() {
        let mut app = make_app();
        handle_agent_event(
            &mut app,
            AgentEvent::AgentStart {
                agent_id: NodeId::from("a1"),
            },
        );

        assert!(matches!(app.state, AppState::WaitingForAgent));
        assert_eq!(app.status_bar.agent_state, "Working");
    }

    #[test]
    fn test_agent_end_sets_idle_state() {
        let mut app = make_app();
        app.state = AppState::Streaming;

        handle_agent_event(
            &mut app,
            AgentEvent::AgentEnd {
                agent_id: NodeId::from("a1"),
                node_ids: vec![],
            },
        );

        assert!(matches!(app.state, AppState::Idle));
        assert_eq!(app.status_bar.agent_state, "Idle");
    }

    #[test]
    fn test_message_delta_appends_to_chat() {
        let mut app = make_app();

        handle_agent_event(
            &mut app,
            AgentEvent::MessageStart {
                node_id: NodeId::from("msg-1"),
            },
        );
        assert_eq!(app.chat.messages.len(), 1);
        assert!(matches!(app.state, AppState::Streaming));

        handle_agent_event(
            &mut app,
            AgentEvent::MessageDelta {
                node_id: NodeId::from("msg-1"),
                delta: StreamEvent::TextDelta("Hello".to_string()),
            },
        );
        handle_agent_event(
            &mut app,
            AgentEvent::MessageDelta {
                node_id: NodeId::from("msg-1"),
                delta: StreamEvent::TextDelta(", world!".to_string()),
            },
        );

        assert_eq!(app.chat.messages.last().unwrap().content, "Hello, world!");
    }

    #[test]
    fn test_tool_start_adds_tool_message() {
        let mut app = make_app();
        handle_agent_event(
            &mut app,
            AgentEvent::ToolStart {
                response_node_id: NodeId::from("t1"),
                call_id: "call_1".to_string(),
                tool_name: "bash".to_string(),
            },
        );

        assert_eq!(app.chat.messages.len(), 1);
        assert!(app.chat.messages[0].is_tool_call);
        assert_eq!(app.chat.messages[0].tool_name, Some("bash".to_string()));
    }

    #[test]
    fn test_tool_end_updates_status() {
        let mut app = make_app();
        handle_agent_event(
            &mut app,
            AgentEvent::ToolEnd {
                node_id: NodeId::from("t1"),
                is_error: false,
            },
        );
        assert_eq!(app.status_bar.agent_state, "Working");

        handle_agent_event(
            &mut app,
            AgentEvent::ToolEnd {
                node_id: NodeId::from("t2"),
                is_error: true,
            },
        );
        assert_eq!(app.status_bar.agent_state, "Tool Error");
    }
}
