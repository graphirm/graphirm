// Agent events: streaming events for UI consumption

use graphirm_graph::edges::EdgeId;
use graphirm_graph::nodes::NodeId;
use graphirm_llm::StreamEvent;
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub enum AgentEvent {
    AgentStart {
        agent_id: NodeId,
    },
    AgentEnd {
        agent_id: NodeId,
        node_ids: Vec<NodeId>,
    },
    TurnStart {
        turn_index: u32,
    },
    TurnEnd {
        response_id: NodeId,
        tool_result_ids: Vec<NodeId>,
    },
    MessageStart {
        node_id: NodeId,
    },
    MessageDelta {
        node_id: NodeId,
        delta: StreamEvent,
    },
    MessageEnd {
        node_id: NodeId,
    },
    ToolStart {
        node_id: NodeId,
        tool_name: String,
    },
    ToolEnd {
        node_id: NodeId,
        is_error: bool,
    },
    GraphUpdate {
        node_id: NodeId,
        edge_ids: Vec<EdgeId>,
    },
}

pub struct EventBus {
    subscribers: Vec<mpsc::Sender<AgentEvent>>,
}

impl EventBus {
    pub fn new() -> Self {
        Self {
            subscribers: Vec::new(),
        }
    }

    pub fn subscribe(&mut self) -> mpsc::Receiver<AgentEvent> {
        let (tx, rx) = mpsc::channel(256);
        self.subscribers.push(tx);
        rx
    }

    pub async fn emit(&self, event: AgentEvent) {
        for sender in &self.subscribers {
            let _ = sender.send(event.clone()).await;
        }
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_event_clone() {
        let event = AgentEvent::TurnStart { turn_index: 3 };
        let cloned = event.clone();
        assert!(matches!(cloned, AgentEvent::TurnStart { turn_index: 3 }));
    }

    #[test]
    fn test_agent_event_debug() {
        let event = AgentEvent::ToolStart {
            node_id: NodeId::from("node-1"),
            tool_name: "bash".to_string(),
        };
        let debug = format!("{:?}", event);
        assert!(debug.contains("ToolStart"));
        assert!(debug.contains("bash"));
    }

    #[tokio::test]
    async fn test_event_bus_single_subscriber() {
        let mut bus = EventBus::new();
        let mut rx = bus.subscribe();

        bus.emit(AgentEvent::TurnStart { turn_index: 0 }).await;

        let event = rx.recv().await.unwrap();
        assert!(matches!(event, AgentEvent::TurnStart { turn_index: 0 }));
    }

    #[tokio::test]
    async fn test_event_bus_multiple_subscribers() {
        let mut bus = EventBus::new();
        let mut rx1 = bus.subscribe();
        let mut rx2 = bus.subscribe();

        bus.emit(AgentEvent::AgentStart {
            agent_id: NodeId::from("a1"),
        })
        .await;

        let e1 = rx1.recv().await.unwrap();
        let e2 = rx2.recv().await.unwrap();
        assert!(matches!(e1, AgentEvent::AgentStart { .. }));
        assert!(matches!(e2, AgentEvent::AgentStart { .. }));
    }

    #[tokio::test]
    async fn test_event_bus_dropped_subscriber_does_not_block() {
        let mut bus = EventBus::new();
        let rx = bus.subscribe();
        drop(rx);

        // Should not panic or block
        bus.emit(AgentEvent::TurnStart { turn_index: 0 }).await;
    }
}
