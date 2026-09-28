//! Bridges `graphirm_tools::ToolEventSink` to the agent loop's `EventBus`.
//!
//! Long-running tools (e.g. `delegate_pi`) report sub-steps through the
//! `ToolEventSink` trait so they do not need to depend on `graphirm-agent`.
//! `EventBusSink` maps those calls onto `AgentEvent`s so TUI/SSE consumers
//! see them exactly as they see the agent loop's own tool events.

use std::sync::Arc;

use graphirm_graph::GraphStore;
use graphirm_graph::nodes::NodeId;
use graphirm_tools::ToolEventSink;
use tokio::runtime::Handle;

use crate::event::{AgentEvent, EventBus};

/// `ToolEventSink` implementation that forwards to an [`EventBus`].
///
/// `tool_started` / `tool_finished` emit synchronously via `EventBus::emit`
/// (which uses `try_send`). `graph_changed` builds a `GraphUpdate` payload
/// from the graph store, which requires async work; it is spawned onto the
/// tokio runtime captured at construction so the call is safe from a
/// `spawn_blocking` thread.
pub struct EventBusSink {
    bus: Arc<EventBus>,
    graph: Arc<GraphStore>,
    handle: Handle,
}

impl EventBusSink {
    /// Create a sink bound to `bus` and `graph`.
    ///
    /// Must be called from within a tokio runtime context; the runtime
    /// handle is captured so later `graph_changed` calls can spawn work
    /// even when invoked from a blocking thread.
    pub fn new(bus: Arc<EventBus>, graph: Arc<GraphStore>) -> Self {
        Self {
            bus,
            graph,
            handle: Handle::current(),
        }
    }
}

impl ToolEventSink for EventBusSink {
    fn tool_started(&self, response_node_id: &NodeId, call_id: &str, tool_name: &str) {
        self.bus.emit(AgentEvent::ToolStart {
            response_node_id: response_node_id.clone(),
            call_id: call_id.to_string(),
            tool_name: tool_name.to_string(),
        });
    }

    fn tool_finished(&self, node_id: &NodeId, is_error: bool) {
        self.bus.emit(AgentEvent::ToolEnd {
            node_id: node_id.clone(),
            is_error,
        });
    }

    fn graph_changed(&self, anchor: &NodeId, touched: &[NodeId]) {
        let bus = self.bus.clone();
        let graph = self.graph.clone();
        let anchor = anchor.clone();
        let touched = touched.to_vec();
        self.handle.spawn(async move {
            crate::workflow::emit_graph_update_for(graph, &anchor, touched, &bus).await;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphirm_graph::nodes::{GraphNode, InteractionData, NodeType};
    use std::time::Duration;

    fn interaction_node(graph: &GraphStore) -> NodeId {
        graph
            .add_node(GraphNode::new(NodeType::Interaction(InteractionData {
                role: "assistant".into(),
                content: "x".into(),
                token_count: None,
            })))
            .unwrap()
    }

    #[tokio::test]
    async fn sink_maps_to_agent_events() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let mut bus = EventBus::new();
        let mut rx = bus.subscribe();
        let sink = EventBusSink::new(Arc::new(bus), graph.clone());
        let n = interaction_node(&graph);

        sink.tool_started(&n, "pi:1", "bash");
        sink.tool_finished(&n, true);
        sink.graph_changed(&n, std::slice::from_ref(&n));

        let e1 = rx.recv().await.unwrap();
        assert!(matches!(
            e1,
            AgentEvent::ToolStart { ref response_node_id, ref call_id, ref tool_name }
                if *response_node_id == n && call_id == "pi:1" && tool_name == "bash"
        ));
        let e2 = rx.recv().await.unwrap();
        assert!(matches!(
            e2,
            AgentEvent::ToolEnd { ref node_id, is_error: true } if *node_id == n
        ));
        let e3 = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .unwrap()
            .unwrap();
        match e3 {
            AgentEvent::GraphUpdate {
                node_id,
                recent_nodes,
                ..
            } => {
                assert_eq!(node_id, n);
                assert!(recent_nodes.iter().any(|g| g.id == n));
            }
            other => panic!("expected GraphUpdate, got {other:?}"),
        }
    }

    /// `graph_changed` must work from a thread with no tokio context at all
    /// (a bare `tokio::spawn` would panic there). `spawn_blocking` threads do
    /// carry a runtime context, so a raw `std::thread` is the stricter check.
    #[tokio::test]
    async fn sink_is_usable_from_thread_without_runtime_context() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let mut bus = EventBus::new();
        let mut rx = bus.subscribe();
        let sink: Arc<dyn ToolEventSink> =
            Arc::new(EventBusSink::new(Arc::new(bus), graph.clone()));
        let n = interaction_node(&graph);

        let s = sink.clone();
        let anchor = n.clone();
        let join = std::thread::spawn(move || {
            assert!(Handle::try_current().is_err(), "must run off-runtime");
            s.tool_started(&anchor, "pi:2", "read");
            s.graph_changed(&anchor, std::slice::from_ref(&anchor));
            s.tool_finished(&anchor, false);
        });
        tokio::task::spawn_blocking(move || join.join().expect("thread panicked"))
            .await
            .unwrap();

        let mut saw_start = false;
        let mut saw_end = false;
        let mut saw_graph = false;
        for _ in 0..3 {
            let e = tokio::time::timeout(Duration::from_secs(2), rx.recv())
                .await
                .unwrap()
                .unwrap();
            match e {
                AgentEvent::ToolStart { .. } => saw_start = true,
                AgentEvent::ToolEnd {
                    is_error: false, ..
                } => saw_end = true,
                AgentEvent::GraphUpdate { .. } => saw_graph = true,
                other => panic!("unexpected event {other:?}"),
            }
        }
        assert!(saw_start && saw_end && saw_graph);
    }
}
