//! Bridges `graphirm_tools::ToolEventSink` to the agent loop's `EventBus`.
//!
//! Long-running tools (e.g. `delegate_pi`) report sub-steps through the
//! `ToolEventSink` trait so they do not need to depend on `graphirm-agent`.
//! `EventBusSink` maps those calls onto `AgentEvent`s so TUI/SSE consumers
//! see them exactly as they see the agent loop's own tool events.

use std::collections::HashSet;
use std::sync::Arc;

use graphirm_graph::GraphStore;
use graphirm_graph::nodes::NodeId;
use graphirm_tools::ToolEventSink;
use tokio::sync::mpsc;

use crate::event::{AgentEvent, EventBus};

/// A pending `graph_changed` notification: `(anchor, touched)`.
type GraphChange = (NodeId, Vec<NodeId>);

/// `ToolEventSink` implementation that forwards to an [`EventBus`].
///
/// `tool_started` / `tool_finished` emit synchronously via `EventBus::emit`
/// (which uses `try_send`), so their relative ordering is preserved.
///
/// `graph_changed` builds a `GraphUpdate` snapshot from the graph store,
/// which requires async work. Rather than spawning a task per call (which
/// could deliver snapshots out of order — consumers replace their node list
/// wholesale, so a stale snapshot arriving last regresses the view — and
/// creates unbounded tasks), calls are queued on an unbounded channel and
/// drained by a single worker task. The worker coalesces everything queued
/// since its last emission into one `GraphUpdate`, so emissions are strictly
/// in order and at most one is in flight at a time.
pub struct EventBusSink {
    bus: Arc<EventBus>,
    changes: mpsc::UnboundedSender<GraphChange>,
}

impl EventBusSink {
    /// Create a sink bound to `bus` and `graph` and start its worker task.
    ///
    /// The worker exits on its own once the sink (the only sender) is dropped.
    ///
    /// # Panics
    ///
    /// Panics if called outside a tokio runtime context, because the worker
    /// task is spawned with `tokio::spawn`. The only planned construction
    /// site is inside `run_agent_loop`, which is always async.
    pub fn new(bus: Arc<EventBus>, graph: Arc<GraphStore>) -> Self {
        let (tx, rx) = mpsc::unbounded_channel::<GraphChange>();
        tokio::spawn(graph_update_worker(rx, graph, bus.clone()));
        Self { bus, changes: tx }
    }
}

/// Drain queued `graph_changed` notifications and emit one `GraphUpdate` per
/// batch. Uses the most recent anchor; `touched` is the deduplicated
/// concatenation of every queued `touched` list, in first-seen order.
async fn graph_update_worker(
    mut rx: mpsc::UnboundedReceiver<GraphChange>,
    graph: Arc<GraphStore>,
    bus: Arc<EventBus>,
) {
    while let Some((mut anchor, mut touched)) = rx.recv().await {
        while let Ok((next_anchor, next_touched)) = rx.try_recv() {
            anchor = next_anchor;
            touched.extend(next_touched);
        }
        let mut seen = HashSet::with_capacity(touched.len());
        touched.retain(|id| seen.insert(id.clone()));
        crate::workflow::emit_graph_update_for(graph.clone(), &anchor, touched, &bus).await;
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
        // Unbounded `send` is synchronous and never blocks, so this is safe
        // from any thread. It only fails if the worker has exited, which
        // cannot happen while `self` (the sender) is alive.
        if self
            .changes
            .send((anchor.clone(), touched.to_vec()))
            .is_err()
        {
            tracing::warn!("EventBusSink: GraphUpdate worker gone, dropping graph_changed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphirm_graph::nodes::{GraphNode, InteractionData, NodeType};
    use std::time::Duration;
    use tokio::runtime::Handle;

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

    /// Rapid `graph_changed` bursts are serialised through one worker: the
    /// stream of GraphUpdates ends with the most recent anchor (no stale
    /// snapshot can arrive last) and never exceeds one event per call.
    #[tokio::test]
    async fn rapid_graph_changes_are_ordered_and_coalesced() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let mut bus = EventBus::new();
        let mut rx = bus.subscribe();
        let sink = EventBusSink::new(Arc::new(bus), graph.clone());
        let anchors: Vec<NodeId> = (0..5).map(|_| interaction_node(&graph)).collect();

        for a in &anchors {
            sink.graph_changed(a, std::slice::from_ref(a));
        }
        // Dropping the sink lets the worker drain and exit; once it does, every
        // subscriber sender is gone and `rx.recv()` yields `None`.
        drop(sink);

        let mut updates = Vec::new();
        loop {
            match tokio::time::timeout(Duration::from_secs(5), rx.recv())
                .await
                .expect("worker should finish promptly")
            {
                Some(AgentEvent::GraphUpdate { node_id, .. }) => updates.push(node_id),
                Some(other) => panic!("unexpected event {other:?}"),
                None => break,
            }
        }

        assert!(
            !updates.is_empty(),
            "at least one GraphUpdate must be emitted"
        );
        assert!(
            updates.len() <= anchors.len(),
            "never more updates than calls"
        );
        assert_eq!(
            updates.last(),
            anchors.last(),
            "last update carries the last anchor"
        );
        // Every emitted anchor must appear in send order (no reordering).
        let positions: Vec<usize> = updates
            .iter()
            .map(|u| anchors.iter().position(|a| a == u).expect("known anchor"))
            .collect();
        assert!(
            positions.windows(2).all(|w| w[0] < w[1]),
            "updates out of order: {positions:?}"
        );
    }
}
