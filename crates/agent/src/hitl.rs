use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use tokio::sync::oneshot;

use graphirm_graph::nodes::NodeId;

/// Decision sent through the gate to unblock the agent loop.
#[derive(Debug)]
pub enum HitlDecision {
    /// Run the tool call as-is.
    Approve,
    /// Block the tool call; inject the reason as a synthetic tool result.
    Reject(String),
    /// Replace the tool arguments with these before executing.
    Modify(serde_json::Value),
}

/// Shared gate between the agent loop (awaits) and the server route (resolves).
///
/// Clone the `Arc<HitlGate>` — one copy into `Session`, one into `SessionHandle`.
pub struct HitlGate {
    pending: Mutex<HashMap<String, oneshot::Sender<HitlDecision>>>,
    paused: AtomicBool,
}

impl HitlGate {
    pub fn new() -> Self {
        Self {
            pending: Mutex::new(HashMap::new()),
            paused: AtomicBool::new(false),
        }
    }

    /// Register a pending gate for `node_id` and return the receiver the
    /// agent loop should await.
    pub fn gate(&self, node_id: NodeId) -> oneshot::Receiver<HitlDecision> {
        let (tx, rx) = oneshot::channel();
        self.pending.lock().unwrap().insert(node_id.0, tx);
        rx
    }

    /// Resolve a pending gate. Returns `true` if a gate was found and sent to.
    pub fn resolve(&self, node_id: &NodeId, decision: HitlDecision) -> bool {
        if let Some(tx) = self.pending.lock().unwrap().remove(&node_id.0) {
            tx.send(decision).is_ok()
        } else {
            false
        }
    }

    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::SeqCst)
    }

    pub fn set_paused(&self, v: bool) {
        self.paused.store(v, Ordering::SeqCst);
    }
}

impl Default for HitlGate {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns `true` for tools that can modify the filesystem or execute arbitrary
/// commands. These are the only tools gated by the HITL approval flow.
pub fn is_destructive_tool(name: &str) -> bool {
    matches!(name, "write" | "edit" | "bash")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_destructive_tool_returns_true_for_write_edit_bash() {
        assert!(is_destructive_tool("write"));
        assert!(is_destructive_tool("edit"));
        assert!(is_destructive_tool("bash"));
    }

    #[test]
    fn is_destructive_tool_returns_false_for_read_only_tools() {
        assert!(!is_destructive_tool("read"));
        assert!(!is_destructive_tool("grep"));
        assert!(!is_destructive_tool("ls"));
        assert!(!is_destructive_tool("find"));
    }

    #[tokio::test]
    async fn gate_and_resolve_approve() {
        let gate = HitlGate::new();
        let node_id = NodeId::from("n1");
        let rx = gate.gate(node_id.clone());
        let resolved = gate.resolve(&node_id, HitlDecision::Approve);
        assert!(resolved);
        let decision = rx.await.unwrap();
        assert!(matches!(decision, HitlDecision::Approve));
    }

    #[tokio::test]
    async fn resolve_returns_false_when_no_pending_gate() {
        let gate = HitlGate::new();
        let node_id = NodeId::from("nope");
        let resolved = gate.resolve(&node_id, HitlDecision::Approve);
        assert!(!resolved);
    }

    #[test]
    fn pause_and_resume_flags() {
        let gate = HitlGate::new();
        assert!(!gate.is_paused());
        gate.set_paused(true);
        assert!(gate.is_paused());
        gate.set_paused(false);
        assert!(!gate.is_paused());
    }
}
