use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::{Mutex, oneshot};

use graphirm_graph::nodes::NodeId;

use crate::hitl_judge::{DestructiveJudge, JUDGE_VERSION, JudgeVerdict};

/// What the judge did with one auto-approved call. Recorded on the tool node.
#[derive(Debug, Clone, PartialEq)]
pub struct JudgeOutcome {
    pub verdict: JudgeVerdict,
    /// `true` when the call must go through the human gate despite auto-approve.
    pub pause: bool,
    /// `"paused"` | `"recorded"` (over threshold but headless) | `"approved"`;
    /// `"observed"` (`JUDGE_ACTION_OBSERVED`) for a delegated executor's calls,
    /// which are scored but never gated.
    pub action: &'static str,
}

impl JudgeOutcome {
    /// Metadata block stored under `hitl_judge` on the tool result node.
    pub fn to_metadata(&self) -> serde_json::Value {
        serde_json::json!({
            "version": JUDGE_VERSION,
            "p_irreversible": self.verdict.p_irreversible,
            "threshold": self.verdict.threshold,
            "action": self.action,
            "latency_ms": self.verdict.latency_ms,
        })
    }
}

/// Decision sent through the gate to unblock the agent loop.
#[derive(Debug, Clone, PartialEq)]
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
    auto_approve: AtomicBool,
    /// No human can answer a pause (session created with `auto_approve: true`).
    headless: AtomicBool,
    judge: Option<Arc<DestructiveJudge>>,
}

impl HitlGate {
    pub fn new() -> Self {
        Self {
            pending: Mutex::new(HashMap::new()),
            paused: AtomicBool::new(false),
            auto_approve: AtomicBool::new(false),
            headless: AtomicBool::new(false),
            judge: None,
        }
    }

    /// Attach the additive judge (see `hitl_judge`).
    pub fn with_judge(mut self, judge: Arc<DestructiveJudge>) -> Self {
        self.judge = Some(judge);
        self
    }

    pub fn has_judge(&self) -> bool {
        self.judge.is_some()
    }

    pub fn is_headless(&self) -> bool {
        self.headless.load(Ordering::Relaxed)
    }

    pub fn set_headless(&self, v: bool) {
        self.headless.store(v, Ordering::Relaxed);
    }

    /// In auto-approve mode, ask the judge whether this call should pause anyway.
    ///
    /// `None` when auto-approve is off (the human gate runs regardless), no judge
    /// is attached, or the judge failed — all of which keep today's behaviour.
    /// Over threshold: `pause` unless headless, where the verdict is only recorded.
    pub async fn judge_auto_approve(
        &self,
        tool_name: &str,
        arguments: &serde_json::Value,
    ) -> Option<JudgeOutcome> {
        if !self.is_auto_approve() {
            return None;
        }
        let judge = self.judge.as_ref()?;
        let verdict = match judge.judge(tool_name, arguments).await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(tool = tool_name, error = %e, "hitl_judge failed; auto-approving as before");
                return None;
            }
        };
        let outcome = match (verdict.over_threshold, self.is_headless()) {
            (true, false) => JudgeOutcome {
                verdict,
                pause: true,
                action: "paused",
            },
            (true, true) => JudgeOutcome {
                verdict,
                pause: false,
                action: "recorded",
            },
            (false, _) => JudgeOutcome {
                verdict,
                pause: false,
                action: "approved",
            },
        };
        tracing::info!(
            tool = tool_name,
            p_irreversible = outcome.verdict.p_irreversible,
            action = outcome.action,
            "hitl_judge"
        );
        Some(outcome)
    }

    /// Register a pending gate for `node_id` and return the receiver the
    /// agent loop should await.
    pub async fn gate(&self, node_id: &NodeId) -> oneshot::Receiver<HitlDecision> {
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(node_id.0.clone(), tx);
        rx
    }

    /// Resolve a pending gate. Returns `true` if a gate was found and sent to.
    pub async fn resolve(&self, node_id: &NodeId, decision: HitlDecision) -> bool {
        if let Some(tx) = self.pending.lock().await.remove(&node_id.0) {
            tx.send(decision).is_ok()
        } else {
            false
        }
    }

    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }

    pub fn set_paused(&self, v: bool) {
        self.paused.store(v, Ordering::Relaxed);
    }

    pub fn is_auto_approve(&self) -> bool {
        self.auto_approve.load(Ordering::Relaxed)
    }

    pub fn set_auto_approve(&self, v: bool) {
        self.auto_approve.store(v, Ordering::Relaxed);
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
        let rx = gate.gate(&node_id).await;
        let resolved = gate.resolve(&node_id, HitlDecision::Approve).await;
        assert!(resolved);
        let decision = rx.await.unwrap();
        assert!(matches!(decision, HitlDecision::Approve));
    }

    #[tokio::test]
    async fn resolve_returns_false_when_no_pending_gate() {
        let gate = HitlGate::new();
        let node_id = NodeId::from("nope");
        let resolved = gate.resolve(&node_id, HitlDecision::Approve).await;
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

    mod judge {
        use std::time::Duration;

        use async_trait::async_trait;
        use graphirm_llm::{DecisionsClient, DecisionsTransport, LlmError};
        use serde_json::{Value, json};

        use super::*;
        use crate::hitl_judge::JUDGE_QUESTION_ID;

        struct Reply(Value);

        #[async_trait]
        impl DecisionsTransport for Reply {
            async fn post(&self, _body: &Value) -> Result<Value, LlmError> {
                Ok(self.0.clone())
            }
        }

        struct Failing;

        #[async_trait]
        impl DecisionsTransport for Failing {
            async fn post(&self, _body: &Value) -> Result<Value, LlmError> {
                Err(LlmError::provider("down"))
            }
        }

        fn gate_with(transport: Arc<dyn DecisionsTransport>) -> HitlGate {
            let client = Arc::new(DecisionsClient::with_transport(transport));
            HitlGate::new().with_judge(Arc::new(DestructiveJudge::new(
                client,
                Duration::from_millis(200),
                0.8,
            )))
        }

        fn scoring(p: f64) -> Arc<dyn DecisionsTransport> {
            Arc::new(Reply(
                json!({"answers": {JUDGE_QUESTION_ID: {"type": "noul", "noul": p}}}),
            ))
        }

        #[tokio::test]
        async fn no_judge_means_no_opinion() {
            let gate = HitlGate::new();
            gate.set_auto_approve(true);
            assert!(gate.judge_auto_approve("bash", &json!({})).await.is_none());
            assert!(!gate.has_judge());
        }

        #[tokio::test]
        async fn manual_mode_never_consults_judge() {
            // Additive only: when the human gate already runs, the judge is silent.
            let gate = gate_with(scoring(0.99));
            assert!(!gate.is_auto_approve());
            assert!(
                gate.judge_auto_approve("bash", &json!({"command": "rm -rf /"}))
                    .await
                    .is_none()
            );
        }

        #[tokio::test]
        async fn auto_approve_over_threshold_pauses() {
            let gate = gate_with(scoring(0.93));
            gate.set_auto_approve(true);
            let out = gate
                .judge_auto_approve("bash", &json!({"command": "git reset --hard"}))
                .await
                .unwrap();
            assert!(out.pause);
            assert_eq!(out.action, "paused");
            assert_eq!(out.to_metadata()["action"], "paused");
            assert_eq!(out.to_metadata()["version"], JUDGE_VERSION);
        }

        #[tokio::test]
        async fn auto_approve_under_threshold_approves() {
            let gate = gate_with(scoring(0.05));
            gate.set_auto_approve(true);
            let out = gate
                .judge_auto_approve("bash", &json!({"command": "cargo test"}))
                .await
                .unwrap();
            assert!(!out.pause);
            assert_eq!(out.action, "approved");
        }

        #[tokio::test]
        async fn headless_over_threshold_records_but_does_not_pause() {
            let gate = gate_with(scoring(0.93));
            gate.set_auto_approve(true);
            gate.set_headless(true);
            let out = gate
                .judge_auto_approve("bash", &json!({"command": "rm -rf build"}))
                .await
                .unwrap();
            assert!(!out.pause);
            assert_eq!(out.action, "recorded");
        }

        #[tokio::test]
        async fn judge_failure_is_no_opinion() {
            let gate = gate_with(Arc::new(Failing));
            gate.set_auto_approve(true);
            assert!(gate.judge_auto_approve("bash", &json!({})).await.is_none());
        }
    }

    #[tokio::test]
    async fn gate_resolves_across_tasks() {
        use std::sync::Arc;
        let gate = Arc::new(HitlGate::new());
        let gate2 = gate.clone();
        let node_id = NodeId::from("concurrent");
        let rx = gate.gate(&node_id).await;
        let node_id2 = node_id.clone();
        tokio::spawn(async move {
            gate2.resolve(&node_id2, HitlDecision::Approve).await;
        });
        assert!(matches!(rx.await.unwrap(), HitlDecision::Approve));
    }
}
