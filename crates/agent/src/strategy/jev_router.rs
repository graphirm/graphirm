//! `JevRouter` — typed cheap/smart routing via the Jev Decisions API.
//!
//! Replaces the generative classifier in `prompt_router.rs` with a single Jev
//! `choice` question over the named `TurnSignals` fields. Jev returns the
//! option plus calibrated probabilities, so callers can threshold on
//! `RoutingDecision::confidence` (= P(chosen tier)).
//!
//! Fail-soft: any client error or timeout falls back to `RuleRouter::select`
//! and records the fallback in `RoutingDecision::reason`.
//!
//! The question id, instructions and option keys below are the stored schema —
//! `routing_reason` metadata on Interaction nodes and any offline comparison
//! depend on them. Changing an option key is a breaking change; bump
//! `JEV_QUESTION_SET_VERSION` when any wording changes.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use graphirm_llm::{Answer, DecisionsClient, LlmError};
use serde_json::{Value, json};

use crate::router::{ModelTier, TaskPhase, TurnSignals};
use crate::strategy::rule_router::RuleRouter;
use crate::strategy::{
    ModelCandidate, ObjectiveWeights, RoutingDecision, RoutingStrategy, TurnOutcome,
};

/// Bump when any question text, option key, or description changes.
pub const JEV_QUESTION_SET_VERSION: u32 = 1;

/// Question id in the request/reply `questions` / `answers` maps.
pub const JEV_TIER_QUESTION_ID: &str = "tier";

/// Option key mapped to [`ModelTier::Cheap`]. Stored schema — do not rename.
pub const JEV_OPTION_CHEAP: &str = "cheap";

/// Option key mapped to [`ModelTier::Smart`]. Stored schema — do not rename.
pub const JEV_OPTION_SMART: &str = "smart";

/// Instructions for the tier question. Scoped to the named state fields; says
/// what to ignore; gives the numeric thresholds the rule router uses so the
/// question is absolute, not relative.
pub const JEV_TIER_INSTRUCTIONS: &str = "Pick the model tier for this coding-agent turn. \
Judge only these state fields: turn_number, last_tool_errored, last_response_tool_only, \
user_message_tokens, task_phase. Ignore anything else. \
Thresholds: user_message_tokens of 200 or more counts as a long message; \
turn_number of 15 or more counts as the agent being stuck. \
If any smart condition holds, choose smart; otherwise choose cheap.";

/// When the cheap tier applies. Mirrors the rule router's default/tool-only tiers.
pub const JEV_CHEAP_DESCRIPTION: &str = "A fast, inexpensive model is enough. \
All of these hold: turn_number is between 2 and 14, last_tool_errored is false, \
user_message_tokens is below 200. Typical routine turns: last_response_tool_only is true \
(the agent is mid-way through mechanical tool work) or task_phase is implementation.";

/// When the smart tier applies. Mirrors the rule router's first_turn,
/// error_recovery, high_complexity and stuck_detection rules.
pub const JEV_SMART_DESCRIPTION: &str = "A capable, slower model is needed. \
Any one of these holds: turn_number is 1 (first turn, the agent must understand the task \
and plan); last_tool_errored is true (the agent must recover from a failed tool call); \
user_message_tokens is 200 or more (long or complex user request); \
turn_number is 15 or more (the agent has run many turns and may be stuck).";

/// `RoutingDecision::strategy_name` in shadow mode.
pub const SHADOW_STRATEGY_NAME: &str = "jev_shadow";

/// Maximum characters of an error message kept in the fallback reason.
const REASON_ERROR_CHARS: usize = 120;

/// The typed result of one tier classification, before candidate lookup.
#[derive(Debug, Clone, PartialEq)]
pub struct JevTierDecision {
    /// Tier chosen by Jev.
    pub tier: ModelTier,
    /// P(chosen tier) — what `RoutingDecision::confidence` carries.
    pub probability: f64,
    /// P(cheap) as reported.
    pub p_cheap: f64,
    /// P(smart) as reported.
    pub p_smart: f64,
    /// Jev's own confidence field for the choice.
    pub confidence: f64,
    /// Round-trip latency of the Decisions call.
    pub latency: Duration,
    /// Cost of the call in USD as reported by the endpoint.
    pub cost_usd: f64,
}

/// Routing strategy backed by a Jev `choice` question, with `RuleRouter` fallback.
pub struct JevRouter {
    client: Arc<DecisionsClient>,
    fallback: RuleRouter,
    timeout: Duration,
    /// Shadow mode: rules decide, Jev is recorded. See [`JevRouter::with_shadow`].
    shadow: bool,
}

impl JevRouter {
    pub fn new(client: Arc<DecisionsClient>, fallback: RuleRouter, timeout: Duration) -> Self {
        Self {
            client,
            fallback,
            timeout,
            shadow: false,
        }
    }

    /// In shadow mode the rule router's decision is returned unchanged except
    /// for `reason`, which gains ` | jev:<tier> p=.. conf=.. agree=<bool> v<N>`
    /// (or ` | jev_err(..)`), and `strategy_name`, which becomes `jev_shadow`.
    /// This is how agreement is measured before Jev is allowed to decide.
    pub fn with_shadow(mut self, shadow: bool) -> Self {
        self.shadow = shadow;
        self
    }

    pub fn is_shadow(&self) -> bool {
        self.shadow
    }

    async fn select_shadow(
        &self,
        signals: &TurnSignals,
        candidates: &[ModelCandidate],
        objective: &ObjectiveWeights,
    ) -> RoutingDecision {
        let (rule, jev) = tokio::join!(
            self.fallback.select(signals, candidates, objective),
            self.classify(signals)
        );
        let mut d = rule;
        match jev {
            Ok(c) => {
                d.reason = format!(
                    "{} | jev:{} p={:.2} conf={:.2} agree={} v{}",
                    d.reason,
                    tier_option(c.tier),
                    c.probability,
                    c.confidence,
                    c.tier == d.tier,
                    JEV_QUESTION_SET_VERSION
                );
            }
            Err(e) => {
                d.reason = format!(
                    "{} | jev_err({})",
                    d.reason,
                    truncate_chars(&e.to_string(), REASON_ERROR_CHARS)
                );
            }
        }
        d.strategy_name = SHADOW_STRATEGY_NAME.to_string();
        d
    }

    /// The `state` object: `TurnSignals` fields as named JSON keys.
    pub fn build_state(signals: &TurnSignals) -> Value {
        json!({
            "turn_number": signals.turn_number,
            "last_tool_errored": signals.last_tool_errored,
            "last_response_tool_only": signals.last_response_tool_only,
            "user_message_tokens": signals.user_message_tokens,
            "task_phase": task_phase_name(signals.task_phase),
        })
    }

    /// The single `choice` question, built from the schema constants.
    pub fn build_questions() -> Value {
        json!({
            JEV_TIER_QUESTION_ID: {
                "type": "choice",
                "instructions": JEV_TIER_INSTRUCTIONS,
                "criteria": {
                    JEV_OPTION_CHEAP: JEV_CHEAP_DESCRIPTION,
                    JEV_OPTION_SMART: JEV_SMART_DESCRIPTION,
                }
            }
        })
    }

    /// Ask Jev for the tier. `Err` on any client failure or an unusable answer;
    /// callers that want a threshold read `probability` from the `Ok`.
    pub async fn classify(&self, signals: &TurnSignals) -> Result<JevTierDecision, LlmError> {
        let decision = self
            .client
            .decide(
                Self::build_state(signals),
                Self::build_questions(),
                self.timeout,
            )
            .await?;

        let answer = decision.answer(JEV_TIER_QUESTION_ID).ok_or_else(|| {
            LlmError::provider(format!("no {JEV_TIER_QUESTION_ID:?} answer in reply"))
        })?;
        let (choice, confidence) = match answer {
            Answer::Choice {
                choice, confidence, ..
            } => (choice.as_str(), *confidence),
            other => {
                return Err(LlmError::provider(format!(
                    "expected choice answer, got {other:?}"
                )));
            }
        };
        let tier = tier_from_option(choice)
            .ok_or_else(|| LlmError::provider(format!("unknown tier option {choice:?}")))?;
        let p_cheap = answer.probability_of(JEV_OPTION_CHEAP).unwrap_or(0.0);
        let p_smart = answer.probability_of(JEV_OPTION_SMART).unwrap_or(0.0);
        let probability = match tier {
            ModelTier::Cheap => p_cheap,
            ModelTier::Smart => p_smart,
        };

        Ok(JevTierDecision {
            tier,
            probability,
            p_cheap,
            p_smart,
            confidence,
            latency: decision.latency,
            cost_usd: decision.cost_usd,
        })
    }
}

fn task_phase_name(phase: TaskPhase) -> &'static str {
    match phase {
        TaskPhase::Planning => "planning",
        TaskPhase::Implementation => "implementation",
        TaskPhase::Verification => "verification",
    }
}

fn tier_from_option(option: &str) -> Option<ModelTier> {
    match option {
        JEV_OPTION_CHEAP => Some(ModelTier::Cheap),
        JEV_OPTION_SMART => Some(ModelTier::Smart),
        _ => None,
    }
}

fn tier_option(tier: ModelTier) -> &'static str {
    match tier {
        ModelTier::Cheap => JEV_OPTION_CHEAP,
        ModelTier::Smart => JEV_OPTION_SMART,
    }
}

fn truncate_chars(s: &str, max: usize) -> String {
    let single_line = s.replace(['\n', '\r'], " ");
    if single_line.chars().count() <= max {
        single_line
    } else {
        let mut out: String = single_line.chars().take(max).collect();
        out.push('…');
        out
    }
}

#[async_trait]
impl RoutingStrategy for JevRouter {
    async fn select(
        &self,
        signals: &TurnSignals,
        candidates: &[ModelCandidate],
        objective: &ObjectiveWeights,
    ) -> RoutingDecision {
        if self.shadow {
            return self.select_shadow(signals, candidates, objective).await;
        }
        let classified = match self.classify(signals).await {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(error = %e, "jev_router failed, falling back to rules");
                let mut d = self.fallback.select(signals, candidates, objective).await;
                d.reason = format!(
                    "jev_fallback({}) -> {}",
                    truncate_chars(&e.to_string(), REASON_ERROR_CHARS),
                    d.reason
                );
                d.strategy_name = self.strategy_name().to_string();
                return d;
            }
        };

        let candidate = candidates
            .iter()
            .find(|c| c.tier == classified.tier)
            .or_else(|| candidates.first());
        let (model, final_tier) = candidate
            .map(|c| (c.model.clone(), c.tier))
            .unwrap_or_else(|| (String::new(), classified.tier));

        tracing::debug!(
            tier = tier_option(classified.tier),
            p_cheap = classified.p_cheap,
            p_smart = classified.p_smart,
            confidence = classified.confidence,
            latency_ms = classified.latency.as_millis() as u64,
            cost_usd = classified.cost_usd,
            "jev_router decision"
        );

        RoutingDecision {
            model,
            tier: final_tier,
            confidence: classified.probability,
            reason: format!(
                "jev:{} p={:.2} conf={:.2} v{}",
                tier_option(classified.tier),
                classified.probability,
                classified.confidence,
                JEV_QUESTION_SET_VERSION
            ),
            strategy_name: self.strategy_name().to_string(),
        }
    }

    fn strategy_name(&self) -> &str {
        "jev_router"
    }

    fn record_outcome(&self, _outcome: &TurnOutcome) {}
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use async_trait::async_trait;
    use graphirm_llm::{DecisionsClient, DecisionsTransport, LlmError};
    use serde_json::{Value, json};

    use super::*;
    use crate::router::{ModelRoutingConfig, ModelTier, RoutingRule, TaskPhase, TurnSignals};
    use crate::strategy::rule_router::RuleRouter;
    use crate::strategy::{ModelCandidate, ObjectiveWeights, RoutingStrategy};

    // ---- fakes ----

    struct ReplyTransport {
        reply: Value,
        seen: Mutex<Option<Value>>,
    }

    impl ReplyTransport {
        fn new(reply: Value) -> Arc<Self> {
            Arc::new(Self {
                reply,
                seen: Mutex::new(None),
            })
        }
    }

    #[async_trait]
    impl DecisionsTransport for ReplyTransport {
        async fn post(&self, body: &Value) -> Result<Value, LlmError> {
            *self.seen.lock().expect("lock") = Some(body.clone());
            Ok(self.reply.clone())
        }
    }

    struct HangingTransport;

    #[async_trait]
    impl DecisionsTransport for HangingTransport {
        async fn post(&self, _body: &Value) -> Result<Value, LlmError> {
            tokio::time::sleep(Duration::from_secs(3600)).await;
            Ok(json!({}))
        }
    }

    fn tier_reply(choice: &str, p_cheap: f64, p_smart: f64, confidence: f64) -> Value {
        json!({
            "model": "typesafe/jev-1.13-20260917",
            "answers": {
                JEV_TIER_QUESTION_ID: {
                    "type": "choice",
                    "choice": choice,
                    "probabilities": {"cheap": p_cheap, "smart": p_smart},
                    "confidence": confidence
                }
            },
            "usage": {"cost": 0.00002}
        })
    }

    fn candidates() -> Vec<ModelCandidate> {
        vec![
            ModelCandidate {
                model: "cheap-model".into(),
                tier: ModelTier::Cheap,
                cost_per_1k_input: 0.001,
                cost_per_1k_output: 0.002,
                avg_latency_ms: None,
            },
            ModelCandidate {
                model: "smart-model".into(),
                tier: ModelTier::Smart,
                cost_per_1k_input: 0.01,
                cost_per_1k_output: 0.03,
                avg_latency_ms: None,
            },
        ]
    }

    /// Fallback rules: first turn → smart, otherwise cheap.
    fn fallback() -> RuleRouter {
        RuleRouter::new(ModelRoutingConfig {
            cheap: vec!["cheap-model".into()],
            smart: vec!["smart-model".into()],
            default_tier: ModelTier::Cheap,
            rules: vec![RoutingRule::FirstTurn {
                tier: ModelTier::Smart,
            }],
        })
    }

    fn signals() -> TurnSignals {
        TurnSignals {
            turn_number: 2,
            last_tool_errored: false,
            last_response_tool_only: true,
            user_message_tokens: 80,
            task_phase: TaskPhase::Implementation,
        }
    }

    fn router_with(transport: Arc<dyn DecisionsTransport>) -> JevRouter {
        JevRouter::new(
            Arc::new(DecisionsClient::with_transport(transport)),
            fallback(),
            Duration::from_millis(500),
        )
    }

    // ---- schema is the stored contract ----

    #[test]
    fn question_set_is_exactly_the_consts() {
        let q = JevRouter::build_questions();
        let obj = q.as_object().expect("questions object");
        assert_eq!(obj.len(), 1, "exactly one question");
        let tier = &q[JEV_TIER_QUESTION_ID];
        assert_eq!(tier["type"], "choice");
        assert_eq!(tier["instructions"], JEV_TIER_INSTRUCTIONS);
        let criteria = tier["criteria"].as_object().expect("criteria object");
        let mut keys: Vec<&str> = criteria.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, vec![JEV_OPTION_CHEAP, JEV_OPTION_SMART]);
        assert_eq!(criteria[JEV_OPTION_CHEAP], JEV_CHEAP_DESCRIPTION);
        assert_eq!(criteria[JEV_OPTION_SMART], JEV_SMART_DESCRIPTION);
    }

    #[test]
    fn option_keys_match_model_tier_names() {
        assert_eq!(JEV_OPTION_CHEAP, "cheap");
        assert_eq!(JEV_OPTION_SMART, "smart");
        assert_eq!(JEV_TIER_QUESTION_ID, "tier");
    }

    #[test]
    fn state_uses_named_turn_signal_fields() {
        let state = JevRouter::build_state(&signals());
        let obj = state.as_object().expect("state object");
        let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec![
                "last_response_tool_only",
                "last_tool_errored",
                "task_phase",
                "turn_number",
                "user_message_tokens",
            ]
        );
        assert_eq!(state["turn_number"], 2);
        assert_eq!(state["last_tool_errored"], false);
        assert_eq!(state["last_response_tool_only"], true);
        assert_eq!(state["user_message_tokens"], 80);
        assert_eq!(state["task_phase"], "implementation");
    }

    // ---- choice → tier mapping ----

    #[tokio::test]
    async fn smart_choice_selects_smart_with_probability() {
        let transport = ReplyTransport::new(tier_reply("smart", 0.13, 0.87, 0.74));
        let router = router_with(transport.clone());
        let d = router
            .select(&signals(), &candidates(), &ObjectiveWeights::default())
            .await;
        assert_eq!(d.tier, ModelTier::Smart);
        assert_eq!(d.model, "smart-model");
        assert!((d.confidence - 0.87).abs() < 1e-9, "confidence = P(smart)");
        assert_eq!(d.strategy_name, "jev_router");
        assert!(d.reason.starts_with("jev:smart"), "reason = {}", d.reason);
        assert!(!d.reason.contains("fallback"), "reason = {}", d.reason);

        let body = transport.seen.lock().expect("lock").clone().expect("sent");
        assert_eq!(body["state"], JevRouter::build_state(&signals()));
        assert_eq!(body["questions"], JevRouter::build_questions());
    }

    #[tokio::test]
    async fn cheap_choice_selects_cheap_with_probability() {
        let router = router_with(ReplyTransport::new(tier_reply("cheap", 0.93, 0.07, 0.9)));
        let d = router
            .select(&signals(), &candidates(), &ObjectiveWeights::default())
            .await;
        assert_eq!(d.tier, ModelTier::Cheap);
        assert_eq!(d.model, "cheap-model");
        assert!((d.confidence - 0.93).abs() < 1e-9);
        assert!(d.reason.starts_with("jev:cheap"), "reason = {}", d.reason);
    }

    #[tokio::test]
    async fn classify_exposes_tier_and_probabilities() {
        let router = router_with(ReplyTransport::new(tier_reply("smart", 0.2, 0.8, 0.6)));
        let c = router.classify(&signals()).await.expect("classify");
        assert_eq!(c.tier, ModelTier::Smart);
        assert!((c.probability - 0.8).abs() < 1e-9);
        assert!((c.p_smart - 0.8).abs() < 1e-9);
        assert!((c.p_cheap - 0.2).abs() < 1e-9);
        assert!((c.confidence - 0.6).abs() < 1e-9);
    }

    // ---- shadow mode: rules decide, Jev is recorded ----

    #[tokio::test]
    async fn shadow_returns_rule_decision_and_records_disagreement() {
        // Rules: turn 2 → default cheap. Jev says smart with p=0.87.
        let transport = ReplyTransport::new(tier_reply("smart", 0.13, 0.87, 0.74));
        let router = router_with(transport.clone()).with_shadow(true);
        assert!(router.is_shadow());
        let d = router
            .select(&signals(), &candidates(), &ObjectiveWeights::default())
            .await;
        assert_eq!(d.tier, ModelTier::Cheap, "rules decide in shadow mode");
        assert_eq!(d.model, "cheap-model");
        assert!(
            (d.confidence - 1.0).abs() < 1e-9,
            "rule confidence, not Jev's"
        );
        assert_eq!(d.strategy_name, "jev_shadow");
        assert!(d.reason.starts_with("rule:"), "reason = {}", d.reason);
        assert!(
            d.reason
                .contains("| jev:smart p=0.87 conf=0.74 agree=false v1"),
            "reason = {}",
            d.reason
        );
        assert!(
            transport.seen.lock().expect("lock").is_some(),
            "Jev was still asked"
        );
    }

    #[tokio::test]
    async fn shadow_records_agreement_when_tiers_match() {
        let router = router_with(ReplyTransport::new(tier_reply("smart", 0.01, 0.99, 0.99)))
            .with_shadow(true);
        let first_turn = TurnSignals {
            turn_number: 1,
            ..signals()
        };
        let d = router
            .select(&first_turn, &candidates(), &ObjectiveWeights::default())
            .await;
        assert_eq!(d.tier, ModelTier::Smart);
        assert!(
            d.reason.contains("rule:first_turn"),
            "reason = {}",
            d.reason
        );
        assert!(d.reason.contains("agree=true"), "reason = {}", d.reason);
    }

    #[tokio::test(start_paused = true)]
    async fn shadow_on_jev_error_still_returns_rule_decision() {
        let router = router_with(Arc::new(HangingTransport)).with_shadow(true);
        let d = router
            .select(&signals(), &candidates(), &ObjectiveWeights::default())
            .await;
        assert_eq!(d.tier, ModelTier::Cheap);
        assert_eq!(d.strategy_name, "jev_shadow");
        assert!(d.reason.contains("| jev_err("), "reason = {}", d.reason);
        assert!(!d.reason.contains("jev_fallback"), "reason = {}", d.reason);
    }

    #[tokio::test]
    async fn shadow_off_by_default() {
        let router = router_with(ReplyTransport::new(tier_reply("smart", 0.13, 0.87, 0.74)));
        assert!(!router.is_shadow());
        let d = router
            .select(&signals(), &candidates(), &ObjectiveWeights::default())
            .await;
        assert_eq!(d.tier, ModelTier::Smart, "non-shadow: Jev decides");
    }

    // ---- fail-soft: fallback to RuleRouter ----

    #[tokio::test(start_paused = true)]
    async fn timeout_falls_back_to_rule_router() {
        let router = router_with(Arc::new(HangingTransport));
        let first_turn = TurnSignals {
            turn_number: 1,
            ..signals()
        };
        let d = router
            .select(&first_turn, &candidates(), &ObjectiveWeights::default())
            .await;
        assert_eq!(d.tier, ModelTier::Smart, "rule first_turn → smart");
        assert_eq!(d.model, "smart-model");
        assert_eq!(d.strategy_name, "jev_router");
        assert!(
            d.reason.starts_with("jev_fallback"),
            "reason = {}",
            d.reason
        );
        assert!(
            d.reason.contains("rule:first_turn"),
            "reason = {}",
            d.reason
        );
        assert!(d.reason.contains("timed out"), "reason = {}", d.reason);
    }

    #[tokio::test]
    async fn malformed_reply_falls_back_to_rule_router() {
        let router = router_with(ReplyTransport::new(json!({"answers": "garbage"})));
        let d = router
            .select(&signals(), &candidates(), &ObjectiveWeights::default())
            .await;
        assert_eq!(d.tier, ModelTier::Cheap, "rule default → cheap");
        assert!(
            d.reason.starts_with("jev_fallback"),
            "reason = {}",
            d.reason
        );
        assert!(d.reason.contains("rule:default"), "reason = {}", d.reason);
    }

    #[tokio::test]
    async fn unknown_option_falls_back_to_rule_router() {
        let router = router_with(ReplyTransport::new(tier_reply("medium", 0.5, 0.5, 0.5)));
        let first_turn = TurnSignals {
            turn_number: 1,
            ..signals()
        };
        let d = router
            .select(&first_turn, &candidates(), &ObjectiveWeights::default())
            .await;
        assert_eq!(d.tier, ModelTier::Smart);
        assert!(
            d.reason.starts_with("jev_fallback"),
            "reason = {}",
            d.reason
        );
        assert!(d.reason.contains("medium"), "reason = {}", d.reason);
    }

    #[tokio::test]
    async fn missing_tier_answer_falls_back_to_rule_router() {
        let reply = json!({"answers": {"other": {"type": "noul", "noul": 0.5}}});
        let router = router_with(ReplyTransport::new(reply));
        let d = router
            .select(&signals(), &candidates(), &ObjectiveWeights::default())
            .await;
        assert_eq!(d.tier, ModelTier::Cheap);
        assert!(
            d.reason.starts_with("jev_fallback"),
            "reason = {}",
            d.reason
        );
    }

    /// Live smoke against the real endpoint. Needs `OPENROUTER_API_KEY`.
    ///
    /// ```bash
    /// OPENROUTER_API_KEY=... cargo test -p graphirm-agent --lib jev_live_smoke -- --ignored --nocapture
    /// ```
    #[tokio::test]
    #[ignore = "hits the network; run explicitly with --ignored"]
    async fn jev_live_smoke() {
        let client = match DecisionsClient::from_env() {
            Ok(c) => Arc::new(c),
            Err(e) => panic!("OPENROUTER_API_KEY required for live smoke: {e}"),
        };
        let router = JevRouter::new(client, fallback(), Duration::from_secs(5));
        let cases = [
            (
                "first_turn",
                TurnSignals {
                    turn_number: 1,
                    last_tool_errored: false,
                    last_response_tool_only: false,
                    user_message_tokens: 40,
                    task_phase: TaskPhase::Planning,
                },
            ),
            (
                "error_recovery",
                TurnSignals {
                    turn_number: 5,
                    last_tool_errored: true,
                    last_response_tool_only: false,
                    user_message_tokens: 30,
                    task_phase: TaskPhase::Implementation,
                },
            ),
            (
                "long_user_message",
                TurnSignals {
                    turn_number: 3,
                    last_tool_errored: false,
                    last_response_tool_only: false,
                    user_message_tokens: 450,
                    task_phase: TaskPhase::Planning,
                },
            ),
            (
                "routine_tool_only",
                TurnSignals {
                    turn_number: 4,
                    last_tool_errored: false,
                    last_response_tool_only: true,
                    user_message_tokens: 30,
                    task_phase: TaskPhase::Implementation,
                },
            ),
        ];
        let shadow = JevRouter::new(router.client.clone(), fallback(), Duration::from_secs(5))
            .with_shadow(true);
        for (label, signals) in cases {
            let c = router.classify(&signals).await.expect("live classify");
            println!(
                "SMOKE {label}: tier={:?} p_cheap={:.3} p_smart={:.3} conf={:.2} latency_ms={} cost_usd={:.8}",
                c.tier,
                c.p_cheap,
                c.p_smart,
                c.confidence,
                c.latency.as_millis(),
                c.cost_usd
            );
            let d = shadow
                .select(&signals, &candidates(), &ObjectiveWeights::default())
                .await;
            println!(
                "SHADOW {label}: tier={:?} {} [{}]",
                d.tier, d.reason, d.strategy_name
            );
        }
    }

    #[tokio::test]
    async fn fallback_reason_is_bounded_in_length() {
        let huge = "x".repeat(5000);
        let reply = json!({"error": {"message": huge}});
        let router = router_with(ReplyTransport::new(reply));
        let d = router
            .select(&signals(), &candidates(), &ObjectiveWeights::default())
            .await;
        assert!(d.reason.len() < 400, "reason too long: {}", d.reason.len());
    }
}
