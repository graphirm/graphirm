use async_trait::async_trait;

use crate::router::{ModelRouter, ModelRoutingConfig};
use crate::strategy::{ModelCandidate, ObjectiveWeights, RoutingDecision, RoutingStrategy, TurnOutcome};

pub struct RuleRouter {
    config: ModelRoutingConfig,
}

impl RuleRouter {
    pub fn new(config: ModelRoutingConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl RoutingStrategy for RuleRouter {
    async fn select(
        &self,
        signals: &crate::router::TurnSignals,
        candidates: &[ModelCandidate],
        _objective: &ObjectiveWeights,
    ) -> RoutingDecision {
        let router = ModelRouter::new(&self.config);
        let (model_str, tier, rule) = router.select(signals);

        // Find the matching candidate or fall back to first with matching tier.
        let matched = candidates
            .iter()
            .find(|c| c.model == model_str)
            .or_else(|| candidates.iter().find(|c| c.tier == tier));

        let (model, final_tier) = matched
            .map(|c| (c.model.clone(), c.tier))
            .unwrap_or_else(|| (model_str.to_string(), tier));

        RoutingDecision {
            model,
            tier: final_tier,
            confidence: 1.0,
            reason: format!("rule:{rule}"),
            strategy_name: self.strategy_name().to_string(),
        }
    }

    fn strategy_name(&self) -> &str {
        "rule_router"
    }

    fn record_outcome(&self, _outcome: &TurnOutcome) {
        // Rule-based router doesn't learn.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::{ModelRoutingConfig, ModelTier, RoutingRule};
    use crate::strategy::{ModelCandidate, ObjectiveWeights};

    fn cheap_candidate() -> ModelCandidate {
        ModelCandidate {
            model: "deepseek/deepseek-chat".into(),
            tier: ModelTier::Cheap,
            cost_per_1k_input: 0.00014,
            cost_per_1k_output: 0.00028,
            avg_latency_ms: None,
        }
    }

    fn smart_candidate() -> ModelCandidate {
        ModelCandidate {
            model: "anthropic/claude-sonnet-4".into(),
            tier: ModelTier::Smart,
            cost_per_1k_input: 0.003,
            cost_per_1k_output: 0.015,
            avg_latency_ms: None,
        }
    }

    #[tokio::test]
    async fn selects_smart_on_first_turn() {
        let config = ModelRoutingConfig {
            cheap: "deepseek/deepseek-chat".into(),
            smart: "anthropic/claude-sonnet-4".into(),
            default_tier: ModelTier::Cheap,
            rules: vec![RoutingRule::FirstTurn { tier: ModelTier::Smart }],
        };
        let router = RuleRouter::new(config);
        let signals = crate::router::TurnSignals {
            turn_number: 1,
            last_tool_errored: false,
            last_response_tool_only: false,
            user_message_tokens: 50,
            task_phase: crate::router::TaskPhase::Planning,
        };
        let decision = router
            .select(&signals, &[cheap_candidate(), smart_candidate()], &ObjectiveWeights::default())
            .await;
        assert_eq!(decision.model, "anthropic/claude-sonnet-4");
        assert_eq!(decision.strategy_name, "rule_router");
    }

    #[tokio::test]
    async fn falls_back_to_default_when_no_rule_matches() {
        let config = ModelRoutingConfig {
            cheap: "deepseek/deepseek-chat".into(),
            smart: "anthropic/claude-sonnet-4".into(),
            default_tier: ModelTier::Cheap,
            rules: vec![],
        };
        let router = RuleRouter::new(config);
        let signals = crate::router::TurnSignals {
            turn_number: 3,
            last_tool_errored: false,
            last_response_tool_only: false,
            user_message_tokens: 30,
            task_phase: crate::router::TaskPhase::Implementation,
        };
        let decision = router
            .select(&signals, &[cheap_candidate(), smart_candidate()], &ObjectiveWeights::default())
            .await;
        assert_eq!(decision.model, "deepseek/deepseek-chat");
    }
}
