use std::sync::Arc;

use async_trait::async_trait;

use crate::router::TurnSignals;
use crate::strategy::{
    ModelCandidate, ObjectiveWeights, RoutingDecision, RoutingStrategy, TurnOutcome,
};

pub struct ExperimentRouter {
    strategy_a: Arc<dyn RoutingStrategy>,
    strategy_b: Arc<dyn RoutingStrategy>,
    /// Fraction [0.0, 1.0] of turns routed to strategy_a. Rest go to strategy_b.
    split: f64,
}

impl ExperimentRouter {
    pub fn new(
        strategy_a: Arc<dyn RoutingStrategy>,
        strategy_b: Arc<dyn RoutingStrategy>,
        split: f64,
    ) -> Self {
        Self {
            strategy_a,
            strategy_b,
            split: split.clamp(0.0, 1.0),
        }
    }

    fn pick_strategy(&self) -> &dyn RoutingStrategy {
        let r = rand_f64();
        if r < self.split {
            self.strategy_a.as_ref()
        } else {
            self.strategy_b.as_ref()
        }
    }
}

/// Minimal thread-safe float in [0,1) without pulling in the `rand` crate.
fn rand_f64() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    (nanos % 1_000_000) as f64 / 1_000_000.0
}

#[async_trait]
impl RoutingStrategy for ExperimentRouter {
    async fn select(
        &self,
        signals: &TurnSignals,
        candidates: &[ModelCandidate],
        objective: &ObjectiveWeights,
    ) -> RoutingDecision {
        let strategy = self.pick_strategy();
        let mut decision = strategy.select(signals, candidates, objective).await;
        // Tag with experiment label so metadata is distinguishable.
        decision.strategy_name = format!("experiment:{}", strategy.strategy_name());
        decision
    }

    fn strategy_name(&self) -> &str {
        "experiment"
    }

    fn record_outcome(&self, outcome: &TurnOutcome) {
        // Fan out to both so each strategy can update its own internal state.
        self.strategy_a.record_outcome(outcome);
        self.strategy_b.record_outcome(outcome);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::router::{ModelTier, TurnSignals};
    use crate::strategy::{ModelCandidate, ObjectiveWeights, RoutingDecision, RoutingStrategy};

    struct FixedRouter(ModelTier, &'static str);

    #[async_trait::async_trait]
    impl RoutingStrategy for FixedRouter {
        async fn select(
            &self,
            _signals: &TurnSignals,
            candidates: &[ModelCandidate],
            _obj: &ObjectiveWeights,
        ) -> RoutingDecision {
            let c = candidates
                .iter()
                .find(|c| c.tier == self.0)
                .expect("candidate not found");
            RoutingDecision {
                model: c.model.clone(),
                tier: self.0,
                confidence: 1.0,
                reason: "fixed".into(),
                strategy_name: self.1.into(),
            }
        }

        fn strategy_name(&self) -> &str {
            self.1
        }
    }

    fn candidates() -> Vec<ModelCandidate> {
        vec![
            ModelCandidate {
                model: "cheap".into(),
                tier: ModelTier::Cheap,
                cost_per_1k_input: 0.001,
                cost_per_1k_output: 0.002,
                avg_latency_ms: None,
            },
            ModelCandidate {
                model: "smart".into(),
                tier: ModelTier::Smart,
                cost_per_1k_input: 0.01,
                cost_per_1k_output: 0.03,
                avg_latency_ms: None,
            },
        ]
    }

    fn signals() -> TurnSignals {
        TurnSignals {
            turn_number: 1,
            last_tool_errored: false,
            last_response_tool_only: false,
            user_message_tokens: 50,
            task_phase: crate::router::TaskPhase::Planning,
        }
    }

    #[tokio::test]
    async fn split_1_0_always_uses_strategy_a() {
        let router = ExperimentRouter::new(
            Arc::new(FixedRouter(ModelTier::Cheap, "a")),
            Arc::new(FixedRouter(ModelTier::Smart, "b")),
            1.0, // always A
        );
        let d = router
            .select(&signals(), &candidates(), &ObjectiveWeights::default())
            .await;
        assert_eq!(d.strategy_name, "experiment:a");
    }

    #[tokio::test]
    async fn split_0_0_always_uses_strategy_b() {
        let router = ExperimentRouter::new(
            Arc::new(FixedRouter(ModelTier::Cheap, "a")),
            Arc::new(FixedRouter(ModelTier::Smart, "b")),
            0.0, // always B
        );
        let d = router
            .select(&signals(), &candidates(), &ObjectiveWeights::default())
            .await;
        assert_eq!(d.strategy_name, "experiment:b");
    }
}
