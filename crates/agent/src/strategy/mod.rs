use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::router::ModelTier;

/// A single model candidate available for routing.
#[derive(Debug, Clone)]
pub struct ModelCandidate {
    pub model: String,
    pub tier: ModelTier,
    pub cost_per_1k_input: f64,
    pub cost_per_1k_output: f64,
    pub avg_latency_ms: Option<u64>,
}

impl ModelCandidate {
    /// Estimated cost in USD for a turn with the given token counts.
    pub fn cost_estimate(&self, input_tokens: u32, output_tokens: u32) -> f64 {
        (input_tokens as f64 / 1000.0) * self.cost_per_1k_input
            + (output_tokens as f64 / 1000.0) * self.cost_per_1k_output
    }
}

/// The outcome of a routing decision.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub model: String,
    pub tier: ModelTier,
    /// 0.0–1.0 confidence from the strategy.
    pub confidence: f64,
    /// Human-readable reason, persisted in Interaction metadata.
    pub reason: String,
    /// Name of the strategy that produced this decision.
    pub strategy_name: String,
}

/// Composite objective weights. Must sum to 1.0 (call `normalized()` before use).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ObjectiveWeights {
    pub cost_weight: f64,
    pub quality_weight: f64,
    pub speed_weight: f64,
}

impl Default for ObjectiveWeights {
    fn default() -> Self {
        Self::preset("balanced")
    }
}

impl ObjectiveWeights {
    /// Built-in presets.
    pub fn preset(name: &str) -> Self {
        match name {
            "cost_focused" => Self { cost_weight: 0.7, quality_weight: 0.2, speed_weight: 0.1 },
            "quality_first" => Self { cost_weight: 0.1, quality_weight: 0.7, speed_weight: 0.2 },
            "speed" => Self { cost_weight: 0.2, quality_weight: 0.2, speed_weight: 0.6 },
            _ => Self { cost_weight: 0.4, quality_weight: 0.4, speed_weight: 0.2 }, // balanced
        }
    }

    /// Return a copy with weights normalised so they sum to 1.0.
    pub fn normalized(&self) -> Self {
        let sum = self.cost_weight + self.quality_weight + self.speed_weight;
        if sum == 0.0 {
            return Self::default();
        }
        Self {
            cost_weight: self.cost_weight / sum,
            quality_weight: self.quality_weight / sum,
            speed_weight: self.speed_weight / sum,
        }
    }
}

/// Per-turn outcome, appended to Interaction node metadata after the LLM call.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TurnOutcome {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_read_tokens: Option<u32>,
    pub latency_ms: u64,
    pub model_used: String,
    pub routing_strategy: String,
    pub routing_decision_ms: u64,
    pub escalation_triggered: bool,
    pub tool_errors: u32,
    pub turn_number: u32,
    /// 1–5, set via PATCH /rating. None until user rates.
    pub user_rating: Option<u8>,
}

/// Aggregated score for a session, persisted on the Agent node.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionScore {
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_latency_ms: u64,
    pub turn_count: u32,
    pub error_turns: u32,
    pub escalation_count: u32,
    pub avg_user_rating: Option<f32>,
    pub completion_status: String,
    pub composite_score: f64,
}

/// Compute a composite score for a session given its outcomes and weights.
/// All signals are normalised 0–1 within this session's own range.
pub fn compute_session_score(outcomes: &[TurnOutcome], weights: &ObjectiveWeights) -> SessionScore {
    if outcomes.is_empty() {
        return SessionScore::default();
    }
    let w = weights.normalized();
    let total_latency: u64 = outcomes.iter().map(|o| o.latency_ms).sum();
    let error_turns = outcomes.iter().filter(|o| o.tool_errors > 0).count() as u32;
    let escalation_count = outcomes.iter().filter(|o| o.escalation_triggered).count() as u32;

    let ratings: Vec<f32> = outcomes
        .iter()
        .filter_map(|o| o.user_rating.map(|r| r as f32))
        .collect();
    let avg_user_rating = if ratings.is_empty() {
        None
    } else {
        Some(ratings.iter().sum::<f32>() / ratings.len() as f32)
    };

    // Quality signal: 0=many errors, 1=no errors; boosted by user rating if present.
    let max_possible_errors = outcomes.len() as f64;
    let quality_signal = if max_possible_errors == 0.0 {
        1.0
    } else {
        let structural =
            1.0 - (error_turns as f64 + escalation_count as f64 * 0.5) / max_possible_errors;
        if let Some(rating) = avg_user_rating {
            // Blend structural (60%) with normalised user rating (40%).
            structural * 0.6 + ((rating - 1.0) / 4.0) as f64 * 0.4
        } else {
            structural
        }
    };
    // Cost and speed signals are placeholders; meaningful when comparing strategies cross-session.
    let cost_signal = 0.5_f64;
    let speed_signal = 0.5_f64;

    let composite_score = w.cost_weight * (1.0 - cost_signal)
        + w.quality_weight * quality_signal.clamp(0.0, 1.0)
        + w.speed_weight * (1.0 - speed_signal);

    SessionScore {
        total_input_tokens: outcomes.iter().map(|o| o.input_tokens as u64).sum(),
        total_output_tokens: outcomes.iter().map(|o| o.output_tokens as u64).sum(),
        total_latency_ms: total_latency,
        turn_count: outcomes.len() as u32,
        error_turns,
        escalation_count,
        avg_user_rating,
        completion_status: "completed".into(),
        composite_score,
    }
}

/// Core routing abstraction. All strategies implement this.
#[async_trait]
pub trait RoutingStrategy: Send + Sync {
    async fn select(
        &self,
        signals: &crate::router::TurnSignals,
        candidates: &[ModelCandidate],
        objective: &ObjectiveWeights,
    ) -> RoutingDecision;

    fn strategy_name(&self) -> &str;

    /// Called after each turn. Strategies that learn override this.
    fn record_outcome(&self, _outcome: &TurnOutcome) {}
}

pub mod builder;
pub mod experiment;
pub mod prompt_router;
pub mod rule_router;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn objective_weights_preset_balanced() {
        let w = ObjectiveWeights::preset("balanced");
        assert!((w.cost_weight - 0.4).abs() < 1e-9);
        assert!((w.quality_weight - 0.4).abs() < 1e-9);
        assert!((w.speed_weight - 0.2).abs() < 1e-9);
    }

    #[test]
    fn objective_weights_normalize() {
        let w = ObjectiveWeights { cost_weight: 2.0, quality_weight: 2.0, speed_weight: 1.0 };
        let n = w.normalized();
        assert!((n.cost_weight - 0.4).abs() < 1e-9);
        assert!((n.speed_weight - 0.2).abs() < 1e-9);
    }

    #[test]
    fn model_candidate_cost_estimate() {
        let c = ModelCandidate {
            model: "test".into(),
            tier: crate::router::ModelTier::Cheap,
            cost_per_1k_input: 0.001,
            cost_per_1k_output: 0.002,
            avg_latency_ms: None,
        };
        // 500 input + 200 output tokens → 0.5*0.001 + 0.2*0.002 = 0.0005 + 0.0004 = 0.0009
        let cost = c.cost_estimate(500, 200);
        assert!((cost - 0.0009).abs() < 1e-9);
    }
}
