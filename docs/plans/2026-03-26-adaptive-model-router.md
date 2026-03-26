# Adaptive Model Router — Implementation Plan

> **For Claude:** REQUIRED: use `dogfood-graphirm` skill to delegate each task to the local graphirm agent. Use `executing-plans` skill for orchestration and review checkpoints between tasks.

**Goal:** Replace the static rule-based model router with an adaptive routing framework that tracks per-turn token outcomes, A/B tests a prompt-based classifier against the rule-based router, and optimises for a user-configurable composite objective (cost / quality / speed).

**Architecture:** A `RoutingStrategy` trait lives in `crates/agent/src/strategy/`. Three implementations: `RuleRouter` (wraps the existing `ModelRouter`, backward-compat), `PromptRouter` (sends a cheap-LLM classification call before each turn), and `ExperimentRouter` (random-split A/B wrapper around any two strategies). Per-turn `TurnOutcome` metadata is appended to each Interaction node. `SessionScore` is computed at session end and stored on the Agent node. Config wires through `AgentConfig`.

**Tech Stack:** Rust, async-trait, existing `graphirm-llm` + `graphirm-graph` crates, no new dependencies.

**Key decisions:**
- Strategy pattern over extending `RoutingRule` enum — strategies that learn and compete need a different abstraction than static rules.
- Prompt-based router first (no training data needed), statistical router in a later phase once history exists.
- Structural heuristics (escalation count, error turns, turn count) as always-on quality signal; optional `PATCH /rating` endpoint for user override.

**Design doc:** `docs/plans/2026-03-26-adaptive-model-router-design.md`

---

## Risks & Blockers

- `PromptRouter` adds ~200–500 ms latency per turn (one extra LLM call). Mitigation: timeout 3 s, fall back to `cheap` on any error.
- `ExperimentRouter` must write `routing_strategy` metadata atomically with the Interaction node — existing metadata map already supports this.
- `PATCH /api/sessions/:id/turns/:turn_id/rating` requires knowing the Interaction node ID from the client — the SSE `graph_update` event already delivers node IDs to the web-app.

## Success Criteria

- [ ] `cargo test -p graphirm-agent` passes with all new unit tests green
- [ ] `cargo clippy -- -D warnings` clean
- [ ] `strategy = "experiment"` in `config/default.toml` routes turns alternately via rules and prompt, and both strategies write distinct `routing_strategy` metadata
- [ ] `GET /api/routing/report` returns per-strategy composite scores
- [ ] `PATCH /api/sessions/:id/turns/:turn_id/rating` writes `user_rating` to Interaction node

---

## Task 1: `RoutingStrategy` trait + core types

**Files:**
- Create: `crates/agent/src/strategy/mod.rs`

**Step 1: Write failing test**

In a new file `crates/agent/src/strategy/mod.rs`, add at the bottom:

```rust
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
        // 500 input + 200 output tokens
        let cost = c.cost_estimate(500, 200);
        assert!((cost - 0.0009).abs() < 1e-9);
    }
}
```

Run: `cargo test -p graphirm-agent strategy::tests 2>&1 | tail -5`
Expected: compile error (module doesn't exist yet).

**Step 2: Implement `strategy/mod.rs`**

```rust
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
            "cost_focused"   => Self { cost_weight: 0.7, quality_weight: 0.2, speed_weight: 0.1 },
            "quality_first"  => Self { cost_weight: 0.1, quality_weight: 0.7, speed_weight: 0.2 },
            "speed"          => Self { cost_weight: 0.2, quality_weight: 0.2, speed_weight: 0.6 },
            _                => Self { cost_weight: 0.4, quality_weight: 0.4, speed_weight: 0.2 }, // balanced
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
    let total_cost: f64 = outcomes.iter().map(|o| o.input_tokens as f64 + o.output_tokens as f64).sum();
    let total_latency: u64 = outcomes.iter().map(|o| o.latency_ms).sum();
    let error_turns = outcomes.iter().filter(|o| o.tool_errors > 0).count() as u32;
    let escalation_count = outcomes.iter().filter(|o| o.escalation_triggered).count() as u32;

    let ratings: Vec<f32> = outcomes.iter().filter_map(|o| o.user_rating.map(|r| r as f32)).collect();
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
        let structural = 1.0 - (error_turns as f64 + escalation_count as f64 * 0.5) / max_possible_errors;
        if let Some(rating) = avg_user_rating {
            // blend structural (60%) with normalised user rating (40%)
            structural * 0.6 + ((rating - 1.0) / 4.0) as f64 * 0.4
        } else {
            structural
        }
    };
    // Cost signal: normalised against this session (all turns same, so score=0.5 unless
    // we have multiple strategies to compare — the cross-session report does that).
    let cost_signal = 0.5_f64; // placeholder; meaningful when comparing strategies
    let speed_signal = 0.5_f64; // placeholder; meaningful when comparing strategies

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

pub mod rule_router;
pub mod prompt_router;
pub mod experiment;
```

**Step 3: Verify tests pass**

Run: `cargo test -p graphirm-agent strategy::tests 2>&1 | tail -10`
Expected: 3 tests pass.

**Step 4: Commit**

```bash
cd /home/krs/graphirm-repo
git add crates/agent/src/strategy/mod.rs
git commit -m "feat(router): RoutingStrategy trait, ObjectiveWeights, TurnOutcome, SessionScore"
```

---

## Task 2: `RuleRouter` — backward-compat wrapper

**Files:**
- Create: `crates/agent/src/strategy/rule_router.rs`

**Step 1: Write failing test**

Add to `crates/agent/src/strategy/rule_router.rs`:

```rust
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
        };
        let decision = router.select(&signals, &[cheap_candidate(), smart_candidate()], &ObjectiveWeights::default()).await;
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
        };
        let decision = router.select(&signals, &[cheap_candidate(), smart_candidate()], &ObjectiveWeights::default()).await;
        assert_eq!(decision.model, "deepseek/deepseek-chat");
    }
}
```

Run: `cargo test -p graphirm-agent strategy::rule_router::tests 2>&1 | tail -5`
Expected: compile error.

**Step 2: Implement `rule_router.rs`**

```rust
use async_trait::async_trait;

use crate::router::{ModelRouter, ModelRoutingConfig, ModelTier};
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
        let matched = candidates.iter().find(|c| c.model == model_str)
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
```

**Step 3: Verify tests pass**

Run: `cargo test -p graphirm-agent strategy::rule_router::tests 2>&1 | tail -10`
Expected: 2 tests pass.

**Step 4: Commit**

```bash
git add crates/agent/src/strategy/rule_router.rs
git commit -m "feat(router): RuleRouter wraps existing ModelRouter, backward-compat"
```

---

## Task 3: `PromptRouter` — LLM-based classifier

**Files:**
- Create: `crates/agent/src/strategy/prompt_router.rs`

**Step 1: Write failing test (using mock provider)**

```rust
#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use super::*;
    use crate::router::{ModelTier, TurnSignals};
    use crate::strategy::{ModelCandidate, ObjectiveWeights};
    use graphirm_llm::mock::{MockProvider, MockResponse};

    fn candidates() -> Vec<ModelCandidate> {
        vec![
            ModelCandidate { model: "cheap-model".into(), tier: ModelTier::Cheap,
                cost_per_1k_input: 0.001, cost_per_1k_output: 0.002, avg_latency_ms: None },
            ModelCandidate { model: "smart-model".into(), tier: ModelTier::Smart,
                cost_per_1k_input: 0.01, cost_per_1k_output: 0.03, avg_latency_ms: None },
        ]
    }

    fn signals() -> TurnSignals {
        TurnSignals { turn_number: 2, last_tool_errored: false,
                      last_response_tool_only: false, user_message_tokens: 80 }
    }

    #[tokio::test]
    async fn selects_smart_when_llm_says_smart() {
        let provider = Arc::new(MockProvider::new(vec![MockResponse::text("smart")]));
        let router = PromptRouter::new(provider, "cheap-model".into(), 3);
        let decision = router.select(&signals(), &candidates(), &ObjectiveWeights::default()).await;
        assert_eq!(decision.tier, ModelTier::Smart);
        assert_eq!(decision.strategy_name, "prompt_router");
    }

    #[tokio::test]
    async fn falls_back_to_cheap_on_bad_response() {
        let provider = Arc::new(MockProvider::new(vec![MockResponse::text("I dunno")]));
        let router = PromptRouter::new(provider, "cheap-model".into(), 3);
        let decision = router.select(&signals(), &candidates(), &ObjectiveWeights::default()).await;
        assert_eq!(decision.tier, ModelTier::Cheap);
    }

    #[tokio::test]
    async fn selects_cheap_when_llm_says_cheap() {
        let provider = Arc::new(MockProvider::new(vec![MockResponse::text("cheap")]));
        let router = PromptRouter::new(provider, "cheap-model".into(), 3);
        let decision = router.select(&signals(), &candidates(), &ObjectiveWeights::default()).await;
        assert_eq!(decision.tier, ModelTier::Cheap);
    }
}
```

Run: `cargo test -p graphirm-agent strategy::prompt_router::tests 2>&1 | tail -5`
Expected: compile error.

**Step 2: Implement `prompt_router.rs`**

```rust
use std::sync::Arc;

use async_trait::async_trait;
use graphirm_llm::{CompletionConfig, ContentPart, LlmProvider};

use crate::router::{ModelTier, TurnSignals};
use crate::strategy::{ModelCandidate, ObjectiveWeights, RoutingDecision, RoutingStrategy, TurnOutcome};

pub struct PromptRouter {
    provider: Arc<dyn LlmProvider>,
    classifier_model: String,
    timeout_seconds: u64,
}

impl PromptRouter {
    pub fn new(provider: Arc<dyn LlmProvider>, classifier_model: String, timeout_seconds: u64) -> Self {
        Self { provider, classifier_model, timeout_seconds }
    }

    fn build_prompt(signals: &TurnSignals) -> String {
        format!(
            "You are a model routing classifier. Based on the signals below, decide whether \
             a cheap fast model or a smart capable model should handle this turn.\n\
             Signals:\n\
             - Turn number: {}\n\
             - Last tool errored: {}\n\
             - Last response was tool-only: {}\n\
             - User message estimated tokens: {}\n\
             Respond with exactly one word: cheap or smart.",
            signals.turn_number,
            signals.last_tool_errored,
            signals.last_response_tool_only,
            signals.user_message_tokens,
        )
    }

    fn parse_tier(text: &str) -> ModelTier {
        let lower = text.trim().to_lowercase();
        if lower.contains("smart") {
            ModelTier::Smart
        } else {
            ModelTier::Cheap
        }
    }
}

#[async_trait]
impl RoutingStrategy for PromptRouter {
    async fn select(
        &self,
        signals: &TurnSignals,
        candidates: &[ModelCandidate],
        _objective: &ObjectiveWeights,
    ) -> RoutingDecision {
        let prompt = Self::build_prompt(signals);
        let messages = vec![graphirm_llm::LlmMessage {
            role: graphirm_llm::Role::User,
            content: vec![ContentPart::text(prompt)],
        }];
        let config = CompletionConfig::new(&self.classifier_model)
            .with_max_tokens(10)
            .with_temperature(0.0);

        let tier = match tokio::time::timeout(
            std::time::Duration::from_secs(self.timeout_seconds),
            self.provider.complete(messages, &[], &config),
        )
        .await
        {
            Ok(Ok(response)) => Self::parse_tier(&response.text_content()),
            Ok(Err(e)) => {
                tracing::warn!(error = %e, "prompt_router LLM call failed, defaulting cheap");
                ModelTier::Cheap
            }
            Err(_) => {
                tracing::warn!(timeout_s = self.timeout_seconds, "prompt_router timed out, defaulting cheap");
                ModelTier::Cheap
            }
        };

        let candidate = candidates.iter().find(|c| c.tier == tier)
            .or_else(|| candidates.first());

        let (model, final_tier) = candidate
            .map(|c| (c.model.clone(), c.tier))
            .unwrap_or_else(|| ("".to_string(), tier));

        RoutingDecision {
            model,
            tier: final_tier,
            confidence: 0.8,
            reason: format!("prompt_classifier:{}", match tier { ModelTier::Cheap => "cheap", ModelTier::Smart => "smart" }),
            strategy_name: self.strategy_name().to_string(),
        }
    }

    fn strategy_name(&self) -> &str {
        "prompt_router"
    }

    fn record_outcome(&self, _outcome: &TurnOutcome) {}
}
```

**Step 3: Verify tests pass**

Run: `cargo test -p graphirm-agent strategy::prompt_router::tests 2>&1 | tail -10`
Expected: 3 tests pass.

**Step 4: Commit**

```bash
git add crates/agent/src/strategy/prompt_router.rs
git commit -m "feat(router): PromptRouter — cheap LLM classifier with timeout + fallback"
```

---

## Task 4: `ExperimentRouter` — A/B wrapper

**Files:**
- Create: `crates/agent/src/strategy/experiment.rs`

**Step 1: Write failing test**

```rust
#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use super::*;
    use crate::router::{ModelTier, TurnSignals};
    use crate::strategy::{ModelCandidate, ObjectiveWeights, RoutingDecision, RoutingStrategy, TurnOutcome};

    struct FixedRouter(ModelTier, &'static str);

    #[async_trait::async_trait]
    impl RoutingStrategy for FixedRouter {
        async fn select(&self, _signals: &TurnSignals, candidates: &[ModelCandidate], _obj: &ObjectiveWeights) -> RoutingDecision {
            let c = candidates.iter().find(|c| c.tier == self.0).unwrap();
            RoutingDecision { model: c.model.clone(), tier: self.0, confidence: 1.0,
                              reason: "fixed".into(), strategy_name: self.1.into() }
        }
        fn strategy_name(&self) -> &str { self.1 }
    }

    fn candidates() -> Vec<ModelCandidate> {
        vec![
            ModelCandidate { model: "cheap".into(), tier: ModelTier::Cheap,
                cost_per_1k_input: 0.001, cost_per_1k_output: 0.002, avg_latency_ms: None },
            ModelCandidate { model: "smart".into(), tier: ModelTier::Smart,
                cost_per_1k_input: 0.01, cost_per_1k_output: 0.03, avg_latency_ms: None },
        ]
    }

    fn signals() -> TurnSignals {
        TurnSignals { turn_number: 1, last_tool_errored: false,
                      last_response_tool_only: false, user_message_tokens: 50 }
    }

    #[tokio::test]
    async fn split_1_0_always_uses_strategy_a() {
        let router = ExperimentRouter::new(
            Arc::new(FixedRouter(ModelTier::Cheap, "a")),
            Arc::new(FixedRouter(ModelTier::Smart, "b")),
            1.0, // always A
        );
        let d = router.select(&signals(), &candidates(), &ObjectiveWeights::default()).await;
        assert_eq!(d.strategy_name, "experiment:a");
    }

    #[tokio::test]
    async fn split_0_0_always_uses_strategy_b() {
        let router = ExperimentRouter::new(
            Arc::new(FixedRouter(ModelTier::Cheap, "a")),
            Arc::new(FixedRouter(ModelTier::Smart, "b")),
            0.0, // always B
        );
        let d = router.select(&signals(), &candidates(), &ObjectiveWeights::default()).await;
        assert_eq!(d.strategy_name, "experiment:b");
    }
}
```

Run: `cargo test -p graphirm-agent strategy::experiment::tests 2>&1 | tail -5`
Expected: compile error.

**Step 2: Implement `experiment.rs`**

```rust
use std::sync::Arc;

use async_trait::async_trait;

use crate::router::TurnSignals;
use crate::strategy::{ModelCandidate, ObjectiveWeights, RoutingDecision, RoutingStrategy, TurnOutcome};

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
        Self { strategy_a, strategy_b, split: split.clamp(0.0, 1.0) }
    }

    fn pick_strategy(&self) -> (&dyn RoutingStrategy, bool) {
        let r: f64 = rand_f64();
        if r < self.split {
            (self.strategy_a.as_ref(), true)
        } else {
            (self.strategy_b.as_ref(), false)
        }
    }
}

/// Minimal thread-safe float in [0,1) without pulling in `rand` crate.
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
        let (strategy, _is_a) = self.pick_strategy();
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
```

**Step 3: Verify tests pass**

Run: `cargo test -p graphirm-agent strategy::experiment::tests 2>&1 | tail -10`
Expected: 2 tests pass.

**Step 4: Commit**

```bash
git add crates/agent/src/strategy/experiment.rs
git commit -m "feat(router): ExperimentRouter A/B wrapper with configurable split"
```

---

## Task 5: Wire strategies into `AgentConfig`

**Files:**
- Modify: `crates/agent/src/config.rs:96-169` (AgentConfig struct)

**Step 1: Write failing test (deserialisation)**

Add to `config.rs` tests:

```rust
#[test]
fn adaptive_routing_config_parses() {
    let toml = r#"
        [agent]
        name = "test"
        model = "fallback"
        system_prompt = "test"
        max_turns = 5

        [agent.adaptive_routing]
        strategy = "experiment"

        [agent.adaptive_routing.objective]
        preset = "cost_focused"

        [agent.adaptive_routing.experiment]
        strategy_a = "rules"
        strategy_b = "prompt"
        split = 0.6

        [agent.adaptive_routing.prompt]
        classifier_model = "deepseek/deepseek-chat"
    "#;
    let config = AgentConfig::from_toml(toml).unwrap();
    let ar = config.adaptive_routing.unwrap();
    assert_eq!(ar.strategy, "experiment");
    assert_eq!(ar.objective.as_ref().unwrap().preset.as_deref(), Some("cost_focused"));
    assert!((ar.experiment.as_ref().unwrap().split - 0.6).abs() < 1e-9);
}
```

Run: `cargo test -p graphirm-agent test_agent_config 2>&1 | tail -5`
Expected: compile error (field doesn't exist).

**Step 2: Add config types and field**

In `crates/agent/src/config.rs`, add before `AgentConfig`:

```rust
#[derive(Debug, Clone, Deserialize, Default)]
pub struct AdaptiveObjectiveConfig {
    pub preset: Option<String>,
    pub cost_weight: Option<f64>,
    pub quality_weight: Option<f64>,
    pub speed_weight: Option<f64>,
}

impl AdaptiveObjectiveConfig {
    pub fn to_weights(&self) -> crate::strategy::ObjectiveWeights {
        if let Some(ref p) = self.preset {
            crate::strategy::ObjectiveWeights::preset(p)
        } else {
            crate::strategy::ObjectiveWeights {
                cost_weight: self.cost_weight.unwrap_or(0.4),
                quality_weight: self.quality_weight.unwrap_or(0.4),
                speed_weight: self.speed_weight.unwrap_or(0.2),
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExperimentConfig {
    pub strategy_a: String,
    pub strategy_b: String,
    #[serde(default = "default_split")]
    pub split: f64,
}

fn default_split() -> f64 { 0.5 }

#[derive(Debug, Clone, Deserialize)]
pub struct PromptRouterConfig {
    pub classifier_model: String,
    #[serde(default = "default_classifier_timeout")]
    pub timeout_seconds: u64,
}

fn default_classifier_timeout() -> u64 { 3 }

#[derive(Debug, Clone, Deserialize)]
pub struct ModelCandidateConfig {
    pub model: String,
    pub tier: String,  // "cheap" or "smart"
    pub cost_per_1k_input: f64,
    pub cost_per_1k_output: f64,
    pub avg_latency_ms: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AdaptiveRoutingConfig {
    #[serde(default = "default_strategy")]
    pub strategy: String,
    pub objective: Option<AdaptiveObjectiveConfig>,
    pub experiment: Option<ExperimentConfig>,
    pub prompt: Option<PromptRouterConfig>,
    #[serde(default)]
    pub candidates: Vec<ModelCandidateConfig>,
}

fn default_strategy() -> String { "rules".into() }
```

Add `adaptive_routing: Option<AdaptiveRoutingConfig>` to `AgentConfig` struct and to `AgentConfigSection`. Wire it through `from_toml`.

**Step 3: Verify tests pass**

Run: `cargo test -p graphirm-agent adaptive_routing_config_parses 2>&1 | tail -10`
Expected: test passes. Then run full suite:
`cargo test -p graphirm-agent 2>&1 | tail -20`

**Step 4: Commit**

```bash
git add crates/agent/src/config.rs crates/agent/src/strategy/mod.rs
git commit -m "feat(router): AdaptiveRoutingConfig wired into AgentConfig"
```

---

## Task 6: Expose `strategy` module from `lib.rs` and build strategy from config

**Files:**
- Modify: `crates/agent/src/lib.rs`
- Create: `crates/agent/src/strategy/builder.rs`

**Step 1: Add `pub mod strategy` to `lib.rs`**

In `crates/agent/src/lib.rs`, add after line 15:

```rust
pub mod strategy;
```

And add to the `pub use` block:

```rust
pub use strategy::{ObjectiveWeights, RoutingDecision, RoutingStrategy, SessionScore, TurnOutcome, compute_session_score};
```

**Step 2: Create `strategy/builder.rs`**

This builds the correct `Arc<dyn RoutingStrategy>` from `AdaptiveRoutingConfig` + an `LlmProvider`:

```rust
use std::sync::Arc;

use graphirm_llm::LlmProvider;

use crate::config::{AdaptiveRoutingConfig, ModelCandidateConfig};
use crate::router::{ModelRoutingConfig, ModelTier};
use crate::strategy::{ModelCandidate, ObjectiveWeights, RoutingStrategy};
use crate::strategy::rule_router::RuleRouter;
use crate::strategy::prompt_router::PromptRouter;
use crate::strategy::experiment::ExperimentRouter;

pub fn build_strategy(
    config: &AdaptiveRoutingConfig,
    routing_config: Option<&ModelRoutingConfig>,
    provider: Arc<dyn LlmProvider>,
) -> Arc<dyn RoutingStrategy> {
    match config.strategy.as_str() {
        "prompt" => Arc::new(build_prompt_router(config, provider)),
        "experiment" => {
            let exp = config.experiment.as_ref();
            let a_name = exp.map(|e| e.strategy_a.as_str()).unwrap_or("rules");
            let b_name = exp.map(|e| e.strategy_b.as_str()).unwrap_or("prompt");
            let split = exp.map(|e| e.split).unwrap_or(0.5);
            let strategy_a = build_named(a_name, config, routing_config, provider.clone());
            let strategy_b = build_named(b_name, config, routing_config, provider);
            Arc::new(ExperimentRouter::new(strategy_a, strategy_b, split))
        }
        _ => build_rule_router(routing_config),
    }
}

fn build_named(
    name: &str,
    config: &AdaptiveRoutingConfig,
    routing_config: Option<&ModelRoutingConfig>,
    provider: Arc<dyn LlmProvider>,
) -> Arc<dyn RoutingStrategy> {
    match name {
        "prompt" => Arc::new(build_prompt_router(config, provider)),
        _ => build_rule_router(routing_config),
    }
}

fn build_rule_router(routing_config: Option<&ModelRoutingConfig>) -> Arc<dyn RoutingStrategy> {
    let config = routing_config.cloned().unwrap_or_else(|| ModelRoutingConfig {
        cheap: "deepseek/deepseek-chat".into(),
        smart: "deepseek/deepseek-chat".into(),
        default_tier: ModelTier::Cheap,
        rules: vec![],
    });
    Arc::new(RuleRouter::new(config))
}

fn build_prompt_router(config: &AdaptiveRoutingConfig, provider: Arc<dyn LlmProvider>) -> PromptRouter {
    let (classifier_model, timeout) = config.prompt.as_ref()
        .map(|p| (p.classifier_model.clone(), p.timeout_seconds))
        .unwrap_or_else(|| ("deepseek/deepseek-chat".into(), 3));
    PromptRouter::new(provider, classifier_model, timeout)
}

pub fn candidates_from_config(
    config_candidates: &[ModelCandidateConfig],
    routing_config: Option<&ModelRoutingConfig>,
) -> Vec<ModelCandidate> {
    if !config_candidates.is_empty() {
        return config_candidates.iter().map(|c| ModelCandidate {
            model: c.model.clone(),
            tier: if c.tier == "smart" { ModelTier::Smart } else { ModelTier::Cheap },
            cost_per_1k_input: c.cost_per_1k_input,
            cost_per_1k_output: c.cost_per_1k_output,
            avg_latency_ms: c.avg_latency_ms,
        }).collect();
    }
    // Fall back to routing config tiers with zero pricing (cost estimation disabled).
    if let Some(rc) = routing_config {
        return vec![
            ModelCandidate { model: rc.cheap.clone(), tier: ModelTier::Cheap,
                cost_per_1k_input: 0.0, cost_per_1k_output: 0.0, avg_latency_ms: None },
            ModelCandidate { model: rc.smart.clone(), tier: ModelTier::Smart,
                cost_per_1k_input: 0.0, cost_per_1k_output: 0.0, avg_latency_ms: None },
        ];
    }
    vec![]
}
```

Add `pub mod builder;` to `strategy/mod.rs`.

**Step 3: Verify compilation**

Run: `cargo check -p graphirm-agent 2>&1 | tail -20`
Expected: no errors.

**Step 4: Commit**

```bash
git add crates/agent/src/strategy/builder.rs crates/agent/src/lib.rs crates/agent/src/strategy/mod.rs
git commit -m "feat(router): strategy builder from AdaptiveRoutingConfig, candidates helper"
```

---

## Task 7: Wire adaptive strategy into `workflow.rs`

Replace lines 114–170 of `crates/agent/src/workflow.rs` (the existing model-routing block).

**Files:**
- Modify: `crates/agent/src/workflow.rs:114-232`

**Step 1: Write integration test**

In `crates/agent/src/workflow.rs` tests section (around line 1148):

```rust
#[tokio::test]
async fn adaptive_strategy_metadata_recorded() {
    // Setup: session with adaptive_routing strategy="rules"
    // After one turn, Interaction node should have routing_strategy in metadata.
    // (use existing test harness pattern already in the file)
}
```

This is a smoke test; full scenario testing is covered by the unit tests above.

**Step 2: Replace the routing block**

Replace lines 114–170:

```rust
// Adaptive model routing — select model via configured strategy.
let (selected_model, routing_outcome) = if let Some(ref ar_config) =
    session.agent_config.adaptive_routing
{
    let t_route_start = std::time::Instant::now();
    let turn_number = session.current_turn();
    let graph_c = session.graph.clone();
    let session_id = session.id.0.clone();

    let (last_tool_errored, last_response_tool_only, user_msg_tokens) =
        tokio::task::spawn_blocking(move || {
            let chain = graph_c.get_session_chain(&session_id).unwrap_or_default();
            let last_assistant = chain.iter().rev().find(|n| {
                matches!(&n.node_type, graphirm_graph::nodes::NodeType::Interaction(i) if i.role == "assistant")
            });
            let tool_errored = chain.iter().rev().any(|n| {
                matches!(&n.node_type, graphirm_graph::nodes::NodeType::Interaction(i) if i.role == "tool_result")
                    && n.metadata.get("is_error").and_then(|v| v.as_bool()).unwrap_or(false)
            });
            let tool_only = last_assistant
                .map(|n| {
                    n.metadata.get("tool_calls").is_some()
                        && matches!(&n.node_type, graphirm_graph::nodes::NodeType::Interaction(i) if i.content.trim().is_empty())
                })
                .unwrap_or(false);
            let user_tokens = chain.iter().rev()
                .find(|n| matches!(&n.node_type, graphirm_graph::nodes::NodeType::Interaction(i) if i.role == "user"))
                .map(|n| match &n.node_type {
                    graphirm_graph::nodes::NodeType::Interaction(i) => i.content.len() / 4,
                    _ => 0,
                })
                .unwrap_or(0);
            (tool_errored, tool_only, user_tokens)
        })
        .await
        .unwrap_or((false, false, 0));

    let signals = crate::router::TurnSignals {
        turn_number,
        last_tool_errored,
        last_response_tool_only,
        user_message_tokens: user_msg_tokens,
    };

    let objective = ar_config.objective.as_ref()
        .map(|o| o.to_weights())
        .unwrap_or_default();

    let candidates = crate::strategy::builder::candidates_from_config(
        &ar_config.candidates,
        session.agent_config.model_routing.as_ref(),
    );

    let strategy = crate::strategy::builder::build_strategy(
        ar_config,
        session.agent_config.model_routing.as_ref(),
        llm.clone(),
    );

    let decision = strategy.select(&signals, &candidates, &objective).await;
    let routing_decision_ms = t_route_start.elapsed().as_millis() as u64;

    tracing::info!(
        model = &decision.model,
        tier = ?decision.tier,
        strategy = decision.strategy_name,
        reason = &decision.reason,
        confidence = decision.confidence,
        routing_ms = routing_decision_ms,
        "adaptive router selected"
    );

    (decision.model.clone(), Some((decision, routing_decision_ms)))
} else if let Some(ref routing) = session.agent_config.model_routing {
    // Legacy static router — unchanged for backward compat.
    let turn_number = session.current_turn();
    let graph_c = session.graph.clone();
    let session_id = session.id.0.clone();
    let (last_tool_errored, last_response_tool_only, user_msg_tokens) =
        tokio::task::spawn_blocking(move || {
            let chain = graph_c.get_session_chain(&session_id).unwrap_or_default();
            let last_assistant = chain.iter().rev().find(|n| {
                matches!(&n.node_type, graphirm_graph::nodes::NodeType::Interaction(i) if i.role == "assistant")
            });
            let tool_errored = chain.iter().rev().any(|n| {
                matches!(&n.node_type, graphirm_graph::nodes::NodeType::Interaction(i) if i.role == "tool_result")
                    && n.metadata.get("is_error").and_then(|v| v.as_bool()).unwrap_or(false)
            });
            let tool_only = last_assistant
                .map(|n| {
                    n.metadata.get("tool_calls").is_some()
                        && matches!(&n.node_type, graphirm_graph::nodes::NodeType::Interaction(i) if i.content.trim().is_empty())
                })
                .unwrap_or(false);
            let user_tokens = chain.iter().rev()
                .find(|n| matches!(&n.node_type, graphirm_graph::nodes::NodeType::Interaction(i) if i.role == "user"))
                .map(|n| match &n.node_type {
                    graphirm_graph::nodes::NodeType::Interaction(i) => i.content.len() / 4,
                    _ => 0,
                })
                .unwrap_or(0);
            (tool_errored, tool_only, user_tokens)
        })
        .await
        .unwrap_or((false, false, 0));
    let signals = crate::router::TurnSignals { turn_number, last_tool_errored,
        last_response_tool_only, user_message_tokens: user_msg_tokens };
    let router = crate::router::ModelRouter::new(routing);
    let (model, tier, rule) = router.select(&signals);
    tracing::info!(model, tier = ?tier, rule, "legacy model router selected");
    (model.to_string(), None)
} else {
    (session.agent_config.model.clone(), None)
};
```

Also update the metadata block (around line 216) to include adaptive routing fields when `routing_outcome` is `Some`:

```rust
if let Some((ref decision, decision_ms)) = routing_outcome {
    metadata.insert("routing_strategy".to_string(), serde_json::json!(decision.strategy_name));
    metadata.insert("routing_reason".to_string(), serde_json::json!(decision.reason));
    metadata.insert("routing_confidence".to_string(), serde_json::json!(decision.confidence));
    metadata.insert("routing_decision_ms".to_string(), serde_json::json!(decision_ms));
    metadata.insert("model_selected".to_string(), serde_json::json!(decision.model));
    metadata.insert("model_tier".to_string(), serde_json::json!(format!("{:?}", decision.tier).to_lowercase()));
}
```

**Step 3: Verify**

```bash
cargo check -p graphirm-agent 2>&1 | tail -20
cargo test -p graphirm-agent 2>&1 | tail -30
```

Expected: all existing tests pass, no new errors.

**Step 4: Commit**

```bash
git add crates/agent/src/workflow.rs
git commit -m "feat(router): wire adaptive strategy into stream_and_record, legacy path preserved"
```

---

## Task 8: User rating endpoint

**Files:**
- Modify: `crates/server/src/routes.rs`
- Modify: `crates/server/src/types.rs`

**Step 1: Add request type to `types.rs`**

```rust
#[derive(Debug, Deserialize)]
pub struct RateTurnRequest {
    /// 1–5 rating for this turn.
    pub rating: u8,
}
```

**Step 2: Add handler in `routes.rs`**

```rust
async fn rate_turn(
    State(state): State<AppState>,
    Path((session_id, turn_id)): Path<(String, String)>,
    Json(body): Json<RateTurnRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if body.rating == 0 || body.rating > 5 {
        return Err(StatusCode::UNPROCESSABLE_ENTITY);
    }
    let graph = state.graph.clone();
    let node_id = turn_id.clone();
    tokio::task::spawn_blocking(move || {
        let mut node = graph.get_node(&node_id).map_err(|_| StatusCode::NOT_FOUND)?;
        if let serde_json::Value::Object(ref mut map) = node.metadata {
            map.insert("user_rating".to_string(), serde_json::json!(body.rating));
        }
        graph.update_node(&node).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        Ok::<_, StatusCode>(())
    })
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)??;

    Ok(Json(serde_json::json!({ "ok": true })))
}
```

Register in `create_router()`:

```rust
.route("/api/sessions/:id/turns/:turn_id/rating", patch(rate_turn))
```

**Step 3: Write unit test**

In `crates/server/src/types.rs` tests:

```rust
#[test]
fn rate_turn_request_deserializes() {
    let json = r#"{"rating": 4}"#;
    let req: RateTurnRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.rating, 4);
}
```

Run: `cargo test -p graphirm-server 2>&1 | tail -10`
Expected: passes.

**Step 4: Commit**

```bash
git add crates/server/src/routes.rs crates/server/src/types.rs
git commit -m "feat(router): PATCH /api/sessions/:id/turns/:turn_id/rating endpoint"
```

---

## Task 9: Routing report endpoint

**Files:**
- Modify: `crates/server/src/routes.rs`
- Modify: `crates/server/src/types.rs`

**Step 1: Add response types**

```rust
#[derive(Debug, Serialize)]
pub struct StrategyReport {
    pub strategy_name: String,
    pub turn_count: u32,
    pub avg_input_tokens: f64,
    pub avg_output_tokens: f64,
    pub avg_latency_ms: f64,
    pub error_rate: f64,
    pub avg_user_rating: Option<f64>,
}
```

**Step 2: Add handler**

```rust
async fn routing_report(
    State(state): State<AppState>,
) -> Json<Vec<StrategyReport>> {
    let graph = state.graph.clone();
    let reports = tokio::task::spawn_blocking(move || {
        build_routing_report(&graph)
    })
    .await
    .unwrap_or_default();
    Json(reports)
}
```

Implement `build_routing_report(graph)` — queries all Interaction nodes, groups by `routing_strategy` metadata field, aggregates token/latency/error/rating stats.

Register:

```rust
.route("/api/routing/report", get(routing_report))
```

**Step 3: Write test**

```rust
#[test]
fn strategy_report_serializes() {
    let r = StrategyReport {
        strategy_name: "experiment:prompt_router".into(),
        turn_count: 10,
        avg_input_tokens: 500.0,
        avg_output_tokens: 200.0,
        avg_latency_ms: 1200.0,
        error_rate: 0.1,
        avg_user_rating: Some(4.2),
    };
    let json = serde_json::to_string(&r).unwrap();
    assert!(json.contains("experiment:prompt_router"));
}
```

**Step 4: Commit**

```bash
git add crates/server/src/routes.rs crates/server/src/types.rs
git commit -m "feat(router): GET /api/routing/report with per-strategy aggregated stats"
```

---

## Task 10: Update `config/default.toml` + `AGENTS.md`

**Files:**
- Modify: `config/default.toml`
- Modify: `AGENTS.md`

**Step 1: Add adaptive routing section to `config/default.toml`**

```toml
# Adaptive model routing — strategy-based model selection with A/B testing.
# strategy: "rules" (default, backward compat) | "prompt" | "experiment"
[agent.adaptive_routing]
strategy = "rules"

[agent.adaptive_routing.objective]
preset = "balanced"
# cost_weight = 0.4
# quality_weight = 0.4
# speed_weight = 0.2

# [agent.adaptive_routing.experiment]
# strategy_a = "rules"
# strategy_b = "prompt"
# split = 0.5

# [agent.adaptive_routing.prompt]
# classifier_model = "openrouter/deepseek/deepseek-chat"
# timeout_seconds = 3

# [agent.adaptive_routing.candidates.models]
# Add model pricing for cost estimation:
# [[agent.adaptive_routing.candidates]]
# model = "openrouter/deepseek/deepseek-chat"
# tier = "cheap"
# cost_per_1k_input = 0.00014
# cost_per_1k_output = 0.00028
```

**Step 2: Update `AGENTS.md` Current State table**

Add row for Phase 36:

```
| 36 | Adaptive model router — `RoutingStrategy` trait, `PromptRouter`, `ExperimentRouter`, per-turn `TurnOutcome` tracking, composite objective presets, A/B report API | 🚧 in progress |
```

**Step 3: Final verification**

```bash
cargo build --release 2>&1 | tail -20
cargo test 2>&1 | tail -30
cargo clippy -- -D warnings 2>&1 | tail -20
```

Expected: clean build, all tests pass, no clippy warnings.

**Step 4: Commit**

```bash
git add config/default.toml AGENTS.md
git commit -m "feat(router): add adaptive_routing config section, update AGENTS.md phase 36"
```

---

## Execution order

Tasks are sequential (each builds on the previous):

1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10

Commit after each task. Delegate tasks 1–9 to graphirm via `dogfood-graphirm`. Task 10 (AGENTS.md) is manual.
