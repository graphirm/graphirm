# Adaptive Model Router — Design

> **For Claude:** use `executing-plans` skill after this design is approved.

**Date:** 2026-03-26
**Phase:** 36
**Size:** L
**Depends on:** Phase 34 (Model Router)

---

## Problem

The current model router (Phase 34) is static — five hardcoded rule types evaluated in order,
picking between `cheap` and `smart` tiers. It doesn't learn from outcomes, doesn't track token
spend holistically, and can't be tuned by the user beyond reordering rules.

## Goal

Replace the static router with an adaptive routing framework that:

1. **Tracks token consumption and turn outcomes** per role, session, and model
2. **Routes prompts through competing strategies** (rule-based, prompt-based, learned)
3. **A/B tests strategies** against each other with graph-persisted experiment data
4. **Optimizes for a composite objective** (cost, quality, speed) with user-tunable weights and presets
5. **Learns from historical data** — structural heuristics as always-on signal, optional user feedback as override

## Decisions

| Decision | Chosen | Alternatives considered | Why |
|----------|--------|------------------------|-----|
| Optimization target | Composite score with tunable weights + presets | Pure cost, pure quality, pure speed | Users have different priorities; presets give sensible defaults, manual weights give control |
| Decision brain | A/B test: statistical model (A+C) vs prompt-based router (D) | Single strategy, layered on existing router | Don't guess — measure which performs better; graph is the experiment log |
| Outcome signal | Structural heuristics + optional user feedback override | Pure user feedback, pure LLM-as-judge | Structural signals are free and always-on; user feedback sharpens without requiring constant input |
| Architecture | `RoutingStrategy` trait with strategy pattern | Extend existing `RoutingRule` enum | Strategies that learn and compete are fundamentally different from static rules; trait gives clean abstraction |
| Scope (phase 1) | Full vertical slice — prompt-based router first, tracking infra for learned router | Data layer only, or both routers minimal | Ship something usable immediately; A/B framework ready for when statistical router has data |

---

## Architecture

### Core Abstraction: `RoutingStrategy` Trait

```rust
#[async_trait]
trait RoutingStrategy: Send + Sync {
    /// Pick a model for this turn.
    async fn select(
        &self,
        signals: &TurnSignals,
        candidates: &[ModelCandidate],
        objective: &ObjectiveWeights,
    ) -> RoutingDecision;

    /// Name for tracking/logging.
    fn strategy_name(&self) -> &str;

    /// Called post-turn with outcome data. Strategies that learn use this.
    fn record_outcome(&self, _outcome: &TurnOutcome) {}
}
```

### Supporting Types

```rust
struct ModelCandidate {
    model: String,
    tier: ModelTier,
    cost_per_1k_input: f64,
    cost_per_1k_output: f64,
    avg_latency_ms: Option<u64>,
}

struct RoutingDecision {
    model: String,
    tier: ModelTier,
    confidence: f64,
    reason: String,
}

struct ObjectiveWeights {
    cost_weight: f64,
    quality_weight: f64,
    speed_weight: f64,
}
```

### Strategy Implementations

#### 1. `RuleRouter`

Wraps the existing `ModelRouter` logic. Backward compatible. `record_outcome` is a no-op.
Evaluates the same five rule types (`first_turn`, `error_recovery`, `high_complexity`,
`tool_only_turn`, `stuck_detection`).

#### 2. `PromptRouter`

Sends a short classification prompt to the cheapest available model:

> "Given this user message and recent context summary, which model tier should handle this
> turn? Consider: task complexity, need for reasoning, code generation requirements.
> Respond with exactly one word: cheap or smart."

Parses the one-word response. Falls back to `cheap` on timeout/parse-error.
Overhead: ~200-500ms + ~50 tokens per turn.

#### 3. `ExperimentRouter`

Wraps two strategies. Per-turn, randomly assigns one (configurable split, default 50/50).
Records which strategy was active in Interaction node metadata (`routing_strategy`,
`routing_experiment`). Has a `report()` method that queries the graph and compares
composite scores between the two arms.

---

## Data Model

### TurnOutcome (persisted in Interaction node metadata)

```rust
struct TurnOutcome {
    input_tokens: u32,
    output_tokens: u32,
    cache_read_tokens: Option<u32>,
    latency_ms: u64,
    model_used: String,
    routing_strategy: String,
    routing_decision_ms: Option<u64>,
    escalation_triggered: bool,
    tool_errors: u32,
    turn_number: u32,
    user_rating: Option<u8>,
}
```

Already tracked: `usage_input`, `usage_output`, `model_tier`, `routing_rule`.
New fields: `latency_ms`, `routing_strategy`, `routing_decision_ms`,
`escalation_triggered`, `tool_errors`, `user_rating`.

### SessionScore (persisted on Agent node metadata)

```rust
struct SessionScore {
    total_input_tokens: u64,
    total_output_tokens: u64,
    total_latency_ms: u64,
    turn_count: u32,
    error_turns: u32,
    escalation_count: u32,
    avg_user_rating: Option<f32>,
    completion_status: String,
    cost_estimate_usd: Option<f64>,
    composite_score: f64,
}
```

### ObjectiveWeights (TOML config)

```toml
[routing.objective]
preset = "balanced"
# manual overrides (normalized to sum=1.0)
cost_weight = 0.4
quality_weight = 0.4
speed_weight = 0.2
```

| Preset | Cost | Quality | Speed |
|--------|------|---------|-------|
| `balanced` | 0.4 | 0.4 | 0.2 |
| `cost_focused` | 0.7 | 0.2 | 0.1 |
| `quality_first` | 0.1 | 0.7 | 0.2 |
| `speed` | 0.2 | 0.2 | 0.6 |

Composite score formula:
`score = w_cost * (1 - normalized_cost) + w_quality * quality_signal + w_speed * (1 - normalized_latency)`

Signals normalized 0-1 against historical baseline (or session range when insufficient history).

### User Feedback

```
PATCH /api/sessions/:id/turns/:turn_id/rating
{ "rating": 4 }
```

Writes `user_rating` to the Interaction node metadata. Optional — structural heuristics
work without it. When present, overrides the quality signal for that turn.

---

## Configuration

```toml
[routing]
strategy = "experiment"  # "rules" | "prompt" | "statistical" | "experiment"

[routing.objective]
preset = "balanced"
# cost_weight = 0.4
# quality_weight = 0.4
# speed_weight = 0.2

[routing.candidates]
# model pricing for cost estimation
[[routing.candidates.models]]
model = "deepseek/deepseek-chat"
tier = "cheap"
cost_per_1k_input = 0.00014
cost_per_1k_output = 0.00028

[[routing.candidates.models]]
model = "anthropic/claude-sonnet-4"
tier = "smart"
cost_per_1k_input = 0.003
cost_per_1k_output = 0.015

[routing.experiment]
strategy_a = "rules"
strategy_b = "prompt"
split = 0.5

[routing.prompt]
classifier_model = "deepseek/deepseek-chat"
```

---

## Workflow Integration

In `stream_and_record` (crates/agent/src/workflow.rs):

**Before LLM call:**
```
let t0 = Instant::now();
let decision = strategy.select(&signals, &candidates, &objective).await;
let routing_decision_ms = t0.elapsed().as_millis();
// use decision.model for the LLM call
```

**After LLM response:**
```
let outcome = TurnOutcome {
    input_tokens: response.usage.input_tokens,
    output_tokens: response.usage.output_tokens,
    latency_ms: turn_elapsed.as_millis(),
    model_used: decision.model.clone(),
    routing_strategy: strategy.strategy_name().into(),
    routing_decision_ms: Some(routing_decision_ms),
    escalation_triggered: ...,
    tool_errors: ...,
    turn_number: ...,
    user_rating: None,
    cache_read_tokens: response.usage.cache_read_tokens,
};
strategy.record_outcome(&outcome);
// persist outcome fields in Interaction node metadata
```

**At session end (or on-demand):**
```
let score = compute_session_score(graph, agent_id, &objective);
// persist on Agent node metadata
```

---

## API Additions

| Method | Path | Purpose |
|--------|------|---------|
| `PATCH` | `/api/sessions/:id/turns/:turn_id/rating` | Set user rating (1-5) on a turn |
| `GET` | `/api/sessions/:id/routing-report` | A/B experiment results for this session |
| `GET` | `/api/routing/report` | Global A/B experiment results across all sessions |
| `GET` | `/api/routing/presets` | List available objective presets |
| `PATCH` | `/api/sessions/:id/objective` | Override objective weights for a session |

---

## File Layout (new/modified)

| File | Change |
|------|--------|
| `crates/agent/src/strategy.rs` | **New.** `RoutingStrategy` trait, `RoutingDecision`, `ModelCandidate`, `ObjectiveWeights` |
| `crates/agent/src/strategy/rule_router.rs` | **New.** Wraps existing `ModelRouter` |
| `crates/agent/src/strategy/prompt_router.rs` | **New.** Prompt-based classifier |
| `crates/agent/src/strategy/experiment.rs` | **New.** A/B test wrapper |
| `crates/agent/src/strategy/scoring.rs` | **New.** `TurnOutcome`, `SessionScore`, `compute_session_score` |
| `crates/agent/src/router.rs` | Keep for backward compat. `RuleRouter` delegates to it. |
| `crates/agent/src/config.rs` | Add `RoutingConfig`, `ObjectiveConfig`, `ExperimentConfig` |
| `crates/agent/src/workflow.rs` | Wire strategy.select/record_outcome, capture latency |
| `crates/server/src/routes.rs` | Rating endpoint, routing report endpoints |
| `crates/server/src/types.rs` | Request/response types for new endpoints |
| `config/default.toml` | Add `[routing]` section |

---

## Phase 1 Scope (this implementation)

1. `RoutingStrategy` trait + `ModelCandidate` + `ObjectiveWeights` + `RoutingDecision`
2. `TurnOutcome` tracking in workflow (latency, strategy name, decision overhead)
3. `RuleRouter` — wraps existing `ModelRouter`, backward compatible
4. `PromptRouter` — classification via cheap LLM call
5. `ExperimentRouter` — A/B wrapper with random split
6. `ObjectiveWeights` with presets in TOML config
7. `SessionScore` computation + Agent node persistence
8. User rating API endpoint
9. Routing report API endpoint (per-session + global)
10. Wire into `stream_and_record`

## Phase 2 (future, after data collection)

- `StatisticalRouter` — logistic regression or decision tree trained on graph history
- LLM-as-judge bootstrapping for training labels
- Auto-preset selection based on historical composite scores
- Cost estimation with model pricing table
- Dashboard visualization in web-app

---

## Testing

- Unit tests for each strategy (mock LLM for `PromptRouter`)
- Unit tests for `TurnOutcome` + `SessionScore` computation
- Unit tests for `ObjectiveWeights` presets + normalization
- Integration test: `ExperimentRouter` records correct metadata in graph
- Integration test: rating endpoint updates Interaction node
- All existing model router tests remain passing (RuleRouter delegates to ModelRouter)
