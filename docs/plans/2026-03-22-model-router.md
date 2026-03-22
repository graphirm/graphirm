# Model Router: Cheap/Smart Per-Turn Model Selection

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `ModelRouter` that automatically selects between a cheap and smart LLM per turn, using graph-derived session signals. Routing decisions are stored on Interaction node metadata for cost analysis and heuristic tuning.

**Architecture:** A `ModelRouter` struct evaluates ordered rules against recent session state (turn number, tool failures, message complexity) and returns the model string to use. It hooks into `stream_and_record` before `CompletionConfig` is built. When `model_routing` is `None` in config, behaviour is unchanged (single model, as today).

**Tech Stack:** Rust (agent crate, config, workflow)

**Key decisions:**
- Router lives in `crates/agent/src/router.rs` — new file, no circular deps
- Rules are evaluated in declaration order, first match wins; unmatched falls through to `default_tier`
- Routing decisions stamped as metadata on the assistant Interaction node (`model_tier`, `model_selected`, `routing_rule`)
- No LLM-based meta-routing (too expensive); no cascade/retry (too slow) — heuristics only
- Two providers can share a single `LlmProvider` instance if they're on the same backend (e.g. both OpenRouter); otherwise the router must hold two provider instances
- Provider construction deferred: the router returns a model string, not a provider — `stream_and_record` already builds `CompletionConfig` from a model string

**Success criteria:**
- [ ] `[agent.routing]` config section parsed from TOML; absent = single-model (backward compatible)
- [ ] `ModelRouter::select()` returns correct model string for each rule type (verified by unit tests)
- [ ] First turn of a session always uses smart model (when routing enabled)
- [ ] Tool-only turns (previous response was all tool calls, no errors) use cheap model
- [ ] Error recovery turns (last tool call failed) use smart model
- [ ] Routing decision metadata appears on Interaction nodes
- [ ] All existing tests pass unchanged
- [ ] `cargo clippy -D warnings` clean

**Risks:**
- Two-provider sessions need two API keys and two `LlmProvider` instances if they span different backends (e.g. DeepSeek cheap + Anthropic smart). Mitigation: document that both providers must have valid keys configured. OpenRouter as a single backend for both tiers avoids this entirely.
- Cheap model might produce malformed tool calls that waste a turn. Mitigation: the existing soft escalation mechanism already catches repeated failures and can be combined with the router (escalate to smart after N cheap failures).

---

## Task 1: Define `ModelRoutingConfig` and `ModelTier` types

**Files:**
- Create: `crates/agent/src/router.rs`
- Modify: `crates/agent/src/lib.rs` (add `pub mod router;`)

**Step 1: Create the router module with config types**

`crates/agent/src/router.rs`:

```rust
use serde::Deserialize;

/// Which model tier to use for a given turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelTier {
    Cheap,
    Smart,
}

/// A single routing rule — evaluated in order, first match wins.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RoutingRule {
    /// First turn of a session → always use this tier.
    FirstTurn { tier: ModelTier },
    /// Previous tool call returned an error → use this tier.
    ErrorRecovery { tier: ModelTier },
    /// User message exceeds `min_tokens` estimated tokens → use this tier.
    HighComplexity { min_tokens: usize, tier: ModelTier },
    /// Previous assistant response contained only tool calls (no reasoning) → use this tier.
    ToolOnlyTurn { tier: ModelTier },
    /// Agent has been running for more than `turn` turns → use this tier.
    StuckDetection { turn: u32, tier: ModelTier },
}

/// Two-tier model routing configuration.
///
/// When present in `AgentConfig`, the router selects between `cheap` and `smart`
/// models per turn based on ordered rules. When absent, single-model behaviour
/// is preserved.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelRoutingConfig {
    /// Model string for cheap/fast turns (e.g. "openrouter/deepseek/deepseek-chat").
    pub cheap: String,
    /// Model string for complex/smart turns (e.g. "openrouter/anthropic/claude-sonnet-4").
    pub smart: String,
    /// Which tier to use when no rule matches.
    #[serde(default = "default_tier")]
    pub default_tier: ModelTier,
    /// Ordered list of routing rules. First match wins.
    #[serde(default)]
    pub rules: Vec<RoutingRule>,
}

fn default_tier() -> ModelTier {
    ModelTier::Cheap
}

impl ModelRoutingConfig {
    /// Resolve a tier to its concrete model string.
    pub fn model_for_tier(&self, tier: ModelTier) -> &str {
        match tier {
            ModelTier::Cheap => &self.cheap,
            ModelTier::Smart => &self.smart,
        }
    }
}
```

**Step 2: Register the module**

In `crates/agent/src/lib.rs`, add `pub mod router;` alongside the existing module declarations.

**Step 3: Verify**

```bash
cargo check -p graphirm-agent
```

**Step 4: Commit**

```bash
git add crates/agent/src/router.rs crates/agent/src/lib.rs
git commit -m "feat(agent): add ModelRoutingConfig and ModelTier types"
```

---

## Task 2: Implement `ModelRouter::select()` with rule evaluation

**Files:**
- Modify: `crates/agent/src/router.rs`

**Step 1: Add the `TurnSignals` struct and `ModelRouter`**

Append to `router.rs`:

```rust
/// Signals extracted from recent session state, used by the router to pick a tier.
///
/// The caller (workflow) is responsible for populating these from the graph
/// before calling `select()`.
#[derive(Debug, Clone)]
pub struct TurnSignals {
    /// 1-based turn number within the session.
    pub turn_number: u32,
    /// Whether the most recent tool execution returned an error.
    pub last_tool_errored: bool,
    /// Whether the previous assistant response contained only tool calls (no text).
    pub last_response_tool_only: bool,
    /// Estimated token count of the current user message.
    pub user_message_tokens: usize,
}

/// Stateless router — evaluates rules against turn signals.
pub struct ModelRouter<'a> {
    config: &'a ModelRoutingConfig,
}

impl<'a> ModelRouter<'a> {
    pub fn new(config: &'a ModelRoutingConfig) -> Self {
        Self { config }
    }

    /// Evaluate rules in order against the current turn's signals.
    /// Returns `(selected_model_string, tier, rule_name)`.
    pub fn select(&self, signals: &TurnSignals) -> (&str, ModelTier, &'static str) {
        for rule in &self.config.rules {
            if let Some((tier, name)) = self.evaluate_rule(rule, signals) {
                return (self.config.model_for_tier(tier), tier, name);
            }
        }
        (
            self.config.model_for_tier(self.config.default_tier),
            self.config.default_tier,
            "default",
        )
    }

    fn evaluate_rule(
        &self,
        rule: &RoutingRule,
        signals: &TurnSignals,
    ) -> Option<(ModelTier, &'static str)> {
        match rule {
            RoutingRule::FirstTurn { tier } if signals.turn_number == 1 => {
                Some((*tier, "first_turn"))
            }
            RoutingRule::ErrorRecovery { tier } if signals.last_tool_errored => {
                Some((*tier, "error_recovery"))
            }
            RoutingRule::HighComplexity { min_tokens, tier }
                if signals.user_message_tokens >= *min_tokens =>
            {
                Some((*tier, "high_complexity"))
            }
            RoutingRule::ToolOnlyTurn { tier } if signals.last_response_tool_only => {
                Some((*tier, "tool_only_turn"))
            }
            RoutingRule::StuckDetection { turn, tier }
                if signals.turn_number >= *turn =>
            {
                Some((*tier, "stuck_detection"))
            }
            _ => None,
        }
    }
}
```

**Step 2: Add unit tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ModelRoutingConfig {
        ModelRoutingConfig {
            cheap: "deepseek/deepseek-chat".into(),
            smart: "anthropic/claude-sonnet-4".into(),
            default_tier: ModelTier::Cheap,
            rules: vec![
                RoutingRule::FirstTurn { tier: ModelTier::Smart },
                RoutingRule::ErrorRecovery { tier: ModelTier::Smart },
                RoutingRule::HighComplexity { min_tokens: 200, tier: ModelTier::Smart },
                RoutingRule::ToolOnlyTurn { tier: ModelTier::Cheap },
                RoutingRule::StuckDetection { turn: 15, tier: ModelTier::Smart },
            ],
        }
    }

    #[test]
    fn first_turn_selects_smart() {
        let cfg = test_config();
        let router = ModelRouter::new(&cfg);
        let signals = TurnSignals {
            turn_number: 1,
            last_tool_errored: false,
            last_response_tool_only: false,
            user_message_tokens: 50,
        };
        let (model, tier, rule) = router.select(&signals);
        assert_eq!(model, "anthropic/claude-sonnet-4");
        assert_eq!(tier, ModelTier::Smart);
        assert_eq!(rule, "first_turn");
    }

    #[test]
    fn error_recovery_selects_smart() {
        let cfg = test_config();
        let router = ModelRouter::new(&cfg);
        let signals = TurnSignals {
            turn_number: 5,
            last_tool_errored: true,
            last_response_tool_only: false,
            user_message_tokens: 30,
        };
        let (_, tier, rule) = router.select(&signals);
        assert_eq!(tier, ModelTier::Smart);
        assert_eq!(rule, "error_recovery");
    }

    #[test]
    fn tool_only_turn_selects_cheap() {
        let cfg = test_config();
        let router = ModelRouter::new(&cfg);
        let signals = TurnSignals {
            turn_number: 3,
            last_tool_errored: false,
            last_response_tool_only: true,
            user_message_tokens: 20,
        };
        let (model, tier, rule) = router.select(&signals);
        assert_eq!(model, "deepseek/deepseek-chat");
        assert_eq!(tier, ModelTier::Cheap);
        assert_eq!(rule, "tool_only_turn");
    }

    #[test]
    fn high_complexity_selects_smart() {
        let cfg = test_config();
        let router = ModelRouter::new(&cfg);
        let signals = TurnSignals {
            turn_number: 4,
            last_tool_errored: false,
            last_response_tool_only: false,
            user_message_tokens: 300,
        };
        let (_, tier, rule) = router.select(&signals);
        assert_eq!(tier, ModelTier::Smart);
        assert_eq!(rule, "high_complexity");
    }

    #[test]
    fn stuck_detection_selects_smart() {
        let cfg = test_config();
        let router = ModelRouter::new(&cfg);
        let signals = TurnSignals {
            turn_number: 20,
            last_tool_errored: false,
            last_response_tool_only: false,
            user_message_tokens: 30,
        };
        let (_, tier, rule) = router.select(&signals);
        assert_eq!(tier, ModelTier::Smart);
        assert_eq!(rule, "stuck_detection");
    }

    #[test]
    fn no_rule_matches_uses_default() {
        let cfg = test_config();
        let router = ModelRouter::new(&cfg);
        let signals = TurnSignals {
            turn_number: 3,
            last_tool_errored: false,
            last_response_tool_only: false,
            user_message_tokens: 50,
        };
        let (model, tier, rule) = router.select(&signals);
        assert_eq!(model, "deepseek/deepseek-chat");
        assert_eq!(tier, ModelTier::Cheap);
        assert_eq!(rule, "default");
    }

    #[test]
    fn empty_rules_uses_default() {
        let cfg = ModelRoutingConfig {
            cheap: "cheap-model".into(),
            smart: "smart-model".into(),
            default_tier: ModelTier::Smart,
            rules: vec![],
        };
        let router = ModelRouter::new(&cfg);
        let signals = TurnSignals {
            turn_number: 1,
            last_tool_errored: true,
            last_response_tool_only: true,
            user_message_tokens: 999,
        };
        let (model, tier, rule) = router.select(&signals);
        assert_eq!(model, "smart-model");
        assert_eq!(tier, ModelTier::Smart);
        assert_eq!(rule, "default");
    }

    #[test]
    fn first_matching_rule_wins() {
        // ErrorRecovery is listed after FirstTurn — FirstTurn should win on turn 1 with error
        let cfg = test_config();
        let router = ModelRouter::new(&cfg);
        let signals = TurnSignals {
            turn_number: 1,
            last_tool_errored: true,
            last_response_tool_only: false,
            user_message_tokens: 500,
        };
        let (_, _, rule) = router.select(&signals);
        assert_eq!(rule, "first_turn");
    }

    #[test]
    fn model_routing_config_deserialize() {
        let toml_str = r#"
            cheap = "deepseek/deepseek-chat"
            smart = "anthropic/claude-sonnet-4"
            default_tier = "cheap"

            [[rules]]
            type = "first_turn"
            tier = "smart"

            [[rules]]
            type = "error_recovery"
            tier = "smart"

            [[rules]]
            type = "high_complexity"
            min_tokens = 200
            tier = "smart"

            [[rules]]
            type = "tool_only_turn"
            tier = "cheap"

            [[rules]]
            type = "stuck_detection"
            turn = 15
            tier = "smart"
        "#;
        let cfg: ModelRoutingConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.cheap, "deepseek/deepseek-chat");
        assert_eq!(cfg.smart, "anthropic/claude-sonnet-4");
        assert_eq!(cfg.default_tier, ModelTier::Cheap);
        assert_eq!(cfg.rules.len(), 5);
    }
}
```

**Step 3: Verify**

```bash
cargo test -p graphirm-agent -- router
```

**Step 4: Commit**

```bash
git add crates/agent/src/router.rs
git commit -m "feat(agent): implement ModelRouter::select() with rule evaluation and tests"
```

---

## Task 3: Wire `ModelRoutingConfig` into `AgentConfig`

**Files:**
- Modify: `crates/agent/src/config.rs`

**Step 1: Add the field to `AgentConfig`**

Import `ModelRoutingConfig` from the router module. Add to `AgentConfig`:

```rust
/// Two-tier model routing. When `Some`, the router selects between cheap and
/// smart models per turn. When `None`, `model` is used for every turn.
#[serde(default)]
pub model_routing: Option<ModelRoutingConfig>,
```

Add the same field to `AgentConfigSection` (the TOML parsing struct) and wire it through in `AgentConfig::from_toml()`.

Update `AgentConfig::default()` to set `model_routing: None`.

**Step 2: Add deserialization test**

```rust
#[test]
fn model_routing_parsed_from_toml() {
    let toml = r#"
        [agent]
        name = "test"
        model = "fallback-model"
        system_prompt = "test"
        max_turns = 5

        [agent.routing]
        cheap = "deepseek/deepseek-chat"
        smart = "anthropic/claude-sonnet-4"
        default_tier = "cheap"

        [[agent.routing.rules]]
        type = "first_turn"
        tier = "smart"
    "#;
    let config = AgentConfig::from_toml(toml).unwrap();
    let routing = config.model_routing.unwrap();
    assert_eq!(routing.cheap, "deepseek/deepseek-chat");
    assert_eq!(routing.smart, "anthropic/claude-sonnet-4");
    assert_eq!(routing.rules.len(), 1);
}

#[test]
fn model_routing_absent_by_default() {
    let config = AgentConfig::default();
    assert!(config.model_routing.is_none());
}
```

**Step 3: Verify**

```bash
cargo test -p graphirm-agent -- config
```

**Step 4: Commit**

```bash
git add crates/agent/src/config.rs
git commit -m "feat(agent): add model_routing to AgentConfig with TOML deserialization"
```

---

## Task 4: Wire the router into `stream_and_record`

**Files:**
- Modify: `crates/agent/src/workflow.rs`

**Step 1: Build `TurnSignals` from session state**

Before `CompletionConfig::new(...)` (around line 113), add signal extraction and routing:

```rust
// Model routing: select cheap or smart model based on session signals.
let (selected_model, routing_meta) = if let Some(ref routing) = session.agent_config.model_routing {
    let signals = build_turn_signals(session).await;
    let router = crate::router::ModelRouter::new(routing);
    let (model, tier, rule) = router.select(&signals);
    tracing::info!(
        model = model,
        tier = ?tier,
        rule = rule,
        turn = signals.turn_number,
        "model router selected"
    );
    (
        model.to_string(),
        Some((tier, rule)),
    )
} else {
    (session.agent_config.model.clone(), None)
};

let config = CompletionConfig::new(&selected_model)
    .with_max_tokens(/* ... existing logic ... */)
    .with_temperature(/* ... existing logic ... */);
```

**Step 2: Implement `build_turn_signals`**

Add a helper function in `workflow.rs`:

```rust
/// Extract routing signals from the current session state.
async fn build_turn_signals(session: &Session) -> crate::router::TurnSignals {
    let graph = session.graph.clone();
    let agent_id = session.id.clone();
    let signals = tokio::task::spawn_blocking(move || {
        // Count interactions with this session_id to determine turn number.
        let chain = graph.get_session_chain(&agent_id.0).unwrap_or_default();
        let turn_number = chain.iter()
            .filter(|n| matches!(&n.node_type, NodeType::Interaction(i) if i.role == "human"))
            .count() as u32;

        // Check last assistant interaction for tool-only and error signals.
        let last_assistant = chain.iter().rev()
            .find(|n| matches!(&n.node_type, NodeType::Interaction(i) if i.role == "assistant"));
        let last_tool_errored = last_assistant
            .and_then(|n| n.metadata.get("tool_calls"))
            .is_some()
            && chain.iter().rev()
                .find(|n| matches!(&n.node_type, NodeType::Interaction(i) if i.role == "tool_result"))
                .and_then(|n| n.metadata.get("is_error"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
        let last_response_tool_only = last_assistant
            .map(|n| {
                n.metadata.get("tool_calls").is_some()
                    && matches!(&n.node_type, NodeType::Interaction(i) if i.content.trim().is_empty())
            })
            .unwrap_or(false);

        // Estimate tokens for the most recent human message.
        let last_human_tokens = chain.iter().rev()
            .find(|n| matches!(&n.node_type, NodeType::Interaction(i) if i.role == "human"))
            .map(|n| match &n.node_type {
                NodeType::Interaction(i) => i.content.len() / 4, // rough estimate
                _ => 0,
            })
            .unwrap_or(0);

        crate::router::TurnSignals {
            turn_number: turn_number.max(1),
            last_tool_errored,
            last_response_tool_only,
            user_message_tokens: last_human_tokens,
        }
    })
    .await
    .unwrap_or_else(|_| crate::router::TurnSignals {
        turn_number: 1,
        last_tool_errored: false,
        last_response_tool_only: false,
        user_message_tokens: 0,
    });

    signals
}
```

**Step 3: Stamp routing metadata on the Interaction node**

After the metadata map is built (around line 155), add:

```rust
if let Some((tier, rule)) = routing_meta {
    metadata.insert("model_tier".to_string(), serde_json::json!(format!("{tier:?}").to_lowercase()));
    metadata.insert("model_selected".to_string(), serde_json::json!(selected_model));
    metadata.insert("routing_rule".to_string(), serde_json::json!(rule));
}
```

**Step 4: Verify**

```bash
cargo check -p graphirm-agent
cargo test -p graphirm-agent
```

**Step 5: Commit**

```bash
git add crates/agent/src/workflow.rs
git commit -m "feat(agent): wire model router into stream_and_record with signal extraction"
```

---

## Task 5: Add TOML config example and update `default.toml`

**Files:**
- Modify: `config/default.toml`

**Step 1: Add commented-out routing section**

After the `system_prompt` closing `"""` and before `[knowledge]`, add:

```toml
# Two-tier model routing — uncomment to enable automatic cheap/smart selection.
# When enabled, the router evaluates rules per turn and selects the appropriate model.
# When disabled (default), `model` above is used for every turn.
#
# [agent.routing]
# cheap = "openrouter/deepseek/deepseek-chat"
# smart = "openrouter/anthropic/claude-sonnet-4"
# default_tier = "cheap"
#
# [[agent.routing.rules]]
# type = "first_turn"
# tier = "smart"
#
# [[agent.routing.rules]]
# type = "error_recovery"
# tier = "smart"
#
# [[agent.routing.rules]]
# type = "high_complexity"
# min_tokens = 200
# tier = "smart"
#
# [[agent.routing.rules]]
# type = "tool_only_turn"
# tier = "cheap"
#
# [[agent.routing.rules]]
# type = "stuck_detection"
# turn = 15
# tier = "smart"
```

**Step 2: Verify config still parses**

```bash
cargo check
```

**Step 3: Commit**

```bash
git add config/default.toml
git commit -m "docs: add commented model routing config example to default.toml"
```

---

## Task 6: Handle dual-provider construction

**Files:**
- Modify: `crates/server/src/routes.rs` (or wherever `LlmProvider` is constructed for a session)

**Step 1: Assess current provider construction**

The router returns a model string, but `stream_and_record` receives a single `&dyn LlmProvider`. If `cheap` and `smart` are on the same backend (e.g. both OpenRouter), one provider instance works — the model is just a parameter in `CompletionConfig`.

If they're on different backends (e.g. DeepSeek cheap + Anthropic smart), we need two providers. The simplest approach for now: when `model_routing` is set, extract the provider prefix from both model strings. If they match, use one provider. If they differ, construct both and wrap in a `DualProvider` that delegates based on model string.

**Step 2: Implement `DualProvider` (only if needed)**

For now, document that both models should use the same provider backend (e.g. both via OpenRouter). This avoids complexity. If the provider prefixes differ, log a warning and fall back to single-model mode.

Add to `router.rs`:

```rust
impl ModelRoutingConfig {
    /// Check whether both tiers use the same provider backend.
    pub fn same_provider(&self) -> bool {
        let cheap_provider = self.cheap.split('/').next().unwrap_or("");
        let smart_provider = self.smart.split('/').next().unwrap_or("");
        cheap_provider == smart_provider
    }
}
```

In session construction, if `!routing.same_provider()`, log a warning:

```rust
if let Some(ref routing) = config.model_routing {
    if !routing.same_provider() {
        tracing::warn!(
            cheap = %routing.cheap,
            smart = %routing.smart,
            "model routing tiers use different providers — routing disabled, using single model"
        );
        config.model_routing = None;
    }
}
```

**Step 3: Verify + commit**

```bash
cargo check
git add crates/agent/src/router.rs crates/server/src/routes.rs
git commit -m "feat(agent): validate same-provider constraint for model routing"
```

---

## Task 7: Update AGENTS.md and backlog

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/backlog.md`

**Step 1: Add Phase 34 to AGENTS.md**

Add to the phase table:

```markdown
| 34 | Model router — automatic cheap/smart per-turn model selection | ✅ done |
```

Add a detailed section after the last phase description:

```markdown
**Model router (Phase 34):**
- `crates/agent/src/router.rs` — `ModelRoutingConfig`, `ModelTier`, `RoutingRule`, `TurnSignals`, `ModelRouter`
- Five built-in rules: `first_turn`, `error_recovery`, `high_complexity`, `tool_only_turn`, `stuck_detection`
- Rules evaluated in declaration order; first match wins; unmatched → `default_tier`
- Routing decision stamped on Interaction node metadata: `model_tier`, `model_selected`, `routing_rule`
- `AgentConfig.model_routing: Option<ModelRoutingConfig>` — absent = single-model (backward compatible)
- Same-provider constraint: both tiers must use the same backend (e.g. both OpenRouter); mismatched providers fall back to single-model with warning
- Config: `[agent.routing]` section in TOML with `cheap`, `smart`, `default_tier`, and `[[agent.routing.rules]]` array
- 9 unit tests (all rule types, default fallback, empty rules, first-match priority, TOML deserialization)
```

**Step 2: Add to backlog**

Under the appropriate section:

```markdown
### ✅ Model router — P2 · M
Done YYYY-MM-DD. Automatic per-turn cheap/smart model selection. `ModelRouter` evaluates ordered rules (first_turn, error_recovery, high_complexity, tool_only_turn, stuck_detection) against graph-derived session signals. Routing decisions stored on Interaction node metadata for cost analysis. Same-provider constraint (both tiers via OpenRouter). Config: `[agent.routing]` in TOML.
Plan: `docs/plans/2026-03-22-model-router.md`
```

**Step 3: Commit**

```bash
git add AGENTS.md docs/backlog.md
git commit -m "docs: add Phase 34 model router to AGENTS.md and backlog"
```

---

## Future extensions (not in this plan)

These are natural follow-ups once the base router is proven:

1. **`graphirm model-stats` CLI command** — aggregate routing decisions from the graph: cost per tier, turns per tier, error rates by tier. Uses existing Interaction node metadata.

2. **Cascade/retry mode** — try cheap first; if response quality is below threshold (too short, generic, malformed tool calls), automatically retry with smart. Adds latency but catches cheap model failures.

3. **Cross-provider routing** — `DualProvider` wrapper holding two `Box<dyn LlmProvider>` instances, delegating `complete()`/`stream()` based on the model string in `CompletionConfig`. Removes the same-provider constraint.

4. **Cost tracking** — per-model token pricing table in config; running cost accumulator per session; budget limits that force cheap-only after threshold.

5. **Adaptive heuristics** — use historical routing decision data (from Interaction metadata) to auto-tune rule thresholds. E.g., if cheap model error rate exceeds 20%, lower the `min_tokens` threshold for `high_complexity`.
