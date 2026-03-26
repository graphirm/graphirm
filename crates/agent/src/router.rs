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
    /// Resolve a tier to its concrete model ID, stripping the leading provider
    /// prefix if present.
    ///
    /// The routing config uses the same `"provider/model"` format as `agent.model`
    /// (e.g. `"openrouter/qwen/qwen3-coder:free"`). Since the provider has already
    /// been selected at session creation, only the model portion is passed to the
    /// LLM provider — identical to how `parse_model_string` works for the main model.
    ///
    /// Examples:
    /// - `"openrouter/qwen/qwen3-coder:free"` → `"qwen/qwen3-coder:free"`
    /// - `"deepseek/deepseek-chat"` → `"deepseek-chat"`
    /// - `"just-a-model"` → `"just-a-model"` (no prefix, returned as-is)
    pub fn model_for_tier(&self, tier: ModelTier) -> &str {
        let raw = match tier {
            ModelTier::Cheap => &self.cheap,
            ModelTier::Smart => &self.smart,
        };
        raw.split_once('/').map(|x| x.1).unwrap_or(raw.as_str())
    }

    /// Check whether both tiers use the same provider backend.
    pub fn same_provider(&self) -> bool {
        let cheap_provider = self.cheap.split('/').next().unwrap_or("");
        let smart_provider = self.smart.split('/').next().unwrap_or("");
        cheap_provider == smart_provider
    }
}

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
            RoutingRule::StuckDetection { turn, tier } if signals.turn_number >= *turn => {
                Some((*tier, "stuck_detection"))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ModelRoutingConfig {
        ModelRoutingConfig {
            cheap: "openrouter/deepseek/deepseek-chat".into(),
            smart: "openrouter/anthropic/claude-sonnet-4".into(),
            default_tier: ModelTier::Cheap,
            rules: vec![
                RoutingRule::FirstTurn {
                    tier: ModelTier::Smart,
                },
                RoutingRule::ErrorRecovery {
                    tier: ModelTier::Smart,
                },
                RoutingRule::HighComplexity {
                    min_tokens: 200,
                    tier: ModelTier::Smart,
                },
                RoutingRule::ToolOnlyTurn {
                    tier: ModelTier::Cheap,
                },
                RoutingRule::StuckDetection {
                    turn: 15,
                    tier: ModelTier::Smart,
                },
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
        assert_eq!(model, "anthropic/claude-sonnet-4"); // openrouter/ prefix stripped
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
        assert_eq!(model, "deepseek/deepseek-chat"); // openrouter/ prefix stripped
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
        assert_eq!(model, "deepseek/deepseek-chat"); // openrouter/ prefix stripped
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

    #[test]
    fn model_for_tier_strips_provider_prefix() {
        let cfg = ModelRoutingConfig {
            cheap: "openrouter/qwen/qwen3-coder:free".into(),
            smart: "openrouter/anthropic/claude-sonnet-4".into(),
            default_tier: ModelTier::Cheap,
            rules: vec![],
        };
        assert_eq!(cfg.model_for_tier(ModelTier::Cheap), "qwen/qwen3-coder:free");
        assert_eq!(
            cfg.model_for_tier(ModelTier::Smart),
            "anthropic/claude-sonnet-4"
        );
    }

    #[test]
    fn model_for_tier_no_prefix_passthrough() {
        // Model strings without a leading provider/ segment are returned as-is.
        let cfg = ModelRoutingConfig {
            cheap: "just-a-model".into(),
            smart: "another-model".into(),
            default_tier: ModelTier::Cheap,
            rules: vec![],
        };
        assert_eq!(cfg.model_for_tier(ModelTier::Cheap), "just-a-model");
        assert_eq!(cfg.model_for_tier(ModelTier::Smart), "another-model");
    }

    #[test]
    fn same_provider_both_openrouter() {
        let cfg = ModelRoutingConfig {
            cheap: "openrouter/deepseek/deepseek-chat".into(),
            smart: "openrouter/anthropic/claude-sonnet-4".into(),
            default_tier: ModelTier::Cheap,
            rules: vec![],
        };
        assert!(cfg.same_provider());
    }

    #[test]
    fn same_provider_different_backends() {
        let cfg = ModelRoutingConfig {
            cheap: "deepseek/deepseek-chat".into(),
            smart: "anthropic/claude-sonnet-4".into(),
            default_tier: ModelTier::Cheap,
            rules: vec![],
        };
        assert!(!cfg.same_provider());
    }
}
