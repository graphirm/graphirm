use std::sync::Arc;

use graphirm_llm::LlmProvider;

use crate::config::{AdaptiveRoutingConfig, ModelCandidateConfig};
use crate::router::{ModelRoutingConfig, ModelTier};
use crate::strategy::experiment::ExperimentRouter;
use crate::strategy::prompt_router::PromptRouter;
use crate::strategy::rule_router::RuleRouter;
use crate::strategy::{ModelCandidate, RoutingStrategy};

/// Construct the correct `RoutingStrategy` from `AdaptiveRoutingConfig`.
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

fn build_prompt_router(
    config: &AdaptiveRoutingConfig,
    provider: Arc<dyn LlmProvider>,
) -> PromptRouter {
    let (classifier_model, timeout) = config
        .prompt
        .as_ref()
        .map(|p| (p.classifier_model.clone(), p.timeout_seconds))
        .unwrap_or_else(|| ("deepseek/deepseek-chat".into(), 3));
    PromptRouter::new(provider, classifier_model, timeout)
}

/// Build `ModelCandidate` list from config, falling back to routing config tiers.
pub fn candidates_from_config(
    config_candidates: &[ModelCandidateConfig],
    routing_config: Option<&ModelRoutingConfig>,
) -> Vec<ModelCandidate> {
    if !config_candidates.is_empty() {
        return config_candidates
            .iter()
            .map(|c| ModelCandidate {
                model: c.model.clone(),
                tier: if c.tier == "smart" { ModelTier::Smart } else { ModelTier::Cheap },
                cost_per_1k_input: c.cost_per_1k_input,
                cost_per_1k_output: c.cost_per_1k_output,
                avg_latency_ms: c.avg_latency_ms,
            })
            .collect();
    }
    // Fall back to routing config tiers with zero pricing (cost estimation disabled).
    // Use model_for_tier() to strip any provider prefix (e.g. "openrouter/vendor/model"
    // becomes "vendor/model"), consistent with how the legacy router path strips prefixes.
    if let Some(rc) = routing_config {
        return vec![
            ModelCandidate {
                model: rc.model_for_tier(ModelTier::Cheap).to_string(),
                tier: ModelTier::Cheap,
                cost_per_1k_input: 0.0,
                cost_per_1k_output: 0.0,
                avg_latency_ms: None,
            },
            ModelCandidate {
                model: rc.model_for_tier(ModelTier::Smart).to_string(),
                tier: ModelTier::Smart,
                cost_per_1k_input: 0.0,
                cost_per_1k_output: 0.0,
                avg_latency_ms: None,
            },
        ];
    }
    vec![]
}
