use std::sync::Arc;
use std::time::Duration;

use graphirm_llm::{DecisionsClient, DecisionsConfig, LlmProvider};

use crate::config::{AdaptiveRoutingConfig, JevRouterConfig, ModelCandidateConfig};
use crate::router::{ModelRoutingConfig, ModelTier};
use crate::strategy::experiment::ExperimentRouter;
use crate::strategy::jev_router::JevRouter;
use crate::strategy::prompt_router::PromptRouter;
use crate::strategy::rule_router::RuleRouter;
use crate::strategy::{ModelCandidate, RoutingStrategy};

/// Construct the correct `RoutingStrategy` from `AdaptiveRoutingConfig`.
///
/// Strategy names: `"rules"` (default), `"prompt"`, `"jev"`, `"experiment"`.
pub fn build_strategy(
    config: &AdaptiveRoutingConfig,
    routing_config: Option<&ModelRoutingConfig>,
    provider: Arc<dyn LlmProvider>,
) -> Arc<dyn RoutingStrategy> {
    match config.strategy.as_str() {
        "prompt" => Arc::new(build_prompt_router(config, provider)),
        "jev" => build_jev_router(config, routing_config, decisions_client_from_env(config)),
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
    let client = if name == "jev" {
        decisions_client_from_env(config)
    } else {
        None
    };
    build_named_with_client(name, config, routing_config, provider, client)
}

fn build_named_with_client(
    name: &str,
    config: &AdaptiveRoutingConfig,
    routing_config: Option<&ModelRoutingConfig>,
    provider: Arc<dyn LlmProvider>,
    client: Option<DecisionsClient>,
) -> Arc<dyn RoutingStrategy> {
    match name {
        "prompt" => Arc::new(build_prompt_router(config, provider)),
        "jev" => build_jev_router(config, routing_config, client),
        _ => build_rule_router(routing_config),
    }
}

/// Resolve `[agent.adaptive_routing.jev]` plus the process environment into a
/// client; `None` (with a warning) when a required key is missing.
fn decisions_client_from_env(config: &AdaptiveRoutingConfig) -> Option<DecisionsClient> {
    let jev = config.jev.clone().unwrap_or_default();
    match resolve_decisions_config(&jev, |name| std::env::var(name).ok()) {
        Ok(cfg) => Some(DecisionsClient::from_config(cfg)),
        Err(e) => {
            tracing::warn!(reason = %e, "jev strategy requested but no Decisions client; using rules");
            None
        }
    }
}

/// Turn `JevRouterConfig` into a concrete `DecisionsConfig`.
///
/// Key resolution: an explicit `api_key_env` must be set and non-blank; the
/// OpenRouter default requires `OPENROUTER_API_KEY`; a custom endpoint without
/// `api_key_env` sends no key. `lookup` is the environment (injected for tests).
/// The error string names the variable but never its value.
fn resolve_decisions_config(
    jev: &JevRouterConfig,
    lookup: impl Fn(&str) -> Option<String>,
) -> Result<DecisionsConfig, String> {
    let defaults = DecisionsConfig::default();
    let endpoint = jev.endpoint.clone().unwrap_or(defaults.endpoint);
    let model = jev.model.clone().unwrap_or(defaults.model);

    let key_env: Option<&str> = match (&jev.api_key_env, jev.endpoint.is_some()) {
        (Some(name), _) => Some(name.as_str()),
        (None, false) => Some("OPENROUTER_API_KEY"),
        (None, true) => None,
    };
    let api_key = match key_env {
        Some(name) => match lookup(name) {
            Some(v) if !v.trim().is_empty() => Some(v),
            _ => return Err(format!("{name} not set")),
        },
        None => None,
    };

    Ok(DecisionsConfig {
        endpoint,
        api_key,
        model,
    })
}

fn default_routing_config(routing_config: Option<&ModelRoutingConfig>) -> ModelRoutingConfig {
    routing_config
        .cloned()
        .unwrap_or_else(|| ModelRoutingConfig {
            cheap: vec!["deepseek/deepseek-chat".into()],
            smart: vec!["deepseek/deepseek-chat".into()],
            default_tier: ModelTier::Cheap,
            rules: vec![],
        })
}

fn build_rule_router(routing_config: Option<&ModelRoutingConfig>) -> Arc<dyn RoutingStrategy> {
    Arc::new(RuleRouter::new(default_routing_config(routing_config)))
}

fn jev_timeout(config: &AdaptiveRoutingConfig) -> Duration {
    let ms = config
        .jev
        .as_ref()
        .map(|j| j.timeout_ms)
        .unwrap_or_else(|| JevRouterConfig::default().timeout_ms);
    Duration::from_millis(ms)
}

/// `JevRouter` over `client`, falling back to `RuleRouter` per turn on error and
/// entirely when no client is available.
fn build_jev_router(
    config: &AdaptiveRoutingConfig,
    routing_config: Option<&ModelRoutingConfig>,
    client: Option<DecisionsClient>,
) -> Arc<dyn RoutingStrategy> {
    match client {
        Some(client) => Arc::new(JevRouter::new(
            Arc::new(client),
            RuleRouter::new(default_routing_config(routing_config)),
            jev_timeout(config),
        )),
        None => build_rule_router(routing_config),
    }
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
                tier: if c.tier == "smart" {
                    ModelTier::Smart
                } else {
                    ModelTier::Cheap
                },
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use graphirm_llm::{DecisionsClient, DecisionsTransport, LlmError};
    use serde_json::{Value, json};

    use super::*;
    use crate::config::JevRouterConfig;

    struct NullTransport;

    #[async_trait]
    impl DecisionsTransport for NullTransport {
        async fn post(&self, _body: &Value) -> Result<Value, LlmError> {
            Ok(json!({}))
        }
    }

    fn jev_config(timeout_ms: Option<u64>) -> AdaptiveRoutingConfig {
        AdaptiveRoutingConfig {
            strategy: "jev".into(),
            objective: None,
            experiment: None,
            prompt: None,
            jev: timeout_ms.map(|timeout_ms| JevRouterConfig {
                timeout_ms,
                ..JevRouterConfig::default()
            }),
            candidates: vec![],
        }
    }

    fn env_with(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: Vec<(String, String)> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |name: &str| map.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone())
    }

    #[test]
    fn resolve_default_provider_uses_openrouter_key() {
        let cfg = JevRouterConfig::default();
        let resolved = resolve_decisions_config(&cfg, env_with(&[("OPENROUTER_API_KEY", "sk-or")]))
            .expect("resolved");
        assert_eq!(
            resolved.endpoint,
            graphirm_llm::decisions::OPENROUTER_DECISIONS_URL
        );
        assert_eq!(resolved.api_key.as_deref(), Some("sk-or"));
        assert_eq!(resolved.model, graphirm_llm::decisions::JEV_MODEL);
    }

    #[test]
    fn resolve_default_provider_without_key_is_error() {
        let cfg = JevRouterConfig::default();
        let err = resolve_decisions_config(&cfg, env_with(&[])).unwrap_err();
        assert!(err.contains("OPENROUTER_API_KEY"), "{err}");
    }

    #[test]
    fn resolve_local_endpoint_without_key_env_sends_no_key() {
        let cfg = JevRouterConfig {
            endpoint: Some("http://localhost:8787/v1/systemone".into()),
            model: Some("laya/systemone-1".into()),
            ..JevRouterConfig::default()
        };
        let resolved = resolve_decisions_config(&cfg, env_with(&[])).expect("resolved");
        assert_eq!(resolved.endpoint, "http://localhost:8787/v1/systemone");
        assert_eq!(resolved.api_key, None);
        assert_eq!(resolved.model, "laya/systemone-1");
    }

    #[test]
    fn resolve_explicit_key_env_reads_that_variable() {
        let cfg = JevRouterConfig {
            endpoint: Some("http://localhost:8787/v1/systemone".into()),
            api_key_env: Some("LAYA_API_KEY".into()),
            ..JevRouterConfig::default()
        };
        let resolved =
            resolve_decisions_config(&cfg, env_with(&[("LAYA_API_KEY", "lk")])).expect("resolved");
        assert_eq!(resolved.api_key.as_deref(), Some("lk"));
    }

    #[test]
    fn resolve_explicit_key_env_missing_is_error() {
        let cfg = JevRouterConfig {
            endpoint: Some("http://localhost:8787/v1/systemone".into()),
            api_key_env: Some("LAYA_API_KEY".into()),
            ..JevRouterConfig::default()
        };
        let err = resolve_decisions_config(&cfg, env_with(&[])).unwrap_err();
        assert!(err.contains("LAYA_API_KEY"), "{err}");
    }

    #[test]
    fn resolve_blank_key_counts_as_missing() {
        let cfg = JevRouterConfig::default();
        assert!(resolve_decisions_config(&cfg, env_with(&[("OPENROUTER_API_KEY", "  ")])).is_err());
    }

    #[test]
    fn jev_strategy_with_client_builds_jev_router() {
        let client = DecisionsClient::with_transport(Arc::new(NullTransport));
        let strategy = build_jev_router(&jev_config(Some(700)), None, Some(client));
        assert_eq!(strategy.strategy_name(), "jev_router");
    }

    #[test]
    fn jev_strategy_without_client_falls_back_to_rule_router() {
        let strategy = build_jev_router(&jev_config(None), None, None);
        assert_eq!(strategy.strategy_name(), "rule_router");
    }

    #[test]
    fn jev_timeout_defaults_when_section_absent() {
        assert_eq!(jev_timeout(&jev_config(None)).as_millis(), 1500);
        assert_eq!(jev_timeout(&jev_config(Some(250))).as_millis(), 250);
    }

    #[test]
    fn experiment_can_name_jev_as_an_arm() {
        // Without a key in the environment the jev arm degrades to rules; the
        // point is that "jev" is an accepted arm name, not an unknown one.
        let client = DecisionsClient::with_transport(Arc::new(NullTransport));
        let strategy = build_named_with_client(
            "jev",
            &jev_config(None),
            None,
            Arc::new(graphirm_llm::MockProvider::new(vec![])),
            Some(client),
        );
        assert_eq!(strategy.strategy_name(), "jev_router");
    }
}
