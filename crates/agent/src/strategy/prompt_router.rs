use std::sync::Arc;

use async_trait::async_trait;
use graphirm_llm::{CompletionConfig, ContentPart, LlmMessage, LlmProvider, Role};

use crate::router::{ModelTier, TurnSignals};
use crate::strategy::{ModelCandidate, ObjectiveWeights, RoutingDecision, RoutingStrategy, TurnOutcome};

pub struct PromptRouter {
    provider: Arc<dyn LlmProvider>,
    classifier_model: String,
    timeout_seconds: u64,
}

impl PromptRouter {
    pub fn new(
        provider: Arc<dyn LlmProvider>,
        classifier_model: String,
        timeout_seconds: u64,
    ) -> Self {
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
        let messages = vec![LlmMessage {
            role: Role::Human,
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
                tracing::warn!(
                    timeout_s = self.timeout_seconds,
                    "prompt_router timed out, defaulting cheap"
                );
                ModelTier::Cheap
            }
        };

        let candidate = candidates.iter().find(|c| c.tier == tier).or_else(|| candidates.first());

        let (model, final_tier) = candidate
            .map(|c| (c.model.clone(), c.tier))
            .unwrap_or_else(|| ("".to_string(), tier));

        RoutingDecision {
            model,
            tier: final_tier,
            confidence: 0.8,
            reason: format!(
                "prompt_classifier:{}",
                match tier {
                    ModelTier::Cheap => "cheap",
                    ModelTier::Smart => "smart",
                }
            ),
            strategy_name: self.strategy_name().to_string(),
        }
    }

    fn strategy_name(&self) -> &str {
        "prompt_router"
    }

    fn record_outcome(&self, _outcome: &TurnOutcome) {}
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::router::{ModelTier, TurnSignals};
    use crate::strategy::{ModelCandidate, ObjectiveWeights};
    use graphirm_llm::mock::{MockProvider, MockResponse};

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

    fn signals() -> TurnSignals {
        TurnSignals {
            turn_number: 2,
            last_tool_errored: false,
            last_response_tool_only: false,
            user_message_tokens: 80,
        }
    }

    #[tokio::test]
    async fn selects_smart_when_llm_says_smart() {
        let provider = Arc::new(MockProvider::new(vec![MockResponse::text("smart")]));
        let router = PromptRouter::new(provider, "cheap-model".into(), 3);
        let decision =
            router.select(&signals(), &candidates(), &ObjectiveWeights::default()).await;
        assert_eq!(decision.tier, ModelTier::Smart);
        assert_eq!(decision.strategy_name, "prompt_router");
    }

    #[tokio::test]
    async fn falls_back_to_cheap_on_bad_response() {
        let provider = Arc::new(MockProvider::new(vec![MockResponse::text("I dunno")]));
        let router = PromptRouter::new(provider, "cheap-model".into(), 3);
        let decision =
            router.select(&signals(), &candidates(), &ObjectiveWeights::default()).await;
        assert_eq!(decision.tier, ModelTier::Cheap);
    }

    #[tokio::test]
    async fn selects_cheap_when_llm_says_cheap() {
        let provider = Arc::new(MockProvider::new(vec![MockResponse::text("cheap")]));
        let router = PromptRouter::new(provider, "cheap-model".into(), 3);
        let decision =
            router.select(&signals(), &candidates(), &ObjectiveWeights::default()).await;
        assert_eq!(decision.tier, ModelTier::Cheap);
    }
}
