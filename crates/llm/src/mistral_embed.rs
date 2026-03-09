//! Mistral AI embedding provider — `mistral-embed` and `codestral-embed`.
//!
//! Both models produce 1024-dimensional f32 vectors.
//! API docs: https://docs.mistral.ai/api/#tag/embeddings

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::LlmError;
use crate::provider::EmbeddingProvider;

/// Which Mistral embedding model to use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MistralEmbedModel {
    /// General-purpose text embeddings (1024-dim).
    MistralEmbed,
    /// Code-optimized embeddings (1024-dim).
    CodestralEmbed,
}

impl MistralEmbedModel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::MistralEmbed => "mistral-embed",
            Self::CodestralEmbed => "codestral-embed",
        }
    }
}

/// Mistral embedding provider using the `/v1/embeddings` REST endpoint.
pub struct MistralEmbeddingProvider {
    client: reqwest::Client,
    api_key: String,
    pub model: MistralEmbedModel,
}

impl MistralEmbeddingProvider {
    pub fn new(api_key: impl Into<String>, model: MistralEmbedModel) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model,
        }
    }

    /// Embedding dimension for the configured model.
    ///
    /// - `mistral-embed`: 1024 dimensions
    /// - `codestral-embed`: 1536 dimensions (not 1024 — verified by benchmark 2026-03-09)
    pub fn dim(&self) -> usize {
        match self.model {
            MistralEmbedModel::MistralEmbed => 1024,
            MistralEmbedModel::CodestralEmbed => 1536,
        }
    }
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

#[derive(Deserialize)]
struct EmbedData {
    embedding: Vec<f32>,
}

#[async_trait]
impl EmbeddingProvider for MistralEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, LlmError> {
        let body = EmbedRequest {
            model: self.model.as_str(),
            input: vec![text],
        };

        let resp = self
            .client
            .post("https://api.mistral.ai/v1/embeddings")
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::provider(format!("Mistral embed request: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::provider(format!(
                "Mistral embed HTTP {status}: {body}"
            )));
        }

        let parsed: EmbedResponse = resp
            .json()
            .await
            .map_err(|e| LlmError::provider(format!("Mistral embed parse: {e}")))?;

        parsed
            .data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .ok_or_else(|| LlmError::provider("Mistral embed returned empty data"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mistral_embedding_provider_compile() {
        fn assert_provider<T: crate::provider::EmbeddingProvider + Send + Sync>() {}
        assert_provider::<MistralEmbeddingProvider>();
    }

    #[test]
    fn test_mistral_embed_model_names() {
        assert_eq!(MistralEmbedModel::MistralEmbed.as_str(), "mistral-embed");
        assert_eq!(MistralEmbedModel::CodestralEmbed.as_str(), "codestral-embed");
    }

    #[test]
    fn test_mistral_embedding_provider_new() {
        let p = MistralEmbeddingProvider::new("dummy-key", MistralEmbedModel::MistralEmbed);
        assert_eq!(p.model.as_str(), "mistral-embed");
        assert_eq!(p.dim(), 1024);

        let p2 = MistralEmbeddingProvider::new("dummy-key", MistralEmbedModel::CodestralEmbed);
        assert_eq!(p2.model.as_str(), "codestral-embed");
        assert_eq!(p2.dim(), 1536);
    }
}
