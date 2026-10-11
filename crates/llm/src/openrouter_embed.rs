//! OpenRouter embeddings. OpenAI-compatible `POST /api/v1/embeddings`.
//!
//! `mistralai/codestral-embed-2505` returns 1536-dimensional f32 vectors
//! (one live call, 2026-10-11).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::LlmError;
use crate::provider::EmbeddingProvider;

const EMBEDDINGS_URL: &str = "https://openrouter.ai/api/v1/embeddings";
const CODESTRAL_EMBED_2505: &str = "mistralai/codestral-embed-2505";
/// Measured 2026-10-11: one embeddings response, `len(data[0].embedding) == 1536`.
const CODESTRAL_EMBED_2505_DIM: usize = 1536;

/// OpenRouter embedding provider.
pub struct OpenRouterEmbeddingProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl OpenRouterEmbeddingProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Result<Self, LlmError> {
        let model = model.into();
        if model != CODESTRAL_EMBED_2505 {
            return Err(LlmError::invalid_model(format!(
                "Unknown OpenRouter embed model '{model}'. Use {CODESTRAL_EMBED_2505}"
            )));
        }
        Ok(Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model,
        })
    }

    pub fn dim(&self) -> usize {
        CODESTRAL_EMBED_2505_DIM
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
impl EmbeddingProvider for OpenRouterEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, LlmError> {
        let body = EmbedRequest {
            model: &self.model,
            input: vec![text],
        };

        let resp = self
            .client
            .post(EMBEDDINGS_URL)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::provider(format!("OpenRouter embed request: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::provider(format!(
                "OpenRouter embed HTTP {status}: {body}"
            )));
        }

        let parsed: EmbedResponse = resp
            .json()
            .await
            .map_err(|e| LlmError::provider(format!("OpenRouter embed parse: {e}")))?;

        let embedding = parsed
            .data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .ok_or_else(|| LlmError::provider("OpenRouter embed returned empty data"))?;

        if embedding.len() != self.dim() {
            return Err(LlmError::provider(format!(
                "OpenRouter embed returned {} dimensions, expected {}",
                embedding.len(),
                self.dim()
            )));
        }

        Ok(embedding)
    }
}
