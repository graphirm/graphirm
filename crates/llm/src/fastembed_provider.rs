//! Local ONNX embedding provider using fastembed-rs.
//!
//! Models are downloaded from HuggingFace on first call and cached
//! at `~/.cache/huggingface/hub/`. No API key required.

use std::sync::Arc;

use async_trait::async_trait;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use tokio::sync::Mutex;

use crate::error::LlmError;
use crate::provider::EmbeddingProvider;

/// Local ONNX embedding provider via fastembed-rs.
///
/// Uses `spawn_blocking` internally so the async interface is maintained
/// while the synchronous fastembed inference runs on a blocking thread.
pub struct FastEmbedProvider {
    model: Arc<Mutex<TextEmbedding>>,
    embedding_dim: usize,
    model_name: String,
}

impl FastEmbedProvider {
    /// Create a provider for the given model name.
    ///
    /// Supported model names:
    /// - `"nomic-embed-text-v1"` — 768-dim, ~270 MB, top MTEB (recommended)
    /// - `"bge-small-en-v1.5"` — 384-dim, ~125 MB, fastest
    /// - `"bge-base-en-v1.5"` — 768-dim, ~435 MB
    /// - `"bge-large-en-v1.5"` — 1024-dim, ~1.3 GB
    ///
    /// Model files are downloaded on first call and cached in
    /// `~/.cache/huggingface/hub/`.
    pub fn new(model_name: &str) -> Result<Self, String> {
        let (embed_model, dim) = match model_name {
            "nomic-embed-text-v1" => (EmbeddingModel::NomicEmbedTextV1, 768),
            "bge-small-en-v1.5" => (EmbeddingModel::BGESmallENV15, 384),
            "bge-base-en-v1.5" => (EmbeddingModel::BGEBaseENV15, 768),
            "bge-large-en-v1.5" => (EmbeddingModel::BGELargeENV15, 1024),
            other => return Err(format!("Unknown fastembed model '{other}'")),
        };

        let te = TextEmbedding::try_new(InitOptions::new(embed_model))
            .map_err(|e| format!("fastembed init: {e}"))?;

        Ok(Self {
            model: Arc::new(Mutex::new(te)),
            embedding_dim: dim,
            model_name: model_name.to_string(),
        })
    }

    pub fn dim(&self) -> usize {
        self.embedding_dim
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[async_trait]
impl EmbeddingProvider for FastEmbedProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, LlmError> {
        let text = text.to_string();
        let model = self.model.clone();

        tokio::task::spawn_blocking(move || {
            let mut guard = model.blocking_lock();
            let embeddings = guard
                .embed(vec![text], None)
                .map_err(|e| LlmError::provider(format!("fastembed embed: {e}")))?;
            embeddings
                .into_iter()
                .next()
                .ok_or_else(|| LlmError::provider("fastembed returned no embeddings"))
        })
        .await
        .map_err(|e| LlmError::provider(format!("spawn_blocking panicked: {e}")))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fastembed_provider_compile() {
        fn assert_provider<T: crate::provider::EmbeddingProvider + Send + Sync>() {}
        assert_provider::<FastEmbedProvider>();
    }

    #[test]
    fn test_fastembed_model_name() {
        let result = FastEmbedProvider::new("nomic-embed-text-v1");
        assert!(result.is_ok());
        let p = result.unwrap();
        assert_eq!(p.dim(), 768);
    }
}
