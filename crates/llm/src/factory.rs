use crate::anthropic::AnthropicProvider;
use crate::deepseek::DeepSeekProvider;
use crate::error::LlmError;
use crate::mistral_embed::{MistralEmbedModel, MistralEmbeddingProvider};
use crate::ollama::OllamaProvider;
use crate::openai::OpenAiProvider;
use crate::provider::{EmbeddingProvider, LlmProvider};

/// Parse a `"provider/model"` string into `(provider, model)`.
pub fn parse_model_string(model_string: &str) -> Result<(&str, &str), LlmError> {
    let parts: Vec<&str> = model_string.splitn(2, '/').collect();
    if parts.len() != 2 {
        return Err(LlmError::invalid_model(format!(
            "Expected 'provider/model' format, got '{model_string}'"
        )));
    }
    if parts[0].is_empty() || parts[1].is_empty() {
        return Err(LlmError::invalid_model(format!(
            "Provider and model must not be empty in '{model_string}'"
        )));
    }
    Ok((parts[0], parts[1]))
}

/// Create a provider by name and optional API key.
///
/// `api_key` is ignored for `"ollama"` (no key needed).
/// For `"ollama"`, set `OLLAMA_HOST` env var to override the default
/// `http://localhost:11434`.
pub fn create_provider(
    provider_name: &str,
    api_key: &str,
) -> Result<Box<dyn LlmProvider>, LlmError> {
    match provider_name {
        "anthropic" => Ok(Box::new(AnthropicProvider::new(api_key))),
        "openai" => Ok(Box::new(OpenAiProvider::new(api_key))),
        "deepseek" => Ok(Box::new(DeepSeekProvider::new(api_key))),
        "ollama" => Ok(Box::new(OllamaProvider::from_env()?)),
        _ => Err(LlmError::invalid_model(format!(
            "Unknown provider: '{provider_name}'. Supported: anthropic, openai, deepseek, ollama"
        ))),
    }
}

/// Parse a `"provider/model"` string and create the corresponding provider.
/// Returns `(provider, model_name)`.
///
/// `api_key` is only used for cloud providers. Pass `""` for `"ollama"`.
pub fn create_provider_from_model_string(
    model_string: &str,
    api_key: &str,
) -> Result<(Box<dyn LlmProvider>, String), LlmError> {
    let (provider_name, model_name) = parse_model_string(model_string)?;
    let provider = create_provider(provider_name, api_key)?;
    Ok((provider, model_name.to_string()))
}

/// Create an embedding provider from a `"backend/model"` string.
///
/// Supported:
/// - `"mistral/mistral-embed"` — 1024-dim, API key from `MISTRAL_API_KEY`
/// - `"mistral/codestral-embed"` — 1024-dim, code-optimised
/// - `"fastembed/nomic-embed-text-v1"` — 768-dim, local ONNX, no key (requires `local-embed` feature)
pub fn create_embedding_provider(
    spec: &str,
    mistral_key: Option<&str>,
) -> Result<(Box<dyn EmbeddingProvider>, usize), LlmError> {
    let (backend, model) = parse_model_string(spec)?;
    match backend {
        "mistral" => {
            let key = mistral_key.unwrap_or("").to_string();
            if key.is_empty() {
                return Err(LlmError::provider("MISTRAL_API_KEY not set"));
            }
            let embed_model = match model {
                "mistral-embed" => MistralEmbedModel::MistralEmbed,
                "codestral-embed" => MistralEmbedModel::CodestralEmbed,
                other => {
                    return Err(LlmError::invalid_model(format!(
                        "Unknown Mistral embed model '{other}'. Use mistral-embed or codestral-embed"
                    )))
                }
            };
            let provider = MistralEmbeddingProvider::new(key, embed_model);
            let dim = provider.dim();
            Ok((Box::new(provider), dim))
        }
        #[cfg(feature = "local-embed")]
        "fastembed" => {
            let provider = crate::fastembed_provider::FastEmbedProvider::new(model)
                .map_err(|e| LlmError::provider(format!("fastembed init: {e}")))?;
            let dim = provider.dim();
            Ok((Box::new(provider), dim))
        }
        #[cfg(not(feature = "local-embed"))]
        "fastembed" => Err(LlmError::provider(
            "fastembed backend requires the 'local-embed' feature flag",
        )),
        other => Err(LlmError::invalid_model(format!(
            "Unknown embedding backend '{other}'. Use mistral or fastembed"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_model_string_valid() {
        let (provider, model) = parse_model_string("anthropic/claude-3-5-sonnet").unwrap();
        assert_eq!(provider, "anthropic");
        assert_eq!(model, "claude-3-5-sonnet");
    }

    #[test]
    fn test_parse_model_string_openai() {
        let (provider, model) = parse_model_string("openai/gpt-4o").unwrap();
        assert_eq!(provider, "openai");
        assert_eq!(model, "gpt-4o");
    }

    #[test]
    fn test_parse_model_string_model_with_slashes() {
        let (provider, model) = parse_model_string("anthropic/claude-3/some-variant").unwrap();
        assert_eq!(provider, "anthropic");
        assert_eq!(model, "claude-3/some-variant");
    }

    #[test]
    fn test_parse_model_string_no_slash() {
        let err = parse_model_string("just-a-model").unwrap_err();
        assert!(err.to_string().contains("provider/model"));
    }

    #[test]
    fn test_parse_model_string_empty_provider() {
        let err = parse_model_string("/gpt-4o").unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn test_parse_model_string_empty_model() {
        let err = parse_model_string("openai/").unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn test_create_provider_anthropic() {
        let provider = create_provider("anthropic", "dummy-key").unwrap();
        assert_eq!(provider.provider_name(), "anthropic");
    }

    #[test]
    fn test_create_provider_openai() {
        let provider = create_provider("openai", "dummy-key").unwrap();
        assert_eq!(provider.provider_name(), "openai");
    }

    #[test]
    fn test_create_provider_deepseek() {
        let provider = create_provider("deepseek", "dummy-key").unwrap();
        assert_eq!(provider.provider_name(), "deepseek");
    }

    #[test]
    fn test_create_provider_ollama() {
        let provider = create_provider("ollama", "").unwrap();
        assert_eq!(provider.provider_name(), "ollama");
    }

    #[test]
    fn test_create_provider_unknown() {
        let result = create_provider("cohere", "dummy-key");
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("Unknown provider"));
        assert!(err.to_string().contains("cohere"));
    }

    #[test]
    fn test_create_provider_from_model_string() {
        let (provider, model) =
            create_provider_from_model_string("anthropic/claude-3-5-sonnet", "dummy-key").unwrap();
        assert_eq!(provider.provider_name(), "anthropic");
        assert_eq!(model, "claude-3-5-sonnet");
    }

    #[test]
    fn test_create_provider_from_model_string_invalid() {
        let result = create_provider_from_model_string("bad-format", "key");
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("provider/model"));
    }

    #[test]
    fn test_create_provider_from_model_string_unknown_provider() {
        let result = create_provider_from_model_string("groq/llama-3.1", "dummy-key");
        assert!(result.is_err());
        assert!(
            result
                .err()
                .unwrap()
                .to_string()
                .contains("Unknown provider")
        );
    }
}
