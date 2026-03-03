use crate::anthropic::AnthropicProvider;
use crate::error::LlmError;
use crate::openai::OpenAiProvider;
use crate::provider::LlmProvider;

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

/// Create a provider by name and API key.
pub fn create_provider(
    provider_name: &str,
    api_key: &str,
) -> Result<Box<dyn LlmProvider>, LlmError> {
    match provider_name {
        "anthropic" => Ok(Box::new(AnthropicProvider::new(api_key))),
        "openai" => Ok(Box::new(OpenAiProvider::new(api_key))),
        _ => Err(LlmError::invalid_model(format!(
            "Unknown provider: '{provider_name}'. Supported: anthropic, openai"
        ))),
    }
}

/// Parse a `"provider/model"` string and create the corresponding provider.
/// Returns `(provider, model_name)`.
pub fn create_provider_from_model_string(
    model_string: &str,
    api_key: &str,
) -> Result<(Box<dyn LlmProvider>, String), LlmError> {
    let (provider_name, model_name) = parse_model_string(model_string)?;
    let provider = create_provider(provider_name, api_key)?;
    Ok((provider, model_name.to_string()))
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
