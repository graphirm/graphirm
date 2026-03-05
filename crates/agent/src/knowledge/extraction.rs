//! Post-turn knowledge extraction from conversations.

use serde::{Deserialize, Serialize};

fn default_entity_types() -> Vec<String> {
    vec![
        "function".into(),
        "api".into(),
        "pattern".into(),
        "decision".into(),
        "bug".into(),
        "architecture".into(),
        "convention".into(),
        "library".into(),
    ]
}

fn default_model() -> String {
    "gpt-4o-mini".into()
}

fn default_min_confidence() -> f64 {
    0.7
}

/// Configuration for post-turn knowledge extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    /// Whether knowledge extraction is enabled.
    #[serde(default)]
    pub enabled: bool,

    /// LLM model to use for extraction.
    #[serde(default = "default_model")]
    pub model: String,

    /// Entity types to extract (e.g. function, api, pattern).
    #[serde(default = "default_entity_types")]
    pub entity_types: Vec<String>,

    /// Minimum confidence score [0.0, 1.0] for a node to be persisted.
    #[serde(default = "default_min_confidence")]
    pub min_confidence: f64,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: default_model(),
            entity_types: default_entity_types(),
            min_confidence: default_min_confidence(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extraction_config_defaults() {
        let config = ExtractionConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.model, "gpt-4o-mini");
        assert_eq!(config.min_confidence, 0.7);
        assert!(!config.entity_types.is_empty());
    }

    #[test]
    fn test_extraction_config_deserialize_partial() {
        let toml_str = r#"
            enabled = true
            model = "claude-3-haiku"
        "#;
        let config: ExtractionConfig = toml::from_str(toml_str).unwrap();
        assert!(config.enabled);
        assert_eq!(config.model, "claude-3-haiku");
        assert_eq!(config.min_confidence, 0.7); // default
        assert!(!config.entity_types.is_empty()); // default
    }

    #[test]
    fn test_extraction_config_serialize_roundtrip() {
        let config = ExtractionConfig {
            enabled: true,
            model: "deepseek-chat".to_string(),
            entity_types: vec!["function".to_string(), "api".to_string()],
            min_confidence: 0.85,
        };
        let json = serde_json::to_string(&config).unwrap();
        let back: ExtractionConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.enabled, true);
        assert_eq!(back.model, "deepseek-chat");
        assert_eq!(back.entity_types.len(), 2);
        assert!((back.min_confidence - 0.85).abs() < f64::EPSILON);
    }
}
