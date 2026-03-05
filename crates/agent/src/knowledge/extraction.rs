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

/// A knowledge entity extracted from a conversation turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    pub entity_type: String,
    pub name: String,
    pub description: String,
    pub confidence: f64,
    pub relationships: Vec<EntityRelationship>,
}

/// A directed relationship from an extracted entity to a named target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRelationship {
    pub target_name: String,
    pub relationship: String,
}

/// Top-level wrapper for the JSON object returned by the extraction LLM call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResponse {
    pub entities: Vec<ExtractedEntity>,
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
        assert_eq!(config.entity_types.len(), 8);
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
        assert_eq!(config.entity_types.len(), 8); // default
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
        assert!(back.enabled);
        assert_eq!(back.model, "deepseek-chat");
        assert_eq!(back.entity_types.len(), 2);
        assert!((back.min_confidence - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_extracted_entity_roundtrip() {
        let entity = ExtractedEntity {
            entity_type: "function".to_string(),
            name: "parse_config".to_string(),
            description: "Parses TOML configuration files and returns a Config struct".to_string(),
            confidence: 0.92,
            relationships: vec![
                EntityRelationship {
                    target_name: "Config".to_string(),
                    relationship: "returns".to_string(),
                },
                EntityRelationship {
                    target_name: "toml".to_string(),
                    relationship: "uses".to_string(),
                },
            ],
        };
        let json = serde_json::to_string(&entity).unwrap();
        let back: ExtractedEntity = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "parse_config");
        assert_eq!(back.entity_type, "function");
        assert_eq!(back.relationships.len(), 2);
        assert_eq!(back.relationships[0].target_name, "Config");
        assert!((back.confidence - 0.92).abs() < f64::EPSILON);
    }

    #[test]
    fn test_extracted_entity_from_llm_json() {
        let llm_output = r#"{
            "entities": [
                {
                    "entity_type": "pattern",
                    "name": "Repository Pattern",
                    "description": "Data access abstraction using trait objects",
                    "confidence": 0.88,
                    "relationships": [
                        { "target_name": "GraphStore", "relationship": "implements" }
                    ]
                },
                {
                    "entity_type": "decision",
                    "name": "Use rusqlite over sqlitegraph",
                    "description": "Chose MIT-licensed rusqlite to avoid GPL infection",
                    "confidence": 0.95,
                    "relationships": []
                }
            ]
        }"#;
        let parsed: ExtractionResponse = serde_json::from_str(llm_output).unwrap();
        assert_eq!(parsed.entities.len(), 2);
        assert_eq!(parsed.entities[0].name, "Repository Pattern");
        assert_eq!(parsed.entities[1].entity_type, "decision");
    }

    #[test]
    fn test_entity_relationship_display() {
        let rel = EntityRelationship {
            target_name: "tokio".to_string(),
            relationship: "depends_on".to_string(),
        };
        let debug = format!("{:?}", rel);
        assert!(debug.contains("tokio"));
        assert!(debug.contains("depends_on"));
    }
}
