// Agent configuration: model selection, temperature, tool permissions

use std::path::PathBuf;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct AgentConfig {
    pub name: String,
    pub model: String,
    pub system_prompt: String,
    pub max_turns: u32,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub tools: Vec<String>,
    /// Working directory for file and shell tools. Defaults to the current
    /// process working directory at the time `AgentConfig::default()` is called.
    #[serde(default = "default_working_dir")]
    pub working_dir: PathBuf,
    /// Maximum number of interaction messages included in each LLM context
    /// window. `None` means no cap (all interactions are included). Set this
    /// to guard against unbounded context growth until Phase 6 compaction lands.
    pub max_context_messages: Option<usize>,
}

fn default_working_dir() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: "graphirm".to_string(),
            model: "claude-sonnet-4-20250514".to_string(),
            system_prompt: "You are a helpful coding assistant.".to_string(),
            max_turns: 50,
            max_tokens: Some(8192),
            temperature: Some(0.7),
            tools: vec![],
            working_dir: default_working_dir(),
            max_context_messages: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_config_defaults() {
        let config = AgentConfig::default();
        assert_eq!(config.name, "graphirm");
        assert_eq!(config.max_turns, 50);
        assert!(config.tools.is_empty());
        assert_eq!(config.max_context_messages, None);
    }

    #[test]
    fn test_agent_config_from_toml() {
        let toml_str = r#"
            name = "test-agent"
            model = "claude-sonnet-4-20250514"
            system_prompt = "You are a coding assistant."
            max_turns = 10
            max_tokens = 4096
            temperature = 0.5
            tools = ["bash", "read", "write"]
            working_dir = "/tmp/project"
            max_context_messages = 20
        "#;
        let config: AgentConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.name, "test-agent");
        assert_eq!(config.model, "claude-sonnet-4-20250514");
        assert_eq!(config.max_turns, 10);
        assert_eq!(config.max_tokens, Some(4096));
        assert_eq!(config.temperature, Some(0.5));
        assert_eq!(config.tools, vec!["bash", "read", "write"]);
        assert_eq!(config.working_dir, PathBuf::from("/tmp/project"));
        assert_eq!(config.max_context_messages, Some(20));
    }

    #[test]
    fn test_agent_config_from_toml_minimal() {
        let toml_str = r#"
            name = "minimal"
            model = "gpt-4o"
            system_prompt = "Help."
            max_turns = 5
            tools = []
        "#;
        let config: AgentConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.name, "minimal");
        assert_eq!(config.max_tokens, None);
        assert_eq!(config.temperature, None);
        assert_eq!(config.max_context_messages, None);
    }
}
