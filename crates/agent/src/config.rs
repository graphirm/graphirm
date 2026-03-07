// Agent configuration: model selection, temperature, tool permissions

use std::collections::HashMap;
use std::path::PathBuf;

use serde::Deserialize;

use crate::error::AgentError;
use crate::knowledge::extraction::ExtractionConfig;

/// Whether an agent operates as the primary coordinator or a spawned subagent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AgentMode {
    #[default]
    Primary,
    Subagent,
}

/// Whether a specific tool is explicitly allowed or denied for an agent.
/// Tools not listed in permissions default to allowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Permission {
    Allow,
    Deny,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AgentConfig {
    pub name: String,
    #[serde(default)]
    pub mode: AgentMode,
    pub model: String,
    #[serde(default)]
    pub description: String,
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
    /// window. `None` means no cap. Set to guard against unbounded context growth.
    pub max_context_messages: Option<usize>,
    /// Per-tool permissions. Tools not listed default to allowed.
    #[serde(default)]
    pub permissions: HashMap<String, Permission>,
    /// Knowledge extraction config. `None` disables post-turn extraction.
    #[serde(default)]
    pub extraction: Option<ExtractionConfig>,
    /// Turn at which soft escalation checks begin (e.g., turn 8)
    #[serde(default = "default_soft_escalation_turn")]
    pub soft_escalation_turn: u32,
    /// Number of repeated identical tool calls to trigger soft escalation
    #[serde(default = "default_soft_escalation_threshold")]
    pub soft_escalation_threshold: usize,
}

fn default_working_dir() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn default_soft_escalation_turn() -> u32 {
    8
}

fn default_soft_escalation_threshold() -> usize {
    2
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: "graphirm".to_string(),
            mode: AgentMode::Primary,
            model: "claude-sonnet-4-20250514".to_string(),
            description: String::new(),
            system_prompt: "You are a helpful coding assistant.".to_string(),
            max_turns: 50,
            max_tokens: Some(8192),
            temperature: Some(0.7),
            tools: vec![],
            working_dir: default_working_dir(),
            max_context_messages: None,
            permissions: HashMap::new(),
            extraction: None,
            soft_escalation_turn: 8,
            soft_escalation_threshold: 2,
        }
    }
}

/// TOML file layout: `[agent]` section + optional `[permissions]` section.
/// This is the multi-agent config format; flat deserialization still works
/// for legacy single-agent TOML via `toml::from_str::<AgentConfig>()`.
#[derive(Debug, Deserialize)]
struct AgentConfigFile {
    agent: AgentConfigSection,
    #[serde(default)]
    permissions: HashMap<String, Permission>,
}

#[derive(Debug, Deserialize)]
struct AgentConfigSection {
    name: String,
    #[serde(default)]
    mode: AgentMode,
    model: String,
    #[serde(default)]
    description: String,
    #[serde(default = "default_system_prompt")]
    system_prompt: String,
    #[serde(default = "default_max_turns")]
    max_turns: u32,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    tools: Vec<String>,
    #[serde(default = "default_working_dir")]
    working_dir: PathBuf,
    #[serde(default)]
    max_context_messages: Option<usize>,
    #[serde(default)]
    extraction: Option<ExtractionConfig>,
    #[serde(default = "default_soft_escalation_turn")]
    soft_escalation_turn: u32,
    #[serde(default = "default_soft_escalation_threshold")]
    soft_escalation_threshold: usize,
}

fn default_system_prompt() -> String {
    "You are a helpful coding assistant.".to_string()
}

fn default_max_turns() -> u32 {
    50
}

impl AgentConfig {
    /// Parse an AgentConfig from TOML with `[agent]` + optional `[permissions]` sections.
    pub fn from_toml(toml_str: &str) -> Result<Self, AgentError> {
        let file: AgentConfigFile =
            toml::from_str(toml_str).map_err(|e| AgentError::Workflow(e.to_string()))?;

        Ok(Self {
            name: file.agent.name,
            mode: file.agent.mode,
            model: file.agent.model,
            description: file.agent.description,
            system_prompt: file.agent.system_prompt,
            max_turns: file.agent.max_turns,
            max_tokens: file.agent.max_tokens,
            temperature: file.agent.temperature,
            tools: file.agent.tools,
            working_dir: file.agent.working_dir,
            max_context_messages: file.agent.max_context_messages,
            permissions: file.permissions,
            extraction: file.agent.extraction,
            soft_escalation_turn: file.agent.soft_escalation_turn,
            soft_escalation_threshold: file.agent.soft_escalation_threshold,
        })
    }

    /// Load an AgentConfig from a TOML file path using the sectioned format.
    pub fn from_file(path: &std::path::Path) -> Result<Self, AgentError> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            AgentError::Workflow(format!("Failed to read {}: {}", path.display(), e))
        })?;
        Self::from_toml(&content)
    }

    /// Check whether a named tool is allowed by this config's permissions.
    /// Default: allow (tools not listed in permissions are permitted).
    pub fn is_tool_allowed(&self, tool_name: &str) -> bool {
        !matches!(self.permissions.get(tool_name), Some(Permission::Deny))
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
    fn test_agent_config_from_toml_flat() {
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
    fn test_agent_mode_deserialize() {
        #[derive(Deserialize)]
        struct W {
            v: AgentMode,
        }
        let primary: W = toml::from_str("v = \"primary\"").unwrap();
        assert_eq!(primary.v, AgentMode::Primary);
        let sub: W = toml::from_str("v = \"subagent\"").unwrap();
        assert_eq!(sub.v, AgentMode::Subagent);
    }

    #[test]
    fn test_permission_deserialize() {
        #[derive(Deserialize)]
        struct W {
            v: Permission,
        }
        let allow: W = toml::from_str("v = \"allow\"").unwrap();
        assert_eq!(allow.v, Permission::Allow);
        let deny: W = toml::from_str("v = \"deny\"").unwrap();
        assert_eq!(deny.v, Permission::Deny);
    }

    #[test]
    fn test_agent_config_from_toml_with_sections() {
        let toml_str = r#"
            [agent]
            name = "build"
            mode = "primary"
            model = "anthropic/claude-sonnet-4"
            description = "Default agent with full tool access"
            system_prompt = "You are a coding assistant."
            max_turns = 50
            tools = ["bash", "read", "write", "edit"]

            [permissions]
            bash = "allow"
            write = "allow"
            edit = "allow"
        "#;
        let config = AgentConfig::from_toml(toml_str).unwrap();
        assert_eq!(config.name, "build");
        assert_eq!(config.mode, AgentMode::Primary);
        assert_eq!(config.model, "anthropic/claude-sonnet-4");
        assert_eq!(config.description, "Default agent with full tool access");
        assert_eq!(config.permissions.get("bash"), Some(&Permission::Allow));
        assert_eq!(config.permissions.get("write"), Some(&Permission::Allow));
    }

    #[test]
    fn test_subagent_config_from_toml() {
        let toml_str = r#"
            [agent]
            name = "explore"
            mode = "subagent"
            model = "anthropic/claude-haiku-4"
            description = "Fast, read-only codebase exploration"
            system_prompt = "You explore code. Read files and report findings."
            max_turns = 10
            tools = ["read", "grep", "find", "ls"]

            [permissions]
            bash = "deny"
            write = "deny"
            edit = "deny"
        "#;
        let config = AgentConfig::from_toml(toml_str).unwrap();
        assert_eq!(config.name, "explore");
        assert_eq!(config.mode, AgentMode::Subagent);
        assert_eq!(config.max_turns, 10);
        assert_eq!(config.permissions.get("bash"), Some(&Permission::Deny));
        assert_eq!(config.permissions.get("write"), Some(&Permission::Deny));
        assert_eq!(config.permissions.get("edit"), Some(&Permission::Deny));
        assert!(!config.permissions.contains_key("read"));
    }

    #[test]
    fn test_agent_config_default_still_works() {
        let config = AgentConfig::default();
        assert_eq!(config.mode, AgentMode::Primary);
        assert!(config.permissions.is_empty());
        assert_eq!(config.description, "");
    }

    #[test]
    fn test_is_tool_allowed() {
        let toml_str = r#"
            [agent]
            name = "explore"
            mode = "subagent"
            model = "test"
            system_prompt = "test"
            max_turns = 5

            [permissions]
            bash = "deny"
            write = "deny"
        "#;
        let config = AgentConfig::from_toml(toml_str).unwrap();
        assert!(!config.is_tool_allowed("bash"));
        assert!(!config.is_tool_allowed("write"));
        assert!(config.is_tool_allowed("read")); // not listed → allowed
        assert!(config.is_tool_allowed("grep")); // not listed → allowed
    }

    #[test]
    fn test_agent_config_from_toml_minimal_sectioned() {
        let toml_str = r#"
            [agent]
            name = "minimal"
            model = "gpt-4o"
            system_prompt = "Help."
            max_turns = 5
            tools = []
        "#;
        let config = AgentConfig::from_toml(toml_str).unwrap();
        assert_eq!(config.name, "minimal");
        assert_eq!(config.mode, AgentMode::Primary); // default
        assert!(config.permissions.is_empty()); // no [permissions] section
    }
}
