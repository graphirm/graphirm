// Agent configuration: model selection, temperature, tool permissions

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::AgentError;
use crate::knowledge::extraction::ExtractionConfig;

/// Configuration for the embedding provider used by cross-session memory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbeddingConfig {
    /// Backend/model spec, e.g. `"mistral/codestral-embed"` or `"fastembed/nomic-embed-text-v1"`.
    #[serde(rename = "embedding_backend")]
    pub backend: String,
    /// Vector dimension produced by this model. Must match the HNSW index.
    #[serde(rename = "embedding_dim")]
    pub dim: usize,
}

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

/// Configuration for structured LLM response segmentation.
#[derive(Debug, Clone, Deserialize)]
pub struct SegmentConfig {
    /// Whether segmentation is active for this agent.
    #[serde(default)]
    pub enabled: bool,
    /// Segment type labels to detect (e.g. "code", "reasoning").
    #[serde(default = "default_segment_labels")]
    pub labels: Vec<String>,
    /// If true, append segment format instructions to the system prompt and expect structured JSON output.
    #[serde(default = "default_structured_output")]
    pub structured_output: bool,
    /// If true, fall back to GLiNER2 ONNX span extraction when structured output parsing fails.
    #[serde(default = "default_gliner2_fallback")]
    pub gliner2_fallback: bool,
    /// Minimum confidence threshold for GLiNER2 spans (0.0–1.0).
    #[serde(default = "default_segment_min_confidence")]
    pub min_confidence: f64,
}

fn default_segment_labels() -> Vec<String> {
    vec![
        "observation".into(),
        "reasoning".into(),
        "code".into(),
        "plan".into(),
        "answer".into(),
    ]
}

fn default_structured_output() -> bool {
    true
}

fn default_gliner2_fallback() -> bool {
    true
}

fn default_segment_min_confidence() -> f64 {
    0.5
}

impl Default for SegmentConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            labels: default_segment_labels(),
            structured_output: default_structured_output(),
            gliner2_fallback: default_gliner2_fallback(),
            min_confidence: default_segment_min_confidence(),
        }
    }
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
    /// Embedding config for cross-session memory. `None` disables memory retrieval.
    #[serde(default, flatten)]
    pub embedding: Option<EmbeddingConfig>,
    /// Turn at which soft escalation checks begin (e.g., turn 8)
    #[serde(default = "default_soft_escalation_turn")]
    pub soft_escalation_turn: u32,
    /// Number of repeated identical tool calls to trigger soft escalation
    #[serde(default = "default_soft_escalation_threshold")]
    pub soft_escalation_threshold: usize,
    /// Segment extraction config. `None` disables response segmentation.
    #[serde(default)]
    pub segments: Option<SegmentConfig>,
    /// When segments are enabled, restrict context window reconstruction to
    /// only these segment types. `None` includes all content (default).
    /// Example: `["reasoning", "code"]`
    #[serde(default)]
    pub segment_filter: Option<Vec<String>>,
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
            model: "deepseek-chat".to_string(),
            description: String::new(),
            system_prompt: concat!(
                "You are Graphirm, a graph-native coding agent. Every message you send and ",
                "receive is stored as a node in a persistent knowledge graph.\n\n",
                "## Tools\n\n",
                "You have access to these tools:\n",
                "- bash       — run shell commands in the working directory\n",
                "- read       — read a file with line numbers\n",
                "- write      — create or overwrite a file\n",
                "- edit       — replace an exact string in a file\n",
                "- grep       — search file contents by regex\n",
                "- find       — find files by name pattern\n",
                "- ls         — list directory contents\n\n",
                "## When to use tools\n\n",
                "ONLY reach for a tool when the task genuinely requires it. Ask yourself: ",
                "\"Does answering this require reading a file, running a command, or touching ",
                "the filesystem?\" If no, answer directly.\n\n",
                "NEVER use bash just to echo or print your answer. If you already know the ",
                "answer, write it directly in your response — never wrap it in `echo` or any ",
                "shell command.\n\n",
                "DO use tools for: reading/editing code, running tests, checking errors, ",
                "searching for a specific symbol, executing commands the user asks for.\n\n",
                "DO NOT use tools for: general questions, explanations, brainstorming, ",
                "or any task that doesn't involve this project's files.\n\n",
                "## How to act\n\n",
                "- Think before acting. State your plan in one sentence, then execute it.\n",
                "- Prefer the minimal number of tool calls needed.\n",
                "- If a task is ambiguous, ask one clarifying question before starting.\n",
                "- If a command fails, diagnose the error before retrying.\n",
            ).to_string(),
            max_turns: 50,
            max_tokens: Some(8192),
            temperature: Some(0.7),
            tools: vec![],
            working_dir: default_working_dir(),
            max_context_messages: None,
            permissions: HashMap::new(),
            extraction: None,
            embedding: None,
            soft_escalation_turn: 8,
            soft_escalation_threshold: 2,
            segments: None,
            segment_filter: None,
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
    #[serde(default, flatten)]
    embedding: Option<EmbeddingConfig>,
    #[serde(default = "default_soft_escalation_turn")]
    soft_escalation_turn: u32,
    #[serde(default = "default_soft_escalation_threshold")]
    soft_escalation_threshold: usize,
    #[serde(default)]
    segments: Option<SegmentConfig>,
    #[serde(default)]
    segment_filter: Option<Vec<String>>,
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
            embedding: file.agent.embedding,
            soft_escalation_turn: file.agent.soft_escalation_turn,
            soft_escalation_threshold: file.agent.soft_escalation_threshold,
            segments: file.agent.segments,
            segment_filter: file.agent.segment_filter,
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
    fn test_embedding_config_default() {
        let config = AgentConfig::default();
        assert!(config.embedding.is_none());
    }

    #[test]
    fn test_embedding_config_deserialize() {
        let toml_str = r#"
            embedding_backend = "mistral/codestral-embed"
            embedding_dim = 1536
        "#;
        let cfg: EmbeddingConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.backend, "mistral/codestral-embed");
        assert_eq!(cfg.dim, 1536);
    }

    #[test]
    fn test_segment_config_defaults() {
        let config = SegmentConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.labels.len(), 5);
        assert!(config.structured_output);
        assert!(config.gliner2_fallback);
        assert!((config.min_confidence - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_segment_config_deserialize() {
        let toml_str = r#"
            enabled = true
            labels = ["code", "answer"]
            structured_output = false
            gliner2_fallback = true
            min_confidence = 0.6
        "#;
        let cfg: SegmentConfig = toml::from_str(toml_str).unwrap();
        assert!(cfg.enabled);
        assert_eq!(cfg.labels, vec!["code", "answer"]);
        assert!(!cfg.structured_output);
        assert!((cfg.min_confidence - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_agent_config_segments_default_is_none() {
        let config = AgentConfig::default();
        assert!(config.segments.is_none());
    }

    #[test]
    fn test_agent_config_segment_filter_default_is_none() {
        let config = AgentConfig::default();
        assert!(config.segment_filter.is_none());
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

    #[test]
    fn test_agent_config_from_toml_with_segments() {
        let toml_str = r#"
            [agent]
            name = "test"
            model = "test-model"
            system_prompt = "test"
            max_turns = 5

            [agent.segments]
            enabled = true
            labels = ["code", "answer"]
            structured_output = true
            gliner2_fallback = false
            min_confidence = 0.7
        "#;
        let config = AgentConfig::from_toml(toml_str).unwrap();
        let seg = config.segments.unwrap();
        assert!(seg.enabled);
        assert_eq!(seg.labels, vec!["code", "answer"]);
        assert!(seg.structured_output);
        assert!(!seg.gliner2_fallback);
        assert!((seg.min_confidence - 0.7).abs() < f64::EPSILON);
    }
}
