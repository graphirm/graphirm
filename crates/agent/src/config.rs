// Agent configuration: model selection, temperature, tool permissions

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::AgentError;
use crate::knowledge::extraction::ExtractionConfig;
use crate::router::ModelRoutingConfig;

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
    /// Maximum output tokens for LLM completions. `None` means no limit.
    /// This is separate from max_tokens which controls context window budget.
    pub max_output_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub tools: Vec<String>,
    /// Working directory for file and shell tools. Defaults to the current
    /// process working directory at the time `AgentConfig::default()` is called.
    #[serde(default = "default_working_dir")]
    pub working_dir: PathBuf,
    /// Root directory under which per-session workspace subdirectories are created.
    /// When `None`, `working_dir` is used directly (no per-session isolation).
    #[serde(default)]
    pub workspaces_root: Option<PathBuf>,
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
    /// Resolved workspace name (set by the server after resolving workspaces_root + name).
    /// Stored here so it can be persisted to the Agent node and restored on restart.
    #[serde(default)]
    pub workspace_name: Option<String>,
    /// Resolved absolute path of the workspace directory.
    /// Only set when the workspace was successfully resolved (root + name joined and directory exists).
    /// Distinct from `working_dir` to avoid false positives when workspace resolution is unavailable.
    #[serde(default)]
    pub workspace_dir: Option<PathBuf>,
    /// When true, destructive tool calls receive a structural impact brief
    /// (dependent files, prior Knowledge notes, risk score) before execution.
    #[serde(default = "default_pre_edit_impact")]
    pub pre_edit_impact: bool,
    /// When true, inject a compact repo briefing into the system prompt at session start.
    #[serde(default = "default_repo_briefing")]
    pub repo_briefing: bool,
    /// Per-turn LLM call timeout in seconds. If the provider doesn't respond
    /// within this window the turn is aborted and the session marked as error.
    #[serde(default = "default_timeout_seconds")]
    pub timeout_seconds: u64,
    /// Two-tier model routing. When `Some`, the router selects between cheap and
    /// smart models per turn. When `None`, `model` is used for every turn.
    #[serde(default)]
    pub model_routing: Option<ModelRoutingConfig>,
    /// Adaptive model routing framework. When `Some`, replaces the static `model_routing` path
    /// with a strategy-based selection (rules, prompt classifier, or A/B experiment).
    #[serde(default)]
    pub adaptive_routing: Option<AdaptiveRoutingConfig>,
    /// When true, automatically compact old interactions when context usage exceeds
    /// `compaction_threshold` (default 0.80). Disabled by default; enable in `[agent]` config.
    #[serde(default)]
    pub enable_compaction: bool,
    /// Maximum number of auto-continuation turns injected after a text-only response when
    /// the agent has already executed tool calls in this session. Prevents the agent from
    /// stopping mid-task. 0 disables. Default 2.
    #[serde(default = "default_max_continuations")]
    pub max_continuations: u32,
    /// When true, intercept the first text-only turn after tool work and inject a
    /// verification checklist (run tests, check lint, re-read task requirements).
    /// Forces the agent to validate its output before declaring the task complete.
    /// Default true.
    #[serde(default = "default_pre_completion_verify")]
    pub pre_completion_verify: bool,
    /// Number of times the agent may write/edit the same file in one session before
    /// an advisory is injected urging it to step back and reconsider. 0 disables.
    /// Default 5.
    #[serde(default = "default_doom_loop_threshold")]
    pub doom_loop_threshold: u32,
    /// Token budget thresholds at which a warning is appended to the system prompt for
    /// the current turn. Each value is a fraction of `max_tokens` (e.g. 0.7 = 70%).
    /// An empty list disables budget warnings. Default: [0.7, 0.9].
    #[serde(default = "default_budget_warning_thresholds")]
    pub budget_warning_thresholds: Vec<f64>,
    /// When true, inject a Plan→Build→Verify→Fix problem-solving framework into the
    /// system prompt at session start. Guides the agent from planning directly into
    /// execution without excessive discussion. Default true.
    #[serde(default = "default_enforce_work_loop")]
    pub enforce_work_loop: bool,
}

/// Objective weights for composite score optimisation.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct AdaptiveObjectiveConfig {
    pub preset: Option<String>,
    pub cost_weight: Option<f64>,
    pub quality_weight: Option<f64>,
    pub speed_weight: Option<f64>,
}

impl AdaptiveObjectiveConfig {
    pub fn to_weights(&self) -> crate::strategy::ObjectiveWeights {
        if let Some(ref p) = self.preset {
            crate::strategy::ObjectiveWeights::preset(p)
        } else {
            crate::strategy::ObjectiveWeights {
                cost_weight: self.cost_weight.unwrap_or(0.4),
                quality_weight: self.quality_weight.unwrap_or(0.4),
                speed_weight: self.speed_weight.unwrap_or(0.2),
            }
        }
    }
}

/// Configuration for the A/B experiment strategy.
#[derive(Debug, Clone, Deserialize)]
pub struct ExperimentConfig {
    pub strategy_a: String,
    pub strategy_b: String,
    #[serde(default = "default_split")]
    pub split: f64,
}

fn default_split() -> f64 {
    0.5
}

/// Configuration for the prompt-based LLM classifier strategy.
#[derive(Debug, Clone, Deserialize)]
pub struct PromptRouterConfig {
    pub classifier_model: String,
    #[serde(default = "default_classifier_timeout")]
    pub timeout_seconds: u64,
}

fn default_classifier_timeout() -> u64 {
    3
}

/// A single model candidate with pricing metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelCandidateConfig {
    pub model: String,
    pub tier: String, // "cheap" or "smart"
    pub cost_per_1k_input: f64,
    pub cost_per_1k_output: f64,
    pub avg_latency_ms: Option<u64>,
}

/// Top-level adaptive routing configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct AdaptiveRoutingConfig {
    #[serde(default = "default_adaptive_strategy")]
    pub strategy: String,
    pub objective: Option<AdaptiveObjectiveConfig>,
    pub experiment: Option<ExperimentConfig>,
    pub prompt: Option<PromptRouterConfig>,
    #[serde(default)]
    pub candidates: Vec<ModelCandidateConfig>,
}

fn default_adaptive_strategy() -> String {
    "rules".into()
}

fn default_timeout_seconds() -> u64 {
    300
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

fn default_pre_edit_impact() -> bool {
    true
}

fn default_repo_briefing() -> bool {
    true
}

fn default_max_continuations() -> u32 {
    0
}

fn default_pre_completion_verify() -> bool {
    true
}

fn default_doom_loop_threshold() -> u32 {
    5
}

fn default_budget_warning_thresholds() -> Vec<f64> {
    vec![0.7, 0.9]
}

fn default_enforce_work_loop() -> bool {
    true
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
            )
            .to_string(),
            max_turns: 50,
            max_tokens: Some(8192),
            max_output_tokens: None,
            temperature: Some(0.7),
            tools: vec![],
            working_dir: default_working_dir(),
            workspaces_root: None,
            max_context_messages: None,
            permissions: HashMap::new(),
            extraction: None,
            embedding: None,
            soft_escalation_turn: 8,
            soft_escalation_threshold: 2,
            segments: None,
            segment_filter: None,
            workspace_name: None,
            workspace_dir: None,
            pre_edit_impact: true,
            repo_briefing: true,
            timeout_seconds: default_timeout_seconds(),
            model_routing: None,
            adaptive_routing: None,
            enable_compaction: false,
            max_continuations: default_max_continuations(),
            pre_completion_verify: true,
            doom_loop_threshold: default_doom_loop_threshold(),
            budget_warning_thresholds: default_budget_warning_thresholds(),
            enforce_work_loop: default_enforce_work_loop(),
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
    max_output_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    tools: Vec<String>,
    #[serde(default = "default_working_dir")]
    working_dir: PathBuf,
    #[serde(default)]
    workspaces_root: Option<PathBuf>,
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
    // Set at runtime by the server after resolving workspaces_root; not read from TOML.
    #[serde(default)]
    workspace_name: Option<String>,
    #[serde(default = "default_pre_edit_impact")]
    pre_edit_impact: bool,
    #[serde(default = "default_repo_briefing")]
    repo_briefing: bool,
    #[serde(default = "default_timeout_seconds")]
    timeout_seconds: u64,
    #[serde(default)]
    routing: Option<ModelRoutingConfig>,
    #[serde(default)]
    adaptive_routing: Option<AdaptiveRoutingConfig>,
    #[serde(default)]
    enable_compaction: bool,
    #[serde(default = "default_max_continuations")]
    max_continuations: u32,
    #[serde(default = "default_pre_completion_verify")]
    pre_completion_verify: bool,
    #[serde(default = "default_doom_loop_threshold")]
    doom_loop_threshold: u32,
    #[serde(default = "default_budget_warning_thresholds")]
    budget_warning_thresholds: Vec<f64>,
    #[serde(default = "default_enforce_work_loop")]
    enforce_work_loop: bool,
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
            max_output_tokens: file.agent.max_output_tokens,
            temperature: file.agent.temperature,
            tools: file.agent.tools,
            working_dir: file.agent.working_dir,
            workspaces_root: file.agent.workspaces_root,
            max_context_messages: file.agent.max_context_messages,
            permissions: file.permissions,
            extraction: file.agent.extraction,
            embedding: file.agent.embedding,
            soft_escalation_turn: file.agent.soft_escalation_turn,
            soft_escalation_threshold: file.agent.soft_escalation_threshold,
            segments: file.agent.segments,
            segment_filter: file.agent.segment_filter,
            workspace_name: file.agent.workspace_name,
            workspace_dir: None,
            pre_edit_impact: file.agent.pre_edit_impact,
            repo_briefing: file.agent.repo_briefing,
            timeout_seconds: file.agent.timeout_seconds,
            model_routing: file.agent.routing,
            adaptive_routing: file.agent.adaptive_routing,
            enable_compaction: file.agent.enable_compaction,
            max_continuations: file.agent.max_continuations,
            pre_completion_verify: file.agent.pre_completion_verify,
            doom_loop_threshold: file.agent.doom_loop_threshold,
            budget_warning_thresholds: file.agent.budget_warning_thresholds,
            enforce_work_loop: file.agent.enforce_work_loop,
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
    fn test_pre_completion_verify_default_true() {
        let config = AgentConfig::default();
        assert!(
            config.pre_completion_verify,
            "pre_completion_verify should default to true"
        );
    }

    #[test]
    fn test_pre_completion_verify_toml_parse() {
        let toml_str = r#"
            [agent]
            name = "test"
            model = "gpt-4"
            system_prompt = "test"
            pre_completion_verify = false
        "#;
        let config = AgentConfig::from_toml(toml_str).unwrap();
        assert!(
            !config.pre_completion_verify,
            "pre_completion_verify should read false from TOML"
        );
    }

    #[test]
    fn test_doom_loop_threshold_default() {
        let config = AgentConfig::default();
        assert_eq!(
            config.doom_loop_threshold, 5,
            "doom_loop_threshold should default to 5"
        );
    }

    #[test]
    fn test_doom_loop_threshold_toml_parse() {
        let toml_str = r#"
            [agent]
            name = "test"
            model = "gpt-4"
            system_prompt = "test"
            doom_loop_threshold = 3
        "#;
        let config = AgentConfig::from_toml(toml_str).unwrap();
        assert_eq!(
            config.doom_loop_threshold, 3,
            "doom_loop_threshold should read 3 from TOML"
        );
    }

    #[test]
    fn test_budget_warning_thresholds_default() {
        let config = AgentConfig::default();
        assert_eq!(
            config.budget_warning_thresholds,
            vec![0.7, 0.9],
            "budget_warning_thresholds should default to [0.7, 0.9]"
        );
    }

    #[test]
    fn test_budget_warning_thresholds_toml_parse() {
        let toml_str = r#"
            [agent]
            name = "test"
            model = "gpt-4"
            system_prompt = "test"
            budget_warning_thresholds = [0.5, 0.8]
        "#;
        let config = AgentConfig::from_toml(toml_str).unwrap();
        assert_eq!(
            config.budget_warning_thresholds,
            vec![0.5, 0.8],
            "budget_warning_thresholds should read [0.5, 0.8] from TOML"
        );
    }

    #[test]
    fn test_enforce_work_loop_default_true() {
        let config = AgentConfig::default();
        assert!(
            config.enforce_work_loop,
            "enforce_work_loop should default to true"
        );
    }

    #[test]
    fn test_enforce_work_loop_toml_parse() {
        let toml_str = r#"
            [agent]
            name = "test"
            model = "gpt-4"
            system_prompt = "test"
            enforce_work_loop = false
        "#;
        let config = AgentConfig::from_toml(toml_str).unwrap();
        assert!(
            !config.enforce_work_loop,
            "enforce_work_loop should read false from TOML"
        );
    }

    #[test]
    fn test_agent_config_from_toml_flat() {
        let toml_str = r#"
            name = "test-agent"
            model = "claude-sonnet-4-20250514"
            system_prompt = "You are a coding assistant."
            max_turns = 10
            max_tokens = 4096
            max_output_tokens = 1500
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
        assert_eq!(config.max_output_tokens, Some(1500));
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

    #[test]
    fn workspaces_root_disabled_by_default() {
        let config = AgentConfig::default();
        assert!(config.workspaces_root.is_none());
    }

    #[test]
    fn workspaces_root_parsed_from_toml() {
        let toml = r#"
            name = "test"
            model = "deepseek-chat"
            system_prompt = "hi"
            max_turns = 10
            workspaces_root = "/workspaces"
            tools = ["bash"]
        "#;
        let config: AgentConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.workspaces_root, Some(PathBuf::from("/workspaces")));
    }

    #[test]
    fn test_agent_config_from_toml_segment_filter_roundtrip() {
        let toml = r#"
[agent]
name = "test"
model = "claude-opus-4-5"
system_prompt = "test"
max_turns = 5
segment_filter = ["reasoning", "code"]
"#;
        let config = AgentConfig::from_toml(toml).expect("TOML parse failed");
        assert_eq!(
            config.segment_filter,
            Some(vec!["reasoning".to_string(), "code".to_string()])
        );
    }

    #[test]
    fn pre_edit_impact_defaults_to_true() {
        let config = AgentConfig::default();
        assert!(config.pre_edit_impact);
    }

    #[test]
    fn pre_edit_impact_can_be_disabled() {
        let toml = r#"
            [agent]
            name = "test"
            model = "test"
            system_prompt = "test"
            max_turns = 5
            pre_edit_impact = false
        "#;
        let config = AgentConfig::from_toml(toml).unwrap();
        assert!(!config.pre_edit_impact);
    }

    #[test]
    fn repo_briefing_defaults_to_true() {
        let config = AgentConfig::default();
        assert!(config.repo_briefing);
    }

    #[test]
    fn repo_briefing_can_be_disabled() {
        let toml = r#"
            [agent]
            name = "test"
            model = "test"
            system_prompt = "test"
            max_turns = 5
            repo_briefing = false
        "#;
        let config = AgentConfig::from_toml(toml).unwrap();
        assert!(!config.repo_briefing);
    }

    #[test]
    fn timeout_seconds_defaults_to_300() {
        let config = AgentConfig::default();
        assert_eq!(config.timeout_seconds, 300);
    }

    #[test]
    fn timeout_seconds_parsed_from_toml() {
        let toml = r#"
            [agent]
            name = "test"
            model = "test"
            system_prompt = "test"
            max_turns = 5
            timeout_seconds = 60
        "#;
        let config = AgentConfig::from_toml(toml).unwrap();
        assert_eq!(config.timeout_seconds, 60);
    }

    #[test]
    fn model_routing_parsed_from_toml() {
        let toml = r#"
            [agent]
            name = "test"
            model = "fallback-model"
            system_prompt = "test"
            max_turns = 5

            [agent.routing]
            cheap = "deepseek/deepseek-chat"
            smart = "anthropic/claude-sonnet-4"
            default_tier = "cheap"

            [[agent.routing.rules]]
            type = "first_turn"
            tier = "smart"
        "#;
        let config = AgentConfig::from_toml(toml).unwrap();
        let routing = config.model_routing.unwrap();
        assert_eq!(routing.cheap, "deepseek/deepseek-chat");
        assert_eq!(routing.smart, "anthropic/claude-sonnet-4");
        assert_eq!(routing.rules.len(), 1);
    }

    #[test]
    fn model_routing_absent_by_default() {
        let config = AgentConfig::default();
        assert!(config.model_routing.is_none());
    }

    #[test]
    fn adaptive_routing_config_parses() {
        let toml = r#"
            [agent]
            name = "test"
            model = "fallback"
            system_prompt = "test"
            max_turns = 5

            [agent.adaptive_routing]
            strategy = "experiment"

            [agent.adaptive_routing.objective]
            preset = "cost_focused"

            [agent.adaptive_routing.experiment]
            strategy_a = "rules"
            strategy_b = "prompt"
            split = 0.6

            [agent.adaptive_routing.prompt]
            classifier_model = "deepseek/deepseek-chat"
        "#;
        let config = AgentConfig::from_toml(toml).unwrap();
        let ar = config.adaptive_routing.unwrap();
        assert_eq!(ar.strategy, "experiment");
        assert_eq!(ar.objective.as_ref().unwrap().preset.as_deref(), Some("cost_focused"));
        assert!((ar.experiment.as_ref().unwrap().split - 0.6).abs() < 1e-9);
    }
}
