# Phase 5: Multi-Agent System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a hand-rolled multi-agent coordinator that routes tasks to specialized subagents, each running their own graph-scoped agent loop with parallel execution via tokio::spawn.

**Architecture:** A `Coordinator` struct orchestrates the primary agent and its subagents. An `AgentRegistry` loads typed agent configs (Primary/Subagent) from TOML files. The primary agent gets a `delegate` tool that spawns subagents — each subagent runs `run_agent_loop()` from Phase 4 inside a `tokio::spawn` task with a scoped ToolRegistry (filtered by permissions) and scoped context (task description + referenced nodes only). Subagents write to the shared `Arc<GraphStore>`; the parent reads their output nodes via graph traversal after they complete. Multiple delegate tool calls in a single LLM turn execute in parallel via Phase 4's JoinSet.

**Tech Stack:** tokio (spawn + JoinHandle), tokio-util (CancellationToken), serde + toml (config deserialization), graphirm-graph (Phase 1), graphirm-llm (Phase 2), graphirm-tools (Phase 3), graphirm-agent (Phase 4)

**Status:** ✅ Complete — branch `phase-5/multi-agent`, commit `ead1962`.

---

## Post-Implementation Spec Review Findings (2026-03-05)

Spec review identified and resolved the following issues before merge:

**Critical (fixed):**
- `wait_for_dependencies` polled indefinitely if an upstream task reached `Failed` status — no timeout, no cancel check. Fixed: added `cancel: &CancellationToken` + `timeout: Duration` parameters; match on `TaskStatus::Failed` returns early; `tokio::select!` handles cancellation and timeout.
- `SubagentTool` created a disconnected `CancellationToken` per delegation — parent cancellation did not propagate to subagents. Fixed: `SubagentTool` stores the parent token and passes `self.cancel.child_token()` to each `spawn_subagent` call.

**Important (fixed):**
- `Coordinator::run_primary` used `self.tools` directly without the `delegate` tool, so a `Coordinator` could not actually orchestrate multi-agent work. Fixed: `Coordinator` moved to `coordinator.rs` (avoids `multi` ↔ `delegate` circular import); `run_primary` now constructs a primary tool registry by cloning base tools and auto-injecting `SubagentTool`.
- `wait_for_subagents` pushed failed subagents into `results` alongside successful ones; the guard `results.is_empty()` was never true for application failures. Fixed: failed subagents are not pushed to `results`; any error is surfaced to the caller.
- `TaskData.status: String` made status comparisons fragile (typos cause silent failures). Fixed: `TaskStatus` enum (`Pending`, `Running`, `Completed`, `Failed`) added to `graphirm_graph::nodes`; all string status comparisons replaced.
- `AgentRegistry::from_configs` accepted any number of primary agents silently. Fixed: returns `Result<Self, AgentError>` and rejects registries with more than one primary agent.

**New tests added by review:**
- `test_registry_rejects_multiple_primaries`
- `test_coordinator_injects_delegate_tool`
- `test_subagent_tool_cancel_propagation`
- `test_coordinator_does_not_require_manual_delegate_wiring`
- Integration test updated to use `Coordinator::run_primary` end-to-end (previously routed around it)

---

## Prerequisites (APIs from Phases 1–4)

Phase 5 builds directly on types defined in earlier phases. The code below references these APIs. If signatures differ after earlier phases land, adapt accordingly — the logic and test structure remain the same.

### From `graphirm-graph` (Phase 1)

```rust
pub struct GraphStore { /* r2d2 pool + Arc<RwLock<petgraph::Graph>> */ }

pub struct NodeId(pub String);
pub struct EdgeId(pub String);

pub struct GraphNode {
    pub id: NodeId,
    pub node_type: NodeType,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: serde_json::Value,
}

// NodeType is a tagged enum with data structs
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum NodeType {
    Interaction(InteractionData),
    Agent(AgentData),
    Content(ContentData),
    Task(TaskData),
    Knowledge(KnowledgeData),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeType {
    RespondsTo, SpawnedBy, DelegatesTo, DependsOn,
    Produces, Reads, Modifies, Summarizes,
    Contains, FollowsUp, Steers, RelatesTo,
}

impl GraphStore {
    pub fn open_memory() -> Result<Self, GraphError>;
    pub fn add_node(&self, node: GraphNode) -> Result<NodeId, GraphError>;
    pub fn get_node(&self, id: &NodeId) -> Result<GraphNode, GraphError>;
    pub fn add_edge(&self, edge: GraphEdge) -> Result<EdgeId, GraphError>;
    pub fn neighbors(&self, id: &NodeId, edge_type: Option<EdgeType>, direction: Direction) -> Result<Vec<GraphNode>, GraphError>;
    pub fn traverse(&self, start: &NodeId, edge_types: &[EdgeType], max_depth: usize) -> Result<Vec<GraphNode>, GraphError>;
}
```

### From `graphirm-llm` (Phase 2)

```rust
#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn complete(
        &self,
        messages: Vec<LlmMessage>,
        tools: &[ToolDefinition],
        config: &CompletionConfig,
    ) -> Result<LlmResponse, LlmError>;

    async fn stream(...) -> Result<Pin<Box<dyn Stream<Item = StreamEvent> + Send>>, LlmError>;
    fn provider_name(&self) -> &str;
}

pub struct MockProvider { /* returns canned responses */ }
pub struct MockResponse { /* text or tool call */ }
```

### From `graphirm-tools` (Phase 3)

```rust
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    async fn execute(&self, args: serde_json::Value) -> Result<String, ToolError>;
}

pub struct ToolRegistry { tools: HashMap<String, Arc<dyn Tool>> }
impl ToolRegistry {
    pub fn new() -> Self;
    pub fn register(&mut self, tool: Arc<dyn Tool>);
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>>;
    pub fn definitions(&self) -> Vec<ToolDefinition>;
    pub fn names(&self) -> Vec<String>;
}
```

### From `graphirm-agent` (Phase 4)

```rust
pub struct AgentConfig {
    pub name: String,
    pub model: String,
    pub system_prompt: String,
    pub max_turns: u32,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub tools: Vec<String>,
}

pub struct Session {
    pub id: NodeId,
    pub agent_config: AgentConfig,
    pub graph: Arc<GraphStore>,
    pub created_at: DateTime<Utc>,
}
impl Session {
    pub fn new(graph: Arc<GraphStore>, config: AgentConfig) -> Result<Self, AgentError>;
    pub fn add_user_message(&self, content: &str) -> Result<NodeId, AgentError>;
}

pub struct EventBus { subscribers: Vec<mpsc::Sender<AgentEvent>> }
impl EventBus {
    pub fn new() -> Self;
    pub fn subscribe(&mut self) -> mpsc::Receiver<AgentEvent>;
    pub async fn emit(&self, event: AgentEvent);
}

pub enum AgentEvent { AgentStart{..}, AgentEnd{..}, TurnStart{..}, TurnEnd{..}, ... }

pub async fn run_agent_loop(
    session: &Session,
    llm: &dyn LlmProvider,
    tools: &ToolRegistry,
    events: &EventBus,
    cancel: &CancellationToken,
) -> Result<(), AgentError>;

pub enum AgentError {
    Graph(..), Llm(..), Tool(..), SessionNotFound(..),
    Workflow(..), Context(..), RecursionLimit(..), Cancelled, Join(..),
}
```

---

## Task 1: Define AgentMode, Permission, and extend AgentConfig

- [x] Complete

**Files:**
- Modify: `crates/agent/src/config.rs`
- Modify: `crates/agent/Cargo.toml` (add `toml` to dependencies)

### Step 1: Add `toml` dependency

Add `toml = "0.8"` to `[dependencies]` in `crates/agent/Cargo.toml` (it's currently only in `[dev-dependencies]` from Phase 4):

```toml
[dependencies]
# ... existing deps from Phase 4 ...
toml = "0.8"
```

### Step 2: Write failing tests

Add to `crates/agent/src/config.rs`, inside the existing `#[cfg(test)] mod tests` block:

```rust
    #[test]
    fn test_agent_mode_deserialize() {
        let primary: AgentMode = toml::from_str("\"primary\"").unwrap();
        assert_eq!(primary, AgentMode::Primary);
        let sub: AgentMode = toml::from_str("\"subagent\"").unwrap();
        assert_eq!(sub, AgentMode::Subagent);
    }

    #[test]
    fn test_permission_deserialize() {
        let allow: Permission = toml::from_str("\"allow\"").unwrap();
        assert_eq!(allow, Permission::Allow);
        let deny: Permission = toml::from_str("\"deny\"").unwrap();
        assert_eq!(deny, Permission::Deny);
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
            system_prompt = "You are a fast code exploration agent. Read files and report findings."
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
        assert!(config.permissions.get("read").is_none());
    }

    #[test]
    fn test_agent_config_default_still_works() {
        let config = AgentConfig::default();
        assert_eq!(config.mode, AgentMode::Primary);
        assert!(config.permissions.is_empty());
        assert_eq!(config.description, "");
    }
```

Run: `cargo test -p graphirm-agent config::tests 2>&1`
Expected: Fails — `AgentMode`, `Permission`, `from_toml` not defined yet.

### Step 3: Implement AgentMode, Permission, and extend AgentConfig

Rewrite `crates/agent/src/config.rs`:

```rust
use std::collections::HashMap;

use serde::Deserialize;

use crate::error::AgentError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentMode {
    Primary,
    Subagent,
}

impl Default for AgentMode {
    fn default() -> Self {
        Self::Primary
    }
}

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
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub tools: Vec<String>,
    #[serde(default)]
    pub permissions: HashMap<String, Permission>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: "graphirm".to_string(),
            mode: AgentMode::Primary,
            model: "anthropic/claude-sonnet-4".to_string(),
            description: String::new(),
            system_prompt: "You are a helpful coding assistant.".to_string(),
            max_turns: 50,
            max_tokens: Some(8192),
            temperature: Some(0.7),
            tools: vec![],
            permissions: HashMap::new(),
        }
    }
}

/// TOML file layout: `[agent]` section + optional `[permissions]` section.
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
}

fn default_system_prompt() -> String {
    "You are a helpful coding assistant.".to_string()
}

fn default_max_turns() -> u32 {
    50
}

impl AgentConfig {
    /// Parse an AgentConfig from TOML string with `[agent]` + `[permissions]` sections.
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
            permissions: file.permissions,
        })
    }

    /// Load an AgentConfig from a TOML file path.
    pub fn from_file(path: &std::path::Path) -> Result<Self, AgentError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| AgentError::Workflow(format!("Failed to read {}: {}", path.display(), e)))?;
        Self::from_toml(&content)
    }

    /// Check whether a named tool is allowed by this config's permissions.
    /// Default: allow (tools not listed in permissions are permitted).
    pub fn is_tool_allowed(&self, tool_name: &str) -> bool {
        match self.permissions.get(tool_name) {
            Some(Permission::Deny) => false,
            _ => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_mode_deserialize() {
        let primary: AgentMode = toml::from_str("\"primary\"").unwrap();
        assert_eq!(primary, AgentMode::Primary);
        let sub: AgentMode = toml::from_str("\"subagent\"").unwrap();
        assert_eq!(sub, AgentMode::Subagent);
    }

    #[test]
    fn test_permission_deserialize() {
        let allow: Permission = toml::from_str("\"allow\"").unwrap();
        assert_eq!(allow, Permission::Allow);
        let deny: Permission = toml::from_str("\"deny\"").unwrap();
        assert_eq!(deny, Permission::Deny);
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
        assert!(config.permissions.get("read").is_none());
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
    fn test_agent_config_from_toml_minimal() {
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
```

### Step 4: Update lib.rs re-exports

In `crates/agent/src/lib.rs`, add:

```rust
pub use config::{AgentConfig, AgentMode, Permission};
```

### Step 5: Verify

Run: `cargo test -p graphirm-agent config::tests 2>&1`
Expected: `test result: ok. 7 passed; 0 failed`

### Step 6: Commit

```bash
git add crates/agent/Cargo.toml crates/agent/src/config.rs crates/agent/src/lib.rs
git commit -m "feat(agent): define AgentMode, Permission, extend AgentConfig for multi-agent TOML configs"
```

---

## Task 2: Implement AgentRegistry

- [x] Complete

**Files:**
- Modify: `crates/agent/src/multi.rs`
- Modify: `crates/agent/src/lib.rs` (add re-exports)

### Step 1: Write failing tests

Write the full test module at the bottom of `crates/agent/src/multi.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn write_toml(dir: &TempDir, name: &str, content: &str) -> std::path::PathBuf {
        let path = dir.path().join(format!("{}.toml", name));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    fn build_toml() -> &'static str {
        r#"
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
"#
    }

    fn explore_toml() -> &'static str {
        r#"
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
"#
    }

    #[test]
    fn test_registry_load_from_dir() {
        let dir = TempDir::new().unwrap();
        write_toml(&dir, "build", build_toml());
        write_toml(&dir, "explore", explore_toml());

        let registry = AgentRegistry::load_from_dir(dir.path()).unwrap();
        assert_eq!(registry.list().len(), 2);
        assert!(registry.get("build").is_some());
        assert!(registry.get("explore").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_primary() {
        let dir = TempDir::new().unwrap();
        write_toml(&dir, "build", build_toml());
        write_toml(&dir, "explore", explore_toml());

        let registry = AgentRegistry::load_from_dir(dir.path()).unwrap();
        let primary = registry.primary().unwrap();
        assert_eq!(primary.name, "build");
        assert_eq!(primary.mode, AgentMode::Primary);
    }

    #[test]
    fn test_registry_list_names() {
        let dir = TempDir::new().unwrap();
        write_toml(&dir, "build", build_toml());
        write_toml(&dir, "explore", explore_toml());

        let registry = AgentRegistry::load_from_dir(dir.path()).unwrap();
        let mut names = registry.list();
        names.sort();
        assert_eq!(names, vec!["build", "explore"]);
    }

    #[test]
    fn test_registry_empty_dir() {
        let dir = TempDir::new().unwrap();
        let registry = AgentRegistry::load_from_dir(dir.path()).unwrap();
        assert!(registry.list().is_empty());
        assert!(registry.primary().is_none());
    }

    #[test]
    fn test_registry_skips_non_toml_files() {
        let dir = TempDir::new().unwrap();
        write_toml(&dir, "build", build_toml());
        std::fs::write(dir.path().join("README.md"), "# Agents").unwrap();

        let registry = AgentRegistry::load_from_dir(dir.path()).unwrap();
        assert_eq!(registry.list().len(), 1);
    }
}
```

Run: `cargo test -p graphirm-agent multi::tests 2>&1`
Expected: Fails — `AgentRegistry` not defined yet.

### Step 2: Implement AgentRegistry

Write the implementation at the top of `crates/agent/src/multi.rs`:

```rust
use std::collections::HashMap;
use std::path::Path;

use crate::config::{AgentConfig, AgentMode};
use crate::error::AgentError;

/// Registry of agent configurations loaded from TOML files.
pub struct AgentRegistry {
    configs: HashMap<String, AgentConfig>,
}

impl AgentRegistry {
    /// Load all `*.toml` agent config files from a directory.
    pub fn load_from_dir(path: &Path) -> Result<Self, AgentError> {
        let mut configs = HashMap::new();

        if !path.exists() {
            return Ok(Self { configs });
        }

        let entries = std::fs::read_dir(path)
            .map_err(|e| AgentError::Workflow(format!("Failed to read dir {}: {}", path.display(), e)))?;

        for entry in entries {
            let entry = entry
                .map_err(|e| AgentError::Workflow(format!("Failed to read entry: {}", e)))?;
            let file_path = entry.path();

            if file_path.extension().and_then(|e| e.to_str()) != Some("toml") {
                continue;
            }

            let config = AgentConfig::from_file(&file_path)?;
            configs.insert(config.name.clone(), config);
        }

        Ok(Self { configs })
    }

    /// Create a registry from a pre-built map (useful for testing).
    pub fn from_configs(configs: HashMap<String, AgentConfig>) -> Self {
        Self { configs }
    }

    /// Get an agent config by name.
    pub fn get(&self, name: &str) -> Option<&AgentConfig> {
        self.configs.get(name)
    }

    /// List all registered agent names.
    pub fn list(&self) -> Vec<&str> {
        self.configs.keys().map(|s| s.as_str()).collect()
    }

    /// Find the primary agent config (the one with `mode = "primary"`).
    pub fn primary(&self) -> Option<&AgentConfig> {
        self.configs.values().find(|c| c.mode == AgentMode::Primary)
    }

    /// Find all subagent configs.
    pub fn subagents(&self) -> Vec<&AgentConfig> {
        self.configs
            .values()
            .filter(|c| c.mode == AgentMode::Subagent)
            .collect()
    }
}
```

### Step 3: Update lib.rs re-exports

In `crates/agent/src/lib.rs`, add:

```rust
pub use multi::AgentRegistry;
```

### Step 4: Verify

Run: `cargo test -p graphirm-agent multi::tests 2>&1`
Expected: `test result: ok. 5 passed; 0 failed`

### Step 5: Commit

```bash
git add crates/agent/src/multi.rs crates/agent/src/lib.rs
git commit -m "feat(agent): implement AgentRegistry — load agent configs from TOML directory"
```

---

## Task 3: Implement scoped context builder for subagents

- [x] Complete

**Files:**
- Modify: `crates/agent/src/context.rs`

### Step 1: Write failing tests

Add to the existing `#[cfg(test)] mod tests` block in `crates/agent/src/context.rs`:

```rust
    #[test]
    fn test_build_subagent_context_basic() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig {
            system_prompt: "You are an exploration agent.".to_string(),
            ..AgentConfig::default()
        };

        // Create a Task node
        let task_node = graphirm_graph::nodes::GraphNode::new(
            graphirm_graph::nodes::NodeType::Task(graphirm_graph::nodes::TaskData {
                title: "Analyze auth module".to_string(),
                description: "Read all files in src/auth/ and summarize patterns.".to_string(),
                status: "pending".to_string(),
                priority: Some(1),
            }),
        );
        let task_id = task_node.id.clone();
        graph.add_node(task_node).unwrap();

        let context = build_subagent_context(&graph, &config, &task_id, &[]).unwrap();

        // System prompt + task description = 2 messages
        assert_eq!(context.len(), 2);
        assert_eq!(context[0].role, Role::System);
        assert_eq!(context[0].content, "You are an exploration agent.");
        assert_eq!(context[1].role, Role::User);
        assert!(context[1].content.contains("Analyze auth module"));
        assert!(context[1].content.contains("summarize patterns"));
    }

    #[test]
    fn test_build_subagent_context_with_referenced_nodes() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig::default();

        // Create a Task node
        let task_node = graphirm_graph::nodes::GraphNode::new(
            graphirm_graph::nodes::NodeType::Task(graphirm_graph::nodes::TaskData {
                title: "Review code".to_string(),
                description: "Review the auth module.".to_string(),
                status: "pending".to_string(),
                priority: None,
            }),
        );
        let task_id = task_node.id.clone();
        graph.add_node(task_node).unwrap();

        // Create Content nodes as context
        let content1 = graphirm_graph::nodes::GraphNode::new(
            graphirm_graph::nodes::NodeType::Content(graphirm_graph::nodes::ContentData {
                content_type: "file".to_string(),
                path: Some("src/auth/mod.rs".to_string()),
                body: "pub mod login;\npub mod session;".to_string(),
                language: Some("rust".to_string()),
            }),
        );
        let c1_id = content1.id.clone();
        graph.add_node(content1).unwrap();

        let knowledge1 = graphirm_graph::nodes::GraphNode::new(
            graphirm_graph::nodes::NodeType::Knowledge(graphirm_graph::nodes::KnowledgeData {
                entity: "auth patterns".to_string(),
                entity_type: "concept".to_string(),
                summary: "Uses JWT with refresh tokens".to_string(),
                confidence: 0.9,
            }),
        );
        let k1_id = knowledge1.id.clone();
        graph.add_node(knowledge1).unwrap();

        let context = build_subagent_context(
            &graph,
            &config,
            &task_id,
            &[c1_id, k1_id],
        )
        .unwrap();

        // System + task + 2 context nodes = 4
        assert_eq!(context.len(), 4);
        assert!(context[2].content.contains("src/auth/mod.rs"));
        assert!(context[3].content.contains("JWT with refresh tokens"));
    }

    #[test]
    fn test_build_subagent_context_empty_context_nodes() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let config = AgentConfig::default();

        let task_node = graphirm_graph::nodes::GraphNode::new(
            graphirm_graph::nodes::NodeType::Task(graphirm_graph::nodes::TaskData {
                title: "Simple task".to_string(),
                description: "Do something simple.".to_string(),
                status: "pending".to_string(),
                priority: None,
            }),
        );
        let task_id = task_node.id.clone();
        graph.add_node(task_node).unwrap();

        let context = build_subagent_context(&graph, &config, &task_id, &[]).unwrap();
        assert_eq!(context.len(), 2); // system + task only
    }
```

Run: `cargo test -p graphirm-agent context::tests::test_build_subagent 2>&1`
Expected: Fails — `build_subagent_context` not defined yet.

### Step 2: Implement build_subagent_context

Add to `crates/agent/src/context.rs`, below the existing `build_context` function:

```rust
use graphirm_graph::{GraphStore, NodeId};
use graphirm_llm::Role;

use crate::config::AgentConfig;

/// Build a scoped LLM message context for a subagent.
///
/// Unlike `build_context` (which reads the full conversation history from a
/// session), this builds a minimal context from:
/// 1. The agent's system prompt
/// 2. The task description (from the Task node)
/// 3. Content from explicitly referenced context nodes
///
/// This ensures subagents see only what's relevant to their task — no access
/// to the parent agent's full conversation.
pub fn build_subagent_context(
    graph: &GraphStore,
    agent_config: &AgentConfig,
    task_id: &NodeId,
    context_node_ids: &[NodeId],
) -> Result<Vec<LlmMessage>, AgentError> {
    let mut messages = Vec::new();

    // 1. System prompt
    messages.push(LlmMessage {
        role: Role::System,
        content: agent_config.system_prompt.clone(),
        tool_calls: vec![],
        tool_call_id: None,
    });

    // 2. Task description
    let task_node = graph
        .get_node(task_id)
        .map_err(|e| AgentError::Context(format!("Failed to read task node: {}", e)))?;

    let task_title = match &task_node.node_type {
        graphirm_graph::nodes::NodeType::Task(data) => {
            format!("## Task: {}\n\n{}", data.title, data.description)
        }
        _ => {
            let fallback = task_node
                .metadata
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("No task description available.");
            fallback.to_string()
        }
    };

    messages.push(LlmMessage {
        role: Role::User,
        content: task_title,
        tool_calls: vec![],
        tool_call_id: None,
    });

    // 3. Referenced context nodes
    for node_id in context_node_ids {
        let node = graph
            .get_node(node_id)
            .map_err(|e| AgentError::Context(format!("Failed to read context node: {}", e)))?;

        let context_text = match &node.node_type {
            graphirm_graph::nodes::NodeType::Content(data) => {
                let header = match &data.path {
                    Some(path) => format!("File: {}", path),
                    None => format!("Content ({})", data.content_type),
                };
                format!("{}\n```\n{}\n```", header, data.body)
            }
            graphirm_graph::nodes::NodeType::Knowledge(data) => {
                format!(
                    "Knowledge — {} ({}): {}",
                    data.entity, data.entity_type, data.summary
                )
            }
            graphirm_graph::nodes::NodeType::Interaction(data) => {
                format!("Previous finding: {}", data.content)
            }
            _ => serde_json::to_string_pretty(&node.node_type)
                .unwrap_or_else(|_| "Unknown context".to_string()),
        };

        messages.push(LlmMessage {
            role: Role::User,
            content: context_text,
            tool_calls: vec![],
            tool_call_id: None,
        });
    }

    Ok(messages)
}
```

### Step 3: Update lib.rs re-exports

In `crates/agent/src/lib.rs`, add:

```rust
pub use context::build_subagent_context;
```

### Step 4: Verify

Run: `cargo test -p graphirm-agent context::tests::test_build_subagent 2>&1`
Expected: `test result: ok. 3 passed; 0 failed`

### Step 5: Commit

```bash
git add crates/agent/src/context.rs crates/agent/src/lib.rs
git commit -m "feat(agent): implement build_subagent_context — scoped context from task + reference nodes"
```

---

## Task 4: Implement SubagentHandle and spawn_subagent()

- [x] Complete

**Files:**
- Modify: `crates/agent/src/multi.rs`
- Modify: `crates/agent/src/error.rs` (add `SubagentFailed` variant)

### Step 1: Extend AgentError

Add a new variant to `AgentError` in `crates/agent/src/error.rs`:

```rust
    #[error("Subagent '{name}' failed: {reason}")]
    SubagentFailed { name: String, reason: String },

    #[error("Agent not found in registry: {0}")]
    AgentNotFound(String),
```

### Step 2: Write failing tests

Add to the `tests` module in `crates/agent/src/multi.rs`:

```rust
    use crate::config::AgentConfig;
    use crate::event::EventBus;
    use graphirm_graph::{GraphStore, EdgeType, Direction};
    use graphirm_graph::nodes::{GraphNode, NodeType, TaskData, AgentData};
    use graphirm_llm::{MockProvider, MockResponse};
    use graphirm_tools::ToolRegistry;
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;

    fn mock_factory_text(text: &str) -> LlmFactory {
        let text = text.to_string();
        Arc::new(move |_model: &str| -> Box<dyn graphirm_llm::LlmProvider> {
            Box::new(MockProvider::new(vec![MockResponse::text(text.clone())]))
        })
    }

    #[tokio::test]
    async fn test_spawn_subagent_creates_graph_nodes() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let mut agents = HashMap::new();
        agents.insert(
            "explore".to_string(),
            AgentConfig {
                name: "explore".to_string(),
                mode: AgentMode::Subagent,
                model: "test-model".to_string(),
                description: "Explore code".to_string(),
                system_prompt: "You explore code.".to_string(),
                max_turns: 5,
                ..AgentConfig::default()
            },
        );
        let registry = AgentRegistry::from_configs(agents);
        let tools = Arc::new(ToolRegistry::new());
        let events = Arc::new(EventBus::new());
        let factory = mock_factory_text("I found some patterns.");

        // Create a parent agent node
        let parent_agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "build".to_string(),
            model: "claude".to_string(),
            system_prompt: None,
            status: "running".to_string(),
        }));
        let parent_id = parent_agent.id.clone();
        graph.add_node(parent_agent).unwrap();

        let cancel = CancellationToken::new();

        let handle = spawn_subagent(
            &graph,
            &registry,
            &factory,
            &tools,
            &events,
            &parent_id,
            "explore",
            "Analyze the auth module",
            vec![],
            cancel,
        )
        .await
        .unwrap();

        // Wait for subagent to complete
        handle.join_handle.await.unwrap().unwrap();

        // Verify Task node was created
        let task_node = graph.get_node(&handle.task_id).unwrap();
        assert!(matches!(task_node.node_type, NodeType::Task(_)));

        // Verify Agent node was created for subagent
        let agent_node = graph.get_node(&handle.agent_id).unwrap();
        assert!(matches!(agent_node.node_type, NodeType::Agent(_)));

        // Verify DelegatesTo edge: parent → task
        let delegated = graph
            .neighbors(&parent_id, Some(EdgeType::DelegatesTo), Direction::Outgoing)
            .unwrap();
        assert_eq!(delegated.len(), 1);
        assert_eq!(delegated[0].id, handle.task_id);

        // Verify SpawnedBy edge: task → subagent
        let spawned = graph
            .neighbors(&handle.task_id, Some(EdgeType::SpawnedBy), Direction::Outgoing)
            .unwrap();
        assert_eq!(spawned.len(), 1);
        assert_eq!(spawned[0].id, handle.agent_id);
    }

    #[tokio::test]
    async fn test_spawn_subagent_unknown_agent() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let registry = AgentRegistry::from_configs(HashMap::new());
        let tools = Arc::new(ToolRegistry::new());
        let events = Arc::new(EventBus::new());
        let factory = mock_factory_text("unused");

        let parent_agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "build".to_string(),
            model: "claude".to_string(),
            system_prompt: None,
            status: "running".to_string(),
        }));
        let parent_id = parent_agent.id.clone();
        graph.add_node(parent_agent).unwrap();

        let cancel = CancellationToken::new();
        let result = spawn_subagent(
            &graph,
            &registry,
            &factory,
            &tools,
            &events,
            &parent_id,
            "nonexistent",
            "Do something",
            vec![],
            cancel,
        )
        .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AgentError::AgentNotFound(_)));
    }
```

Run: `cargo test -p graphirm-agent multi::tests::test_spawn_subagent 2>&1`
Expected: Fails — `spawn_subagent`, `SubagentHandle`, `LlmFactory` not defined.

### Step 3: Implement SubagentHandle, LlmFactory, and spawn_subagent

Add to `crates/agent/src/multi.rs`:

```rust
use std::sync::Arc;

use graphirm_graph::edges::{EdgeType, GraphEdge};
use graphirm_graph::nodes::{AgentData, GraphNode, NodeId, NodeType, TaskData};
use graphirm_graph::GraphStore;
use graphirm_llm::LlmProvider;
use graphirm_tools::ToolRegistry;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::info;

use crate::config::{AgentConfig, AgentMode, Permission};
use crate::context::build_subagent_context;
use crate::error::AgentError;
use crate::event::EventBus;
use crate::session::Session;
use crate::workflow::run_agent_loop;

/// Factory function type: takes a model string, returns a boxed LlmProvider.
pub type LlmFactory = Arc<dyn Fn(&str) -> Box<dyn LlmProvider> + Send + Sync>;

/// Handle to a running subagent. Returned by `spawn_subagent()`.
pub struct SubagentHandle {
    pub agent_id: NodeId,
    pub task_id: NodeId,
    pub name: String,
    pub join_handle: JoinHandle<Result<(), AgentError>>,
}

/// Build a scoped ToolRegistry for a subagent based on its permission config.
/// Tools explicitly denied are excluded. Unlisted tools are allowed.
fn build_scoped_tools(base_tools: &ToolRegistry, config: &AgentConfig) -> ToolRegistry {
    let mut scoped = ToolRegistry::new();
    for name in base_tools.names() {
        if config.is_tool_allowed(&name) {
            if let Some(tool) = base_tools.get(&name) {
                scoped.register(tool);
            }
        }
    }
    scoped
}

/// Spawn a subagent to work on a specific task.
///
/// Creates the graph structure:
/// ```text
/// Parent Agent --DelegatesTo--> Task --SpawnedBy--> Subagent
/// ```
///
/// The subagent runs `run_agent_loop()` inside a `tokio::spawn` with:
/// - A scoped ToolRegistry (filtered by permissions)
/// - Context built from the task description + referenced nodes
/// - Its own Session (creating an Agent node in the graph)
///
/// Returns a `SubagentHandle` with the JoinHandle to await completion.
pub async fn spawn_subagent(
    graph: &Arc<GraphStore>,
    agents: &AgentRegistry,
    llm_factory: &LlmFactory,
    base_tools: &Arc<ToolRegistry>,
    events: &Arc<EventBus>,
    parent_agent_id: &NodeId,
    agent_name: &str,
    task_description: &str,
    context_nodes: Vec<NodeId>,
    cancel: CancellationToken,
) -> Result<SubagentHandle, AgentError> {
    // 1. Look up agent config
    let agent_config = agents
        .get(agent_name)
        .ok_or_else(|| AgentError::AgentNotFound(agent_name.to_string()))?
        .clone();

    // 2. Create Task node
    let task_node = GraphNode::new(NodeType::Task(TaskData {
        title: format!("Delegated to {}", agent_name),
        description: task_description.to_string(),
        status: "pending".to_string(),
        priority: None,
    }));
    let task_id = task_node.id.clone();
    graph.add_node(task_node)?;

    // 3. DelegatesTo edge: parent agent → task
    graph.add_edge(GraphEdge::new(
        EdgeType::DelegatesTo,
        parent_agent_id.clone(),
        task_id.clone(),
    ))?;

    // 4. Create subagent Session (creates Agent node)
    let session = Session::new(graph.clone(), agent_config.clone())?;
    let agent_id = session.id.clone();

    // 5. SpawnedBy edge: task → subagent
    graph.add_edge(GraphEdge::new(
        EdgeType::SpawnedBy,
        task_id.clone(),
        agent_id.clone(),
    ))?;

    // 6. Build scoped context and add as user messages
    let scoped_messages = build_subagent_context(
        graph,
        &agent_config,
        &task_id,
        &context_nodes,
    )?;

    // Add each context message (skip system — Session already handles that) as
    // user messages so build_context() in run_agent_loop picks them up.
    for msg in &scoped_messages {
        if msg.role != graphirm_llm::Role::System {
            session.add_user_message(&msg.content)?;
        }
    }

    // 7. Build scoped tools
    let scoped_tools = build_scoped_tools(base_tools, &agent_config);

    // 8. Create LLM provider
    let llm = (llm_factory)(&agent_config.model);

    // 9. Spawn agent loop
    let events_clone = events.clone();
    let agent_name_owned = agent_name.to_string();
    let task_id_for_status = task_id.clone();
    let graph_for_status = graph.clone();

    info!(
        agent = %agent_name_owned,
        task_id = %task_id,
        agent_id = %agent_id,
        "Spawning subagent"
    );

    let join_handle = tokio::spawn(async move {
        let result = run_agent_loop(&session, llm.as_ref(), &scoped_tools, &events_clone, &cancel)
            .await;

        // Update task status in graph
        let status = if result.is_ok() { "completed" } else { "failed" };
        let mut task_node = graph_for_status.get_node(&task_id_for_status)?;
        if let NodeType::Task(ref mut data) = task_node.node_type {
            data.status = status.to_string();
        }
        graph_for_status.update_node(&task_id_for_status, task_node)?;

        info!(agent = %agent_name_owned, status, "Subagent finished");
        result
    });

    Ok(SubagentHandle {
        agent_id,
        task_id,
        name: agent_name.to_string(),
        join_handle,
    })
}
```

### Step 4: Verify

Run: `cargo test -p graphirm-agent multi::tests::test_spawn_subagent 2>&1`
Expected: `test result: ok. 2 passed; 0 failed`

### Step 5: Commit

```bash
git add crates/agent/src/multi.rs crates/agent/src/error.rs
git commit -m "feat(agent): implement SubagentHandle and spawn_subagent with graph tracking"
```

---

## Task 5: Implement wait_for_subagents()

- [x] Complete

**Files:**
- Modify: `crates/agent/src/multi.rs`

### Step 1: Write failing test

Add to the `tests` module in `crates/agent/src/multi.rs`:

```rust
    #[tokio::test]
    async fn test_wait_for_subagents_all_succeed() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let mut agents = HashMap::new();
        agents.insert(
            "explore".to_string(),
            AgentConfig {
                name: "explore".to_string(),
                mode: AgentMode::Subagent,
                model: "test".to_string(),
                system_prompt: "Explore.".to_string(),
                max_turns: 3,
                ..AgentConfig::default()
            },
        );
        let registry = AgentRegistry::from_configs(agents);
        let tools = Arc::new(ToolRegistry::new());
        let events = Arc::new(EventBus::new());
        let factory = mock_factory_text("Done exploring.");

        let parent_agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "build".to_string(),
            model: "claude".to_string(),
            system_prompt: None,
            status: "running".to_string(),
        }));
        let parent_id = parent_agent.id.clone();
        graph.add_node(parent_agent).unwrap();

        let cancel = CancellationToken::new();

        // Spawn 2 subagents
        let h1 = spawn_subagent(
            &graph, &registry, &factory, &tools, &events,
            &parent_id, "explore", "Task A", vec![], cancel.clone(),
        ).await.unwrap();

        let h2 = spawn_subagent(
            &graph, &registry, &factory, &tools, &events,
            &parent_id, "explore", "Task B", vec![], cancel.clone(),
        ).await.unwrap();

        let results = wait_for_subagents(vec![h1, h2]).await.unwrap();

        // Both should have completed
        assert_eq!(results.len(), 2);

        // Verify task nodes are marked completed
        for (task_id, _agent_id) in &results {
            let task_node = graph.get_node(task_id).unwrap();
            if let NodeType::Task(data) = &task_node.node_type {
                assert_eq!(data.status, "completed");
            }
        }
    }
```

Run: `cargo test -p graphirm-agent multi::tests::test_wait_for_subagents 2>&1`
Expected: Fails — `wait_for_subagents` not defined yet.

### Step 2: Implement wait_for_subagents

Add to `crates/agent/src/multi.rs`:

```rust
/// Wait for all subagent handles to complete.
/// Returns a list of (task_id, agent_id) for each completed subagent.
/// If any subagent fails, collects the error but continues waiting for others.
pub async fn wait_for_subagents(
    handles: Vec<SubagentHandle>,
) -> Result<Vec<(NodeId, NodeId)>, AgentError> {
    let mut results = Vec::new();
    let mut errors = Vec::new();

    for handle in handles {
        match handle.join_handle.await {
            Ok(Ok(())) => {
                info!(agent = %handle.name, "Subagent completed successfully");
                results.push((handle.task_id, handle.agent_id));
            }
            Ok(Err(e)) => {
                info!(
                    agent = %handle.name,
                    error = %e,
                    "Subagent failed"
                );
                errors.push(AgentError::SubagentFailed {
                    name: handle.name.clone(),
                    reason: e.to_string(),
                });
                results.push((handle.task_id, handle.agent_id));
            }
            Err(join_err) => {
                errors.push(AgentError::Join(format!(
                    "Subagent '{}' panicked: {}",
                    handle.name, join_err
                )));
            }
        }
    }

    if !errors.is_empty() && results.is_empty() {
        return Err(errors.remove(0));
    }

    Ok(results)
}
```

### Step 3: Verify

Run: `cargo test -p graphirm-agent multi::tests::test_wait_for_subagents 2>&1`
Expected: `test result: ok. 1 passed; 0 failed`

### Step 4: Commit

```bash
git add crates/agent/src/multi.rs
git commit -m "feat(agent): implement wait_for_subagents — join handles, collect results"
```

---

## Task 6: Implement Coordinator struct and run_primary()

- [x] Complete

**Files:**
- Modify: `crates/agent/src/multi.rs`

### Step 1: Write failing tests

Add to the `tests` module in `crates/agent/src/multi.rs`:

```rust
    #[tokio::test]
    async fn test_coordinator_run_primary_simple() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let dir = TempDir::new().unwrap();
        write_toml(&dir, "build", build_toml());
        let registry = AgentRegistry::load_from_dir(dir.path()).unwrap();
        let tools = Arc::new(ToolRegistry::new());
        let events = Arc::new(EventBus::new());
        let factory = mock_factory_text("Hello from the primary agent!");

        let coordinator = Coordinator::new(
            graph.clone(),
            registry,
            factory,
            tools,
            events,
        );

        let cancel = CancellationToken::new();
        let session_id = coordinator
            .run_primary("What is 2+2?", cancel)
            .await
            .unwrap();

        // Verify session created an agent node
        let agent_node = graph.get_node(&session_id).unwrap();
        assert!(matches!(agent_node.node_type, NodeType::Agent(_)));

        // Verify the user message was recorded
        let neighbors = graph
            .neighbors(&session_id, Some(EdgeType::Produces), Direction::Outgoing)
            .unwrap();
        assert!(neighbors.len() >= 2); // user msg + assistant response
    }

    #[tokio::test]
    async fn test_coordinator_run_primary_no_primary_agent() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let dir = TempDir::new().unwrap();
        write_toml(&dir, "explore", explore_toml()); // only subagent, no primary
        let registry = AgentRegistry::load_from_dir(dir.path()).unwrap();
        let tools = Arc::new(ToolRegistry::new());
        let events = Arc::new(EventBus::new());
        let factory = mock_factory_text("unused");

        let coordinator = Coordinator::new(graph, registry, factory, tools, events);

        let cancel = CancellationToken::new();
        let result = coordinator.run_primary("Hello", cancel).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AgentError::Workflow(_)));
    }
```

Run: `cargo test -p graphirm-agent multi::tests::test_coordinator 2>&1`
Expected: Fails — `Coordinator` not defined yet.

### Step 2: Implement Coordinator

Add to `crates/agent/src/multi.rs`:

```rust
/// Orchestrates multi-agent workflows.
///
/// The Coordinator owns the agent registry, LLM factory, shared graph, tools,
/// and event bus. It runs the primary agent loop and provides the plumbing for
/// subagent spawning via the `delegate` tool.
pub struct Coordinator {
    graph: Arc<GraphStore>,
    agents: AgentRegistry,
    llm_factory: LlmFactory,
    tools: Arc<ToolRegistry>,
    events: Arc<EventBus>,
}

impl Coordinator {
    pub fn new(
        graph: Arc<GraphStore>,
        agents: AgentRegistry,
        llm_factory: LlmFactory,
        tools: Arc<ToolRegistry>,
        events: Arc<EventBus>,
    ) -> Self {
        Self {
            graph,
            agents,
            llm_factory,
            tools,
            events,
        }
    }

    /// Run the primary agent with the given user prompt.
    /// Returns the Session's agent NodeId.
    pub async fn run_primary(
        &self,
        prompt: &str,
        cancel: CancellationToken,
    ) -> Result<NodeId, AgentError> {
        let primary_config = self
            .agents
            .primary()
            .ok_or_else(|| {
                AgentError::Workflow("No primary agent configured in registry".to_string())
            })?
            .clone();

        let session = Session::new(self.graph.clone(), primary_config.clone())?;
        session.add_user_message(prompt)?;

        let llm = (self.llm_factory)(&primary_config.model);

        run_agent_loop(&session, llm.as_ref(), &self.tools, &self.events, &cancel).await?;

        Ok(session.id)
    }

    /// Access the agent registry (for SubagentTool).
    pub fn registry(&self) -> &AgentRegistry {
        &self.agents
    }

    /// Access the graph store.
    pub fn graph(&self) -> &Arc<GraphStore> {
        &self.graph
    }

    /// Access the LLM factory.
    pub fn llm_factory(&self) -> &LlmFactory {
        &self.llm_factory
    }

    /// Access the base tool registry.
    pub fn tools(&self) -> &Arc<ToolRegistry> {
        &self.tools
    }

    /// Access the event bus.
    pub fn events(&self) -> &Arc<EventBus> {
        &self.events
    }
}
```

### Step 3: Update lib.rs re-exports

In `crates/agent/src/lib.rs`, add:

```rust
pub use multi::{AgentRegistry, Coordinator, LlmFactory, SubagentHandle};
pub use multi::{spawn_subagent, wait_for_subagents};
```

### Step 4: Verify

Run: `cargo test -p graphirm-agent multi::tests::test_coordinator 2>&1`
Expected: `test result: ok. 2 passed; 0 failed`

### Step 5: Commit

```bash
git add crates/agent/src/multi.rs crates/agent/src/lib.rs
git commit -m "feat(agent): implement Coordinator with run_primary — orchestrates primary agent loop"
```

---

## Task 7: Implement SubagentTool (delegate tool)

- [x] Complete

**Files:**
- Create: `crates/agent/src/delegate.rs`
- Modify: `crates/agent/src/lib.rs` (add module)

### Step 1: Write failing tests

Create `crates/agent/src/delegate.rs` with tests at the bottom:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AgentConfig, AgentMode};
    use crate::event::EventBus;
    use crate::multi::AgentRegistry;
    use graphirm_graph::edges::EdgeType;
    use graphirm_graph::nodes::{AgentData, GraphNode, NodeType};
    use graphirm_graph::{Direction, GraphStore};
    use graphirm_llm::{MockProvider, MockResponse};
    use graphirm_tools::{Tool, ToolRegistry};
    use std::collections::HashMap;
    use std::sync::Arc;

    fn mock_factory() -> LlmFactory {
        Arc::new(|_model: &str| -> Box<dyn graphirm_llm::LlmProvider> {
            Box::new(MockProvider::new(vec![MockResponse::text(
                "Exploration complete. Found auth patterns using JWT.",
            )]))
        })
    }

    fn test_registry() -> AgentRegistry {
        let mut configs = HashMap::new();
        configs.insert(
            "explore".to_string(),
            AgentConfig {
                name: "explore".to_string(),
                mode: AgentMode::Subagent,
                model: "test-model".to_string(),
                description: "Fast code exploration".to_string(),
                system_prompt: "You explore code.".to_string(),
                max_turns: 5,
                ..AgentConfig::default()
            },
        );
        AgentRegistry::from_configs(configs)
    }

    #[test]
    fn test_delegate_tool_name_and_schema() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let registry = Arc::new(test_registry());
        let tools = Arc::new(ToolRegistry::new());
        let events = Arc::new(EventBus::new());
        let factory = mock_factory();

        let parent_id = graphirm_graph::nodes::NodeId::from("parent-1");

        let delegate = SubagentTool::new(
            graph, registry, factory, tools, events, parent_id,
        );

        assert_eq!(delegate.name(), "delegate");
        assert!(delegate.description().contains("subagent"));

        let schema = delegate.parameters_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["agent"].is_object());
        assert!(schema["properties"]["task"].is_object());
        assert_eq!(schema["required"], serde_json::json!(["agent", "task"]));
    }

    #[tokio::test]
    async fn test_delegate_tool_execute() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let registry = Arc::new(test_registry());
        let tools = Arc::new(ToolRegistry::new());
        let events = Arc::new(EventBus::new());
        let factory = mock_factory();

        // Create parent agent node in graph
        let parent_agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "build".to_string(),
            model: "claude".to_string(),
            system_prompt: None,
            status: "running".to_string(),
        }));
        let parent_id = parent_agent.id.clone();
        graph.add_node(parent_agent).unwrap();

        let delegate = SubagentTool::new(
            graph.clone(),
            registry,
            factory,
            tools,
            events,
            parent_id.clone(),
        );

        let args = serde_json::json!({
            "agent": "explore",
            "task": "Analyze authentication patterns in src/auth/"
        });

        let result = delegate.execute(args).await.unwrap();

        // Result should describe what happened
        assert!(result.contains("explore"));
        assert!(result.contains("completed"));

        // Verify graph structure: parent --DelegatesTo--> task
        let delegated = graph
            .neighbors(&parent_id, Some(EdgeType::DelegatesTo), Direction::Outgoing)
            .unwrap();
        assert_eq!(delegated.len(), 1);
    }

    #[tokio::test]
    async fn test_delegate_tool_unknown_agent() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let registry = Arc::new(test_registry());
        let tools = Arc::new(ToolRegistry::new());
        let events = Arc::new(EventBus::new());
        let factory = mock_factory();
        let parent_id = graphirm_graph::nodes::NodeId::from("parent-1");

        let delegate = SubagentTool::new(
            graph, registry, factory, tools, events, parent_id,
        );

        let args = serde_json::json!({
            "agent": "nonexistent",
            "task": "Do something"
        });

        let result = delegate.execute(args).await;
        assert!(result.is_err());
    }
}
```

Run: `cargo test -p graphirm-agent delegate::tests 2>&1`
Expected: Fails — `SubagentTool` not defined yet.

### Step 2: Implement SubagentTool

Write the full `crates/agent/src/delegate.rs`:

```rust
use std::sync::Arc;

use async_trait::async_trait;

use graphirm_graph::nodes::NodeId;
use graphirm_graph::GraphStore;
use graphirm_tools::{Tool, ToolError, ToolRegistry};
use tokio_util::sync::CancellationToken;

use crate::event::EventBus;
use crate::multi::{spawn_subagent, wait_for_subagents, AgentRegistry, LlmFactory};

/// Tool that allows the primary agent to delegate work to a subagent.
///
/// When the primary agent calls `delegate(agent="explore", task="...")`,
/// this tool:
/// 1. Looks up the agent config from the registry
/// 2. Creates Task + Agent nodes with DelegatesTo/SpawnedBy edges
/// 3. Runs the subagent's agent loop
/// 4. Waits for completion
/// 5. Returns a summary of what the subagent produced
pub struct SubagentTool {
    graph: Arc<GraphStore>,
    agents: Arc<AgentRegistry>,
    llm_factory: LlmFactory,
    base_tools: Arc<ToolRegistry>,
    events: Arc<EventBus>,
    parent_agent_id: NodeId,
}

impl SubagentTool {
    pub fn new(
        graph: Arc<GraphStore>,
        agents: Arc<AgentRegistry>,
        llm_factory: LlmFactory,
        base_tools: Arc<ToolRegistry>,
        events: Arc<EventBus>,
        parent_agent_id: NodeId,
    ) -> Self {
        Self {
            graph,
            agents,
            llm_factory,
            base_tools,
            events,
            parent_agent_id,
        }
    }
}

#[async_trait]
impl Tool for SubagentTool {
    fn name(&self) -> &str {
        "delegate"
    }

    fn description(&self) -> &str {
        "Delegate a task to a specialized subagent. The subagent runs independently \
         with scoped tools and context, writes results to the graph, and returns a \
         summary when complete."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "Name of the subagent to delegate to (e.g. 'explore')"
                },
                "task": {
                    "type": "string",
                    "description": "Description of the task for the subagent to perform"
                },
                "context_nodes": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional list of node IDs to include as context for the subagent"
                }
            },
            "required": ["agent", "task"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let agent_name = args["agent"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("Missing 'agent' field".to_string()))?;

        let task_description = args["task"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("Missing 'task' field".to_string()))?;

        let context_nodes: Vec<NodeId> = args["context_nodes"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| NodeId::from(s)))
                    .collect()
            })
            .unwrap_or_default();

        let cancel = CancellationToken::new();

        let handle = spawn_subagent(
            &self.graph,
            &self.agents,
            &self.llm_factory,
            &self.base_tools,
            &self.events,
            &self.parent_agent_id,
            agent_name,
            task_description,
            context_nodes,
            cancel,
        )
        .await
        .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        let task_id = handle.task_id.clone();
        let agent_id = handle.agent_id.clone();
        let name = handle.name.clone();

        let results = wait_for_subagents(vec![handle])
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        // Collect output from the subagent's conversation
        let output = self.collect_subagent_output(&agent_id)?;

        Ok(format!(
            "Subagent '{}' completed.\nTask: {}\nAgent ID: {}\nTask ID: {}\n\nOutput:\n{}",
            name, task_description, agent_id, task_id, output
        ))
    }
}

impl SubagentTool {
    /// Read the subagent's final assistant response from the graph.
    fn collect_subagent_output(&self, agent_id: &NodeId) -> Result<String, ToolError> {
        use graphirm_graph::edges::EdgeType;
        use graphirm_graph::Direction;

        let neighbors = self
            .graph
            .neighbors(agent_id, Some(EdgeType::Produces), Direction::Outgoing)
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        let mut outputs = Vec::new();
        for node in &neighbors {
            match &node.node_type {
                graphirm_graph::nodes::NodeType::Interaction(data) => {
                    if data.role == "assistant" && !data.content.is_empty() {
                        outputs.push(data.content.clone());
                    }
                }
                graphirm_graph::nodes::NodeType::Knowledge(data) => {
                    outputs.push(format!("[Knowledge] {}: {}", data.entity, data.summary));
                }
                graphirm_graph::nodes::NodeType::Content(data) => {
                    outputs.push(format!(
                        "[Content] {}: {}",
                        data.path.as_deref().unwrap_or("unknown"),
                        &data.body[..data.body.len().min(200)]
                    ));
                }
                _ => {}
            }
        }

        if outputs.is_empty() {
            Ok("(no output produced)".to_string())
        } else {
            Ok(outputs.join("\n\n"))
        }
    }
}
```

### Step 3: Add module to lib.rs

In `crates/agent/src/lib.rs`, add:

```rust
pub mod delegate;

pub use delegate::SubagentTool;
```

### Step 4: Verify

Run: `cargo test -p graphirm-agent delegate::tests 2>&1`
Expected: `test result: ok. 3 passed; 0 failed`

### Step 5: Commit

```bash
git add crates/agent/src/delegate.rs crates/agent/src/lib.rs
git commit -m "feat(agent): implement SubagentTool — delegate tool that spawns subagents"
```

---

## Task 8: Implement task dependency tracking

- [x] Complete

**Files:**
- Modify: `crates/agent/src/multi.rs`

### Step 1: Write failing test

Add to the `tests` module in `crates/agent/src/multi.rs`:

```rust
    #[tokio::test]
    async fn test_task_dependency_waits_for_prerequisite() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let mut agents = HashMap::new();
        agents.insert(
            "worker".to_string(),
            AgentConfig {
                name: "worker".to_string(),
                mode: AgentMode::Subagent,
                model: "test".to_string(),
                system_prompt: "Work.".to_string(),
                max_turns: 3,
                ..AgentConfig::default()
            },
        );
        let registry = AgentRegistry::from_configs(agents);
        let tools = Arc::new(ToolRegistry::new());
        let events = Arc::new(EventBus::new());

        let call_order = Arc::new(std::sync::Mutex::new(Vec::new()));

        let order_a = call_order.clone();
        let factory_a: LlmFactory = Arc::new(move |_| {
            order_a.lock().unwrap().push("A");
            Box::new(MockProvider::new(vec![MockResponse::text("A done")]))
        });

        let order_b = call_order.clone();
        let factory_b: LlmFactory = Arc::new(move |_| {
            order_b.lock().unwrap().push("B");
            Box::new(MockProvider::new(vec![MockResponse::text("B done")]))
        });

        let parent_agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "build".to_string(),
            model: "claude".to_string(),
            system_prompt: None,
            status: "running".to_string(),
        }));
        let parent_id = parent_agent.id.clone();
        graph.add_node(parent_agent).unwrap();

        let cancel = CancellationToken::new();

        // Spawn Task A
        let handle_a = spawn_subagent(
            &graph, &registry, &factory_a, &tools, &events,
            &parent_id, "worker", "Task A", vec![], cancel.clone(),
        ).await.unwrap();

        // Create Task B that depends on Task A
        let task_b_node = GraphNode::new(NodeType::Task(TaskData {
            title: "Task B".to_string(),
            description: "Depends on Task A".to_string(),
            status: "pending".to_string(),
            priority: None,
        }));
        let task_b_id = task_b_node.id.clone();
        graph.add_node(task_b_node).unwrap();

        // Add DependsOn edge: Task B --DependsOn--> Task A
        graph.add_edge(GraphEdge::new(
            EdgeType::DependsOn,
            task_b_id.clone(),
            handle_a.task_id.clone(),
        )).unwrap();

        // Wait for Task A to complete before spawning B
        wait_for_dependencies(&graph, &task_b_id).await.unwrap();

        // Now spawn Task B
        let handle_b = spawn_subagent(
            &graph, &registry, &factory_b, &tools, &events,
            &parent_id, "worker", "Task B", vec![], cancel.clone(),
        ).await.unwrap();

        // Wait for both
        let _ = wait_for_subagents(vec![handle_a]).await.unwrap();
        let _ = wait_for_subagents(vec![handle_b]).await.unwrap();

        // A should have been created before B
        let order = call_order.lock().unwrap();
        assert!(order.len() >= 2);
        let a_idx = order.iter().position(|x| *x == "A").unwrap();
        let b_idx = order.iter().position(|x| *x == "B").unwrap();
        assert!(a_idx < b_idx, "Task A should run before Task B");
    }
```

Run: `cargo test -p graphirm-agent multi::tests::test_task_dependency 2>&1`
Expected: Fails — `wait_for_dependencies` not defined.

### Step 2: Implement wait_for_dependencies

Add to `crates/agent/src/multi.rs`:

```rust
use graphirm_graph::Direction;

/// Wait for all tasks that a given task depends on (via DependsOn edges) to
/// reach "completed" status. Polls the graph every 100ms.
///
/// Used to enforce task ordering: if Task B DependsOn Task A, call
/// `wait_for_dependencies(graph, task_b_id)` before spawning Task B.
pub async fn wait_for_dependencies(
    graph: &GraphStore,
    task_id: &NodeId,
) -> Result<(), AgentError> {
    let deps = graph
        .neighbors(task_id, Some(EdgeType::DependsOn), Direction::Outgoing)
        .map_err(|e| AgentError::Context(format!("Failed to read dependencies: {}", e)))?;

    if deps.is_empty() {
        return Ok(());
    }

    let dep_ids: Vec<NodeId> = deps.iter().map(|n| n.id.clone()).collect();

    loop {
        let mut all_done = true;
        for dep_id in &dep_ids {
            let node = graph
                .get_node(dep_id)
                .map_err(|e| AgentError::Context(format!("Failed to read dep: {}", e)))?;
            if let NodeType::Task(data) = &node.node_type {
                if data.status != "completed" {
                    all_done = false;
                    break;
                }
            }
        }
        if all_done {
            return Ok(());
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}
```

### Step 3: Verify

Run: `cargo test -p graphirm-agent multi::tests::test_task_dependency 2>&1`
Expected: `test result: ok. 1 passed; 0 failed`

### Step 4: Commit

```bash
git add crates/agent/src/multi.rs
git commit -m "feat(agent): implement wait_for_dependencies — poll graph for prerequisite task completion"
```

---

## Task 9: Implement parallel subagent execution test

- [x] Complete

**Files:**
- Modify: `crates/agent/src/multi.rs` (test only)

### Step 1: Write test

Add to the `tests` module in `crates/agent/src/multi.rs`:

```rust
    #[tokio::test]
    async fn test_parallel_subagent_execution() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let mut agents = HashMap::new();
        agents.insert(
            "explore".to_string(),
            AgentConfig {
                name: "explore".to_string(),
                mode: AgentMode::Subagent,
                model: "test".to_string(),
                system_prompt: "Explore.".to_string(),
                max_turns: 3,
                ..AgentConfig::default()
            },
        );
        let registry = AgentRegistry::from_configs(agents);
        let tools = Arc::new(ToolRegistry::new());
        let events = Arc::new(EventBus::new());

        let completion_times = Arc::new(std::sync::Mutex::new(Vec::new()));

        let times = completion_times.clone();
        let factory: LlmFactory = Arc::new(move |_| {
            let times = times.clone();
            Box::new(MockProvider::new(vec![MockResponse::text("done")])) as Box<dyn graphirm_llm::LlmProvider>
        });

        let parent_agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "build".to_string(),
            model: "claude".to_string(),
            system_prompt: None,
            status: "running".to_string(),
        }));
        let parent_id = parent_agent.id.clone();
        graph.add_node(parent_agent).unwrap();

        let cancel = CancellationToken::new();

        // Spawn 3 subagents concurrently
        let mut handles = Vec::new();
        for i in 0..3 {
            let h = spawn_subagent(
                &graph, &registry, &factory, &tools, &events,
                &parent_id, "explore",
                &format!("Task {}", i),
                vec![],
                cancel.clone(),
            ).await.unwrap();
            handles.push(h);
        }

        // Wait for all
        let results = wait_for_subagents(handles).await.unwrap();
        assert_eq!(results.len(), 3);

        // Verify all 3 tasks and 3 agents are in the graph
        let delegated = graph
            .neighbors(&parent_id, Some(EdgeType::DelegatesTo), Direction::Outgoing)
            .unwrap();
        assert_eq!(delegated.len(), 3);

        // Verify each task is completed
        for task_node in &delegated {
            if let NodeType::Task(data) = &task_node.node_type {
                assert_eq!(data.status, "completed");
            }
        }
    }
```

### Step 2: Verify

Run: `cargo test -p graphirm-agent multi::tests::test_parallel_subagent 2>&1`
Expected: `test result: ok. 1 passed; 0 failed`

### Step 3: Commit

```bash
git add crates/agent/src/multi.rs
git commit -m "test(agent): add parallel subagent execution test — 3 concurrent subagents"
```

---

## Task 10: Implement result merging — parent reads subagent outputs

- [x] Complete

**Files:**
- Modify: `crates/agent/src/multi.rs`

### Step 1: Write failing test

Add to the `tests` module in `crates/agent/src/multi.rs`:

```rust
    #[tokio::test]
    async fn test_collect_subagent_results_from_graph() {
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let mut agents = HashMap::new();
        agents.insert(
            "explore".to_string(),
            AgentConfig {
                name: "explore".to_string(),
                mode: AgentMode::Subagent,
                model: "test".to_string(),
                system_prompt: "Explore code.".to_string(),
                max_turns: 3,
                ..AgentConfig::default()
            },
        );
        let registry = AgentRegistry::from_configs(agents);
        let tools = Arc::new(ToolRegistry::new());
        let events = Arc::new(EventBus::new());
        let factory = mock_factory_text("Found: auth uses JWT tokens with 24h expiry.");

        let parent_agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "build".to_string(),
            model: "claude".to_string(),
            system_prompt: None,
            status: "running".to_string(),
        }));
        let parent_id = parent_agent.id.clone();
        graph.add_node(parent_agent).unwrap();

        let cancel = CancellationToken::new();

        let handle = spawn_subagent(
            &graph, &registry, &factory, &tools, &events,
            &parent_id, "explore", "Analyze auth tokens", vec![],
            cancel,
        ).await.unwrap();

        let agent_id = handle.agent_id.clone();
        let task_id = handle.task_id.clone();
        wait_for_subagents(vec![handle]).await.unwrap();

        // Collect results using the public function
        let results = collect_subagent_results(&graph, &task_id).unwrap();

        assert!(!results.is_empty());
        // The subagent's assistant response should be in the results
        let has_jwt_content = results.iter().any(|r| r.contains("JWT"));
        assert!(has_jwt_content, "Results should contain subagent output: {:?}", results);
    }
```

Run: `cargo test -p graphirm-agent multi::tests::test_collect_subagent_results 2>&1`
Expected: Fails — `collect_subagent_results` not defined.

### Step 2: Implement collect_subagent_results

Add to `crates/agent/src/multi.rs`:

```rust
/// Collect output produced by a subagent via graph traversal.
///
/// Starting from a Task node, follows SpawnedBy → Agent → Produces → outputs.
/// Returns text content from assistant responses, knowledge, and content nodes.
pub fn collect_subagent_results(
    graph: &GraphStore,
    task_id: &NodeId,
) -> Result<Vec<String>, AgentError> {
    // task --SpawnedBy--> agent
    let spawned = graph
        .neighbors(task_id, Some(EdgeType::SpawnedBy), Direction::Outgoing)
        .map_err(|e| AgentError::Context(e.to_string()))?;

    let mut results = Vec::new();

    for agent_node in &spawned {
        // agent --Produces--> interactions/content/knowledge
        let outputs = graph
            .neighbors(&agent_node.id, Some(EdgeType::Produces), Direction::Outgoing)
            .map_err(|e| AgentError::Context(e.to_string()))?;

        for node in &outputs {
            match &node.node_type {
                NodeType::Interaction(data) if data.role == "assistant" => {
                    if !data.content.is_empty() {
                        results.push(data.content.clone());
                    }
                }
                NodeType::Knowledge(data) => {
                    results.push(format!(
                        "[Knowledge] {} ({}): {}",
                        data.entity, data.entity_type, data.summary
                    ));
                }
                NodeType::Content(data) => {
                    let path = data.path.as_deref().unwrap_or("unknown");
                    let preview = if data.body.len() > 500 {
                        format!("{}...", &data.body[..500])
                    } else {
                        data.body.clone()
                    };
                    results.push(format!("[File: {}]\n{}", path, preview));
                }
                _ => {}
            }
        }
    }

    Ok(results)
}
```

### Step 3: Update lib.rs re-exports

In `crates/agent/src/lib.rs`, add:

```rust
pub use multi::{collect_subagent_results, wait_for_dependencies};
```

### Step 4: Verify

Run: `cargo test -p graphirm-agent multi::tests::test_collect_subagent_results 2>&1`
Expected: `test result: ok. 1 passed; 0 failed`

### Step 5: Commit

```bash
git add crates/agent/src/multi.rs crates/agent/src/lib.rs
git commit -m "feat(agent): implement collect_subagent_results — traverse graph for subagent output"
```

---

## Task 11: Integration test — full multi-agent flow

- [x] Complete

**Files:**
- Create: `crates/agent/tests/multi_agent_integration.rs`

### Step 1: Write the integration test

This test exercises the complete flow: primary agent receives a prompt, delegates to an explore subagent via the delegate tool, the explore subagent reads files and produces knowledge, and the primary agent uses that knowledge for its final response.

```rust
use std::collections::HashMap;
use std::sync::Arc;

use graphirm_agent::{
    AgentConfig, AgentMode, AgentRegistry, Coordinator, EventBus, LlmFactory,
    SubagentTool,
};
use graphirm_graph::edges::EdgeType;
use graphirm_graph::nodes::{NodeType, GraphNode};
use graphirm_graph::{Direction, GraphStore};
use graphirm_llm::{MockProvider, MockResponse, LlmProvider};
use graphirm_tools::{Tool, ToolRegistry};
use tokio_util::sync::CancellationToken;

/// Build a primary agent config.
fn primary_config() -> AgentConfig {
    AgentConfig {
        name: "build".to_string(),
        mode: AgentMode::Primary,
        model: "test-primary".to_string(),
        description: "Primary coding agent".to_string(),
        system_prompt: "You are the primary coding agent. Use the delegate tool to assign exploration tasks to the explore subagent.".to_string(),
        max_turns: 10,
        tools: vec!["delegate".to_string()],
        ..AgentConfig::default()
    }
}

/// Build a subagent config for exploration.
fn explore_config() -> AgentConfig {
    AgentConfig {
        name: "explore".to_string(),
        mode: AgentMode::Subagent,
        model: "test-explore".to_string(),
        description: "Read-only exploration agent".to_string(),
        system_prompt: "You are an exploration agent. Read code and report findings.".to_string(),
        max_turns: 5,
        tools: vec!["read".to_string()],
        ..AgentConfig::default()
    }
}

#[tokio::test]
async fn test_full_multi_agent_flow() {
    let graph = Arc::new(GraphStore::open_memory().unwrap());

    // Set up registry with both agents
    let mut configs = HashMap::new();
    configs.insert("build".to_string(), primary_config());
    configs.insert("explore".to_string(), explore_config());
    let registry = Arc::new(AgentRegistry::from_configs(configs));

    // The primary agent's first response uses the delegate tool,
    // second response is the final answer incorporating subagent results.
    let factory: LlmFactory = Arc::new(|model: &str| -> Box<dyn LlmProvider> {
        if model == "test-primary" {
            // Turn 1: delegate to explore subagent
            // Turn 2: provide final answer (after tool result)
            Box::new(MockProvider::new(vec![
                MockResponse::with_tool_call(
                    "I'll delegate exploration to the explore agent.",
                    "tc_delegate_1",
                    "delegate",
                    serde_json::json!({
                        "agent": "explore",
                        "task": "Analyze the auth module patterns"
                    }),
                ),
                MockResponse::text(
                    "Based on the exploration results, the auth module uses JWT tokens \
                     with refresh token rotation. The implementation is solid."
                ),
            ]))
        } else {
            // Subagent (explore) returns its findings
            Box::new(MockProvider::new(vec![MockResponse::text(
                "Found: auth uses JWT with 24h access tokens and 7d refresh tokens. \
                 Refresh rotation is implemented in src/auth/refresh.rs.",
            )]))
        }
    });

    // Base tools (empty for this test — subagents won't call real tools)
    let base_tools = Arc::new(ToolRegistry::new());
    let events = Arc::new(EventBus::new());

    // Create Coordinator — it will build the delegate tool internally
    // For this test, we manually create the tool registry with the delegate tool
    let coordinator = Coordinator::new(
        graph.clone(),
        AgentRegistry::from_configs({
            let mut m = HashMap::new();
            m.insert("build".to_string(), primary_config());
            m.insert("explore".to_string(), explore_config());
            m
        }),
        factory.clone(),
        base_tools.clone(),
        events.clone(),
    );

    // We need to set up the primary agent's tool registry with the delegate tool.
    // Since Coordinator::run_primary uses self.tools directly, we construct the
    // full registry ourselves and run the agent loop manually.

    let primary = primary_config();
    let session = graphirm_agent::Session::new(graph.clone(), primary.clone()).unwrap();
    session
        .add_user_message("Review the auth module and give me a summary.")
        .unwrap();

    let delegate_tool = SubagentTool::new(
        graph.clone(),
        registry.clone(),
        factory.clone(),
        base_tools.clone(),
        events.clone(),
        session.id.clone(),
    );

    let mut tool_registry = ToolRegistry::new();
    tool_registry.register(Arc::new(delegate_tool));

    let llm = (factory)(&primary.model);
    let cancel = CancellationToken::new();

    graphirm_agent::run_agent_loop(&session, llm.as_ref(), &tool_registry, &events, &cancel)
        .await
        .unwrap();

    // ---- Verify the graph structure ----

    // 1. Primary agent node exists
    let agent_node = graph.get_node(&session.id).unwrap();
    assert!(matches!(agent_node.node_type, NodeType::Agent(_)));

    // 2. Primary agent delegated a task
    let delegated = graph
        .neighbors(&session.id, Some(EdgeType::DelegatesTo), Direction::Outgoing)
        .unwrap();
    assert_eq!(delegated.len(), 1, "Primary should have delegated 1 task");

    let task_node = &delegated[0];
    assert!(matches!(task_node.node_type, NodeType::Task(_)));
    if let NodeType::Task(data) = &task_node.node_type {
        assert_eq!(data.status, "completed");
    }

    // 3. Task spawned a subagent
    let spawned = graph
        .neighbors(&task_node.id, Some(EdgeType::SpawnedBy), Direction::Outgoing)
        .unwrap();
    assert_eq!(spawned.len(), 1, "Task should have spawned 1 subagent");

    let subagent = &spawned[0];
    assert!(matches!(subagent.node_type, NodeType::Agent(_)));

    // 4. Subagent produced output (assistant responses)
    let subagent_outputs = graph
        .neighbors(&subagent.id, Some(EdgeType::Produces), Direction::Outgoing)
        .unwrap();
    let assistant_responses: Vec<_> = subagent_outputs
        .iter()
        .filter(|n| {
            matches!(&n.node_type, NodeType::Interaction(d) if d.role == "assistant")
        })
        .collect();
    assert!(
        !assistant_responses.is_empty(),
        "Subagent should have produced at least one assistant response"
    );

    // 5. Primary agent produced a final response
    let primary_outputs = graph
        .neighbors(&session.id, Some(EdgeType::Produces), Direction::Outgoing)
        .unwrap();
    let primary_responses: Vec<_> = primary_outputs
        .iter()
        .filter(|n| {
            matches!(&n.node_type, NodeType::Interaction(d) if d.role == "assistant")
        })
        .collect();
    assert!(
        primary_responses.len() >= 2,
        "Primary should have at least 2 assistant responses (delegate + final)"
    );

    // 6. Verify the graph has the full delegation chain
    // Parent Agent --DelegatesTo--> Task --SpawnedBy--> Subagent --Produces--> Results
    let chain = graph
        .traverse(&session.id, &[EdgeType::DelegatesTo, EdgeType::SpawnedBy], 3)
        .unwrap();
    assert!(
        chain.len() >= 2,
        "Delegation chain should include task + subagent"
    );
}

#[tokio::test]
async fn test_multi_agent_graph_isolation() {
    let graph = Arc::new(GraphStore::open_memory().unwrap());

    let mut configs = HashMap::new();
    configs.insert("build".to_string(), primary_config());
    configs.insert("explore".to_string(), explore_config());
    let registry = Arc::new(AgentRegistry::from_configs(configs));

    // The primary has a 3-message conversation before delegating
    let factory: LlmFactory = Arc::new(|model: &str| -> Box<dyn LlmProvider> {
        Box::new(MockProvider::new(vec![MockResponse::text("done")]))
    });

    let base_tools = Arc::new(ToolRegistry::new());
    let events = Arc::new(EventBus::new());

    // Create primary session with 3 user messages
    let primary = primary_config();
    let session = graphirm_agent::Session::new(graph.clone(), primary.clone()).unwrap();
    session.add_user_message("Message 1").unwrap();
    session.add_user_message("Message 2").unwrap();
    session.add_user_message("Message 3").unwrap();

    // Spawn a subagent
    let cancel = CancellationToken::new();
    let handle = graphirm_agent::spawn_subagent(
        &graph,
        &registry,
        &factory,
        &base_tools,
        &events,
        &session.id,
        "explore",
        "Explore something",
        vec![],
        cancel,
    )
    .await
    .unwrap();

    let subagent_id = handle.agent_id.clone();
    graphirm_agent::wait_for_subagents(vec![handle]).await.unwrap();

    // Verify: subagent's context should NOT include the parent's 3 messages
    let subagent_messages = graph
        .neighbors(&subagent_id, Some(EdgeType::Produces), Direction::Outgoing)
        .unwrap();

    // Subagent should have its own messages (task description + response),
    // NOT the parent's 3 messages
    let user_messages: Vec<_> = subagent_messages
        .iter()
        .filter(|n| {
            matches!(&n.node_type, NodeType::Interaction(d) if d.role == "user")
        })
        .collect();

    // The subagent should have 1 user message (the task description),
    // NOT 3 (the parent's messages)
    assert_eq!(
        user_messages.len(),
        1,
        "Subagent should only see its task, not parent conversation"
    );
}
```

### Step 2: Verify

Run: `cargo test -p graphirm-agent --test multi_agent_integration 2>&1`
Expected: `test result: ok. 2 passed; 0 failed`

### Step 3: Run full test suite

Run: `cargo test -p graphirm-agent 2>&1`
Expected: All tests pass (config, event, session, context, workflow, multi, delegate, integration).

### Step 4: Run clippy

Run: `cargo clippy -p graphirm-agent --all-targets 2>&1`
Expected: No errors.

### Step 5: Commit

```bash
git add crates/agent/tests/multi_agent_integration.rs
git commit -m "test(agent): add full multi-agent integration tests — delegation, isolation, graph structure"
```

---

## Final: Update lib.rs with all Phase 5 re-exports

- [x] Complete

**Files:**
- Modify: `crates/agent/src/lib.rs`

Ensure `crates/agent/src/lib.rs` has all public re-exports from Phases 4 + 5:

```rust
pub mod config;
pub mod context;
pub mod delegate;
pub mod error;
pub mod event;
pub mod multi;
pub mod session;
pub mod workflow;

// Phase 4 re-exports
pub use config::AgentConfig;
pub use error::AgentError;
pub use event::{AgentEvent, EventBus};
pub use session::Session;
pub use workflow::run_agent_loop;

// Phase 5 re-exports
pub use config::{AgentMode, Permission};
pub use context::build_subagent_context;
pub use delegate::SubagentTool;
pub use multi::{
    AgentRegistry, Coordinator, LlmFactory, SubagentHandle,
    collect_subagent_results, spawn_subagent, wait_for_dependencies, wait_for_subagents,
};
```

### Verify full build

Run: `cargo build -p graphirm-agent 2>&1`
Expected: Compiles with no errors.

Run: `cargo test -p graphirm-agent 2>&1`
Expected: All tests pass.

Run: `cargo clippy -p graphirm-agent --all-targets 2>&1`
Expected: No errors.

### Commit

```bash
git add crates/agent/src/lib.rs
git commit -m "feat(agent): Phase 5 complete — multi-agent coordinator with delegation, scoped context, and graph tracking"
```

---

## Summary

| Task | Description | Files | Tests |
|------|-------------|-------|-------|
| 1 | AgentMode, Permission, extend AgentConfig | config.rs | 7 |
| 2 | AgentRegistry | multi.rs | 5 |
| 3 | build_subagent_context | context.rs | 3 |
| 4 | SubagentHandle + spawn_subagent | multi.rs, error.rs | 2 |
| 5 | wait_for_subagents | multi.rs | 1 |
| 6 | Coordinator + run_primary | multi.rs | 2 |
| 7 | SubagentTool (delegate) | delegate.rs | 3 |
| 8 | Task dependency tracking | multi.rs | 1 |
| 9 | Parallel subagent execution | multi.rs (test) | 1 |
| 10 | Result merging | multi.rs | 1 |
| 11 | Integration test | tests/multi_agent_integration.rs | 2 |
| **Total** | | **5 source files + 1 test file** | **~28 tests** |

---

## Pre-Requisite: AgentStatus Enum (from Phase 4 code review)

Before implementing Phase 5's multi-agent status tracking, migrate `AgentData.status` from a bare `String` to a typed enum. This makes status values exhaustive and prevents typos across coordinator and subagent code.

**Location:** `graphirm-graph/src/nodes.rs` → `AgentData` struct

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgentStatus {
    Active,
    Completed,
    Cancelled,
    LimitReached,
    Failed(String),
}

impl Default for AgentStatus {
    fn default() -> Self { AgentStatus::Active }
}
```

- Replace `status: String` with `status: AgentStatus` in `AgentData`.
- Update `Session::new` to use `AgentStatus::Active`.
- Update `Session::set_status(&str)` to accept `AgentStatus` directly (remove string-based API).
- Update all call sites in `workflow.rs` (`"completed"`, `"cancelled"`, `"limit_reached"`).
- Update `Session::set_status` test and `test_agent_loop_*` tests.

---

## Graph Structure Reference

After Phase 5, multi-agent interactions produce this graph shape:

```text
Human message
  └─Produces── Primary Agent (build)
                 ├── Produces → Assistant response (delegation decision)
                 ├── Produces → Tool result (delegate output)
                 ├── Produces → Assistant response (final answer)
                 └── DelegatesTo → Task: "Analyze auth"
                                    ├── status: "completed"
                                    └── SpawnedBy → Explore Subagent
                                                     ├── Produces → User msg (task description)
                                                     ├── Produces → Assistant response (findings)
                                                     └── Produces → Knowledge: "auth patterns"
```

## Adaptation Notes

- **Phase 1 type differences:** Phase 1 uses rich `NodeType` with data structs (e.g., `NodeType::Task(TaskData {...})`), while Phase 4 used simplified `serde_json::Value` payloads. This plan uses Phase 1's rich types. If Phase 4's code was implemented with the simplified types, adapt the graph node creation to match.
- **Phase 2 LlmProvider trait:** Phase 2's trait has 3 methods (`complete`, `stream`, `provider_name`) and `complete()` takes `config: &CompletionConfig`. Phase 4's test helpers used a simplified 2-arg version. If Phase 4 was implemented with simplified types, update `run_agent_loop` to pass a `CompletionConfig` from `AgentConfig`.
- **EventBus thread safety:** Phase 4's `EventBus::subscribe()` takes `&mut self`, which conflicts with `Arc<EventBus>`. Phase 5 wraps EventBus in Arc. If needed, update EventBus to use interior mutability (`RwLock<Vec<Sender>>`) so `subscribe()` takes `&self`.
- **ToolRegistry::clone():** Phase 5's `build_scoped_tools` creates a new ToolRegistry from a subset. This requires `ToolRegistry::new()` and `register()` to work, which Phase 3 provides. If ToolRegistry doesn't implement Clone, that's fine — we build a new one from scratch.
- **MockProvider location:** Tests reference `graphirm_llm::MockProvider` from Phase 2. If Phase 2's MockProvider API differs (e.g., `MockResponse::text()` vs `text_response()`), adapt the test helper functions accordingly.
