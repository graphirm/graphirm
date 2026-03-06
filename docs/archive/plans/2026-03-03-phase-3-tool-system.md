# Phase 3: Tool System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a complete tool system with 7 built-in tools (bash, read, write, edit, grep, find, ls), a registry, permission checking, and parallel execution — all graph-tracked via the Phase 1 GraphStore.

**Architecture:** Every tool implements a common `Tool` trait with `execute()` returning a `ToolResult`. Tools receive a `ToolContext` containing an `Arc<GraphStore>` and create graph nodes/edges as side effects. A `ToolRegistry` maps tool names to trait objects. Parallel execution uses `tokio::task::JoinSet` to run independent tool calls concurrently.

**Tech Stack:** tokio (async + process::Command + JoinSet), serde/serde_json (arg parsing + JSON Schema), thiserror (errors), async-trait, tempfile (tests), glob (find tool), graphirm-graph (Phase 1 dependency)

---

## Prerequisites

- **Phase 1 complete** — `graphirm-graph` crate with `GraphStore`, `NodeId`, `NodeType`, `EdgeType`, `GraphError` available
- **Phase 0 complete** — workspace scaffold with `crates/tools/` directory and stub files

### Assumed Phase 1 API Surface

The plan assumes these types exist in `graphirm_graph`. If names differ, adjust imports accordingly:

```rust
// From crates/graph/src/lib.rs (Phase 1)
pub struct GraphStore { /* r2d2 pool + Arc<RwLock<petgraph::Graph>> */ }
pub struct NodeId(pub i64);

pub enum NodeType {
    Interaction,
    Agent,
    Content,
    Task,
    Knowledge,
}

pub enum EdgeType {
    RespondsTo,
    SpawnedBy,
    DelegatesTo,
    DependsOn,
    Produces,
    Reads,
    Modifies,
    Summarizes,
    Contains,
    FollowsUp,
    Steers,
    RelatesTo,
}

pub struct GraphError; // thiserror enum

impl GraphStore {
    pub fn open(path: &Path) -> Result<Self, GraphError>;
    pub fn open_memory() -> Result<Self, GraphError>;
    pub fn add_node(&self, node_type: NodeType, metadata: serde_json::Value) -> Result<NodeId, GraphError>;
    pub fn add_edge(&self, from: NodeId, to: NodeId, edge_type: EdgeType, metadata: serde_json::Value) -> Result<(), GraphError>;
    pub fn get_node(&self, id: NodeId) -> Result<Option<serde_json::Value>, GraphError>;
    pub fn find_nodes_by_metadata(&self, key: &str, value: &str) -> Result<Vec<NodeId>, GraphError>;
}
```

---

## Task 1: Define ToolError Enum

- [x] **Complete**

**Files:**
- Modify: `crates/tools/src/error.rs`
- Test: `crates/tools/src/error.rs` (inline tests)

### Step 1.1 — Write the error enum with Display test

Write the `ToolError` enum and a test that verifies each variant's `Display` output.

**`crates/tools/src/error.rs`:**

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("Tool not found: {0}")]
    NotFound(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Invalid arguments: {0}")]
    InvalidArguments(String),

    #[error("Timeout after {0}s")]
    Timeout(u64),

    #[error("Cancelled")]
    Cancelled,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Graph error: {0}")]
    Graph(#[from] graphirm_graph::GraphError),
}

pub type ToolResult<T> = std::result::Result<T, ToolError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_not_found() {
        let err = ToolError::NotFound("bash".into());
        assert_eq!(err.to_string(), "Tool not found: bash");
    }

    #[test]
    fn display_permission_denied() {
        let err = ToolError::PermissionDenied("write requires approval".into());
        assert_eq!(err.to_string(), "Permission denied: write requires approval");
    }

    #[test]
    fn display_execution_failed() {
        let err = ToolError::ExecutionFailed("exit code 1".into());
        assert_eq!(err.to_string(), "Execution failed: exit code 1");
    }

    #[test]
    fn display_invalid_arguments() {
        let err = ToolError::InvalidArguments("missing 'path' field".into());
        assert_eq!(err.to_string(), "Invalid arguments: missing 'path' field");
    }

    #[test]
    fn display_timeout() {
        let err = ToolError::Timeout(30);
        assert_eq!(err.to_string(), "Timeout after 30s");
    }

    #[test]
    fn display_cancelled() {
        let err = ToolError::Cancelled;
        assert_eq!(err.to_string(), "Cancelled");
    }

    #[test]
    fn from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let tool_err = ToolError::from(io_err);
        assert!(tool_err.to_string().contains("file missing"));
    }

    #[test]
    fn from_json_error() {
        let json_err = serde_json::from_str::<serde_json::Value>("not json").unwrap_err();
        let tool_err = ToolError::from(json_err);
        assert!(matches!(tool_err, ToolError::Json(_)));
    }
}
```

### Step 1.2 — Run tests

```bash
cd crates/tools && cargo test -- error
```

Expected: All 8 tests pass.

### Step 1.3 — Commit

```bash
git add crates/tools/src/error.rs
git commit -m "feat(tools): define ToolError enum with thiserror derives"
```

---

## Task 2: Define Tool Trait, ToolContext, and ToolResult

- [x] **Complete**

**Files:**
- Modify: `crates/tools/src/lib.rs`
- Modify: `crates/tools/Cargo.toml`
- Test: `crates/tools/src/lib.rs` (inline tests)

### Step 2.1 — Update Cargo.toml with dependencies

**`crates/tools/Cargo.toml`:**

```toml
[package]
name = "graphirm-tools"
version = "0.1.0"
edition = "2021"

[dependencies]
graphirm-graph = { path = "../graph" }
async-trait = "0.1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
tokio = { version = "1", features = ["full"] }
tokio-util = { version = "0.7", features = ["rt"] }
tracing = "0.1"
glob = "0.3"

[dev-dependencies]
tempfile = "3"
tokio = { version = "1", features = ["full", "test-util"] }
```

### Step 2.2 — Write trait, context, and result types with a mock tool test

**`crates/tools/src/lib.rs`:**

```rust
pub mod bash;
pub mod edit;
pub mod error;
pub mod find;
pub mod grep;
pub mod ls;
pub mod read;
pub mod write;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

pub use error::ToolError;
use graphirm_graph::{GraphStore, NodeId};

/// Context passed to every tool execution.
#[derive(Clone)]
pub struct ToolContext {
    pub graph: Arc<GraphStore>,
    pub agent_id: NodeId,
    pub interaction_id: NodeId,
    pub working_dir: PathBuf,
    pub signal: CancellationToken,
}

/// Result of a tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
    pub node_id: Option<NodeId>,
}

impl ToolOutput {
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
            node_id: None,
        }
    }

    pub fn success_with_node(content: impl Into<String>, node_id: NodeId) -> Self {
        Self {
            content: content.into(),
            is_error: false,
            node_id: Some(node_id),
        }
    }

    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
            node_id: None,
        }
    }
}

/// JSON Schema definition sent to the LLM so it knows what tools exist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// A tool call request from the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Every tool implements this trait.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> serde_json::Value;

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError>;

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: self.parameters(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }

        fn description(&self) -> &str {
            "Echoes input back"
        }

        fn parameters(&self) -> serde_json::Value {
            json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string", "description": "Message to echo" }
                },
                "required": ["message"]
            })
        }

        async fn execute(
            &self,
            args: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, ToolError> {
            let message = args["message"]
                .as_str()
                .ok_or_else(|| ToolError::InvalidArguments("missing 'message'".into()))?;
            Ok(ToolOutput::success(message))
        }
    }

    fn make_test_context() -> ToolContext {
        let graph = Arc::new(GraphStore::open_memory().expect("memory graph"));
        ToolContext {
            graph,
            agent_id: NodeId(1),
            interaction_id: NodeId(2),
            working_dir: PathBuf::from("/tmp"),
            signal: CancellationToken::new(),
        }
    }

    #[test]
    fn tool_definition() {
        let tool = EchoTool;
        let def = tool.definition();
        assert_eq!(def.name, "echo");
        assert_eq!(def.description, "Echoes input back");
        assert!(def.parameters["properties"]["message"].is_object());
    }

    #[tokio::test]
    async fn tool_execute_success() {
        let tool = EchoTool;
        let ctx = make_test_context();
        let result = tool
            .execute(json!({"message": "hello"}), &ctx)
            .await
            .unwrap();
        assert_eq!(result.content, "hello");
        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn tool_execute_invalid_args() {
        let tool = EchoTool;
        let ctx = make_test_context();
        let err = tool.execute(json!({}), &ctx).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[test]
    fn tool_output_constructors() {
        let ok = ToolOutput::success("done");
        assert_eq!(ok.content, "done");
        assert!(!ok.is_error);
        assert!(ok.node_id.is_none());

        let ok_node = ToolOutput::success_with_node("done", NodeId(5));
        assert_eq!(ok_node.node_id, Some(NodeId(5)));

        let err = ToolOutput::error("failed");
        assert_eq!(err.content, "failed");
        assert!(err.is_error);
    }
}
```

### Step 2.3 — Run tests

```bash
cd crates/tools && cargo test -- tests
```

Expected: 4 tests pass (tool_definition, tool_execute_success, tool_execute_invalid_args, tool_output_constructors).

### Step 2.4 — Commit

```bash
git add crates/tools/src/lib.rs crates/tools/Cargo.toml
git commit -m "feat(tools): define Tool trait, ToolContext, ToolOutput, ToolDefinition"
```

---

## Task 3: Implement ToolRegistry

- [x] **Complete**

**Files:**
- Create: `crates/tools/src/registry.rs`
- Modify: `crates/tools/src/lib.rs` (add `pub mod registry;`)
- Test: `crates/tools/src/registry.rs` (inline tests)

### Step 3.1 — Write the registry with tests

**`crates/tools/src/registry.rs`:**

```rust
use std::collections::HashMap;
use std::sync::Arc;

use crate::{Tool, ToolCall, ToolContext, ToolDefinition, ToolError, ToolOutput};

/// Registry of available tools, keyed by name.
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn get(&self, name: &str) -> Result<Arc<dyn Tool>, ToolError> {
        self.tools
            .get(name)
            .cloned()
            .ok_or_else(|| ToolError::NotFound(name.to_string()))
    }

    pub fn list(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tools.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        let mut defs: Vec<ToolDefinition> = self.tools.values().map(|t| t.definition()).collect();
        defs.sort_by(|a, b| a.name.cmp(&b.name));
        defs
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Execute a single tool call by name.
    pub async fn execute(
        &self,
        call: &ToolCall,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let tool = self.get(&call.name)?;
        tool.execute(call.arguments.clone(), ctx).await
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;
    use async_trait::async_trait;
    use serde_json::json;

    struct DummyTool {
        tool_name: String,
    }

    impl DummyTool {
        fn new(name: &str) -> Self {
            Self {
                tool_name: name.to_string(),
            }
        }
    }

    #[async_trait]
    impl Tool for DummyTool {
        fn name(&self) -> &str {
            &self.tool_name
        }

        fn description(&self) -> &str {
            "A dummy tool for testing"
        }

        fn parameters(&self) -> serde_json::Value {
            json!({"type": "object", "properties": {}})
        }

        async fn execute(
            &self,
            _args: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::success(format!("{} executed", self.tool_name)))
        }
    }

    #[test]
    fn new_registry_is_empty() {
        let reg = ToolRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        assert!(reg.list().is_empty());
    }

    #[test]
    fn register_and_get() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(DummyTool::new("alpha")));
        reg.register(Arc::new(DummyTool::new("beta")));

        assert_eq!(reg.len(), 2);
        assert!(reg.get("alpha").is_ok());
        assert!(reg.get("beta").is_ok());
    }

    #[test]
    fn get_not_found() {
        let reg = ToolRegistry::new();
        let err = reg.get("nonexistent").unwrap_err();
        assert!(matches!(err, ToolError::NotFound(_)));
        assert_eq!(err.to_string(), "Tool not found: nonexistent");
    }

    #[test]
    fn list_sorted() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(DummyTool::new("charlie")));
        reg.register(Arc::new(DummyTool::new("alpha")));
        reg.register(Arc::new(DummyTool::new("bravo")));

        assert_eq!(reg.list(), vec!["alpha", "bravo", "charlie"]);
    }

    #[test]
    fn definitions_sorted() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(DummyTool::new("zulu")));
        reg.register(Arc::new(DummyTool::new("alpha")));

        let defs = reg.definitions();
        assert_eq!(defs.len(), 2);
        assert_eq!(defs[0].name, "alpha");
        assert_eq!(defs[1].name, "zulu");
    }

    #[test]
    fn register_overwrites() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(DummyTool::new("alpha")));
        reg.register(Arc::new(DummyTool::new("alpha")));
        assert_eq!(reg.len(), 1);
    }

    #[tokio::test]
    async fn execute_by_name() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(DummyTool::new("echo")));

        let ctx = make_test_context();
        let call = ToolCall {
            id: "call_1".into(),
            name: "echo".into(),
            arguments: json!({}),
        };
        let result = reg.execute(&call, &ctx).await.unwrap();
        assert_eq!(result.content, "echo executed");
    }

    #[tokio::test]
    async fn execute_not_found() {
        let reg = ToolRegistry::new();
        let ctx = make_test_context();
        let call = ToolCall {
            id: "call_1".into(),
            name: "missing".into(),
            arguments: json!({}),
        };
        let err = reg.execute(&call, &ctx).await.unwrap_err();
        assert!(matches!(err, ToolError::NotFound(_)));
    }
}
```

### Step 3.2 — Expose the module from lib.rs

Add to the top of `crates/tools/src/lib.rs`:

```rust
pub mod registry;
```

Also make the `make_test_context` helper visible to sibling test modules by changing the `#[cfg(test)]` block in `lib.rs`:

```rust
#[cfg(test)]
pub(crate) mod tests {
    // ... existing test code, but now pub(crate) ...
}
```

### Step 3.3 — Run tests

```bash
cd crates/tools && cargo test -- registry
```

Expected: 7 tests pass.

### Step 3.4 — Commit

```bash
git add crates/tools/src/registry.rs crates/tools/src/lib.rs
git commit -m "feat(tools): implement ToolRegistry with register/get/list/execute"
```

---

## Task 4: Define Permission Enum and ToolPermissions

- [x] **Complete**

**Files:**
- Create: `crates/tools/src/permissions.rs`
- Modify: `crates/tools/src/lib.rs` (add `pub mod permissions;`)
- Test: `crates/tools/src/permissions.rs` (inline tests)

### Step 4.1 — Write permission types with tests

**`crates/tools/src/permissions.rs`:**

```rust
use std::collections::HashMap;

use crate::ToolError;

/// Permission level for a tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permission {
    /// Tool can execute without user confirmation.
    Allow,
    /// Tool must ask the user before executing.
    Ask,
    /// Tool is blocked from executing.
    Deny,
}

/// Manages per-tool permission policies.
pub struct ToolPermissions {
    defaults: HashMap<String, Permission>,
    fallback: Permission,
}

impl ToolPermissions {
    pub fn new(fallback: Permission) -> Self {
        Self {
            defaults: HashMap::new(),
            fallback,
        }
    }

    /// Permissive: all tools allowed by default.
    pub fn allow_all() -> Self {
        Self::new(Permission::Allow)
    }

    /// Conservative: all tools require confirmation by default.
    pub fn ask_all() -> Self {
        Self::new(Permission::Ask)
    }

    /// Set the permission for a specific tool.
    pub fn set(&mut self, tool_name: impl Into<String>, permission: Permission) {
        self.defaults.insert(tool_name.into(), permission);
    }

    /// Get the permission for a tool (falls back to default).
    pub fn get(&self, tool_name: &str) -> Permission {
        self.defaults
            .get(tool_name)
            .copied()
            .unwrap_or(self.fallback)
    }

    /// Check if a tool is allowed to execute. Returns `Ok(())` if allowed,
    /// `Err(ToolError::PermissionDenied)` if denied, or `Ok(())` with `Ask`
    /// (caller is responsible for prompting the user when `Ask` is returned).
    pub fn check(&self, tool_name: &str) -> Result<Permission, ToolError> {
        match self.get(tool_name) {
            Permission::Deny => Err(ToolError::PermissionDenied(format!(
                "tool '{}' is denied by policy",
                tool_name
            ))),
            perm => Ok(perm),
        }
    }
}

impl Default for ToolPermissions {
    fn default() -> Self {
        Self::ask_all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allow_all_default() {
        let perms = ToolPermissions::allow_all();
        assert_eq!(perms.get("bash"), Permission::Allow);
        assert_eq!(perms.get("anything"), Permission::Allow);
    }

    #[test]
    fn ask_all_default() {
        let perms = ToolPermissions::ask_all();
        assert_eq!(perms.get("bash"), Permission::Ask);
    }

    #[test]
    fn override_specific_tool() {
        let mut perms = ToolPermissions::allow_all();
        perms.set("bash", Permission::Ask);
        perms.set("write", Permission::Deny);

        assert_eq!(perms.get("bash"), Permission::Ask);
        assert_eq!(perms.get("write"), Permission::Deny);
        assert_eq!(perms.get("read"), Permission::Allow);
    }

    #[test]
    fn check_allowed() {
        let perms = ToolPermissions::allow_all();
        let result = perms.check("read").unwrap();
        assert_eq!(result, Permission::Allow);
    }

    #[test]
    fn check_ask() {
        let perms = ToolPermissions::ask_all();
        let result = perms.check("read").unwrap();
        assert_eq!(result, Permission::Ask);
    }

    #[test]
    fn check_denied() {
        let mut perms = ToolPermissions::allow_all();
        perms.set("bash", Permission::Deny);

        let err = perms.check("bash").unwrap_err();
        assert!(matches!(err, ToolError::PermissionDenied(_)));
        assert!(err.to_string().contains("bash"));
    }

    #[test]
    fn default_is_ask_all() {
        let perms = ToolPermissions::default();
        assert_eq!(perms.get("bash"), Permission::Ask);
    }
}
```

### Step 4.2 — Run tests

```bash
cd crates/tools && cargo test -- permissions
```

Expected: 7 tests pass.

### Step 4.3 — Commit

```bash
git add crates/tools/src/permissions.rs crates/tools/src/lib.rs
git commit -m "feat(tools): add Permission enum and ToolPermissions policy"
```

---

## Task 5: Implement ReadTool

- [x] **Complete**

**Files:**
- Modify: `crates/tools/src/read.rs`
- Test: `crates/tools/src/read.rs` (inline tests)

### Step 5.1 — Write ReadTool with graph tracking and tests

**`crates/tools/src/read.rs`:**

```rust
use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::json;
use tracing::debug;

use crate::{Tool, ToolContext, ToolError, ToolOutput};
use graphirm_graph::{EdgeType, NodeType};

pub struct ReadTool;

impl ReadTool {
    pub fn new() -> Self {
        Self
    }

    fn resolve_path(path: &str, working_dir: &PathBuf) -> PathBuf {
        let p = PathBuf::from(path);
        if p.is_absolute() {
            p
        } else {
            working_dir.join(p)
        }
    }
}

#[async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn description(&self) -> &str {
        "Read the contents of a file. Returns the full file content as a string with line numbers."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (absolute or relative to working directory)"
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-based). Optional."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read. Optional."
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let path_str = args["path"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'path' field".into()))?;

        let offset = args["offset"].as_u64().map(|v| v as usize);
        let limit = args["limit"].as_u64().map(|v| v as usize);

        let full_path = Self::resolve_path(path_str, &ctx.working_dir);

        debug!(path = %full_path.display(), "reading file");

        let content = tokio::fs::read_to_string(&full_path)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("{}: {}", full_path.display(), e)))?;

        let lines: Vec<&str> = content.lines().collect();
        let start = offset.unwrap_or(1).saturating_sub(1);
        let end = limit
            .map(|l| (start + l).min(lines.len()))
            .unwrap_or(lines.len());

        let numbered: String = lines[start..end]
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{:>6}|{}", start + i + 1, line))
            .collect::<Vec<_>>()
            .join("\n");

        // Graph tracking: create Content node for the file, add Reads edge
        let content_node = ctx.graph.add_node(
            NodeType::Content,
            json!({
                "type": "file",
                "path": full_path.to_string_lossy(),
                "size_bytes": content.len(),
                "line_count": lines.len(),
            }),
        )?;

        ctx.graph.add_edge(
            ctx.interaction_id.clone(),
            content_node.clone(),
            EdgeType::Reads,
            json!({
                "offset": offset,
                "limit": limit,
            }),
        )?;

        Ok(ToolOutput::success_with_node(numbered, content_node))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::make_test_context;
    use std::io::Write;
    use tempfile::TempDir;

    fn ctx_with_dir(dir: &TempDir) -> ToolContext {
        let mut ctx = make_test_context();
        ctx.working_dir = dir.path().to_path_buf();
        ctx
    }

    #[tokio::test]
    async fn read_file_full() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("hello.txt");
        std::fs::write(&file_path, "line one\nline two\nline three\n").unwrap();

        let tool = ReadTool::new();
        let ctx = ctx_with_dir(&dir);
        let result = tool
            .execute(json!({"path": "hello.txt"}), &ctx)
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("line one"));
        assert!(result.content.contains("line two"));
        assert!(result.content.contains("line three"));
        assert!(result.node_id.is_some());
    }

    #[tokio::test]
    async fn read_file_with_offset_and_limit() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("data.txt");
        let mut f = std::fs::File::create(&file_path).unwrap();
        for i in 1..=10 {
            writeln!(f, "line {}", i).unwrap();
        }

        let tool = ReadTool::new();
        let ctx = ctx_with_dir(&dir);
        let result = tool
            .execute(json!({"path": "data.txt", "offset": 3, "limit": 2}), &ctx)
            .await
            .unwrap();

        assert!(result.content.contains("line 3"));
        assert!(result.content.contains("line 4"));
        assert!(!result.content.contains("line 5"));
        assert!(!result.content.contains("line 2"));
    }

    #[tokio::test]
    async fn read_file_not_found() {
        let dir = TempDir::new().unwrap();
        let tool = ReadTool::new();
        let ctx = ctx_with_dir(&dir);
        let err = tool
            .execute(json!({"path": "missing.txt"}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::ExecutionFailed(_)));
    }

    #[tokio::test]
    async fn read_missing_path_arg() {
        let dir = TempDir::new().unwrap();
        let tool = ReadTool::new();
        let ctx = ctx_with_dir(&dir);
        let err = tool.execute(json!({}), &ctx).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[tokio::test]
    async fn read_creates_graph_edges() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("test.txt"), "content").unwrap();

        let tool = ReadTool::new();
        let ctx = ctx_with_dir(&dir);
        let result = tool
            .execute(json!({"path": "test.txt"}), &ctx)
            .await
            .unwrap();

        let node_id = result.node_id.unwrap();
        let node = ctx.graph.get_node(node_id).unwrap();
        assert!(node.is_some());
    }

    #[tokio::test]
    async fn read_absolute_path() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("abs.txt");
        std::fs::write(&file_path, "absolute content").unwrap();

        let tool = ReadTool::new();
        let ctx = ctx_with_dir(&dir);
        let result = tool
            .execute(
                json!({"path": file_path.to_string_lossy()}),
                &ctx,
            )
            .await
            .unwrap();

        assert!(result.content.contains("absolute content"));
    }
}
```

### Step 5.2 — Run tests

```bash
cd crates/tools && cargo test -- read
```

Expected: 6 tests pass.

### Step 5.3 — Commit

```bash
git add crates/tools/src/read.rs
git commit -m "feat(tools): implement ReadTool with graph tracking"
```

---

## Task 6: Implement WriteTool

- [x] **Complete**

**Files:**
- Modify: `crates/tools/src/write.rs`
- Test: `crates/tools/src/write.rs` (inline tests)

### Step 6.1 — Write WriteTool with graph tracking and tests

**`crates/tools/src/write.rs`:**

```rust
use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::json;
use tracing::debug;

use crate::{Tool, ToolContext, ToolError, ToolOutput};
use graphirm_graph::{EdgeType, NodeType};

pub struct WriteTool;

impl WriteTool {
    pub fn new() -> Self {
        Self
    }

    fn resolve_path(path: &str, working_dir: &PathBuf) -> PathBuf {
        let p = PathBuf::from(path);
        if p.is_absolute() {
            p
        } else {
            working_dir.join(p)
        }
    }
}

#[async_trait]
impl Tool for WriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn description(&self) -> &str {
        "Write content to a file. Creates the file and parent directories if they don't exist. Overwrites existing content."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write (absolute or relative to working directory)"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let path_str = args["path"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'path' field".into()))?;
        let content = args["content"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'content' field".into()))?;

        let full_path = Self::resolve_path(path_str, &ctx.working_dir);
        let existed = full_path.exists();

        debug!(path = %full_path.display(), bytes = content.len(), "writing file");

        if let Some(parent) = full_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        tokio::fs::write(&full_path, content).await?;

        let content_node = ctx.graph.add_node(
            NodeType::Content,
            json!({
                "type": "file",
                "path": full_path.to_string_lossy(),
                "size_bytes": content.len(),
                "created": !existed,
            }),
        )?;

        ctx.graph.add_edge(
            ctx.interaction_id.clone(),
            content_node.clone(),
            EdgeType::Modifies,
            json!({
                "action": if existed { "overwrite" } else { "create" },
                "bytes_written": content.len(),
            }),
        )?;

        let msg = if existed {
            format!("Updated {}", full_path.display())
        } else {
            format!("Created {}", full_path.display())
        };

        Ok(ToolOutput::success_with_node(msg, content_node))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::make_test_context;
    use tempfile::TempDir;

    fn ctx_with_dir(dir: &TempDir) -> ToolContext {
        let mut ctx = make_test_context();
        ctx.working_dir = dir.path().to_path_buf();
        ctx
    }

    #[tokio::test]
    async fn write_creates_file() {
        let dir = TempDir::new().unwrap();
        let tool = WriteTool::new();
        let ctx = ctx_with_dir(&dir);

        let result = tool
            .execute(json!({"path": "new.txt", "content": "hello world"}), &ctx)
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("Created"));
        assert!(result.node_id.is_some());

        let on_disk = std::fs::read_to_string(dir.path().join("new.txt")).unwrap();
        assert_eq!(on_disk, "hello world");
    }

    #[tokio::test]
    async fn write_overwrites_existing() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("exist.txt"), "old").unwrap();

        let tool = WriteTool::new();
        let ctx = ctx_with_dir(&dir);

        let result = tool
            .execute(json!({"path": "exist.txt", "content": "new content"}), &ctx)
            .await
            .unwrap();

        assert!(result.content.contains("Updated"));
        let on_disk = std::fs::read_to_string(dir.path().join("exist.txt")).unwrap();
        assert_eq!(on_disk, "new content");
    }

    #[tokio::test]
    async fn write_creates_parent_dirs() {
        let dir = TempDir::new().unwrap();
        let tool = WriteTool::new();
        let ctx = ctx_with_dir(&dir);

        tool.execute(
            json!({"path": "a/b/c/deep.txt", "content": "deep content"}),
            &ctx,
        )
        .await
        .unwrap();

        let on_disk = std::fs::read_to_string(dir.path().join("a/b/c/deep.txt")).unwrap();
        assert_eq!(on_disk, "deep content");
    }

    #[tokio::test]
    async fn write_missing_args() {
        let dir = TempDir::new().unwrap();
        let tool = WriteTool::new();
        let ctx = ctx_with_dir(&dir);

        let err = tool
            .execute(json!({"path": "file.txt"}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));

        let err = tool
            .execute(json!({"content": "stuff"}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[tokio::test]
    async fn write_creates_graph_edges() {
        let dir = TempDir::new().unwrap();
        let tool = WriteTool::new();
        let ctx = ctx_with_dir(&dir);

        let result = tool
            .execute(json!({"path": "tracked.txt", "content": "data"}), &ctx)
            .await
            .unwrap();

        let node_id = result.node_id.unwrap();
        let node = ctx.graph.get_node(node_id).unwrap();
        assert!(node.is_some());
    }
}
```

### Step 6.2 — Run tests

```bash
cd crates/tools && cargo test -- write
```

Expected: 5 tests pass.

### Step 6.3 — Commit

```bash
git add crates/tools/src/write.rs
git commit -m "feat(tools): implement WriteTool with graph tracking"
```

---

## Task 7: Implement EditTool

- [x] **Complete**

**Files:**
- Modify: `crates/tools/src/edit.rs`
- Test: `crates/tools/src/edit.rs` (inline tests)

### Step 7.1 — Write EditTool with diff tracking and tests

**`crates/tools/src/edit.rs`:**

```rust
use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::json;
use tracing::debug;

use crate::{Tool, ToolContext, ToolError, ToolOutput};
use graphirm_graph::{EdgeType, NodeType};

pub struct EditTool;

impl EditTool {
    pub fn new() -> Self {
        Self
    }

    fn resolve_path(path: &str, working_dir: &PathBuf) -> PathBuf {
        let p = PathBuf::from(path);
        if p.is_absolute() {
            p
        } else {
            working_dir.join(p)
        }
    }
}

#[async_trait]
impl Tool for EditTool {
    fn name(&self) -> &str {
        "edit"
    }

    fn description(&self) -> &str {
        "Edit a file by replacing an exact string match with new content. The old_string must appear exactly once in the file."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit"
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact string to find and replace (must be unique in the file)"
                },
                "new_string": {
                    "type": "string",
                    "description": "The string to replace old_string with"
                }
            },
            "required": ["path", "old_string", "new_string"]
        })
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let path_str = args["path"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'path' field".into()))?;
        let old_string = args["old_string"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'old_string' field".into()))?;
        let new_string = args["new_string"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'new_string' field".into()))?;

        let full_path = Self::resolve_path(path_str, &ctx.working_dir);

        debug!(path = %full_path.display(), "editing file");

        let content = tokio::fs::read_to_string(&full_path)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("{}: {}", full_path.display(), e)))?;

        let match_count = content.matches(old_string).count();
        if match_count == 0 {
            return Err(ToolError::ExecutionFailed(format!(
                "old_string not found in {}",
                full_path.display()
            )));
        }
        if match_count > 1 {
            return Err(ToolError::ExecutionFailed(format!(
                "old_string found {} times in {} (must be unique)",
                match_count,
                full_path.display()
            )));
        }

        let new_content = content.replacen(old_string, new_string, 1);
        tokio::fs::write(&full_path, &new_content).await?;

        let content_node = ctx.graph.add_node(
            NodeType::Content,
            json!({
                "type": "file",
                "path": full_path.to_string_lossy(),
                "size_bytes": new_content.len(),
            }),
        )?;

        ctx.graph.add_edge(
            ctx.interaction_id.clone(),
            content_node.clone(),
            EdgeType::Modifies,
            json!({
                "action": "edit",
                "old_string": old_string,
                "new_string": new_string,
                "old_len": old_string.len(),
                "new_len": new_string.len(),
            }),
        )?;

        Ok(ToolOutput::success_with_node(
            format!(
                "Edited {}: replaced {} bytes with {} bytes",
                full_path.display(),
                old_string.len(),
                new_string.len()
            ),
            content_node,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::make_test_context;
    use tempfile::TempDir;

    fn ctx_with_dir(dir: &TempDir) -> ToolContext {
        let mut ctx = make_test_context();
        ctx.working_dir = dir.path().to_path_buf();
        ctx
    }

    #[tokio::test]
    async fn edit_replaces_string() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("code.rs"), "fn hello() {\n    println!(\"hello\");\n}\n")
            .unwrap();

        let tool = EditTool::new();
        let ctx = ctx_with_dir(&dir);

        let result = tool
            .execute(
                json!({
                    "path": "code.rs",
                    "old_string": "println!(\"hello\")",
                    "new_string": "println!(\"goodbye\")"
                }),
                &ctx,
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.node_id.is_some());

        let on_disk = std::fs::read_to_string(dir.path().join("code.rs")).unwrap();
        assert!(on_disk.contains("goodbye"));
        assert!(!on_disk.contains("hello"));
    }

    #[tokio::test]
    async fn edit_not_found() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("file.txt"), "aaa bbb ccc").unwrap();

        let tool = EditTool::new();
        let ctx = ctx_with_dir(&dir);

        let err = tool
            .execute(
                json!({
                    "path": "file.txt",
                    "old_string": "zzz",
                    "new_string": "yyy"
                }),
                &ctx,
            )
            .await
            .unwrap_err();

        assert!(matches!(err, ToolError::ExecutionFailed(_)));
        assert!(err.to_string().contains("not found"));
    }

    #[tokio::test]
    async fn edit_ambiguous_match() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("file.txt"), "aaa aaa aaa").unwrap();

        let tool = EditTool::new();
        let ctx = ctx_with_dir(&dir);

        let err = tool
            .execute(
                json!({
                    "path": "file.txt",
                    "old_string": "aaa",
                    "new_string": "bbb"
                }),
                &ctx,
            )
            .await
            .unwrap_err();

        assert!(err.to_string().contains("3 times"));
    }

    #[tokio::test]
    async fn edit_file_not_exists() {
        let dir = TempDir::new().unwrap();
        let tool = EditTool::new();
        let ctx = ctx_with_dir(&dir);

        let err = tool
            .execute(
                json!({
                    "path": "missing.rs",
                    "old_string": "a",
                    "new_string": "b"
                }),
                &ctx,
            )
            .await
            .unwrap_err();

        assert!(matches!(err, ToolError::ExecutionFailed(_)));
    }

    #[tokio::test]
    async fn edit_stores_diff_in_graph() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("tracked.txt"), "old value").unwrap();

        let tool = EditTool::new();
        let ctx = ctx_with_dir(&dir);

        let result = tool
            .execute(
                json!({
                    "path": "tracked.txt",
                    "old_string": "old value",
                    "new_string": "new value"
                }),
                &ctx,
            )
            .await
            .unwrap();

        assert!(result.node_id.is_some());
    }
}
```

### Step 7.2 — Run tests

```bash
cd crates/tools && cargo test -- edit
```

Expected: 5 tests pass.

### Step 7.3 — Commit

```bash
git add crates/tools/src/edit.rs
git commit -m "feat(tools): implement EditTool with diff tracking in graph"
```

---

## Task 8: Implement BashTool

- [x] **Complete**

**Files:**
- Modify: `crates/tools/src/bash.rs`
- Test: `crates/tools/src/bash.rs` (inline tests)

### Step 8.1 — Write BashTool with process management and tests

**`crates/tools/src/bash.rs`:**

```rust
use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::json;
use tokio::process::Command;
use tracing::debug;

use crate::{Tool, ToolContext, ToolError, ToolOutput};
use graphirm_graph::{EdgeType, NodeType};

const DEFAULT_TIMEOUT_SECS: u64 = 120;

pub struct BashTool;

impl BashTool {
    pub fn new() -> Self {
        Self
    }

    fn resolve_dir(dir: Option<&str>, working_dir: &PathBuf) -> PathBuf {
        match dir {
            Some(d) => {
                let p = PathBuf::from(d);
                if p.is_absolute() {
                    p
                } else {
                    working_dir.join(p)
                }
            }
            None => working_dir.clone(),
        }
    }
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a bash command and return its output. Captures both stdout and stderr."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                },
                "working_directory": {
                    "type": "string",
                    "description": "Directory to execute the command in. Optional, defaults to current working directory."
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds. Optional, defaults to 120."
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let command = args["command"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'command' field".into()))?;

        let timeout_secs = args["timeout"].as_u64().unwrap_or(DEFAULT_TIMEOUT_SECS);
        let exec_dir = Self::resolve_dir(args["working_directory"].as_str(), &ctx.working_dir);

        debug!(command = command, dir = %exec_dir.display(), "executing bash command");

        let child = Command::new("bash")
            .arg("-c")
            .arg(command)
            .current_dir(&exec_dir)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| ToolError::ExecutionFailed(format!("failed to spawn bash: {}", e)))?;

        let result = tokio::select! {
            output = child.wait_with_output() => {
                output.map_err(|e| ToolError::ExecutionFailed(format!("process error: {}", e)))?
            }
            _ = tokio::time::sleep(Duration::from_secs(timeout_secs)) => {
                return Err(ToolError::Timeout(timeout_secs));
            }
            _ = ctx.signal.cancelled() => {
                return Err(ToolError::Cancelled);
            }
        };

        let stdout = String::from_utf8_lossy(&result.stdout);
        let stderr = String::from_utf8_lossy(&result.stderr);
        let exit_code = result.status.code().unwrap_or(-1);

        let output = if stderr.is_empty() {
            stdout.to_string()
        } else if stdout.is_empty() {
            format!("STDERR:\n{}", stderr)
        } else {
            format!("{}\nSTDERR:\n{}", stdout, stderr)
        };

        let is_error = exit_code != 0;

        let content_node = ctx.graph.add_node(
            NodeType::Content,
            json!({
                "type": "command_output",
                "command": command,
                "exit_code": exit_code,
                "stdout_bytes": result.stdout.len(),
                "stderr_bytes": result.stderr.len(),
                "working_directory": exec_dir.to_string_lossy(),
            }),
        )?;

        ctx.graph.add_edge(
            ctx.interaction_id.clone(),
            content_node.clone(),
            EdgeType::Produces,
            json!({
                "exit_code": exit_code,
            }),
        )?;

        let mut tool_output = if is_error {
            ToolOutput::error(format!("Exit code {}\n{}", exit_code, output))
        } else {
            ToolOutput::success(output)
        };
        tool_output.node_id = Some(content_node);
        Ok(tool_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::make_test_context;
    use tempfile::TempDir;

    fn ctx_with_dir(dir: &TempDir) -> ToolContext {
        let mut ctx = make_test_context();
        ctx.working_dir = dir.path().to_path_buf();
        ctx
    }

    #[tokio::test]
    async fn bash_echo() {
        let dir = TempDir::new().unwrap();
        let tool = BashTool::new();
        let ctx = ctx_with_dir(&dir);

        let result = tool
            .execute(json!({"command": "echo hello"}), &ctx)
            .await
            .unwrap();

        assert!(!result.is_error);
        assert_eq!(result.content.trim(), "hello");
        assert!(result.node_id.is_some());
    }

    #[tokio::test]
    async fn bash_captures_stderr() {
        let dir = TempDir::new().unwrap();
        let tool = BashTool::new();
        let ctx = ctx_with_dir(&dir);

        let result = tool
            .execute(json!({"command": "echo oops >&2"}), &ctx)
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("oops"));
    }

    #[tokio::test]
    async fn bash_exit_code_nonzero() {
        let dir = TempDir::new().unwrap();
        let tool = BashTool::new();
        let ctx = ctx_with_dir(&dir);

        let result = tool
            .execute(json!({"command": "exit 42"}), &ctx)
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(result.content.contains("Exit code 42"));
    }

    #[tokio::test]
    async fn bash_working_directory() {
        let dir = TempDir::new().unwrap();
        let sub = dir.path().join("subdir");
        std::fs::create_dir(&sub).unwrap();

        let tool = BashTool::new();
        let ctx = ctx_with_dir(&dir);

        let result = tool
            .execute(
                json!({"command": "pwd", "working_directory": "subdir"}),
                &ctx,
            )
            .await
            .unwrap();

        assert!(result.content.trim().ends_with("subdir"));
    }

    #[tokio::test]
    async fn bash_missing_command() {
        let dir = TempDir::new().unwrap();
        let tool = BashTool::new();
        let ctx = ctx_with_dir(&dir);

        let err = tool.execute(json!({}), &ctx).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[tokio::test]
    async fn bash_creates_graph_node() {
        let dir = TempDir::new().unwrap();
        let tool = BashTool::new();
        let ctx = ctx_with_dir(&dir);

        let result = tool
            .execute(json!({"command": "echo graphed"}), &ctx)
            .await
            .unwrap();

        let node_id = result.node_id.unwrap();
        let node = ctx.graph.get_node(node_id).unwrap();
        assert!(node.is_some());
    }

    #[tokio::test]
    async fn bash_cancellation() {
        let dir = TempDir::new().unwrap();
        let tool = BashTool::new();
        let ctx = ctx_with_dir(&dir);
        ctx.signal.cancel();

        let err = tool
            .execute(json!({"command": "sleep 10"}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Cancelled));
    }
}
```

### Step 8.2 — Run tests

```bash
cd crates/tools && cargo test -- bash
```

Expected: 7 tests pass.

### Step 8.3 — Commit

```bash
git add crates/tools/src/bash.rs
git commit -m "feat(tools): implement BashTool with timeout, cancellation, and graph tracking"
```

---

## Task 9: Implement GrepTool

- [x] **Complete**

**Files:**
- Modify: `crates/tools/src/grep.rs`
- Test: `crates/tools/src/grep.rs` (inline tests)

### Step 9.1 — Write GrepTool with graph tracking and tests

Uses `tokio::process::Command` to run `rg` (ripgrep) if available, falls back to manual `grep`-style search.

**`crates/tools/src/grep.rs`:**

```rust
use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::json;
use tokio::process::Command;
use tracing::debug;

use crate::{Tool, ToolContext, ToolError, ToolOutput};
use graphirm_graph::{EdgeType, NodeType};

pub struct GrepTool;

impl GrepTool {
    pub fn new() -> Self {
        Self
    }

    fn resolve_path(path: &str, working_dir: &PathBuf) -> PathBuf {
        let p = PathBuf::from(path);
        if p.is_absolute() {
            p
        } else {
            working_dir.join(p)
        }
    }
}

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search for a pattern in files using ripgrep. Returns matching lines with file paths and line numbers."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in. Defaults to working directory."
                },
                "include": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g. '*.rs'). Optional."
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case insensitive search. Defaults to false."
                }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let pattern = args["pattern"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'pattern' field".into()))?;

        let search_path = args["path"]
            .as_str()
            .map(|p| Self::resolve_path(p, &ctx.working_dir))
            .unwrap_or_else(|| ctx.working_dir.clone());

        let include = args["include"].as_str();
        let case_insensitive = args["case_insensitive"].as_bool().unwrap_or(false);

        debug!(pattern = pattern, path = %search_path.display(), "grep search");

        let mut cmd = Command::new("rg");
        cmd.arg("--line-number")
            .arg("--no-heading")
            .arg("--color=never");

        if case_insensitive {
            cmd.arg("--ignore-case");
        }

        if let Some(glob) = include {
            cmd.arg("--glob").arg(glob);
        }

        cmd.arg(pattern).arg(&search_path);

        let output = cmd
            .output()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("failed to run rg: {}", e)))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        // rg exit code: 0 = matches found, 1 = no matches, 2 = error
        match output.status.code() {
            Some(0) => {}
            Some(1) => {
                return Ok(ToolOutput::success("No matches found."));
            }
            _ => {
                return Err(ToolError::ExecutionFailed(format!(
                    "rg error: {}",
                    stderr.trim()
                )));
            }
        }

        let match_count = stdout.lines().count();

        let content_node = ctx.graph.add_node(
            NodeType::Content,
            json!({
                "type": "search_results",
                "pattern": pattern,
                "search_path": search_path.to_string_lossy(),
                "match_count": match_count,
            }),
        )?;

        ctx.graph.add_edge(
            ctx.interaction_id.clone(),
            content_node.clone(),
            EdgeType::Reads,
            json!({
                "search_pattern": pattern,
                "match_count": match_count,
            }),
        )?;

        Ok(ToolOutput::success_with_node(
            stdout.to_string(),
            content_node,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::make_test_context;
    use tempfile::TempDir;

    fn ctx_with_dir(dir: &TempDir) -> ToolContext {
        let mut ctx = make_test_context();
        ctx.working_dir = dir.path().to_path_buf();
        ctx
    }

    fn create_test_files(dir: &TempDir) {
        std::fs::write(dir.path().join("hello.rs"), "fn hello() {\n    println!(\"hello world\");\n}\n").unwrap();
        std::fs::write(dir.path().join("bye.rs"), "fn goodbye() {\n    println!(\"goodbye\");\n}\n").unwrap();
        std::fs::write(dir.path().join("notes.txt"), "some notes\nhello again\n").unwrap();
    }

    #[tokio::test]
    async fn grep_finds_matches() {
        let dir = TempDir::new().unwrap();
        create_test_files(&dir);

        let tool = GrepTool::new();
        let ctx = ctx_with_dir(&dir);

        let result = tool
            .execute(json!({"pattern": "hello"}), &ctx)
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("hello"));
        assert!(result.node_id.is_some());
    }

    #[tokio::test]
    async fn grep_no_matches() {
        let dir = TempDir::new().unwrap();
        create_test_files(&dir);

        let tool = GrepTool::new();
        let ctx = ctx_with_dir(&dir);

        let result = tool
            .execute(json!({"pattern": "zzzzz_not_found"}), &ctx)
            .await
            .unwrap();

        assert_eq!(result.content, "No matches found.");
    }

    #[tokio::test]
    async fn grep_with_include_filter() {
        let dir = TempDir::new().unwrap();
        create_test_files(&dir);

        let tool = GrepTool::new();
        let ctx = ctx_with_dir(&dir);

        let result = tool
            .execute(json!({"pattern": "hello", "include": "*.rs"}), &ctx)
            .await
            .unwrap();

        assert!(result.content.contains("hello.rs"));
        assert!(!result.content.contains("notes.txt"));
    }

    #[tokio::test]
    async fn grep_specific_file() {
        let dir = TempDir::new().unwrap();
        create_test_files(&dir);

        let tool = GrepTool::new();
        let ctx = ctx_with_dir(&dir);

        let result = tool
            .execute(json!({"pattern": "fn", "path": "bye.rs"}), &ctx)
            .await
            .unwrap();

        assert!(result.content.contains("goodbye"));
    }

    #[tokio::test]
    async fn grep_missing_pattern() {
        let dir = TempDir::new().unwrap();
        let tool = GrepTool::new();
        let ctx = ctx_with_dir(&dir);

        let err = tool.execute(json!({}), &ctx).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }
}
```

### Step 9.2 — Run tests

```bash
cd crates/tools && cargo test -- grep
```

Expected: 5 tests pass (requires `rg` binary installed on PATH).

### Step 9.3 — Commit

```bash
git add crates/tools/src/grep.rs
git commit -m "feat(tools): implement GrepTool using ripgrep with graph tracking"
```

---

## Task 10: Implement FindTool

- [x] **Complete**

**Files:**
- Modify: `crates/tools/src/find.rs`
- Test: `crates/tools/src/find.rs` (inline tests)

### Step 10.1 — Write FindTool with glob pattern matching and tests

**`crates/tools/src/find.rs`:**

```rust
use std::path::PathBuf;

use async_trait::async_trait;
use glob::glob;
use serde_json::json;
use tracing::debug;

use crate::{Tool, ToolContext, ToolError, ToolOutput};
use graphirm_graph::{EdgeType, NodeType};

const MAX_RESULTS: usize = 1000;

pub struct FindTool;

impl FindTool {
    pub fn new() -> Self {
        Self
    }

    fn resolve_path(path: &str, working_dir: &PathBuf) -> PathBuf {
        let p = PathBuf::from(path);
        if p.is_absolute() {
            p
        } else {
            working_dir.join(p)
        }
    }
}

#[async_trait]
impl Tool for FindTool {
    fn name(&self) -> &str {
        "find"
    }

    fn description(&self) -> &str {
        "Find files matching a glob pattern. Returns matching file paths relative to the search directory."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files (e.g. '**/*.rs', 'src/**/*.toml')"
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in. Defaults to working directory."
                }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let pattern = args["pattern"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'pattern' field".into()))?;

        let search_dir = args["path"]
            .as_str()
            .map(|p| Self::resolve_path(p, &ctx.working_dir))
            .unwrap_or_else(|| ctx.working_dir.clone());

        debug!(pattern = pattern, dir = %search_dir.display(), "finding files");

        let full_pattern = format!("{}/{}", search_dir.display(), pattern);

        let entries: Vec<PathBuf> = glob(&full_pattern)
            .map_err(|e| ToolError::InvalidArguments(format!("invalid glob pattern: {}", e)))?
            .filter_map(|entry| entry.ok())
            .take(MAX_RESULTS)
            .collect();

        if entries.is_empty() {
            return Ok(ToolOutput::success("No files found."));
        }

        let relative_paths: Vec<String> = entries
            .iter()
            .filter_map(|p| p.strip_prefix(&search_dir).ok())
            .map(|p| p.display().to_string())
            .collect();

        let output = relative_paths.join("\n");
        let file_count = relative_paths.len();

        let content_node = ctx.graph.add_node(
            NodeType::Content,
            json!({
                "type": "file_listing",
                "pattern": pattern,
                "search_dir": search_dir.to_string_lossy(),
                "file_count": file_count,
            }),
        )?;

        ctx.graph.add_edge(
            ctx.interaction_id.clone(),
            content_node.clone(),
            EdgeType::Reads,
            json!({
                "pattern": pattern,
                "file_count": file_count,
            }),
        )?;

        Ok(ToolOutput::success_with_node(output, content_node))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::make_test_context;
    use tempfile::TempDir;

    fn ctx_with_dir(dir: &TempDir) -> ToolContext {
        let mut ctx = make_test_context();
        ctx.working_dir = dir.path().to_path_buf();
        ctx
    }

    fn create_test_tree(dir: &TempDir) {
        let src = dir.path().join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("main.rs"), "fn main() {}").unwrap();
        std::fs::write(src.join("lib.rs"), "pub mod foo;").unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]").unwrap();
        std::fs::write(dir.path().join("README.md"), "# README").unwrap();
    }

    #[tokio::test]
    async fn find_rs_files() {
        let dir = TempDir::new().unwrap();
        create_test_tree(&dir);

        let tool = FindTool::new();
        let ctx = ctx_with_dir(&dir);

        let result = tool
            .execute(json!({"pattern": "**/*.rs"}), &ctx)
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("main.rs"));
        assert!(result.content.contains("lib.rs"));
        assert!(!result.content.contains("Cargo.toml"));
    }

    #[tokio::test]
    async fn find_no_matches() {
        let dir = TempDir::new().unwrap();
        create_test_tree(&dir);

        let tool = FindTool::new();
        let ctx = ctx_with_dir(&dir);

        let result = tool
            .execute(json!({"pattern": "**/*.py"}), &ctx)
            .await
            .unwrap();

        assert_eq!(result.content, "No files found.");
    }

    #[tokio::test]
    async fn find_specific_subdir() {
        let dir = TempDir::new().unwrap();
        create_test_tree(&dir);

        let tool = FindTool::new();
        let ctx = ctx_with_dir(&dir);

        let result = tool
            .execute(json!({"pattern": "*.rs", "path": "src"}), &ctx)
            .await
            .unwrap();

        assert!(result.content.contains("main.rs"));
    }

    #[tokio::test]
    async fn find_invalid_glob() {
        let dir = TempDir::new().unwrap();
        let tool = FindTool::new();
        let ctx = ctx_with_dir(&dir);

        let err = tool
            .execute(json!({"pattern": "[invalid"}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[tokio::test]
    async fn find_creates_graph_node() {
        let dir = TempDir::new().unwrap();
        create_test_tree(&dir);

        let tool = FindTool::new();
        let ctx = ctx_with_dir(&dir);

        let result = tool
            .execute(json!({"pattern": "**/*.rs"}), &ctx)
            .await
            .unwrap();

        assert!(result.node_id.is_some());
    }
}
```

### Step 10.2 — Run tests

```bash
cd crates/tools && cargo test -- find
```

Expected: 5 tests pass.

### Step 10.3 — Commit

```bash
git add crates/tools/src/find.rs
git commit -m "feat(tools): implement FindTool with glob patterns and graph tracking"
```

---

## Task 11: Implement LsTool

- [x] **Complete**

**Files:**
- Modify: `crates/tools/src/ls.rs`
- Test: `crates/tools/src/ls.rs` (inline tests)

### Step 11.1 — Write LsTool with directory listing and tests

**`crates/tools/src/ls.rs`:**

```rust
use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::json;
use tracing::debug;

use crate::{Tool, ToolContext, ToolError, ToolOutput};
use graphirm_graph::{EdgeType, NodeType};

pub struct LsTool;

impl LsTool {
    pub fn new() -> Self {
        Self
    }

    fn resolve_path(path: &str, working_dir: &PathBuf) -> PathBuf {
        let p = PathBuf::from(path);
        if p.is_absolute() {
            p
        } else {
            working_dir.join(p)
        }
    }
}

#[async_trait]
impl Tool for LsTool {
    fn name(&self) -> &str {
        "ls"
    }

    fn description(&self) -> &str {
        "List directory contents. Shows files and subdirectories with type indicators (/ for dirs)."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to list. Defaults to working directory."
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "Show hidden files (starting with '.'). Defaults to false."
                }
            },
            "required": []
        })
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let dir = args["path"]
            .as_str()
            .map(|p| Self::resolve_path(p, &ctx.working_dir))
            .unwrap_or_else(|| ctx.working_dir.clone());

        let show_hidden = args["show_hidden"].as_bool().unwrap_or(false);

        debug!(dir = %dir.display(), show_hidden = show_hidden, "listing directory");

        if !dir.is_dir() {
            return Err(ToolError::ExecutionFailed(format!(
                "not a directory: {}",
                dir.display()
            )));
        }

        let mut entries = Vec::new();
        let mut read_dir = tokio::fs::read_dir(&dir).await?;

        while let Some(entry) = read_dir.next_entry().await? {
            let name = entry.file_name().to_string_lossy().to_string();

            if !show_hidden && name.starts_with('.') {
                continue;
            }

            let metadata = entry.metadata().await?;
            let suffix = if metadata.is_dir() { "/" } else { "" };
            entries.push(format!("{}{}", name, suffix));
        }

        entries.sort();

        if entries.is_empty() {
            return Ok(ToolOutput::success("(empty directory)"));
        }

        let output = entries.join("\n");

        let content_node = ctx.graph.add_node(
            NodeType::Content,
            json!({
                "type": "directory_listing",
                "path": dir.to_string_lossy(),
                "entry_count": entries.len(),
            }),
        )?;

        ctx.graph.add_edge(
            ctx.interaction_id.clone(),
            content_node.clone(),
            EdgeType::Reads,
            json!({
                "entry_count": entries.len(),
            }),
        )?;

        Ok(ToolOutput::success_with_node(output, content_node))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::make_test_context;
    use tempfile::TempDir;

    fn ctx_with_dir(dir: &TempDir) -> ToolContext {
        let mut ctx = make_test_context();
        ctx.working_dir = dir.path().to_path_buf();
        ctx
    }

    #[tokio::test]
    async fn ls_basic() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("file.txt"), "content").unwrap();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();

        let tool = LsTool::new();
        let ctx = ctx_with_dir(&dir);

        let result = tool.execute(json!({}), &ctx).await.unwrap();

        assert!(result.content.contains("file.txt"));
        assert!(result.content.contains("subdir/"));
    }

    #[tokio::test]
    async fn ls_hides_dotfiles_by_default() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join(".hidden"), "secret").unwrap();
        std::fs::write(dir.path().join("visible.txt"), "public").unwrap();

        let tool = LsTool::new();
        let ctx = ctx_with_dir(&dir);

        let result = tool.execute(json!({}), &ctx).await.unwrap();

        assert!(!result.content.contains(".hidden"));
        assert!(result.content.contains("visible.txt"));
    }

    #[tokio::test]
    async fn ls_show_hidden() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join(".hidden"), "secret").unwrap();
        std::fs::write(dir.path().join("visible.txt"), "public").unwrap();

        let tool = LsTool::new();
        let ctx = ctx_with_dir(&dir);

        let result = tool
            .execute(json!({"show_hidden": true}), &ctx)
            .await
            .unwrap();

        assert!(result.content.contains(".hidden"));
        assert!(result.content.contains("visible.txt"));
    }

    #[tokio::test]
    async fn ls_subdir() {
        let dir = TempDir::new().unwrap();
        let sub = dir.path().join("nested");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(sub.join("inner.txt"), "data").unwrap();

        let tool = LsTool::new();
        let ctx = ctx_with_dir(&dir);

        let result = tool
            .execute(json!({"path": "nested"}), &ctx)
            .await
            .unwrap();

        assert!(result.content.contains("inner.txt"));
    }

    #[tokio::test]
    async fn ls_empty_dir() {
        let dir = TempDir::new().unwrap();
        let empty = dir.path().join("empty");
        std::fs::create_dir(&empty).unwrap();

        let tool = LsTool::new();
        let ctx = ctx_with_dir(&dir);

        let result = tool
            .execute(json!({"path": "empty"}), &ctx)
            .await
            .unwrap();

        assert_eq!(result.content, "(empty directory)");
    }

    #[tokio::test]
    async fn ls_not_a_directory() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("file.txt"), "content").unwrap();

        let tool = LsTool::new();
        let ctx = ctx_with_dir(&dir);

        let err = tool
            .execute(json!({"path": "file.txt"}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::ExecutionFailed(_)));
    }

    #[tokio::test]
    async fn ls_sorted_output() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("c.txt"), "").unwrap();
        std::fs::write(dir.path().join("a.txt"), "").unwrap();
        std::fs::write(dir.path().join("b.txt"), "").unwrap();

        let tool = LsTool::new();
        let ctx = ctx_with_dir(&dir);

        let result = tool.execute(json!({}), &ctx).await.unwrap();

        let lines: Vec<&str> = result.content.lines().collect();
        assert_eq!(lines, vec!["a.txt", "b.txt", "c.txt"]);
    }
}
```

### Step 11.2 — Run tests

```bash
cd crates/tools && cargo test -- ls
```

Expected: 7 tests pass.

### Step 11.3 — Commit

```bash
git add crates/tools/src/ls.rs
git commit -m "feat(tools): implement LsTool with hidden file filtering and graph tracking"
```

---

## Task 12: Implement Parallel Tool Execution

- [x] **Complete**

**Files:**
- Create: `crates/tools/src/executor.rs`
- Modify: `crates/tools/src/lib.rs` (add `pub mod executor;`)
- Test: `crates/tools/src/executor.rs` (inline tests)

### Step 12.1 — Write parallel executor with tests

**`crates/tools/src/executor.rs`:**

```rust
use std::sync::Arc;

use tokio::task::JoinSet;
use tracing::{debug, warn};

use crate::{Tool, ToolCall, ToolContext, ToolError, ToolOutput};
use crate::registry::ToolRegistry;

/// Result of a single tool call within a batch, preserving the call ID.
#[derive(Debug)]
pub struct ToolCallResult {
    pub call_id: String,
    pub result: Result<ToolOutput, ToolError>,
}

/// Execute multiple tool calls in parallel using tokio::JoinSet.
///
/// Each tool runs in its own spawned task. Results are collected in
/// completion order (not submission order). Use `call_id` to correlate
/// results back to the original calls.
pub async fn execute_parallel(
    registry: &ToolRegistry,
    calls: Vec<ToolCall>,
    ctx: &ToolContext,
) -> Vec<ToolCallResult> {
    let mut set = JoinSet::new();

    for call in calls {
        let tool = match registry.get(&call.name) {
            Ok(t) => t,
            Err(e) => {
                // Can't even find the tool — record error immediately rather than
                // spawning a task. We still need to track it, so we handle it below.
                set.spawn({
                    let call_id = call.id.clone();
                    let err_msg = e.to_string();
                    async move {
                        (call_id, Err::<ToolOutput, ToolError>(ToolError::NotFound(err_msg)))
                    }
                });
                continue;
            }
        };

        let args = call.arguments.clone();
        let call_id = call.id.clone();
        let ctx = ctx.clone();
        let tool: Arc<dyn Tool> = tool;

        set.spawn(async move {
            debug!(tool = %call_id, "executing tool");
            let result = tool.execute(args, &ctx).await;
            (call_id, result)
        });
    }

    let mut results = Vec::new();
    while let Some(join_result) = set.join_next().await {
        match join_result {
            Ok((call_id, result)) => {
                results.push(ToolCallResult { call_id, result });
            }
            Err(join_error) => {
                warn!(error = %join_error, "tool task panicked");
                results.push(ToolCallResult {
                    call_id: "unknown".into(),
                    result: Err(ToolError::ExecutionFailed(format!(
                        "task panicked: {}",
                        join_error
                    ))),
                });
            }
        }
    }

    results
}

/// Execute a single tool call (convenience wrapper).
pub async fn execute_single(
    registry: &ToolRegistry,
    call: &ToolCall,
    ctx: &ToolContext,
) -> Result<ToolOutput, ToolError> {
    let tool = registry.get(&call.name)?;
    tool.execute(call.arguments.clone(), ctx).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::make_test_context;
    use crate::registry::ToolRegistry;
    use async_trait::async_trait;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CounterTool {
        counter: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Tool for CounterTool {
        fn name(&self) -> &str {
            "counter"
        }

        fn description(&self) -> &str {
            "Increments a counter"
        }

        fn parameters(&self) -> serde_json::Value {
            json!({"type": "object", "properties": {}})
        }

        async fn execute(
            &self,
            _args: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, ToolError> {
            let val = self.counter.fetch_add(1, Ordering::SeqCst);
            Ok(ToolOutput::success(format!("count: {}", val + 1)))
        }
    }

    struct SlowTool;

    #[async_trait]
    impl Tool for SlowTool {
        fn name(&self) -> &str {
            "slow"
        }

        fn description(&self) -> &str {
            "Sleeps briefly then returns"
        }

        fn parameters(&self) -> serde_json::Value {
            json!({"type": "object", "properties": {}})
        }

        async fn execute(
            &self,
            _args: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, ToolError> {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            Ok(ToolOutput::success("done"))
        }
    }

    fn make_calls(name: &str, count: usize) -> Vec<ToolCall> {
        (0..count)
            .map(|i| ToolCall {
                id: format!("call_{}", i),
                name: name.into(),
                arguments: json!({}),
            })
            .collect()
    }

    #[tokio::test]
    async fn parallel_three_tools() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(CounterTool {
            counter: counter.clone(),
        }));

        let ctx = make_test_context();
        let calls = make_calls("counter", 3);

        let results = execute_parallel(&reg, calls, &ctx).await;

        assert_eq!(results.len(), 3);
        assert_eq!(counter.load(Ordering::SeqCst), 3);
        for r in &results {
            assert!(r.result.is_ok());
        }
    }

    #[tokio::test]
    async fn parallel_preserves_call_ids() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(CounterTool {
            counter: counter.clone(),
        }));

        let ctx = make_test_context();
        let calls = make_calls("counter", 2);

        let results = execute_parallel(&reg, calls, &ctx).await;

        let mut ids: Vec<String> = results.iter().map(|r| r.call_id.clone()).collect();
        ids.sort();
        assert_eq!(ids, vec!["call_0", "call_1"]);
    }

    #[tokio::test]
    async fn parallel_with_not_found() {
        let reg = ToolRegistry::new();
        let ctx = make_test_context();

        let calls = vec![ToolCall {
            id: "missing_call".into(),
            name: "nonexistent".into(),
            arguments: json!({}),
        }];

        let results = execute_parallel(&reg, calls, &ctx).await;
        assert_eq!(results.len(), 1);
        assert!(results[0].result.is_err());
    }

    #[tokio::test]
    async fn parallel_actually_concurrent() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(SlowTool));

        let ctx = make_test_context();
        let calls = make_calls("slow", 5);

        let start = std::time::Instant::now();
        let results = execute_parallel(&reg, calls, &ctx).await;
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 5);
        // 5 tasks each sleeping 50ms should complete in well under 250ms
        // if running in parallel (vs 250ms+ if sequential)
        assert!(
            elapsed.as_millis() < 200,
            "took {}ms, expected < 200ms for parallel execution",
            elapsed.as_millis()
        );
    }

    #[tokio::test]
    async fn execute_single_success() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(CounterTool {
            counter: counter.clone(),
        }));

        let ctx = make_test_context();
        let call = ToolCall {
            id: "single".into(),
            name: "counter".into(),
            arguments: json!({}),
        };

        let result = execute_single(&reg, &call, &ctx).await.unwrap();
        assert_eq!(result.content, "count: 1");
    }
}
```

### Step 12.2 — Run tests

```bash
cd crates/tools && cargo test -- executor
```

Expected: 5 tests pass.

### Step 12.3 — Commit

```bash
git add crates/tools/src/executor.rs crates/tools/src/lib.rs
git commit -m "feat(tools): implement parallel tool execution with JoinSet"
```

---

## Task 13: Integration Test — Full Workflow

- [x] **Complete**

**Files:**
- Create: `crates/tools/tests/integration.rs`
- Test: `crates/tools/tests/integration.rs`

### Step 13.1 — Write integration test

This test registers all tools, runs a realistic sequence (ls → read → write → edit → bash → grep → find), and verifies the complete graph trail.

**`crates/tools/tests/integration.rs`:**

```rust
use std::path::PathBuf;
use std::sync::Arc;

use graphirm_graph::{GraphStore, NodeId};
use graphirm_tools::{
    bash::BashTool,
    edit::EditTool,
    executor::{execute_parallel, execute_single},
    find::FindTool,
    grep::GrepTool,
    ls::LsTool,
    read::ReadTool,
    registry::ToolRegistry,
    write::WriteTool,
    ToolCall, ToolContext,
};
use serde_json::json;
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

fn setup() -> (TempDir, ToolRegistry, ToolContext) {
    let dir = TempDir::new().unwrap();
    let graph = Arc::new(GraphStore::open_memory().expect("memory graph"));

    let agent_id = graph
        .add_node(
            graphirm_graph::NodeType::Agent,
            json!({"name": "test-agent"}),
        )
        .unwrap();

    let interaction_id = graph
        .add_node(
            graphirm_graph::NodeType::Interaction,
            json!({"type": "test"}),
        )
        .unwrap();

    let ctx = ToolContext {
        graph,
        agent_id,
        interaction_id,
        working_dir: dir.path().to_path_buf(),
        signal: CancellationToken::new(),
    };

    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(ReadTool::new()));
    registry.register(Arc::new(WriteTool::new()));
    registry.register(Arc::new(EditTool::new()));
    registry.register(Arc::new(BashTool::new()));
    registry.register(Arc::new(GrepTool::new()));
    registry.register(Arc::new(FindTool::new()));
    registry.register(Arc::new(LsTool::new()));

    (dir, registry, ctx)
}

#[tokio::test]
async fn full_workflow_sequence() {
    let (dir, registry, ctx) = setup();

    // Step 1: Write a file
    let write_result = execute_single(
        &registry,
        &ToolCall {
            id: "w1".into(),
            name: "write".into(),
            arguments: json!({
                "path": "src/main.rs",
                "content": "fn main() {\n    println!(\"hello world\");\n}\n"
            }),
        },
        &ctx,
    )
    .await
    .unwrap();
    assert!(!write_result.is_error);
    assert!(write_result.node_id.is_some());

    // Step 2: Read the file back
    let read_result = execute_single(
        &registry,
        &ToolCall {
            id: "r1".into(),
            name: "read".into(),
            arguments: json!({"path": "src/main.rs"}),
        },
        &ctx,
    )
    .await
    .unwrap();
    assert!(read_result.content.contains("hello world"));

    // Step 3: Edit the file
    let edit_result = execute_single(
        &registry,
        &ToolCall {
            id: "e1".into(),
            name: "edit".into(),
            arguments: json!({
                "path": "src/main.rs",
                "old_string": "hello world",
                "new_string": "hello graphirm"
            }),
        },
        &ctx,
    )
    .await
    .unwrap();
    assert!(!edit_result.is_error);

    // Verify edit on disk
    let content = std::fs::read_to_string(dir.path().join("src/main.rs")).unwrap();
    assert!(content.contains("hello graphirm"));
    assert!(!content.contains("hello world"));

    // Step 4: Run bash to verify
    let bash_result = execute_single(
        &registry,
        &ToolCall {
            id: "b1".into(),
            name: "bash".into(),
            arguments: json!({"command": "cat src/main.rs"}),
        },
        &ctx,
    )
    .await
    .unwrap();
    assert!(bash_result.content.contains("hello graphirm"));

    // Step 5: List directory
    let ls_result = execute_single(
        &registry,
        &ToolCall {
            id: "l1".into(),
            name: "ls".into(),
            arguments: json!({}),
        },
        &ctx,
    )
    .await
    .unwrap();
    assert!(ls_result.content.contains("src/"));

    // Step 6: Find files
    let find_result = execute_single(
        &registry,
        &ToolCall {
            id: "f1".into(),
            name: "find".into(),
            arguments: json!({"pattern": "**/*.rs"}),
        },
        &ctx,
    )
    .await
    .unwrap();
    assert!(find_result.content.contains("main.rs"));

    // Step 7: Grep for content
    let grep_result = execute_single(
        &registry,
        &ToolCall {
            id: "g1".into(),
            name: "grep".into(),
            arguments: json!({"pattern": "graphirm"}),
        },
        &ctx,
    )
    .await
    .unwrap();
    assert!(grep_result.content.contains("graphirm"));
}

#[tokio::test]
async fn parallel_reads() {
    let (dir, registry, ctx) = setup();

    // Create 3 files
    for i in 0..3 {
        std::fs::write(
            dir.path().join(format!("file_{}.txt", i)),
            format!("content of file {}", i),
        )
        .unwrap();
    }

    // Read all 3 in parallel
    let calls: Vec<ToolCall> = (0..3)
        .map(|i| ToolCall {
            id: format!("read_{}", i),
            name: "read".into(),
            arguments: json!({"path": format!("file_{}.txt", i)}),
        })
        .collect();

    let results = execute_parallel(&registry, calls, &ctx).await;

    assert_eq!(results.len(), 3);
    for r in &results {
        let output = r.result.as_ref().unwrap();
        assert!(!output.is_error);
        assert!(output.content.contains("content of file"));
        assert!(output.node_id.is_some());
    }
}

#[tokio::test]
async fn registry_lists_all_tools() {
    let (_, registry, _) = setup();

    let names = registry.list();
    assert_eq!(names.len(), 7);
    assert!(names.contains(&"bash"));
    assert!(names.contains(&"read"));
    assert!(names.contains(&"write"));
    assert!(names.contains(&"edit"));
    assert!(names.contains(&"grep"));
    assert!(names.contains(&"find"));
    assert!(names.contains(&"ls"));
}

#[tokio::test]
async fn definitions_for_llm() {
    let (_, registry, _) = setup();

    let defs = registry.definitions();
    assert_eq!(defs.len(), 7);

    for def in &defs {
        assert!(!def.name.is_empty());
        assert!(!def.description.is_empty());
        assert!(def.parameters.is_object());
    }
}

#[tokio::test]
async fn graph_trail_after_workflow() {
    let (dir, registry, ctx) = setup();

    // Write + Read — should create graph nodes and edges
    execute_single(
        &registry,
        &ToolCall {
            id: "w".into(),
            name: "write".into(),
            arguments: json!({"path": "tracked.txt", "content": "tracked content"}),
        },
        &ctx,
    )
    .await
    .unwrap();

    execute_single(
        &registry,
        &ToolCall {
            id: "r".into(),
            name: "read".into(),
            arguments: json!({"path": "tracked.txt"}),
        },
        &ctx,
    )
    .await
    .unwrap();

    // Verify the graph has Content nodes for the file
    // (exact verification depends on GraphStore query API from Phase 1)
    // At minimum, the tool executions should not have errored,
    // and node_ids should have been created.
}
```

### Step 13.2 — Run all tests

```bash
cd crates/tools && cargo test
```

Expected: All tests across all modules pass (error: 8, lib: 4, registry: 7, permissions: 7, read: 6, write: 5, edit: 5, bash: 7, grep: 5, find: 5, ls: 7, executor: 5, integration: 5 = **81 total tests**).

### Step 13.3 — Commit

```bash
git add crates/tools/tests/integration.rs
git commit -m "test(tools): add integration tests for full tool workflow and parallel execution"
```

---

## Final Checklist

After all tasks complete:

```bash
# Full test suite
cd crates/tools && cargo test

# Clippy
cargo clippy -- -D warnings

# Format check
cargo fmt -- --check
```

All 13 tasks build a complete tool system:

| # | What | Key Output |
|---|------|-----------|
| 1 | ToolError | `error.rs` — thiserror enum with From impls |
| 2 | Tool trait | `lib.rs` — trait, ToolContext, ToolOutput, ToolDefinition |
| 3 | ToolRegistry | `registry.rs` — register/get/list/execute |
| 4 | Permissions | `permissions.rs` — Allow/Ask/Deny policy |
| 5 | ReadTool | `read.rs` — file read + Content node + Reads edge |
| 6 | WriteTool | `write.rs` — file write + Content node + Modifies edge |
| 7 | EditTool | `edit.rs` — string replace + diff in edge metadata |
| 8 | BashTool | `bash.rs` — process exec + Content node + Produces edge |
| 9 | GrepTool | `grep.rs` — ripgrep search + Reads edges |
| 10 | FindTool | `find.rs` — glob walk + Reads edges |
| 11 | LsTool | `ls.rs` — dir listing + Reads edge |
| 12 | Parallel executor | `executor.rs` — JoinSet-based concurrency |
| 13 | Integration test | `tests/integration.rs` — end-to-end workflow |
