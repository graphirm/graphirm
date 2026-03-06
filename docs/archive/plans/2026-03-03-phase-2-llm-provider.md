# Phase 2: LLM Provider Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a provider-agnostic LLM abstraction layer (`crates/llm/`) that wraps `rig-core` behind a stable trait, enabling the agent loop to call any supported LLM without knowing provider specifics.

**Architecture:** The `LlmProvider` trait defines `complete()` and `stream()` methods using our own message/tool/response types. Concrete providers (Anthropic, OpenAI) implement this trait by converting between our types and rig-core's types, then delegating to rig-core's `CompletionModel`. A factory function parses `"provider/model"` strings and returns `Box<dyn LlmProvider>`. A `MockProvider` enables deterministic testing without API keys.

**Tech Stack:** `rig-core` 0.31+ (LLM abstraction, 17+ providers), `thiserror` (error types), `serde`/`serde_json` (serialization), `tokio` (async runtime), `async-trait` (async trait support), `tokio-stream` (Stream utilities), `futures` (Stream trait)

---

## Prerequisites

- Phase 0 (scaffold) and Phase 1 (graph store) complete and merged to `main`
- Workspace `Cargo.toml` exists at repo root with `crates/llm` in `members`
- Rust toolchain installed (`cargo`, `rustc`)

## Branch

```bash
git checkout -b phase-2/llm-provider main
```

---

## Task 1: Set Up Cargo.toml and Crate Skeleton

- [x] **Status: complete**

**Files:**
- Modify: `crates/llm/Cargo.toml`
- Modify: `crates/llm/src/lib.rs`
- Modify: `crates/llm/src/error.rs`
- Modify: `crates/llm/src/provider.rs`
- Modify: `crates/llm/src/stream.rs`
- Modify: `crates/llm/src/tool.rs`

> If these files don't exist yet (Phase 0 wasn't done), create them instead.

### Step 1.1: Write Cargo.toml with all dependencies

```toml
[package]
name = "graphirm-llm"
version = "0.1.0"
edition = "2021"
description = "LLM provider abstraction layer for Graphirm"

[dependencies]
rig-core = { version = "0.31", features = ["anthropic", "openai"] }
async-trait = "0.1"
tokio = { version = "1", features = ["full"] }
tokio-stream = "0.1"
futures = "0.3"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
tracing = "0.1"

[dev-dependencies]
tokio = { version = "1", features = ["full", "test-util"] }
```

### Step 1.2: Write lib.rs with module declarations

```rust
pub mod error;
pub mod provider;
pub mod stream;
pub mod tool;

pub use error::LlmError;
pub use provider::{
    CompletionConfig, ContentPart, LlmMessage, LlmProvider, LlmResponse, Role, StopReason,
};
pub use stream::{StreamEvent, TokenUsage};
pub use tool::ToolDefinition;
```

### Step 1.3: Write minimal stubs for each module

`crates/llm/src/error.rs`:
```rust
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("placeholder")]
    Provider(String),
}
```

`crates/llm/src/provider.rs`:
```rust
// Will be filled in Task 2 and Task 5
```

`crates/llm/src/stream.rs`:
```rust
// Will be filled in Task 3
```

`crates/llm/src/tool.rs`:
```rust
// Will be filled in Task 4
```

### Step 1.4: Verify it compiles

```bash
cargo check -p graphirm-llm
```

**Expected:** Compiles with no errors (warnings about unused items are OK).

### Step 1.5: Commit

```bash
git add crates/llm/
git commit -m "phase-2: scaffold llm crate with dependencies"
```

---

## Task 2: Define LlmError Enum

- [x] **Status: complete**

**Files:**
- Modify: `crates/llm/src/error.rs`

### Step 2.1: Write the full error enum

```rust
use std::fmt;

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("Provider error: {0}")]
    Provider(String),

    #[error("Rate limited, retry after {retry_after_ms}ms")]
    RateLimited { retry_after_ms: u64 },

    #[error("Invalid model: {0}")]
    InvalidModel(String),

    #[error("Stream error: {0}")]
    Stream(String),

    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("Request error: {0}")]
    Request(String),

    #[error("Configuration error: {0}")]
    Config(String),
}

impl LlmError {
    pub fn provider(msg: impl Into<String>) -> Self {
        Self::Provider(msg.into())
    }

    pub fn stream(msg: impl Into<String>) -> Self {
        Self::Stream(msg.into())
    }

    pub fn invalid_model(model: impl Into<String>) -> Self {
        Self::InvalidModel(model.into())
    }

    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }
}
```

### Step 2.2: Write tests

Add at the bottom of `crates/llm/src/error.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_error_display() {
        let err = LlmError::provider("connection failed");
        assert_eq!(err.to_string(), "Provider error: connection failed");
    }

    #[test]
    fn test_rate_limited_display() {
        let err = LlmError::RateLimited { retry_after_ms: 5000 };
        assert_eq!(err.to_string(), "Rate limited, retry after 5000ms");
    }

    #[test]
    fn test_invalid_model_display() {
        let err = LlmError::invalid_model("gpt-99");
        assert_eq!(err.to_string(), "Invalid model: gpt-99");
    }

    #[test]
    fn test_stream_error_display() {
        let err = LlmError::stream("unexpected EOF");
        assert_eq!(err.to_string(), "Stream error: unexpected EOF");
    }

    #[test]
    fn test_serde_error_from() {
        let json_err = serde_json::from_str::<serde_json::Value>("invalid json").unwrap_err();
        let err: LlmError = json_err.into();
        assert!(err.to_string().starts_with("Serialization error:"));
    }

    #[test]
    fn test_config_error_display() {
        let err = LlmError::config("missing API key");
        assert_eq!(err.to_string(), "Configuration error: missing API key");
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LlmError>();
    }
}
```

### Step 2.3: Run tests

```bash
cargo test -p graphirm-llm -- error::tests
```

**Expected:** All 7 tests pass.

### Step 2.4: Commit

```bash
git add crates/llm/src/error.rs
git commit -m "phase-2: define LlmError enum with thiserror"
```

---

## Task 3: Define Message Types

- [x] **Status: complete**

**Files:**
- Modify: `crates/llm/src/provider.rs`

### Step 3.1: Write Role, ContentPart, and LlmMessage

```rust
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    Human,
    Assistant,
    ToolResult,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: Value,
    },
    ToolResult {
        id: String,
        content: String,
        is_error: bool,
    },
}

impl ContentPart {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    pub fn tool_call(id: impl Into<String>, name: impl Into<String>, arguments: Value) -> Self {
        Self::ToolCall {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }

    pub fn tool_result(id: impl Into<String>, content: impl Into<String>, is_error: bool) -> Self {
        Self::ToolResult {
            id: id.into(),
            content: content.into(),
            is_error,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmMessage {
    pub role: Role,
    pub content: Vec<ContentPart>,
}

impl LlmMessage {
    pub fn new(role: Role, content: Vec<ContentPart>) -> Self {
        Self { role, content }
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: vec![ContentPart::text(text)],
        }
    }

    pub fn human(text: impl Into<String>) -> Self {
        Self {
            role: Role::Human,
            content: vec![ContentPart::text(text)],
        }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentPart::text(text)],
        }
    }

    pub fn tool_result(id: impl Into<String>, content: impl Into<String>, is_error: bool) -> Self {
        Self {
            role: Role::ToolResult,
            content: vec![ContentPart::tool_result(id, content, is_error)],
        }
    }
}
```

### Step 3.2: Write tests

Add at the bottom of `crates/llm/src/provider.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_serialization_roundtrip() {
        let roles = vec![Role::System, Role::Human, Role::Assistant, Role::ToolResult];
        for role in roles {
            let json = serde_json::to_string(&role).unwrap();
            let deserialized: Role = serde_json::from_str(&json).unwrap();
            assert_eq!(role, deserialized);
        }
    }

    #[test]
    fn test_role_snake_case_format() {
        assert_eq!(serde_json::to_string(&Role::ToolResult).unwrap(), "\"tool_result\"");
        assert_eq!(serde_json::to_string(&Role::System).unwrap(), "\"system\"");
    }

    #[test]
    fn test_content_part_text_roundtrip() {
        let part = ContentPart::text("hello world");
        let json = serde_json::to_string(&part).unwrap();
        let deserialized: ContentPart = serde_json::from_str(&json).unwrap();
        assert_eq!(part, deserialized);
        assert!(json.contains("\"type\":\"text\""));
    }

    #[test]
    fn test_content_part_tool_call_roundtrip() {
        let part = ContentPart::tool_call(
            "call_123",
            "read_file",
            serde_json::json!({"path": "/tmp/test.txt"}),
        );
        let json = serde_json::to_string(&part).unwrap();
        let deserialized: ContentPart = serde_json::from_str(&json).unwrap();
        assert_eq!(part, deserialized);
        assert!(json.contains("\"type\":\"tool_call\""));
    }

    #[test]
    fn test_content_part_tool_result_roundtrip() {
        let part = ContentPart::tool_result("call_123", "file contents here", false);
        let json = serde_json::to_string(&part).unwrap();
        let deserialized: ContentPart = serde_json::from_str(&json).unwrap();
        assert_eq!(part, deserialized);
        assert!(json.contains("\"type\":\"tool_result\""));
    }

    #[test]
    fn test_llm_message_roundtrip() {
        let msg = LlmMessage::human("What is Rust?");
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: LlmMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(msg, deserialized);
    }

    #[test]
    fn test_llm_message_constructors() {
        let sys = LlmMessage::system("You are helpful");
        assert_eq!(sys.role, Role::System);
        assert_eq!(sys.content.len(), 1);

        let human = LlmMessage::human("Hello");
        assert_eq!(human.role, Role::Human);

        let asst = LlmMessage::assistant("Hi there");
        assert_eq!(asst.role, Role::Assistant);

        let tr = LlmMessage::tool_result("id1", "result data", false);
        assert_eq!(tr.role, Role::ToolResult);
    }

    #[test]
    fn test_llm_message_with_multiple_content_parts() {
        let msg = LlmMessage::new(
            Role::Assistant,
            vec![
                ContentPart::text("Let me read that file."),
                ContentPart::tool_call("tc_1", "read_file", serde_json::json!({"path": "main.rs"})),
            ],
        );
        assert_eq!(msg.content.len(), 2);
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: LlmMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(msg, deserialized);
    }

    #[test]
    fn test_message_types_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Role>();
        assert_send_sync::<ContentPart>();
        assert_send_sync::<LlmMessage>();
    }
}
```

### Step 3.3: Run tests

```bash
cargo test -p graphirm-llm -- provider::tests
```

**Expected:** All 9 tests pass.

### Step 3.4: Commit

```bash
git add crates/llm/src/provider.rs
git commit -m "phase-2: define Role, ContentPart, LlmMessage types"
```

---

## Task 4: Define Stream Types

- [x] **Status: complete**

**Files:**
- Modify: `crates/llm/src/stream.rs`

### Step 4.1: Write TokenUsage and StreamEvent

```rust
use serde::{Deserialize, Serialize};

use crate::error::LlmError;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u32>,
}

impl TokenUsage {
    pub fn new(input_tokens: u32, output_tokens: u32) -> Self {
        Self {
            input_tokens,
            output_tokens,
            cache_read_tokens: None,
            cache_write_tokens: None,
        }
    }

    pub fn total(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }
}

impl std::ops::Add for TokenUsage {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            input_tokens: self.input_tokens + rhs.input_tokens,
            output_tokens: self.output_tokens + rhs.output_tokens,
            cache_read_tokens: match (self.cache_read_tokens, rhs.cache_read_tokens) {
                (Some(a), Some(b)) => Some(a + b),
                (Some(a), None) | (None, Some(a)) => Some(a),
                (None, None) => None,
            },
            cache_write_tokens: match (self.cache_write_tokens, rhs.cache_write_tokens) {
                (Some(a), Some(b)) => Some(a + b),
                (Some(a), None) | (None, Some(a)) => Some(a),
                (None, None) => None,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    TextDelta(String),
    ThinkingDelta(String),
    ToolCallStart { id: String, name: String },
    ToolCallDelta { id: String, arguments_delta: String },
    ToolCallEnd { id: String },
    Done(TokenUsage),
    Error(String),
}

impl StreamEvent {
    pub fn text_delta(text: impl Into<String>) -> Self {
        Self::TextDelta(text.into())
    }

    pub fn thinking_delta(text: impl Into<String>) -> Self {
        Self::ThinkingDelta(text.into())
    }

    pub fn tool_call_start(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self::ToolCallStart {
            id: id.into(),
            name: name.into(),
        }
    }

    pub fn tool_call_delta(id: impl Into<String>, arguments_delta: impl Into<String>) -> Self {
        Self::ToolCallDelta {
            id: id.into(),
            arguments_delta: arguments_delta.into(),
        }
    }

    pub fn tool_call_end(id: impl Into<String>) -> Self {
        Self::ToolCallEnd { id: id.into() }
    }

    pub fn done(usage: TokenUsage) -> Self {
        Self::Done(usage)
    }

    pub fn error(msg: impl Into<String>) -> Self {
        Self::Error(msg.into())
    }

    pub fn is_done(&self) -> bool {
        matches!(self, Self::Done(_))
    }

    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }
}
```

### Step 4.2: Write tests

Add at the bottom of `crates/llm/src/stream.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_usage_default() {
        let usage = TokenUsage::default();
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
        assert_eq!(usage.cache_read_tokens, None);
        assert_eq!(usage.cache_write_tokens, None);
    }

    #[test]
    fn test_token_usage_new() {
        let usage = TokenUsage::new(100, 200);
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 200);
        assert_eq!(usage.total(), 300);
    }

    #[test]
    fn test_token_usage_serialization_roundtrip() {
        let usage = TokenUsage {
            input_tokens: 150,
            output_tokens: 300,
            cache_read_tokens: Some(50),
            cache_write_tokens: None,
        };
        let json = serde_json::to_string(&usage).unwrap();
        let deserialized: TokenUsage = serde_json::from_str(&json).unwrap();
        assert_eq!(usage, deserialized);
    }

    #[test]
    fn test_token_usage_skips_none_in_json() {
        let usage = TokenUsage::new(10, 20);
        let json = serde_json::to_string(&usage).unwrap();
        assert!(!json.contains("cache_read_tokens"));
        assert!(!json.contains("cache_write_tokens"));
    }

    #[test]
    fn test_token_usage_add() {
        let a = TokenUsage {
            input_tokens: 100,
            output_tokens: 200,
            cache_read_tokens: Some(10),
            cache_write_tokens: None,
        };
        let b = TokenUsage {
            input_tokens: 50,
            output_tokens: 75,
            cache_read_tokens: Some(5),
            cache_write_tokens: Some(20),
        };
        let sum = a + b;
        assert_eq!(sum.input_tokens, 150);
        assert_eq!(sum.output_tokens, 275);
        assert_eq!(sum.cache_read_tokens, Some(15));
        assert_eq!(sum.cache_write_tokens, Some(20));
    }

    #[test]
    fn test_stream_event_constructors() {
        let e1 = StreamEvent::text_delta("hello");
        assert!(matches!(e1, StreamEvent::TextDelta(ref s) if s == "hello"));

        let e2 = StreamEvent::thinking_delta("thinking...");
        assert!(matches!(e2, StreamEvent::ThinkingDelta(ref s) if s == "thinking..."));

        let e3 = StreamEvent::tool_call_start("tc_1", "bash");
        assert!(matches!(e3, StreamEvent::ToolCallStart { ref id, ref name } if id == "tc_1" && name == "bash"));

        let e4 = StreamEvent::tool_call_delta("tc_1", "{\"cmd\":");
        assert!(matches!(e4, StreamEvent::ToolCallDelta { ref id, .. } if id == "tc_1"));

        let e5 = StreamEvent::tool_call_end("tc_1");
        assert!(matches!(e5, StreamEvent::ToolCallEnd { ref id } if id == "tc_1"));

        let e6 = StreamEvent::done(TokenUsage::new(10, 20));
        assert!(e6.is_done());
        assert!(!e6.is_error());

        let e7 = StreamEvent::error("oops");
        assert!(e7.is_error());
        assert!(!e7.is_done());
    }

    #[test]
    fn test_stream_event_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<StreamEvent>();
    }
}
```

### Step 4.3: Run tests

```bash
cargo test -p graphirm-llm -- stream::tests
```

**Expected:** All 8 tests pass.

### Step 4.4: Commit

```bash
git add crates/llm/src/stream.rs
git commit -m "phase-2: define TokenUsage and StreamEvent types"
```

---

## Task 5: Define Tool Types

- [x] **Status: complete**

**Files:**
- Modify: `crates/llm/src/tool.rs`

### Step 5.1: Write ToolDefinition

```rust
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl ToolDefinition {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }

    /// Build a ToolDefinition with a JSON Schema for the parameters.
    /// Takes required field names and a map of property name -> (type, description).
    pub fn with_properties(
        name: impl Into<String>,
        description: impl Into<String>,
        properties: Vec<(&str, &str, &str)>, // (name, type, description)
        required: Vec<&str>,
    ) -> Self {
        let mut props = serde_json::Map::new();
        for (prop_name, prop_type, prop_desc) in properties {
            props.insert(
                prop_name.to_string(),
                serde_json::json!({
                    "type": prop_type,
                    "description": prop_desc,
                }),
            );
        }

        Self {
            name: name.into(),
            description: description.into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": props,
                "required": required,
            }),
        }
    }
}

/// Conversion to rig-core's ToolDefinition for passing to CompletionRequest.
impl From<ToolDefinition> for rig::completion::ToolDefinition {
    fn from(td: ToolDefinition) -> Self {
        rig::completion::ToolDefinition {
            name: td.name,
            description: td.description,
            parameters: td.parameters,
        }
    }
}

/// Conversion from rig-core's ToolDefinition.
impl From<rig::completion::ToolDefinition> for ToolDefinition {
    fn from(td: rig::completion::ToolDefinition) -> Self {
        Self {
            name: td.name,
            description: td.description,
            parameters: td.parameters,
        }
    }
}
```

### Step 5.2: Write tests

Add at the bottom of `crates/llm/src/tool.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_definition_new() {
        let tool = ToolDefinition::new(
            "bash",
            "Execute a bash command",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute"
                    }
                },
                "required": ["command"]
            }),
        );
        assert_eq!(tool.name, "bash");
        assert_eq!(tool.description, "Execute a bash command");
    }

    #[test]
    fn test_tool_definition_serialization_roundtrip() {
        let tool = ToolDefinition::new(
            "read_file",
            "Read a file",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                },
                "required": ["path"]
            }),
        );
        let json = serde_json::to_string(&tool).unwrap();
        let deserialized: ToolDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(tool, deserialized);
    }

    #[test]
    fn test_tool_definition_with_properties() {
        let tool = ToolDefinition::with_properties(
            "write_file",
            "Write content to a file",
            vec![
                ("path", "string", "The file path to write to"),
                ("content", "string", "The content to write"),
            ],
            vec!["path", "content"],
        );

        assert_eq!(tool.name, "write_file");
        let params = &tool.parameters;
        assert_eq!(params["type"], "object");
        assert!(params["properties"]["path"]["type"] == "string");
        assert!(params["properties"]["content"]["type"] == "string");
        assert_eq!(params["required"], serde_json::json!(["path", "content"]));
    }

    #[test]
    fn test_tool_definition_json_schema_for_bash() {
        let tool = ToolDefinition::with_properties(
            "bash",
            "Execute a bash command and return the output",
            vec![
                ("command", "string", "The command to execute"),
                ("timeout", "integer", "Timeout in seconds"),
            ],
            vec!["command"],
        );

        let json = serde_json::to_string_pretty(&tool).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["name"], "bash");
        assert_eq!(parsed["parameters"]["properties"]["command"]["type"], "string");
        assert_eq!(parsed["parameters"]["properties"]["timeout"]["type"], "integer");
        assert_eq!(parsed["parameters"]["required"], serde_json::json!(["command"]));
    }

    #[test]
    fn test_convert_to_rig_tool_definition() {
        let our_tool = ToolDefinition::new(
            "test_tool",
            "A test tool",
            serde_json::json!({"type": "object"}),
        );
        let rig_tool: rig::completion::ToolDefinition = our_tool.clone().into();
        assert_eq!(rig_tool.name, "test_tool");
        assert_eq!(rig_tool.description, "A test tool");
        assert_eq!(rig_tool.parameters, serde_json::json!({"type": "object"}));
    }

    #[test]
    fn test_convert_from_rig_tool_definition() {
        let rig_tool = rig::completion::ToolDefinition {
            name: "rig_tool".to_string(),
            description: "From rig".to_string(),
            parameters: serde_json::json!({"type": "object"}),
        };
        let our_tool: ToolDefinition = rig_tool.into();
        assert_eq!(our_tool.name, "rig_tool");
        assert_eq!(our_tool.description, "From rig");
    }

    #[test]
    fn test_tool_definition_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ToolDefinition>();
    }
}
```

### Step 5.3: Run tests

```bash
cargo test -p graphirm-llm -- tool::tests
```

**Expected:** All 7 tests pass.

### Step 5.4: Commit

```bash
git add crates/llm/src/tool.rs
git commit -m "phase-2: define ToolDefinition with rig-core conversions"
```

---

## Task 6: Define LlmProvider Trait + CompletionConfig + LlmResponse + StopReason

- [x] **Status: complete**

**Files:**
- Modify: `crates/llm/src/provider.rs` (append to existing types from Task 3)

### Step 6.1: Add trait and supporting types to provider.rs

Append the following after the existing `LlmMessage` code in `provider.rs`:

```rust
use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::error::LlmError;
use crate::stream::{StreamEvent, TokenUsage};
use crate::tool::ToolDefinition;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionConfig {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
}

impl CompletionConfig {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            max_tokens: None,
            temperature: None,
            stop_sequences: Vec::new(),
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = sequences;
        self
    }
}

#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub content: Vec<ContentPart>,
    pub usage: TokenUsage,
    pub stop_reason: StopReason,
}

impl LlmResponse {
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    pub fn tool_calls(&self) -> Vec<&ContentPart> {
        self.content
            .iter()
            .filter(|part| matches!(part, ContentPart::ToolCall { .. }))
            .collect()
    }

    pub fn has_tool_calls(&self) -> bool {
        self.content
            .iter()
            .any(|part| matches!(part, ContentPart::ToolCall { .. }))
    }
}

#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Send a completion request and get a full response.
    async fn complete(
        &self,
        messages: Vec<LlmMessage>,
        tools: &[ToolDefinition],
        config: &CompletionConfig,
    ) -> Result<LlmResponse, LlmError>;

    /// Send a completion request and get a streaming response.
    async fn stream(
        &self,
        messages: Vec<LlmMessage>,
        tools: &[ToolDefinition],
        config: &CompletionConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = StreamEvent> + Send>>, LlmError>;

    /// Return the provider name (e.g. "anthropic", "openai").
    fn provider_name(&self) -> &str;
}
```

### Step 6.2: Update use statements at the top of provider.rs

Make sure the top of `provider.rs` has all necessary imports. The final file should have these at the top:

```rust
use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::LlmError;
use crate::stream::{StreamEvent, TokenUsage};
use crate::tool::ToolDefinition;
```

### Step 6.3: Add trait-related tests

Add to the existing `#[cfg(test)] mod tests` in `provider.rs`:

```rust
    #[test]
    fn test_stop_reason_serialization() {
        let reasons = vec![
            (StopReason::EndTurn, "\"end_turn\""),
            (StopReason::ToolUse, "\"tool_use\""),
            (StopReason::MaxTokens, "\"max_tokens\""),
            (StopReason::StopSequence, "\"stop_sequence\""),
        ];
        for (reason, expected_json) in reasons {
            let json = serde_json::to_string(&reason).unwrap();
            assert_eq!(json, expected_json);
            let deserialized: StopReason = serde_json::from_str(&json).unwrap();
            assert_eq!(reason, deserialized);
        }
    }

    #[test]
    fn test_completion_config_builder() {
        let config = CompletionConfig::new("claude-sonnet-4-20250514")
            .with_max_tokens(4096)
            .with_temperature(0.7)
            .with_stop_sequences(vec!["STOP".to_string()]);

        assert_eq!(config.model, "claude-sonnet-4-20250514");
        assert_eq!(config.max_tokens, Some(4096));
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.stop_sequences, vec!["STOP"]);
    }

    #[test]
    fn test_completion_config_defaults() {
        let config = CompletionConfig::new("gpt-4o");
        assert_eq!(config.max_tokens, None);
        assert_eq!(config.temperature, None);
        assert!(config.stop_sequences.is_empty());
    }

    #[test]
    fn test_llm_response_text_content() {
        let response = LlmResponse {
            content: vec![
                ContentPart::text("Hello "),
                ContentPart::text("world!"),
            ],
            usage: TokenUsage::new(10, 5),
            stop_reason: StopReason::EndTurn,
        };
        assert_eq!(response.text_content(), "Hello world!");
        assert!(!response.has_tool_calls());
        assert!(response.tool_calls().is_empty());
    }

    #[test]
    fn test_llm_response_tool_calls() {
        let response = LlmResponse {
            content: vec![
                ContentPart::text("I'll read that file."),
                ContentPart::tool_call("tc_1", "read_file", serde_json::json!({"path": "main.rs"})),
                ContentPart::tool_call("tc_2", "bash", serde_json::json!({"command": "ls"})),
            ],
            usage: TokenUsage::new(50, 100),
            stop_reason: StopReason::ToolUse,
        };
        assert!(response.has_tool_calls());
        assert_eq!(response.tool_calls().len(), 2);
        assert_eq!(response.text_content(), "I'll read that file.");
    }

    #[test]
    fn test_llm_provider_trait_is_object_safe() {
        // This compiles only if LlmProvider is object-safe
        fn _accept_provider(_: &dyn LlmProvider) {}
        fn _accept_boxed(_: Box<dyn LlmProvider>) {}
    }
```

### Step 6.4: Update lib.rs re-exports

Ensure `crates/llm/src/lib.rs` re-exports the new types:

```rust
pub mod error;
pub mod provider;
pub mod stream;
pub mod tool;

pub use error::LlmError;
pub use provider::{
    CompletionConfig, ContentPart, LlmMessage, LlmProvider, LlmResponse, Role, StopReason,
};
pub use stream::{StreamEvent, TokenUsage};
pub use tool::ToolDefinition;
```

### Step 6.5: Run all tests

```bash
cargo test -p graphirm-llm
```

**Expected:** All tests pass (error: 7, provider: 15, stream: 8, tool: 7 = 37 total).

### Step 6.6: Commit

```bash
git add crates/llm/
git commit -m "phase-2: define LlmProvider trait, CompletionConfig, LlmResponse, StopReason"
```

---

## Task 7: Implement MockProvider

- [x] **Status: complete**

**Files:**
- Create: `crates/llm/src/mock.rs`
- Modify: `crates/llm/src/lib.rs` (add `pub mod mock;`)

### Step 7.1: Create mock.rs

```rust
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream;
use futures::Stream;

use crate::error::LlmError;
use crate::provider::{
    CompletionConfig, ContentPart, LlmMessage, LlmProvider, LlmResponse, StopReason,
};
use crate::stream::{StreamEvent, TokenUsage};
use crate::tool::ToolDefinition;

/// A mock LLM provider for testing. Returns canned responses in order.
/// Thread-safe: can be shared across async tasks.
#[derive(Debug, Clone)]
pub struct MockProvider {
    responses: Arc<Vec<MockResponse>>,
    call_count: Arc<AtomicUsize>,
}

#[derive(Debug, Clone)]
pub struct MockResponse {
    pub content: Vec<ContentPart>,
    pub stop_reason: StopReason,
    pub usage: TokenUsage,
}

impl MockResponse {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentPart::text(text)],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::new(10, 20),
        }
    }

    pub fn with_tool_call(
        text: impl Into<String>,
        tool_id: impl Into<String>,
        tool_name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self {
            content: vec![
                ContentPart::text(text),
                ContentPart::tool_call(tool_id, tool_name, arguments),
            ],
            stop_reason: StopReason::ToolUse,
            usage: TokenUsage::new(50, 100),
        }
    }

    pub fn with_usage(mut self, usage: TokenUsage) -> Self {
        self.usage = usage;
        self
    }

    pub fn with_stop_reason(mut self, reason: StopReason) -> Self {
        self.stop_reason = reason;
        self
    }
}

impl MockProvider {
    pub fn new(responses: Vec<MockResponse>) -> Self {
        Self {
            responses: Arc::new(responses),
            call_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Create a MockProvider that always returns the same text.
    pub fn fixed(text: impl Into<String>) -> Self {
        Self::new(vec![MockResponse::text(text)])
    }

    /// How many times complete() or stream() has been called.
    pub fn call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }

    fn next_response(&self) -> Result<MockResponse, LlmError> {
        let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
        let effective_idx = if self.responses.is_empty() {
            return Err(LlmError::provider("MockProvider has no responses configured"));
        } else if idx >= self.responses.len() {
            self.responses.len() - 1
        } else {
            idx
        };
        Ok(self.responses[effective_idx].clone())
    }
}

#[async_trait]
impl LlmProvider for MockProvider {
    async fn complete(
        &self,
        _messages: Vec<LlmMessage>,
        _tools: &[ToolDefinition],
        _config: &CompletionConfig,
    ) -> Result<LlmResponse, LlmError> {
        let response = self.next_response()?;
        Ok(LlmResponse {
            content: response.content,
            usage: response.usage,
            stop_reason: response.stop_reason,
        })
    }

    async fn stream(
        &self,
        _messages: Vec<LlmMessage>,
        _tools: &[ToolDefinition],
        _config: &CompletionConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = StreamEvent> + Send>>, LlmError> {
        let response = self.next_response()?;

        let mut events: Vec<StreamEvent> = Vec::new();
        for part in &response.content {
            match part {
                ContentPart::Text { text } => {
                    for chunk in text.as_bytes().chunks(10) {
                        events.push(StreamEvent::text_delta(
                            String::from_utf8_lossy(chunk).to_string(),
                        ));
                    }
                }
                ContentPart::ToolCall { id, name, arguments } => {
                    events.push(StreamEvent::tool_call_start(id.clone(), name.clone()));
                    let args_str = serde_json::to_string(arguments)
                        .unwrap_or_else(|_| "{}".to_string());
                    events.push(StreamEvent::tool_call_delta(id.clone(), args_str));
                    events.push(StreamEvent::tool_call_end(id.clone()));
                }
                ContentPart::ToolResult { .. } => {}
            }
        }
        events.push(StreamEvent::done(response.usage));

        Ok(Box::pin(stream::iter(events)))
    }

    fn provider_name(&self) -> &str {
        "mock"
    }
}
```

### Step 7.2: Add `pub mod mock;` to lib.rs

Update `crates/llm/src/lib.rs`:

```rust
pub mod error;
pub mod mock;
pub mod provider;
pub mod stream;
pub mod tool;

pub use error::LlmError;
pub use mock::{MockProvider, MockResponse};
pub use provider::{
    CompletionConfig, ContentPart, LlmMessage, LlmProvider, LlmResponse, Role, StopReason,
};
pub use stream::{StreamEvent, TokenUsage};
pub use tool::ToolDefinition;
```

### Step 7.3: Write tests

Add at the bottom of `crates/llm/src/mock.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[tokio::test]
    async fn test_mock_provider_complete_text() {
        let provider = MockProvider::fixed("Hello, I'm Claude!");
        let config = CompletionConfig::new("test-model");

        let response = provider
            .complete(vec![LlmMessage::human("Hi")], &[], &config)
            .await
            .unwrap();

        assert_eq!(response.text_content(), "Hello, I'm Claude!");
        assert_eq!(response.stop_reason, StopReason::EndTurn);
        assert!(!response.has_tool_calls());
        assert_eq!(provider.call_count(), 1);
    }

    #[tokio::test]
    async fn test_mock_provider_complete_with_tool_call() {
        let provider = MockProvider::new(vec![MockResponse::with_tool_call(
            "Let me read that file.",
            "tc_1",
            "read_file",
            serde_json::json!({"path": "main.rs"}),
        )]);

        let config = CompletionConfig::new("test-model");
        let response = provider
            .complete(vec![LlmMessage::human("Read main.rs")], &[], &config)
            .await
            .unwrap();

        assert_eq!(response.stop_reason, StopReason::ToolUse);
        assert!(response.has_tool_calls());
        assert_eq!(response.tool_calls().len(), 1);
    }

    #[tokio::test]
    async fn test_mock_provider_multiple_responses_in_sequence() {
        let provider = MockProvider::new(vec![
            MockResponse::text("First response"),
            MockResponse::text("Second response"),
        ]);
        let config = CompletionConfig::new("test-model");

        let r1 = provider
            .complete(vec![LlmMessage::human("1")], &[], &config)
            .await
            .unwrap();
        assert_eq!(r1.text_content(), "First response");

        let r2 = provider
            .complete(vec![LlmMessage::human("2")], &[], &config)
            .await
            .unwrap();
        assert_eq!(r2.text_content(), "Second response");

        // Third call reuses last response
        let r3 = provider
            .complete(vec![LlmMessage::human("3")], &[], &config)
            .await
            .unwrap();
        assert_eq!(r3.text_content(), "Second response");
        assert_eq!(provider.call_count(), 3);
    }

    #[tokio::test]
    async fn test_mock_provider_stream() {
        let provider = MockProvider::fixed("Hello!");
        let config = CompletionConfig::new("test-model");

        let stream = provider
            .stream(vec![LlmMessage::human("Hi")], &[], &config)
            .await
            .unwrap();

        let events: Vec<StreamEvent> = stream.collect().await;

        // "Hello!" is 6 bytes, fits in one 10-byte chunk, so 1 TextDelta + Done
        assert!(events.len() >= 2);

        let has_text = events.iter().any(|e| matches!(e, StreamEvent::TextDelta(_)));
        let has_done = events.iter().any(|e| e.is_done());
        assert!(has_text);
        assert!(has_done);
    }

    #[tokio::test]
    async fn test_mock_provider_stream_with_tool_call() {
        let provider = MockProvider::new(vec![MockResponse::with_tool_call(
            "Reading file",
            "tc_1",
            "read_file",
            serde_json::json!({"path": "src/main.rs"}),
        )]);
        let config = CompletionConfig::new("test-model");

        let stream = provider
            .stream(vec![LlmMessage::human("Read it")], &[], &config)
            .await
            .unwrap();

        let events: Vec<StreamEvent> = stream.collect().await;

        let has_tool_start = events.iter().any(|e| {
            matches!(e, StreamEvent::ToolCallStart { name, .. } if name == "read_file")
        });
        let has_tool_end = events.iter().any(|e| {
            matches!(e, StreamEvent::ToolCallEnd { id } if id == "tc_1")
        });
        assert!(has_tool_start);
        assert!(has_tool_end);
    }

    #[tokio::test]
    async fn test_mock_provider_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockProvider>();

        // Can be used as dyn LlmProvider
        let provider: Box<dyn LlmProvider> = Box::new(MockProvider::fixed("test"));
        let config = CompletionConfig::new("test");
        let response = provider
            .complete(vec![LlmMessage::human("hi")], &[], &config)
            .await
            .unwrap();
        assert_eq!(response.text_content(), "test");
    }

    #[test]
    fn test_mock_response_builders() {
        let r1 = MockResponse::text("hello");
        assert_eq!(r1.content.len(), 1);
        assert_eq!(r1.stop_reason, StopReason::EndTurn);

        let r2 = MockResponse::text("custom")
            .with_usage(TokenUsage::new(100, 200))
            .with_stop_reason(StopReason::MaxTokens);
        assert_eq!(r2.usage.input_tokens, 100);
        assert_eq!(r2.stop_reason, StopReason::MaxTokens);
    }
}
```

### Step 7.4: Run tests

```bash
cargo test -p graphirm-llm -- mock::tests
```

**Expected:** All 7 tests pass.

### Step 7.5: Commit

```bash
git add crates/llm/
git commit -m "phase-2: implement MockProvider for testing"
```

---

## Task 8: Implement Anthropic Provider

- [x] **Status: complete**

**Files:**
- Create: `crates/llm/src/anthropic.rs`
- Modify: `crates/llm/src/lib.rs` (add `pub mod anthropic;`)

### Step 8.1: Create anthropic.rs

This provider wraps rig-core's `rig::providers::anthropic` module. It converts between our types and rig-core's types, then delegates to rig-core for the actual API call.

```rust
use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use rig::completion::message as rig_msg;
use rig::completion::{
    CompletionModel as RigCompletionModel, CompletionRequest as RigCompletionRequest,
};
use rig::message::Message as RigMessage;
use rig::OneOrMany;
use tracing::{debug, warn};

use crate::error::LlmError;
use crate::provider::{
    CompletionConfig, ContentPart, LlmMessage, LlmProvider, LlmResponse, Role, StopReason,
};
use crate::stream::{StreamEvent, TokenUsage};
use crate::tool::ToolDefinition;

/// Anthropic LLM provider wrapping rig-core's Anthropic client.
pub struct AnthropicProvider {
    client: rig::providers::anthropic::Anthropic,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        let client = rig::providers::anthropic::Anthropic::new(&api_key.into());
        Self { client }
    }

    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
            LlmError::config("ANTHROPIC_API_KEY environment variable not set")
        })?;
        Ok(Self::new(api_key))
    }

    fn build_completion_request(
        &self,
        messages: Vec<LlmMessage>,
        tools: &[ToolDefinition],
        config: &CompletionConfig,
    ) -> Result<(rig::providers::anthropic::completion::CompletionModel, RigCompletionRequest), LlmError> {
        let model = self.client.completion_model(&config.model);

        let (system_prompt, chat_messages) = Self::split_system_and_chat(messages)?;

        let mut builder = model.completion_request(RigMessage::User {
            content: OneOrMany::one(rig_msg::UserContent::text(".")),
        });

        if let Some(system) = &system_prompt {
            builder = builder.preamble(system);
        }

        if let Some(max_tokens) = config.max_tokens {
            builder = builder.max_tokens(max_tokens as u64);
        }

        if let Some(temp) = config.temperature {
            builder = builder.temperature(temp as f64);
        }

        for tool in tools {
            let rig_tool: rig::completion::ToolDefinition = tool.clone().into();
            builder = builder.tool(rig_tool);
        }

        let mut request = builder.build();
        request.chat_history = Self::convert_messages_to_rig(chat_messages)?;

        Ok((model, request))
    }

    fn split_system_and_chat(
        messages: Vec<LlmMessage>,
    ) -> Result<(Option<String>, Vec<LlmMessage>), LlmError> {
        let mut system_parts: Vec<String> = Vec::new();
        let mut chat_messages: Vec<LlmMessage> = Vec::new();

        for msg in messages {
            match msg.role {
                Role::System => {
                    for part in &msg.content {
                        if let ContentPart::Text { text } = part {
                            system_parts.push(text.clone());
                        }
                    }
                }
                _ => chat_messages.push(msg),
            }
        }

        let system = if system_parts.is_empty() {
            None
        } else {
            Some(system_parts.join("\n\n"))
        };

        Ok((system, chat_messages))
    }

    fn convert_messages_to_rig(
        messages: Vec<LlmMessage>,
    ) -> Result<OneOrMany<RigMessage>, LlmError> {
        let mut rig_messages: Vec<RigMessage> = Vec::new();

        for msg in messages {
            match msg.role {
                Role::Human => {
                    let content: Vec<rig_msg::UserContent> = msg
                        .content
                        .into_iter()
                        .filter_map(|part| match part {
                            ContentPart::Text { text } => Some(rig_msg::UserContent::text(text)),
                            _ => None,
                        })
                        .collect();
                    if !content.is_empty() {
                        rig_messages.push(RigMessage::User {
                            content: OneOrMany::many(content)
                                .unwrap_or_else(|_| OneOrMany::one(rig_msg::UserContent::text(""))),
                        });
                    }
                }
                Role::Assistant => {
                    let content: Vec<rig_msg::AssistantContent> = msg
                        .content
                        .into_iter()
                        .filter_map(|part| match part {
                            ContentPart::Text { text } => {
                                Some(rig_msg::AssistantContent::text(text))
                            }
                            ContentPart::ToolCall { id, name, arguments } => {
                                Some(rig_msg::AssistantContent::tool_call(id, name, arguments))
                            }
                            _ => None,
                        })
                        .collect();
                    if !content.is_empty() {
                        rig_messages.push(RigMessage::Assistant {
                            id: None,
                            content: OneOrMany::many(content)
                                .unwrap_or_else(|_| OneOrMany::one(rig_msg::AssistantContent::text(""))),
                        });
                    }
                }
                Role::ToolResult => {
                    for part in msg.content {
                        if let ContentPart::ToolResult { id, content, is_error: _ } = part {
                            rig_messages.push(RigMessage::tool_result(id, content));
                        }
                    }
                }
                Role::System => {
                    warn!("System message found in chat messages — should have been extracted");
                }
            }
        }

        OneOrMany::many(rig_messages).map_err(|_| {
            LlmError::provider("No messages to send to Anthropic")
        })
    }

    fn convert_rig_response(
        response: rig::completion::CompletionResponse<
            rig::providers::anthropic::completion::CompletionResponse,
        >,
    ) -> LlmResponse {
        let mut content_parts: Vec<ContentPart> = Vec::new();

        for item in response.choice.iter() {
            match item {
                rig_msg::AssistantContent::Text(text) => {
                    content_parts.push(ContentPart::text(&text.text));
                }
                rig_msg::AssistantContent::ToolCall(tool_call) => {
                    content_parts.push(ContentPart::tool_call(
                        &tool_call.id,
                        &tool_call.function.name,
                        tool_call.function.arguments.clone(),
                    ));
                }
                _ => {
                    debug!("Ignoring unsupported AssistantContent variant");
                }
            }
        }

        let has_tool_calls = content_parts
            .iter()
            .any(|p| matches!(p, ContentPart::ToolCall { .. }));

        let stop_reason = if has_tool_calls {
            StopReason::ToolUse
        } else {
            StopReason::EndTurn
        };

        let usage = TokenUsage {
            input_tokens: response.usage.input_tokens as u32,
            output_tokens: response.usage.output_tokens as u32,
            cache_read_tokens: if response.usage.cached_input_tokens > 0 {
                Some(response.usage.cached_input_tokens as u32)
            } else {
                None
            },
            cache_write_tokens: None,
        };

        LlmResponse {
            content: content_parts,
            usage,
            stop_reason,
        }
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    async fn complete(
        &self,
        messages: Vec<LlmMessage>,
        tools: &[ToolDefinition],
        config: &CompletionConfig,
    ) -> Result<LlmResponse, LlmError> {
        let (model, request) = self.build_completion_request(messages, tools, config)?;

        let response = model
            .completion(request)
            .await
            .map_err(|e| LlmError::provider(e.to_string()))?;

        Ok(Self::convert_rig_response(response))
    }

    async fn stream(
        &self,
        messages: Vec<LlmMessage>,
        tools: &[ToolDefinition],
        config: &CompletionConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = StreamEvent> + Send>>, LlmError> {
        let (model, request) = self.build_completion_request(messages, tools, config)?;

        let rig_stream = model
            .stream(request)
            .await
            .map_err(|e| LlmError::stream(e.to_string()))?;

        // Convert rig's StreamingCompletionResponse into our CompletionResponse
        // then construct stream events from it.
        // rig-core's StreamingCompletionResponse can be converted to CompletionResponse.
        let response: rig::completion::CompletionResponse<
            Option<rig::providers::anthropic::completion::CompletionResponse>,
        > = rig_stream.into();

        let mut events: Vec<StreamEvent> = Vec::new();
        for item in response.choice.iter() {
            match item {
                rig_msg::AssistantContent::Text(text) => {
                    events.push(StreamEvent::text_delta(text.text.clone()));
                }
                rig_msg::AssistantContent::ToolCall(tool_call) => {
                    events.push(StreamEvent::tool_call_start(
                        tool_call.id.clone(),
                        tool_call.function.name.clone(),
                    ));
                    let args_str = serde_json::to_string(&tool_call.function.arguments)
                        .unwrap_or_else(|_| "{}".to_string());
                    events.push(StreamEvent::tool_call_delta(tool_call.id.clone(), args_str));
                    events.push(StreamEvent::tool_call_end(tool_call.id.clone()));
                }
                _ => {}
            }
        }

        let usage = TokenUsage {
            input_tokens: response.usage.input_tokens as u32,
            output_tokens: response.usage.output_tokens as u32,
            cache_read_tokens: if response.usage.cached_input_tokens > 0 {
                Some(response.usage.cached_input_tokens as u32)
            } else {
                None
            },
            cache_write_tokens: None,
        };
        events.push(StreamEvent::done(usage));

        Ok(Box::pin(futures::stream::iter(events)))
    }

    fn provider_name(&self) -> &str {
        "anthropic"
    }
}
```

### Step 8.2: Add module to lib.rs

Add `pub mod anthropic;` to `crates/llm/src/lib.rs` and re-export:

```rust
pub mod anthropic;
pub mod error;
pub mod mock;
pub mod provider;
pub mod stream;
pub mod tool;

pub use anthropic::AnthropicProvider;
pub use error::LlmError;
pub use mock::{MockProvider, MockResponse};
pub use provider::{
    CompletionConfig, ContentPart, LlmMessage, LlmProvider, LlmResponse, Role, StopReason,
};
pub use stream::{StreamEvent, TokenUsage};
pub use tool::ToolDefinition;
```

### Step 8.3: Verify it compiles

```bash
cargo check -p graphirm-llm
```

**Expected:** Compiles. If rig-core API differs slightly (e.g., `Anthropic::new` signature, field names), adapt the code to match actual API. The key contract is: our `AnthropicProvider` implements `LlmProvider` and converts between our types and rig-core types.

**Note:** The exact rig-core API may have changed. If `Anthropic::new()` takes `&str` vs `impl Into<String>`, or if `CompletionResponse` field names differ, adjust accordingly. The streaming implementation here uses `into()` to collect the stream into a `CompletionResponse` — this is a pragmatic first pass. A true event-by-event streaming implementation can be added in a follow-up.

### Step 8.4: Write unit test (no API key needed — tests construction only)

Add at the bottom of `crates/llm/src/anthropic.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_system_and_chat() {
        let messages = vec![
            LlmMessage::system("You are helpful"),
            LlmMessage::system("Be concise"),
            LlmMessage::human("Hello"),
            LlmMessage::assistant("Hi there"),
        ];
        let (system, chat) = AnthropicProvider::split_system_and_chat(messages).unwrap();
        assert_eq!(system, Some("You are helpful\n\nBe concise".to_string()));
        assert_eq!(chat.len(), 2);
        assert_eq!(chat[0].role, Role::Human);
        assert_eq!(chat[1].role, Role::Assistant);
    }

    #[test]
    fn test_split_no_system() {
        let messages = vec![LlmMessage::human("Hello")];
        let (system, chat) = AnthropicProvider::split_system_and_chat(messages).unwrap();
        assert_eq!(system, None);
        assert_eq!(chat.len(), 1);
    }

    #[test]
    fn test_convert_messages_to_rig() {
        let messages = vec![
            LlmMessage::human("What is Rust?"),
            LlmMessage::assistant("Rust is a systems programming language."),
            LlmMessage::human("Tell me more."),
        ];
        let rig_messages = AnthropicProvider::convert_messages_to_rig(messages).unwrap();
        assert_eq!(rig_messages.len(), 3);
    }

    #[test]
    fn test_convert_tool_result_messages() {
        let messages = vec![
            LlmMessage::human("Read this file"),
            LlmMessage::new(
                Role::Assistant,
                vec![ContentPart::tool_call(
                    "tc_1",
                    "read_file",
                    serde_json::json!({"path": "test.rs"}),
                )],
            ),
            LlmMessage::tool_result("tc_1", "file contents here", false),
        ];
        let rig_messages = AnthropicProvider::convert_messages_to_rig(messages).unwrap();
        assert_eq!(rig_messages.len(), 3);
    }

    #[test]
    fn test_provider_name() {
        let provider = AnthropicProvider::new("test-key");
        assert_eq!(provider.provider_name(), "anthropic");
    }
}
```

### Step 8.5: Run tests

```bash
cargo test -p graphirm-llm -- anthropic::tests
```

**Expected:** All 5 tests pass.

### Step 8.6: Commit

```bash
git add crates/llm/
git commit -m "phase-2: implement AnthropicProvider wrapping rig-core"
```

---

## Task 9: Implement OpenAI Provider

- [x] **Status: complete**

**Files:**
- Create: `crates/llm/src/openai.rs`
- Modify: `crates/llm/src/lib.rs` (add `pub mod openai;`)

### Step 9.1: Create openai.rs

Same pattern as Anthropic — wrap rig-core's OpenAI client.

```rust
use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use rig::completion::message as rig_msg;
use rig::completion::{
    CompletionModel as RigCompletionModel, CompletionRequest as RigCompletionRequest,
};
use rig::message::Message as RigMessage;
use rig::OneOrMany;
use tracing::{debug, warn};

use crate::error::LlmError;
use crate::provider::{
    CompletionConfig, ContentPart, LlmMessage, LlmProvider, LlmResponse, Role, StopReason,
};
use crate::stream::{StreamEvent, TokenUsage};
use crate::tool::ToolDefinition;

/// OpenAI LLM provider wrapping rig-core's OpenAI client.
pub struct OpenAiProvider {
    client: rig::providers::openai::Client,
}

impl OpenAiProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        let client = rig::providers::openai::Client::new(&api_key.into());
        Self { client }
    }

    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
            LlmError::config("OPENAI_API_KEY environment variable not set")
        })?;
        Ok(Self::new(api_key))
    }

    fn build_completion_request(
        &self,
        messages: Vec<LlmMessage>,
        tools: &[ToolDefinition],
        config: &CompletionConfig,
    ) -> Result<(rig::providers::openai::CompletionModel, RigCompletionRequest), LlmError> {
        let model = self.client.completion_model(&config.model);

        let (system_prompt, chat_messages) = Self::split_system_and_chat(messages)?;

        let mut builder = model.completion_request(RigMessage::User {
            content: OneOrMany::one(rig_msg::UserContent::text(".")),
        });

        if let Some(system) = &system_prompt {
            builder = builder.preamble(system);
        }

        if let Some(max_tokens) = config.max_tokens {
            builder = builder.max_tokens(max_tokens as u64);
        }

        if let Some(temp) = config.temperature {
            builder = builder.temperature(temp as f64);
        }

        for tool in tools {
            let rig_tool: rig::completion::ToolDefinition = tool.clone().into();
            builder = builder.tool(rig_tool);
        }

        let mut request = builder.build();
        request.chat_history = Self::convert_messages_to_rig(chat_messages)?;

        Ok((model, request))
    }

    fn split_system_and_chat(
        messages: Vec<LlmMessage>,
    ) -> Result<(Option<String>, Vec<LlmMessage>), LlmError> {
        let mut system_parts: Vec<String> = Vec::new();
        let mut chat_messages: Vec<LlmMessage> = Vec::new();

        for msg in messages {
            match msg.role {
                Role::System => {
                    for part in &msg.content {
                        if let ContentPart::Text { text } = part {
                            system_parts.push(text.clone());
                        }
                    }
                }
                _ => chat_messages.push(msg),
            }
        }

        let system = if system_parts.is_empty() {
            None
        } else {
            Some(system_parts.join("\n\n"))
        };

        Ok((system, chat_messages))
    }

    fn convert_messages_to_rig(
        messages: Vec<LlmMessage>,
    ) -> Result<OneOrMany<RigMessage>, LlmError> {
        let mut rig_messages: Vec<RigMessage> = Vec::new();

        for msg in messages {
            match msg.role {
                Role::Human => {
                    let content: Vec<rig_msg::UserContent> = msg
                        .content
                        .into_iter()
                        .filter_map(|part| match part {
                            ContentPart::Text { text } => Some(rig_msg::UserContent::text(text)),
                            _ => None,
                        })
                        .collect();
                    if !content.is_empty() {
                        rig_messages.push(RigMessage::User {
                            content: OneOrMany::many(content)
                                .unwrap_or_else(|_| OneOrMany::one(rig_msg::UserContent::text(""))),
                        });
                    }
                }
                Role::Assistant => {
                    let content: Vec<rig_msg::AssistantContent> = msg
                        .content
                        .into_iter()
                        .filter_map(|part| match part {
                            ContentPart::Text { text } => {
                                Some(rig_msg::AssistantContent::text(text))
                            }
                            ContentPart::ToolCall { id, name, arguments } => {
                                Some(rig_msg::AssistantContent::tool_call(id, name, arguments))
                            }
                            _ => None,
                        })
                        .collect();
                    if !content.is_empty() {
                        rig_messages.push(RigMessage::Assistant {
                            id: None,
                            content: OneOrMany::many(content)
                                .unwrap_or_else(|_| OneOrMany::one(rig_msg::AssistantContent::text(""))),
                        });
                    }
                }
                Role::ToolResult => {
                    for part in msg.content {
                        if let ContentPart::ToolResult { id, content, is_error: _ } = part {
                            rig_messages.push(RigMessage::tool_result(id, content));
                        }
                    }
                }
                Role::System => {
                    warn!("System message found in chat messages — should have been extracted");
                }
            }
        }

        OneOrMany::many(rig_messages).map_err(|_| {
            LlmError::provider("No messages to send to OpenAI")
        })
    }

    fn convert_rig_response(
        response: rig::completion::CompletionResponse<
            rig::providers::openai::CompletionResponse,
        >,
    ) -> LlmResponse {
        let mut content_parts: Vec<ContentPart> = Vec::new();

        for item in response.choice.iter() {
            match item {
                rig_msg::AssistantContent::Text(text) => {
                    content_parts.push(ContentPart::text(&text.text));
                }
                rig_msg::AssistantContent::ToolCall(tool_call) => {
                    content_parts.push(ContentPart::tool_call(
                        &tool_call.id,
                        &tool_call.function.name,
                        tool_call.function.arguments.clone(),
                    ));
                }
                _ => {
                    debug!("Ignoring unsupported AssistantContent variant");
                }
            }
        }

        let has_tool_calls = content_parts
            .iter()
            .any(|p| matches!(p, ContentPart::ToolCall { .. }));

        let stop_reason = if has_tool_calls {
            StopReason::ToolUse
        } else {
            StopReason::EndTurn
        };

        let usage = TokenUsage {
            input_tokens: response.usage.input_tokens as u32,
            output_tokens: response.usage.output_tokens as u32,
            cache_read_tokens: if response.usage.cached_input_tokens > 0 {
                Some(response.usage.cached_input_tokens as u32)
            } else {
                None
            },
            cache_write_tokens: None,
        };

        LlmResponse {
            content: content_parts,
            usage,
            stop_reason,
        }
    }
}

#[async_trait]
impl LlmProvider for OpenAiProvider {
    async fn complete(
        &self,
        messages: Vec<LlmMessage>,
        tools: &[ToolDefinition],
        config: &CompletionConfig,
    ) -> Result<LlmResponse, LlmError> {
        let (model, request) = self.build_completion_request(messages, tools, config)?;

        let response = model
            .completion(request)
            .await
            .map_err(|e| LlmError::provider(e.to_string()))?;

        Ok(Self::convert_rig_response(response))
    }

    async fn stream(
        &self,
        messages: Vec<LlmMessage>,
        tools: &[ToolDefinition],
        config: &CompletionConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = StreamEvent> + Send>>, LlmError> {
        let (model, request) = self.build_completion_request(messages, tools, config)?;

        let rig_stream = model
            .stream(request)
            .await
            .map_err(|e| LlmError::stream(e.to_string()))?;

        let response: rig::completion::CompletionResponse<
            Option<rig::providers::openai::CompletionResponse>,
        > = rig_stream.into();

        let mut events: Vec<StreamEvent> = Vec::new();
        for item in response.choice.iter() {
            match item {
                rig_msg::AssistantContent::Text(text) => {
                    events.push(StreamEvent::text_delta(text.text.clone()));
                }
                rig_msg::AssistantContent::ToolCall(tool_call) => {
                    events.push(StreamEvent::tool_call_start(
                        tool_call.id.clone(),
                        tool_call.function.name.clone(),
                    ));
                    let args_str = serde_json::to_string(&tool_call.function.arguments)
                        .unwrap_or_else(|_| "{}".to_string());
                    events.push(StreamEvent::tool_call_delta(tool_call.id.clone(), args_str));
                    events.push(StreamEvent::tool_call_end(tool_call.id.clone()));
                }
                _ => {}
            }
        }

        let usage = TokenUsage {
            input_tokens: response.usage.input_tokens as u32,
            output_tokens: response.usage.output_tokens as u32,
            cache_read_tokens: if response.usage.cached_input_tokens > 0 {
                Some(response.usage.cached_input_tokens as u32)
            } else {
                None
            },
            cache_write_tokens: None,
        };
        events.push(StreamEvent::done(usage));

        Ok(Box::pin(futures::stream::iter(events)))
    }

    fn provider_name(&self) -> &str {
        "openai"
    }
}
```

### Step 9.2: Add module and re-export to lib.rs

```rust
pub mod anthropic;
pub mod error;
pub mod mock;
pub mod openai;
pub mod provider;
pub mod stream;
pub mod tool;

pub use anthropic::AnthropicProvider;
pub use error::LlmError;
pub use mock::{MockProvider, MockResponse};
pub use openai::OpenAiProvider;
pub use provider::{
    CompletionConfig, ContentPart, LlmMessage, LlmProvider, LlmResponse, Role, StopReason,
};
pub use stream::{StreamEvent, TokenUsage};
pub use tool::ToolDefinition;
```

### Step 9.3: Write unit tests

Add at the bottom of `crates/llm/src/openai.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_system_and_chat() {
        let messages = vec![
            LlmMessage::system("You are helpful"),
            LlmMessage::human("Hello"),
        ];
        let (system, chat) = OpenAiProvider::split_system_and_chat(messages).unwrap();
        assert_eq!(system, Some("You are helpful".to_string()));
        assert_eq!(chat.len(), 1);
    }

    #[test]
    fn test_convert_messages_to_rig() {
        let messages = vec![
            LlmMessage::human("What is 2+2?"),
            LlmMessage::assistant("4"),
            LlmMessage::human("Thanks"),
        ];
        let rig_messages = OpenAiProvider::convert_messages_to_rig(messages).unwrap();
        assert_eq!(rig_messages.len(), 3);
    }

    #[test]
    fn test_provider_name() {
        let provider = OpenAiProvider::new("test-key");
        assert_eq!(provider.provider_name(), "openai");
    }
}
```

### Step 9.4: Run tests

```bash
cargo test -p graphirm-llm -- openai::tests
```

**Expected:** All 3 tests pass.

### Step 9.5: Commit

```bash
git add crates/llm/
git commit -m "phase-2: implement OpenAiProvider wrapping rig-core"
```

---

## Task 10: Implement Provider Factory

- [x] **Status: complete**

**Files:**
- Create: `crates/llm/src/factory.rs`
- Modify: `crates/llm/src/lib.rs` (add `pub mod factory;`)

### Step 10.1: Create factory.rs

```rust
use crate::anthropic::AnthropicProvider;
use crate::error::LlmError;
use crate::openai::OpenAiProvider;
use crate::provider::LlmProvider;

/// Parse a model string like "anthropic/claude-sonnet-4-20250514" into (provider, model).
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

/// Create an LLM provider from a provider name and API key.
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

/// Create an LLM provider from a "provider/model" string and API key.
/// Returns (provider, model_name) where model_name can be used in CompletionConfig.
pub fn create_provider_from_model_string(
    model_string: &str,
    api_key: &str,
) -> Result<(Box<dyn LlmProvider>, String), LlmError> {
    let (provider_name, model_name) = parse_model_string(model_string)?;
    let provider = create_provider(provider_name, api_key)?;
    Ok((provider, model_name.to_string()))
}
```

### Step 10.2: Add module and re-exports to lib.rs

Update `crates/llm/src/lib.rs`:

```rust
pub mod anthropic;
pub mod error;
pub mod factory;
pub mod mock;
pub mod openai;
pub mod provider;
pub mod stream;
pub mod tool;

pub use anthropic::AnthropicProvider;
pub use error::LlmError;
pub use factory::{create_provider, create_provider_from_model_string, parse_model_string};
pub use mock::{MockProvider, MockResponse};
pub use openai::OpenAiProvider;
pub use provider::{
    CompletionConfig, ContentPart, LlmMessage, LlmProvider, LlmResponse, Role, StopReason,
};
pub use stream::{StreamEvent, TokenUsage};
pub use tool::ToolDefinition;
```

### Step 10.3: Write tests

Add at the bottom of `crates/llm/src/factory.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_model_string_valid() {
        let (provider, model) = parse_model_string("anthropic/claude-sonnet-4-20250514").unwrap();
        assert_eq!(provider, "anthropic");
        assert_eq!(model, "claude-sonnet-4-20250514");
    }

    #[test]
    fn test_parse_model_string_openai() {
        let (provider, model) = parse_model_string("openai/gpt-4o").unwrap();
        assert_eq!(provider, "openai");
        assert_eq!(model, "gpt-4o");
    }

    #[test]
    fn test_parse_model_string_with_nested_slash() {
        let (provider, model) = parse_model_string("openai/ft:gpt-4o:my-org:custom").unwrap();
        assert_eq!(provider, "openai");
        assert_eq!(model, "ft:gpt-4o:my-org:custom");
    }

    #[test]
    fn test_parse_model_string_no_slash() {
        let err = parse_model_string("just-a-model").unwrap_err();
        assert!(err.to_string().contains("Expected 'provider/model' format"));
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
        let provider = create_provider("anthropic", "test-key").unwrap();
        assert_eq!(provider.provider_name(), "anthropic");
    }

    #[test]
    fn test_create_provider_openai() {
        let provider = create_provider("openai", "test-key").unwrap();
        assert_eq!(provider.provider_name(), "openai");
    }

    #[test]
    fn test_create_provider_unknown() {
        let err = create_provider("gemini", "test-key").unwrap_err();
        assert!(err.to_string().contains("Unknown provider"));
        assert!(err.to_string().contains("gemini"));
    }

    #[test]
    fn test_create_provider_from_model_string() {
        let (provider, model) =
            create_provider_from_model_string("anthropic/claude-sonnet-4-20250514", "test-key")
                .unwrap();
        assert_eq!(provider.provider_name(), "anthropic");
        assert_eq!(model, "claude-sonnet-4-20250514");
    }

    #[test]
    fn test_create_provider_from_invalid_model_string() {
        let err = create_provider_from_model_string("bad-format", "test-key").unwrap_err();
        assert!(err.to_string().contains("Expected 'provider/model' format"));
    }
}
```

### Step 10.4: Run tests

```bash
cargo test -p graphirm-llm -- factory::tests
```

**Expected:** All 11 tests pass.

### Step 10.5: Commit

```bash
git add crates/llm/
git commit -m "phase-2: implement provider factory with model string parsing"
```

---

## Task 11: Integration Test — Full Round-Trip with MockProvider

- [x] **Status: complete**

**Files:**
- Create: `crates/llm/tests/integration.rs`

### Step 11.1: Create the integration test file

```rust
use futures::StreamExt;
use graphirm_llm::{
    CompletionConfig, ContentPart, LlmMessage, LlmProvider, LlmResponse, MockProvider,
    MockResponse, Role, StopReason, StreamEvent, TokenUsage, ToolDefinition,
};

#[tokio::test]
async fn test_full_roundtrip_text_only() {
    let provider = MockProvider::fixed("Hello! I'm your coding assistant.");
    let config = CompletionConfig::new("mock-model").with_max_tokens(1024);

    let messages = vec![
        LlmMessage::system("You are a helpful coding assistant."),
        LlmMessage::human("Hello, who are you?"),
    ];

    let response = provider.complete(messages, &[], &config).await.unwrap();

    assert_eq!(response.text_content(), "Hello! I'm your coding assistant.");
    assert_eq!(response.stop_reason, StopReason::EndTurn);
    assert!(!response.has_tool_calls());
}

#[tokio::test]
async fn test_full_roundtrip_with_tool_calls() {
    let read_file_tool = ToolDefinition::with_properties(
        "read_file",
        "Read a file from the filesystem",
        vec![("path", "string", "The absolute file path to read")],
        vec!["path"],
    );

    let bash_tool = ToolDefinition::with_properties(
        "bash",
        "Execute a bash command",
        vec![
            ("command", "string", "The command to execute"),
            ("timeout", "integer", "Timeout in seconds"),
        ],
        vec!["command"],
    );

    let tools = vec![read_file_tool, bash_tool];

    // First call: assistant decides to use a tool
    let provider = MockProvider::new(vec![
        MockResponse::with_tool_call(
            "I'll read that file for you.",
            "tc_001",
            "read_file",
            serde_json::json!({"path": "/home/user/main.rs"}),
        ),
        MockResponse::text("The file contains a main function that prints 'Hello, world!'."),
    ]);

    let config = CompletionConfig::new("mock-model");

    // Turn 1: User asks, assistant uses tool
    let messages_turn1 = vec![
        LlmMessage::system("You are a coding assistant."),
        LlmMessage::human("What's in main.rs?"),
    ];

    let response1 = provider.complete(messages_turn1, &tools, &config).await.unwrap();
    assert_eq!(response1.stop_reason, StopReason::ToolUse);
    assert!(response1.has_tool_calls());

    let tool_calls = response1.tool_calls();
    assert_eq!(tool_calls.len(), 1);
    match &tool_calls[0] {
        ContentPart::ToolCall { id, name, arguments } => {
            assert_eq!(id, "tc_001");
            assert_eq!(name, "read_file");
            assert_eq!(arguments["path"], "/home/user/main.rs");
        }
        _ => panic!("Expected ToolCall"),
    }

    // Turn 2: Provide tool result, get final answer
    let messages_turn2 = vec![
        LlmMessage::system("You are a coding assistant."),
        LlmMessage::human("What's in main.rs?"),
        LlmMessage::new(Role::Assistant, response1.content.clone()),
        LlmMessage::tool_result(
            "tc_001",
            "fn main() { println!(\"Hello, world!\"); }",
            false,
        ),
    ];

    let response2 = provider.complete(messages_turn2, &tools, &config).await.unwrap();
    assert_eq!(response2.stop_reason, StopReason::EndTurn);
    assert!(!response2.has_tool_calls());
    assert!(response2.text_content().contains("Hello, world!"));
}

#[tokio::test]
async fn test_streaming_roundtrip() {
    let provider = MockProvider::new(vec![MockResponse::with_tool_call(
        "Let me check that.",
        "tc_1",
        "bash",
        serde_json::json!({"command": "ls -la"}),
    )]);

    let config = CompletionConfig::new("mock-model");
    let tools = vec![ToolDefinition::with_properties(
        "bash",
        "Run a command",
        vec![("command", "string", "Command to run")],
        vec!["command"],
    )];

    let stream = provider
        .stream(vec![LlmMessage::human("List files")], &tools, &config)
        .await
        .unwrap();

    let events: Vec<StreamEvent> = stream.collect().await;

    // Verify we got text deltas
    let text_deltas: Vec<&StreamEvent> = events
        .iter()
        .filter(|e| matches!(e, StreamEvent::TextDelta(_)))
        .collect();
    assert!(!text_deltas.is_empty(), "Should have text deltas");

    // Verify we got tool call events
    let tool_starts: Vec<&StreamEvent> = events
        .iter()
        .filter(|e| matches!(e, StreamEvent::ToolCallStart { .. }))
        .collect();
    assert_eq!(tool_starts.len(), 1, "Should have exactly one tool call start");

    let tool_ends: Vec<&StreamEvent> = events
        .iter()
        .filter(|e| matches!(e, StreamEvent::ToolCallEnd { .. }))
        .collect();
    assert_eq!(tool_ends.len(), 1, "Should have exactly one tool call end");

    // Verify stream ends with Done
    let last = events.last().unwrap();
    assert!(last.is_done(), "Stream should end with Done event");
}

#[tokio::test]
async fn test_provider_as_trait_object() {
    let provider: Box<dyn LlmProvider> = Box::new(MockProvider::fixed("trait object works"));
    let config = CompletionConfig::new("test");

    let response = provider
        .complete(vec![LlmMessage::human("test")], &[], &config)
        .await
        .unwrap();

    assert_eq!(response.text_content(), "trait object works");
}

#[tokio::test]
async fn test_token_usage_tracking() {
    let usage = TokenUsage {
        input_tokens: 150,
        output_tokens: 300,
        cache_read_tokens: Some(50),
        cache_write_tokens: None,
    };
    let provider = MockProvider::new(vec![
        MockResponse::text("response").with_usage(usage.clone())
    ]);
    let config = CompletionConfig::new("test");

    let response = provider
        .complete(vec![LlmMessage::human("test")], &[], &config)
        .await
        .unwrap();

    assert_eq!(response.usage.input_tokens, 150);
    assert_eq!(response.usage.output_tokens, 300);
    assert_eq!(response.usage.cache_read_tokens, Some(50));
    assert_eq!(response.usage.cache_write_tokens, None);
    assert_eq!(response.usage.total(), 450);
}

#[tokio::test]
async fn test_multi_turn_conversation_flow() {
    let provider = MockProvider::new(vec![
        MockResponse::text("I can help with that. What language?"),
        MockResponse::text("Here's a Rust hello world:\n```rust\nfn main() {\n    println!(\"Hello!\");\n}\n```"),
    ]);

    let config = CompletionConfig::new("test");

    // Turn 1
    let r1 = provider
        .complete(
            vec![LlmMessage::human("Write me a hello world program")],
            &[],
            &config,
        )
        .await
        .unwrap();
    assert!(r1.text_content().contains("What language"));

    // Turn 2
    let r2 = provider
        .complete(
            vec![
                LlmMessage::human("Write me a hello world program"),
                LlmMessage::assistant(&r1.text_content()),
                LlmMessage::human("Rust please"),
            ],
            &[],
            &config,
        )
        .await
        .unwrap();
    assert!(r2.text_content().contains("Rust"));
    assert_eq!(provider.call_count(), 2);
}
```

### Step 11.2: Run the integration tests

```bash
cargo test -p graphirm-llm --test integration
```

**Expected:** All 6 integration tests pass.

### Step 11.3: Run all tests together

```bash
cargo test -p graphirm-llm
```

**Expected:** All tests pass — unit tests from Tasks 2-9 + integration tests from this task.

### Step 11.4: Commit

```bash
git add crates/llm/
git commit -m "phase-2: add integration tests for full LLM provider round-trip"
```

---

## Summary

| Task | Description | Files | Tests |
|------|-------------|-------|-------|
| 1 | Crate skeleton + Cargo.toml | 6 files | compile check |
| 2 | LlmError enum | error.rs | 7 |
| 3 | Message types (Role, ContentPart, LlmMessage) | provider.rs | 9 |
| 4 | Stream types (TokenUsage, StreamEvent) | stream.rs | 8 |
| 5 | Tool types (ToolDefinition) | tool.rs | 7 |
| 6 | LlmProvider trait + Config + Response + StopReason | provider.rs | 6 |
| 7 | MockProvider | mock.rs | 7 |
| 8 | AnthropicProvider | anthropic.rs | 5 |
| 9 | OpenAiProvider | openai.rs | 3 |
| 10 | Provider factory | factory.rs | 11 |
| 11 | Integration tests | tests/integration.rs | 6 |
| **Total** | | **10 source files + 1 test file** | **~69 tests** |

## Adaptation Notes

The Anthropic and OpenAI provider implementations are written against the rig-core 0.31 API as documented at docs.rs. Key rig-core types used:

- `rig::completion::CompletionModel` trait — `completion()`, `stream()`, `completion_request()`
- `rig::completion::CompletionRequest` — `chat_history`, `tools`, `temperature`, `max_tokens`, `preamble`
- `rig::completion::CompletionResponse<T>` — `choice: OneOrMany<AssistantContent>`, `usage: Usage`
- `rig::completion::message::Message` — `User { content }`, `Assistant { id, content }`
- `rig::completion::message::AssistantContent` — `Text(Text)`, `ToolCall(ToolCall)`, `Reasoning(Reasoning)`
- `rig::completion::message::UserContent` — `Text(Text)`, `ToolResult(ToolResult)`
- `rig::completion::message::ToolCall` — `id`, `function: ToolFunction { name, arguments }`
- `rig::completion::ToolDefinition` — `name`, `description`, `parameters: Value`
- `rig::completion::Usage` — `input_tokens`, `output_tokens`, `total_tokens`, `cached_input_tokens`
- `rig::providers::anthropic::Anthropic::new(&str)` — client constructor
- `rig::providers::openai::Client::new(&str)` — client constructor

If the actual rig-core API at implementation time differs (method signatures, field names, module paths), adapt the provider implementations while keeping our `LlmProvider` trait interface stable. The trait is the contract; the providers are adapters.

## Post-Phase Follow-Ups (Not Part of This Plan)

- **True streaming:** Current streaming collects rig-core's stream into a response then re-emits events. A future PR should consume rig-core's stream event-by-event for real incremental delivery.
- **Retry logic:** Add configurable retry with exponential backoff for rate limits.
- **Provider config from TOML:** Wire up `config/default.toml` provider sections to the factory.
- **Additional providers:** DeepSeek, Google Gemini, etc. via rig-core's other provider modules.
