//! Task definition types for graphirm-eval.
#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// A single evaluation task.
#[derive(Debug, Clone)]
pub struct EvalTask {
    /// Unique snake_case identifier, e.g. "add-fibonacci"
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Tags for filtering: "basic", "memory", "knowledge", "graph"
    pub tags: Vec<String>,
    /// One or more prompts to send sequentially to the same session.
    pub prompts: Vec<String>,
    /// How to determine pass/fail
    pub verifier: Verifier,
    /// Maximum agent turns before declaring timeout
    pub max_turns: u32,
    /// Seconds to wait for the session to go idle after each prompt
    pub timeout_secs: u64,
    /// Whether to enable structured response segmentation for this task's session.
    pub enable_segments: bool,
}

/// A cross-session task that requires two separate sessions (used for memory recall).
#[derive(Debug, Clone)]
pub struct CrossSessionTask {
    pub id: String,
    pub name: String,
    pub tags: Vec<String>,
    pub session1_prompt: String,
    pub session2_prompt: String,
    /// Applied to session 2's last response
    pub verifier: Verifier,
    pub timeout_secs: u64,
}

/// Verification strategy for an eval task.
#[derive(Debug, Clone)]
pub enum Verifier {
    /// The final assistant message must contain this substring (case-insensitive).
    ResponseContains { substring: String },
    /// The final assistant message must contain at least one of these substrings
    /// (case-insensitive). Useful for checking error phrases where wording varies
    /// ("not found" vs "does not exist" vs "no such file").
    ResponseContainsAny { substrings: Vec<String> },
    /// The final assistant message must NOT contain this substring (case-insensitive).
    /// Use to verify the agent didn't hallucinate content.
    ResponseNotContains { substring: String },
    /// Run a shell command; pass if exit code == 0.
    CommandSucceeds { command: String, args: Vec<String> },
    /// The file at `path` must exist and contain `substring`.
    FileContains { path: String, substring: String },
    /// Run a shell command, trim its stdout, and check the final assistant response
    /// contains that output (case-insensitive). Use to avoid hardcoding values that
    /// change as source files are edited (e.g. line counts).
    ResponseContainsCommandOutput { command: String, args: Vec<String> },
    /// GET /api/graph/{session}/knowledge — pass if count >= min_count.
    KnowledgeNodeCount { min_count: usize },
    /// GET /api/graph/{session} — pass if node count >= min_nodes and
    /// at least one node has node_type matching type_name.
    GraphContains { min_nodes: usize, type_name: String },
    /// GET /api/graph/{session} — pass if at least one Content node exists
    /// with `node_type.content_type` matching the given string.
    GraphContainsContentType { content_type: String },
    /// All verifiers must pass.
    All(Vec<Verifier>),
}

/// The outcome of running one EvalTask.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: String,
    pub passed: bool,
    pub turns_used: u32,
    pub elapsed_secs: f64,
    pub failure_reason: Option<String>,
    pub session_id: Option<String>,
}

impl TaskResult {
    pub fn pass(task_id: &str, turns_used: u32, elapsed_secs: f64) -> Self {
        Self {
            task_id: task_id.to_string(),
            passed: true,
            turns_used,
            elapsed_secs,
            failure_reason: None,
            session_id: None,
        }
    }

    pub fn fail(task_id: &str, reason: impl Into<String>) -> Self {
        Self {
            task_id: task_id.to_string(),
            passed: false,
            turns_used: 0,
            elapsed_secs: 0.0,
            failure_reason: Some(reason.into()),
            session_id: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_has_required_fields() {
        let t = EvalTask {
            id: "test-task".to_string(),
            name: "Test Task".to_string(),
            tags: vec!["basic".to_string()],
            prompts: vec!["Hello".to_string()],
            verifier: Verifier::ResponseContains { substring: "world".to_string() },
            max_turns: 5,
            timeout_secs: 30,
            enable_segments: false,
        };
        assert_eq!(t.id, "test-task");
        assert!(matches!(t.verifier, Verifier::ResponseContains { .. }));
        assert!(!t.enable_segments);
    }

    #[test]
    fn task_result_pass_and_fail() {
        let pass = TaskResult::pass("test-task", 2, 5.0);
        assert!(pass.passed);
        let fail = TaskResult::fail("test-task", "file not found");
        assert!(!fail.passed);
        assert!(fail.failure_reason.is_some());
    }
}
