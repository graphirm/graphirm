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
    /// When set, only these segment types are included in the reconstructed context window.
    pub segment_filter: Option<Vec<String>>,
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

impl Verifier {
    /// Checks that decide the score.
    ///
    /// A file or a command is the task's evidence. Reply-text checks in the
    /// same group are skipped, so a later verification summary cannot hide a
    /// result the file or the command already shows.
    pub fn decisive_checks(&self) -> Vec<&Verifier> {
        match self {
            Verifier::All(verifiers) => {
                let artifact = verifiers.iter().any(Verifier::is_file_or_command);
                if artifact {
                    verifiers
                        .iter()
                        .filter(|verifier| !verifier.is_reply_text())
                        .collect()
                } else {
                    verifiers.iter().collect()
                }
            }
            other => vec![other],
        }
    }

    fn is_file_or_command(&self) -> bool {
        matches!(
            self,
            Verifier::FileContains { .. } | Verifier::CommandSucceeds { .. }
        )
    }

    fn is_reply_text(&self) -> bool {
        matches!(
            self,
            Verifier::ResponseContains { .. }
                | Verifier::ResponseContainsAny { .. }
                | Verifier::ResponseNotContains { .. }
        )
    }
}

/// How one task finished.
///
/// `Error` is infrastructure (a rate limit that retries could not clear, for
/// example). It is recorded and left out of the pass rate so a blip does not
/// change the score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskOutcome {
    Pass,
    Fail,
    /// The verifier already held, and the session was still running at the timeout.
    RanOver,
    Error,
}

/// The outcome of running one EvalTask.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: String,
    pub passed: bool,
    pub outcome: TaskOutcome,
    pub turns_used: u32,
    pub elapsed_secs: f64,
    pub failure_reason: Option<String>,
    pub session_id: Option<String>,
    /// Last assistant text, kept when the task does not pass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_answer: Option<String>,
    /// Tool results from the session, kept when the task does not pass.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_trace: Vec<ToolTrace>,
    /// Newest compaction summary, kept when the task does not pass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compaction_summary: Option<String>,
}

/// One tool result stored with a failed task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolTrace {
    pub name: String,
    pub is_error: bool,
    pub output: String,
}

const TOOL_OUTPUT_CAP: usize = 2000;
const ANSWER_CAP: usize = 4000;

/// Drop commas that group digits (`3,537` → `3537`) so a formatted count
/// still matches the command's digits. Other commas stay put.
pub fn answer_contains_command_output(answer: &str, expected: &str) -> bool {
    let expected = strip_thousands_separators(expected).to_lowercase();
    if expected.is_empty() {
        return false;
    }
    strip_thousands_separators(answer)
        .to_lowercase()
        .contains(&expected)
}

fn strip_thousands_separators(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut index = 0;
    while index < chars.len() {
        let comma_between_digits = chars[index] == ','
            && index > 0
            && chars[index - 1].is_ascii_digit()
            && chars[index + 1..].len() >= 3
            && chars[index + 1..index + 4]
                .iter()
                .all(|ch| ch.is_ascii_digit())
            && chars.get(index + 4).is_none_or(|ch| !ch.is_ascii_digit());
        if comma_between_digits {
            index += 1;
            continue;
        }
        out.push(chars[index]);
        index += 1;
    }
    out
}

/// A timeout whose verifier already holds is a different miss from a timeout
/// that never finished the task.
pub fn timeout_failure_reason(assistants: u32, verifier_passed: bool) -> String {
    if verifier_passed {
        "finished but didn't stop".to_string()
    } else {
        format!("session timed out after {assistants} model responses")
    }
}

fn clip(text: &str, cap: usize) -> String {
    let mut clipped = String::new();
    for (index, ch) in text.chars().enumerate() {
        if index == cap {
            clipped.push('…');
            return clipped;
        }
        clipped.push(ch);
    }
    clipped
}

/// Newest compaction summary from a knowledge-node list.
/// Other knowledge nodes are ignored. An empty summary is ignored.
pub fn latest_compaction_summary(nodes: &[serde_json::Value]) -> Option<String> {
    let mut best: Option<(&str, &str)> = None;
    for node in nodes {
        let data = &node["node_type"];
        let is_summary = data["type"].as_str() == Some("Knowledge")
            && data["entity"].as_str() == Some("session_summary")
            && data["entity_type"].as_str() == Some("compaction");
        if !is_summary {
            continue;
        }
        let summary = data["summary"].as_str().unwrap_or("");
        if summary.is_empty() {
            continue;
        }
        let created = node["created_at"].as_str().unwrap_or("");
        let newer = best.is_none_or(|(at, _)| created > at);
        if newer {
            best = Some((created, summary));
        }
    }
    best.map(|(_, summary)| clip(summary, ANSWER_CAP))
}

/// Final assistant text and tool outputs from a session message list.
pub fn failure_transcript(messages: &[serde_json::Value]) -> (String, Vec<ToolTrace>) {
    let answer = messages
        .iter()
        .rev()
        .find(|message| message["node_type"]["role"].as_str() == Some("assistant"))
        .and_then(|message| message["node_type"]["content"].as_str())
        .unwrap_or("");
    let tools = messages
        .iter()
        .filter(|message| message["node_type"]["role"].as_str() == Some("tool"))
        .map(|message| ToolTrace {
            name: message["metadata"]["tool_name"]
                .as_str()
                .unwrap_or("unknown")
                .to_string(),
            is_error: message["metadata"]["is_error"].as_bool().unwrap_or(false),
            output: clip(
                message["node_type"]["content"].as_str().unwrap_or(""),
                TOOL_OUTPUT_CAP,
            ),
        })
        .collect();
    (clip(answer, ANSWER_CAP), tools)
}

impl TaskResult {
    pub fn pass(task_id: &str, turns_used: u32, elapsed_secs: f64) -> Self {
        Self {
            task_id: task_id.to_string(),
            passed: true,
            outcome: TaskOutcome::Pass,
            turns_used,
            elapsed_secs,
            failure_reason: None,
            session_id: None,
            final_answer: None,
            tool_trace: Vec::new(),
            compaction_summary: None,
        }
    }

    pub fn fail(task_id: &str, reason: impl Into<String>) -> Self {
        Self {
            task_id: task_id.to_string(),
            passed: false,
            outcome: TaskOutcome::Fail,
            turns_used: 0,
            elapsed_secs: 0.0,
            failure_reason: Some(reason.into()),
            session_id: None,
            final_answer: None,
            tool_trace: Vec::new(),
            compaction_summary: None,
        }
    }

    /// The file or answer was already correct, and the session did not stop in time.
    /// `passed` stays false so the on-time rate does not count it.
    pub fn ran_over(task_id: &str, reason: impl Into<String>) -> Self {
        Self {
            task_id: task_id.to_string(),
            passed: false,
            outcome: TaskOutcome::RanOver,
            turns_used: 0,
            elapsed_secs: 0.0,
            failure_reason: Some(reason.into()),
            session_id: None,
            final_answer: None,
            tool_trace: Vec::new(),
            compaction_summary: None,
        }
    }

    /// Infrastructure problem. `passed` stays false so a reader of the bool
    /// does not count it as a success; [`SuiteScore`] leaves it out of the rate.
    pub fn error(task_id: &str, reason: impl Into<String>) -> Self {
        Self {
            task_id: task_id.to_string(),
            passed: false,
            outcome: TaskOutcome::Error,
            turns_used: 0,
            elapsed_secs: 0.0,
            failure_reason: Some(reason.into()),
            session_id: None,
            final_answer: None,
            tool_trace: Vec::new(),
            compaction_summary: None,
        }
    }
}

/// On-time passes, wrong answers, and correct runs that did not stop.
/// Infrastructure errors are counted apart and do not change either rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SuiteScore {
    pub passed: usize,
    pub failed: usize,
    pub ran_over: usize,
    pub errored: usize,
}

impl SuiteScore {
    pub fn from_results(results: &[TaskResult]) -> Self {
        let mut score = Self {
            passed: 0,
            failed: 0,
            ran_over: 0,
            errored: 0,
        };
        for result in results {
            match result.outcome {
                TaskOutcome::Pass => score.passed += 1,
                TaskOutcome::Fail => score.failed += 1,
                TaskOutcome::RanOver => score.ran_over += 1,
                TaskOutcome::Error => score.errored += 1,
            }
        }
        score
    }

    /// Tasks with a right or wrong answer. Infrastructure errors are left out.
    pub fn scored(&self) -> usize {
        self.passed + self.ran_over + self.failed
    }

    /// On-time passes plus correct runs that did not stop.
    pub fn correct(&self) -> usize {
        self.passed + self.ran_over
    }

    /// Share of scored tasks that finished on time.
    pub fn percent(&self) -> f64 {
        let scored = self.scored();
        if scored == 0 {
            0.0
        } else {
            self.passed as f64 / scored as f64 * 100.0
        }
    }

    /// Share of scored tasks whose verifier held, including runs that did not stop.
    pub fn correct_percent(&self) -> f64 {
        let scored = self.scored();
        if scored == 0 {
            0.0
        } else {
            self.correct() as f64 / scored as f64 * 100.0
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
            verifier: Verifier::ResponseContains {
                substring: "world".to_string(),
            },
            max_turns: 5,
            timeout_secs: 30,
            enable_segments: false,
            segment_filter: None,
        };
        assert_eq!(t.id, "test-task");
        assert!(matches!(t.verifier, Verifier::ResponseContains { .. }));
        assert!(!t.enable_segments);
    }

    #[test]
    fn task_result_pass_and_fail() {
        let pass = TaskResult::pass("test-task", 2, 5.0);
        assert!(pass.passed);
        assert_eq!(pass.outcome, TaskOutcome::Pass);
        let fail = TaskResult::fail("test-task", "file not found");
        assert!(!fail.passed);
        assert_eq!(fail.outcome, TaskOutcome::Fail);
        assert!(fail.failure_reason.is_some());
    }

    #[test]
    fn latest_compaction_summary_keeps_the_newest_text() {
        let older = serde_json::json!({
            "created_at": "2026-10-09T09:00:00Z",
            "node_type": {
                "type": "Knowledge",
                "entity": "session_summary",
                "entity_type": "compaction",
                "summary": "STALE_SUMMARY",
                "confidence": 1.0
            }
        });
        let newer = serde_json::json!({
            "created_at": "2026-10-09T09:21:20Z",
            "node_type": {
                "type": "Knowledge",
                "entity": "session_summary",
                "entity_type": "compaction",
                "summary": "PINNED_SUMMARY TOKEN_EARLY_184729",
                "confidence": 1.0
            }
        });
        let other = serde_json::json!({
            "created_at": "2026-10-09T09:30:00Z",
            "node_type": {
                "type": "Knowledge",
                "entity": "file",
                "entity_type": "code",
                "summary": "not a compaction summary",
                "confidence": 0.5
            }
        });
        assert_eq!(
            latest_compaction_summary(&[older, newer, other]).as_deref(),
            Some("PINNED_SUMMARY TOKEN_EARLY_184729")
        );
        assert_eq!(latest_compaction_summary(&[]), None);
    }

    #[test]
    fn a_thousands_separator_still_matches_the_command_output() {
        assert!(answer_contains_command_output(
            "The file has 3,537 lines.",
            "3537"
        ));
        assert!(answer_contains_command_output("3537 lines", "3537"));
        assert!(!answer_contains_command_output(
            "about three thousand",
            "3537"
        ));
        assert!(!answer_contains_command_output(
            "hello, world",
            "helloworld"
        ));
    }

    #[test]
    fn a_file_or_command_check_outranks_the_reply_text() {
        let mixed = Verifier::All(vec![
            Verifier::CommandSucceeds {
                command: "true".to_string(),
                args: vec![],
            },
            Verifier::ResponseContainsAny {
                substrings: vec!["fixed".to_string()],
            },
        ]);
        let checks = mixed.decisive_checks();
        assert_eq!(checks.len(), 1);
        assert!(matches!(checks[0], Verifier::CommandSucceeds { .. }));

        let reply_only = Verifier::All(vec![Verifier::ResponseContains {
            substring: "def ".to_string(),
        }]);
        assert_eq!(reply_only.decisive_checks().len(), 1);
        assert!(matches!(
            reply_only.decisive_checks()[0],
            Verifier::ResponseContains { .. }
        ));
    }

    #[test]
    fn a_timeout_after_the_verifier_would_pass_is_finished_but_not_stopped() {
        assert_eq!(timeout_failure_reason(3, true), "finished but didn't stop");
        assert_eq!(
            timeout_failure_reason(3, false),
            "session timed out after 3 model responses"
        );
    }

    #[test]
    fn infrastructure_error_is_excluded_from_the_score() {
        let results = vec![
            TaskResult::pass("a", 1, 1.0),
            TaskResult::fail("b", "verifier returned false"),
            TaskResult::error("c", "rate limit exhausted"),
        ];
        let score = SuiteScore::from_results(&results);
        assert_eq!(score.passed, 1);
        assert_eq!(score.failed, 1);
        assert_eq!(score.ran_over, 0);
        assert_eq!(score.errored, 1);
        assert_eq!(score.scored(), 2);
        assert_eq!(score.percent(), 50.0);
        assert_eq!(score.correct(), 1);
    }

    #[test]
    fn a_correct_run_that_did_not_stop_is_not_a_wrong_answer() {
        let results = vec![
            TaskResult::pass("a", 1, 1.0),
            TaskResult::ran_over("b", "finished but didn't stop"),
            TaskResult::fail("c", "verifier returned false"),
        ];
        let score = SuiteScore::from_results(&results);
        assert_eq!(score.correct(), 2);
        assert_eq!(score.failed, 1);
        assert_eq!(score.ran_over, 1);
        assert_eq!(score.passed, 1);
        assert_eq!(score.scored(), 3);
        assert!((score.correct_percent() - 200.0 / 3.0).abs() < 0.01);
        assert!((score.percent() - 100.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn failure_transcript_keeps_the_answer_and_the_tool_error() {
        let messages = vec![
            serde_json::json!({
                "node_type": {"role": "user", "content": "count the lines"},
                "metadata": {}
            }),
            serde_json::json!({
                "node_type": {"role": "tool", "content": "No such file"},
                "metadata": {"tool_name": "bash", "is_error": true}
            }),
            serde_json::json!({
                "node_type": {"role": "assistant", "content": "about 100 lines"},
                "metadata": {}
            }),
        ];
        let (answer, tools) = failure_transcript(&messages);
        assert_eq!(answer, "about 100 lines");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "bash");
        assert!(tools[0].is_error);
        assert_eq!(tools[0].output, "No such file");
    }
}
