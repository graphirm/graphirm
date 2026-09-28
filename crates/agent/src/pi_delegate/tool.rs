//! The `delegate_pi` tool: hands one bounded task to Pi and records the run.
//!
//! `execute` is a straight pipeline over the sibling modules (design D2–D5):
//!
//! 1. refuse when `ctx.disable_bash` (Pi runs shell);
//! 2. parse `task` / `context_paths` / `timeout_seconds`;
//! 3. `probe_version` — a missing binary fails here, **before** any graph
//!    write, so an uninstalled Pi never leaves a Task node behind;
//! 4. [`PiRun::begin`] → Task + Pi Agent, `graph_changed` so the UI shows the
//!    delegation immediately;
//! 5. [`spawn_pi`] with the session's cancellation token and the effective
//!    timeout; the process wrapper owns cancel/timeout, so `execute` returns
//!    whenever the turn is cancelled or the deadline passes;
//! 6. drain events: every Pi tool call becomes a `role:"tool"` node with
//!    `ToolStart` / `ToolEnd` on the sink; Pi's `bash` / `write` / `edit` are
//!    scored by the judge (when configured) **observe-only** — the verdict is
//!    stored as `hitl_judge{action:"observed"}` and counted, never gated;
//! 7. map the outcome onto [`PiRunFinish`] + the tool result, and call
//!    [`PiRun::finish`] on every path (the `Drop` net is a fallback only);
//! 8. final `graph_changed` so the Task status flips live.
//!
//! The sink is only ever used through `ctx` — nothing here moves
//! `ctx.event_sink` into a task that could outlive `execute`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use graphirm_graph::nodes::NodeId;
use graphirm_llm::LlmError;
use graphirm_tools::registry::ToolRegistry;
use graphirm_tools::{Tool, ToolContext, ToolError, ToolEventSink, ToolOutput};
use serde_json::{Value, json};
use tokio::task::JoinHandle;

use super::events::{MAX_ERROR_CHARS, PiEvent, flatten_result};
use super::graph::{PiRun, PiRunFinish, PiToolCall, truncate_within};
use super::process::{
    PiProcessError, PiRunHandle, PiRunOutcome, PiSpawnSpec, probe_version, spawn_pi,
};
use crate::config::{AgentConfig, PiConfig};
use crate::hitl::{JudgeOutcome, is_destructive_tool};
use crate::hitl_judge::{DestructiveJudge, JUDGE_ACTION_OBSERVED, JudgeVerdict, build_judge};

/// Registered tool name.
pub const PI_DELEGATE_TOOL_NAME: &str = "delegate_pi";

/// At most this many `Warnings:` bullets in the tool result.
const MAX_SUMMARY_WARNINGS: usize = 10;

/// Pi's `stopReason` for an assistant message that is followed by tool calls.
const STOP_REASON_TOOL_USE: &str = "toolUse";

const DESCRIPTION: &str = "Delegate one bounded, well-specified implementation task to Pi, an \
external coding agent. Pi runs autonomously in the session workspace with its own read, write, \
edit and shell tools and cannot be paused: its tool calls are recorded in the graph and scored \
by the judge, but they are not gated for approval — only this delegate_pi call is. Returns Pi's \
final message plus a summary of what it ran. Pi's reply is a claim, not proof: verify the result \
with your own tools (read, diff, cargo_check, tests) before reporting the work as done. Use it \
for multi-file implementation with a clear brief and a verification command; do not use it for \
reads, questions, single small edits, or anything that needs graph memory or per-step approval.";

/// First line of the system-prompt notice; its presence makes
/// [`apply_pi_delegate_system_notice`] a no-op.
const NOTICE_MARKER: &str = "## Delegating to Pi (`delegate_pi`)";

const NOTICE_BODY: &str = "\
- Delegate when the work is a multi-file implementation with a clear written brief and a \
verification command (build, tests) Pi can run itself.
- Do not delegate reads, questions, small single-file edits, or anything that needs the graph's \
memory, session context, or the user's approval at each step.
- Pi runs autonomously: its tool calls are recorded in the graph and scored by the judge, but \
they cannot be paused. Only your `delegate_pi` call is gated.
- Pi's final message is a claim, not proof. Verify the result with your own tools (`read`, \
`diff`, `cargo_check`, tests) before reporting the work as done.
- Prefer a clean git state or a dedicated branch before delegating so Pi's changes can be \
reviewed and reverted.
";

/// Appends the `delegate_pi` guidance to a system prompt. Idempotent: a
/// prompt that already contains [`NOTICE_MARKER`] is left unchanged.
pub fn apply_pi_delegate_system_notice(prompt: &mut String) {
    if prompt.contains(NOTICE_MARKER) {
        return;
    }
    prompt.push_str("\n\n");
    prompt.push_str(NOTICE_MARKER);
    prompt.push('\n');
    prompt.push_str(NOTICE_BODY);
}

/// `true` for tools that `disable_bash` removes from every tool list: `bash`
/// itself and `delegate_pi` (Pi runs shell). Shared by the director's list in
/// `workflow::stream_and_record` and the subagent scoping in `multi.rs`.
pub(crate) fn hidden_under_disable_bash(name: &str) -> bool {
    name == "bash" || name == PI_DELEGATE_TOOL_NAME
}

/// Registers `delegate_pi` when `[agent.pi]` is registrable
/// ([`PiConfig::is_registrable`]: enabled, non-blank binary/provider/model).
///
/// An enabled-but-invalid config is skipped with a warning; a zero
/// `timeout_seconds` falls back to the default. The binary is probed once so
/// a missing Pi is visible in the logs at startup, but the tool is registered
/// regardless — the failure then surfaces to the model as a tool error rather
/// than as a silently absent tool. Never logs `extra_args` (they may carry
/// secrets) or anything Pi reads its provider key from.
pub async fn register_pi_delegate(registry: &mut ToolRegistry, config: &AgentConfig) {
    let Some(pi) = config.pi.as_ref().filter(|p| p.enabled) else {
        return;
    };
    if !pi.is_registrable() {
        tracing::warn!(
            "[agent.pi] is enabled but `binary`, `provider` or `model` is blank; \
             delegate_pi not registered"
        );
        return;
    }
    let mut pi = pi.clone();
    if pi.timeout_seconds == 0 {
        let default_secs = PiConfig::default().timeout_seconds;
        tracing::warn!(
            default_secs,
            "[agent.pi].timeout_seconds is 0; using the default"
        );
        pi.timeout_seconds = default_secs;
    }
    let judge = build_judge(config);
    match probe_version(&pi).await {
        Ok(version) => tracing::info!(
            %version,
            binary = %pi.binary,
            provider = %pi.provider,
            model = %pi.model,
            judge = judge.is_some(),
            "delegate_pi registered"
        ),
        Err(e) => tracing::warn!(
            binary = %pi.binary,
            error = %e,
            "pi binary not found; delegate_pi registered but will fail at call time"
        ),
    }
    registry.register(Arc::new(PiDelegateTool::new(pi, judge)));
}

/// The `delegate_pi` tool. Stateless across calls; one instance per registry.
pub struct PiDelegateTool {
    config: PiConfig,
    judge: Option<Arc<DestructiveJudge>>,
}

impl PiDelegateTool {
    pub fn new(config: PiConfig, judge: Option<Arc<DestructiveJudge>>) -> Self {
        Self { config, judge }
    }

    /// Wall-clock cap for one run: the per-call `timeout_seconds` when given
    /// and positive, capped by `[agent.pi].timeout_seconds` (a zero config
    /// value means the default). The argument can only lower the cap.
    pub fn effective_timeout(&self, requested: Option<u64>) -> Duration {
        let cap = if self.config.timeout_seconds == 0 {
            PiConfig::default().timeout_seconds
        } else {
            self.config.timeout_seconds
        };
        let secs = match requested {
            Some(r) if r > 0 => r.min(cap),
            _ => cap,
        };
        Duration::from_secs(secs)
    }

    fn parse_args(&self, args: &Value) -> Result<DelegateArgs, ToolError> {
        let task = args
            .get("task")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .ok_or_else(|| {
                ToolError::InvalidArguments("'task' must be a non-empty string".to_string())
            })?;
        let mut text = task.to_string();
        let paths: Vec<&str> = args
            .get("context_paths")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .map(str::trim)
                    .filter(|p| !p.is_empty())
                    .collect()
            })
            .unwrap_or_default();
        if !paths.is_empty() {
            text.push_str("\n\nRelevant files:");
            for p in paths {
                text.push_str("\n- ");
                text.push_str(p);
            }
        }
        let requested = args.get("timeout_seconds").and_then(Value::as_u64);
        Ok(DelegateArgs {
            task: text,
            timeout: self.effective_timeout(requested),
        })
    }
}

/// Parsed, validated tool arguments.
struct DelegateArgs {
    /// Task text as handed to Pi (brief + optional `Relevant files:` list).
    task: String,
    timeout: Duration,
}

#[async_trait]
impl Tool for PiDelegateTool {
    fn name(&self) -> &str {
        PI_DELEGATE_TOOL_NAME
    }

    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Complete, self-contained brief for Pi: what to build or change, \
                        where, constraints, and the command that verifies it (e.g. `cargo test -p x`). \
                        Pi has no access to this conversation or the graph."
                },
                "context_paths": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional workspace-relative files Pi should read first; \
                        appended to the brief as a 'Relevant files' list."
                },
                "timeout_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional wall-clock cap for this run; capped by the server's \
                        [agent.pi].timeout_seconds. Pi is killed when exceeded."
                }
            },
            "required": ["task"]
        })
    }

    /// The parent's HITL gate pauses on *sending work to Pi*; Pi's own calls
    /// are not gated (see module docs).
    fn is_destructive(&self) -> bool {
        true
    }

    async fn execute(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        if ctx.disable_bash {
            return Err(ToolError::ExecutionFailed(
                "delegate_pi is disabled: bash is locked down on this server and Pi runs shell"
                    .to_string(),
            ));
        }
        let parsed = self.parse_args(&args)?;
        // Deliberately probed on every call (not only at registration): this is
        // what guarantees an uninstalled or broken Pi never creates a Task
        // node. Cost is one `pi --version` (typically ~100 ms, capped at 5 s)
        // against runs that last minutes; the result becomes `pi_version`.
        let version = probe_version(&self.config)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("pi is not available: {e}")))?;

        let run = PiRun::begin(
            ctx,
            &parsed.task,
            &self.config.provider,
            &self.config.model,
            Some(&version),
            self.config.max_result_chars,
        )
        .await
        .map_err(|e| ToolError::ExecutionFailed(format!("recording Pi delegation: {e}")))?;
        let driver = RunDriver::new(self, ctx, run);
        driver.notify_run_changed();

        let spec = PiSpawnSpec {
            config: &self.config,
            cwd: &ctx.working_dir,
            task: &parsed.task,
            timeout: parsed.timeout,
        };
        match spawn_pi(spec, ctx.signal.clone()).await {
            Ok(handle) => driver.consume(handle).await,
            Err(e) => driver.finish(Err(e)).await,
        }
    }
}

/// Counters and text gathered while Pi runs; feeds the summary and the Task
/// metadata.
#[derive(Default)]
struct RunStats {
    tool_calls: u32,
    tool_errors: u32,
    judge_over_threshold: u32,
    retries: u32,
    unknown_events: u32,
    /// Pi `error` / `extension_error` payloads, provider errors, and graph
    /// write failures — surfaced as `Warnings:` and `Task.metadata.errors`.
    errors: Vec<String>,
    /// Pi's most recent turn-ending assistant message (not a `toolUse` turn).
    last_text: Option<String>,
}

/// A Pi tool call seen at `tool_execution_start`, waiting for its end event.
struct PendingCall {
    args: Value,
    judge: Option<JoinHandle<Result<JudgeVerdict, LlmError>>>,
}

/// Drives one Pi run: consumes events, writes the graph through [`PiRun`],
/// reports through the sink, and maps the outcome to the tool result.
struct RunDriver<'a> {
    tool: &'a PiDelegateTool,
    ctx: &'a ToolContext,
    run: PiRun,
    task_id: NodeId,
    pi_agent_id: NodeId,
    stats: RunStats,
    pending: HashMap<String, PendingCall>,
    judge_warned: bool,
}

impl<'a> RunDriver<'a> {
    fn new(tool: &'a PiDelegateTool, ctx: &'a ToolContext, run: PiRun) -> Self {
        Self {
            tool,
            ctx,
            task_id: run.task_id.clone(),
            pi_agent_id: run.pi_agent_id.clone(),
            run,
            stats: RunStats::default(),
            pending: HashMap::new(),
            judge_warned: false,
        }
    }

    fn sink(&self) -> Option<&dyn ToolEventSink> {
        self.ctx.event_sink.as_deref()
    }

    /// `graph_changed` anchored on the Task, touching Task + Pi Agent.
    fn notify_run_changed(&self) {
        if let Some(sink) = self.sink() {
            sink.graph_changed(
                &self.task_id,
                &[self.task_id.clone(), self.pi_agent_id.clone()],
            );
        }
    }

    /// Drain every event, then await the process outcome and finish.
    async fn consume(mut self, mut handle: PiRunHandle) -> Result<ToolOutput, ToolError> {
        while let Some(event) = handle.events.recv().await {
            self.on_event(event).await;
        }
        let result = handle.wait().await;
        self.abort_pending_judges();
        self.finish(result).await
    }

    async fn on_event(&mut self, event: PiEvent) {
        match event {
            PiEvent::Session { id, cwd } => {
                if let Err(e) = self.run.record_session(id.as_deref(), cwd.as_deref()).await {
                    self.graph_write_failed("session", e);
                }
            }
            PiEvent::ToolStart {
                tool_call_id,
                tool_name,
                args,
            } => self.on_tool_start(tool_call_id, &tool_name, args),
            PiEvent::ToolEnd {
                tool_call_id,
                tool_name,
                result,
                is_error,
            } => {
                self.on_tool_end(&tool_call_id, &tool_name, &result, is_error)
                    .await
            }
            PiEvent::AssistantMessage {
                text,
                stop_reason,
                error_message,
                usage,
            } => {
                self.on_assistant(text, stop_reason, error_message, usage)
                    .await
            }
            PiEvent::Error(msg) => self.stats.errors.push(msg),
            PiEvent::AgentEnd { will_retry: true } => self.stats.retries += 1,
            PiEvent::Unknown(ty) => {
                self.stats.unknown_events += 1;
                tracing::debug!(event_type = %ty, "unknown pi event");
            }
            PiEvent::AgentEnd { will_retry: false }
            | PiEvent::Lifecycle(_)
            | PiEvent::TextDelta { .. }
            | PiEvent::Ignored => {}
        }
    }

    fn on_tool_start(&mut self, tool_call_id: String, tool_name: &str, args: Value) {
        if let Some(sink) = self.sink() {
            sink.tool_started(&self.ctx.interaction_id, &tool_call_id, tool_name);
        }
        let judge = match &self.tool.judge {
            Some(judge) if is_destructive_tool(tool_name) => {
                let judge = Arc::clone(judge);
                let name = tool_name.to_string();
                let judged_args = args.clone();
                Some(tokio::spawn(async move {
                    judge.judge(&name, &judged_args).await
                }))
            }
            _ => None,
        };
        self.pending
            .insert(tool_call_id, PendingCall { args, judge });
    }

    async fn on_tool_end(
        &mut self,
        tool_call_id: &str,
        tool_name: &str,
        result: &Value,
        is_error: bool,
    ) {
        let pending = self.pending.remove(tool_call_id);
        let (args, judge) = match pending {
            Some(p) => (p.args, p.judge),
            None => (Value::Null, None),
        };
        let hitl_judge = self.take_verdict(judge).await.map(|verdict| {
            if verdict.over_threshold {
                self.stats.judge_over_threshold += 1;
            }
            JudgeOutcome {
                verdict,
                pause: false,
                action: JUDGE_ACTION_OBSERVED,
            }
            .to_metadata()
        });
        self.stats.tool_calls += 1;
        if is_error {
            self.stats.tool_errors += 1;
        }
        let result_text = flatten_result(result);
        let call = PiToolCall {
            tool_call_id,
            tool_name,
            args: &args,
            result_text: &result_text,
            is_error,
            hitl_judge,
        };
        match self.run.record_tool_call(call).await {
            Ok(node) => {
                if let Some(sink) = self.sink() {
                    sink.tool_finished(&node, is_error);
                    sink.graph_changed(&node, std::slice::from_ref(&node));
                }
            }
            Err(e) => self.graph_write_failed(tool_name, e),
        }
    }

    /// Await a judge started at `tool_execution_start`, bounded by the judge's
    /// own timeout. Fail-soft: any failure yields `None` (no `hitl_judge`
    /// key) and one warning per run.
    async fn take_verdict(
        &mut self,
        handle: Option<JoinHandle<Result<JudgeVerdict, LlmError>>>,
    ) -> Option<JudgeVerdict> {
        let handle = handle?;
        let timeout = self.tool.judge.as_ref()?.timeout();
        let abort = handle.abort_handle();
        match tokio::time::timeout(timeout, handle).await {
            Ok(Ok(Ok(verdict))) => Some(verdict),
            Ok(Ok(Err(e))) => {
                self.warn_judge_once(&e.to_string());
                None
            }
            Ok(Err(join)) => {
                self.warn_judge_once(&format!("judge task failed: {join}"));
                None
            }
            Err(_) => {
                // Dropping the JoinHandle only detaches; stop the request too.
                abort.abort();
                self.warn_judge_once("judge did not answer within its timeout");
                None
            }
        }
    }

    fn warn_judge_once(&mut self, reason: &str) {
        if !self.judge_warned {
            self.judge_warned = true;
            tracing::warn!(
                task_id = %self.task_id,
                reason,
                "hitl_judge failed for a Pi tool call; recording without verdict for this run"
            );
        }
    }

    /// Judges whose `tool_execution_end` never arrived (run killed mid-call).
    fn abort_pending_judges(&mut self) {
        for (_, call) in self.pending.drain() {
            if let Some(handle) = call.judge {
                handle.abort();
            }
        }
    }

    async fn on_assistant(
        &mut self,
        text: String,
        stop_reason: Option<String>,
        error_message: Option<String>,
        usage: Option<Value>,
    ) {
        if let Some(err) = error_message {
            // `PiEvent::Error` is capped by the parser; `errorMessage` is not.
            self.stats.errors.push(format!(
                "provider: {}",
                truncate_within(&err, MAX_ERROR_CHARS)
            ));
        }
        if text.trim().is_empty() {
            return;
        }
        if let Err(e) = self
            .run
            .record_assistant_message(&text, stop_reason.as_deref(), usage.as_ref())
            .await
        {
            self.graph_write_failed("assistant message", e);
        }
        // A `toolUse` turn ("Let me check…") is narration before more tool
        // calls, not Pi's answer; only a message that ends a turn counts as
        // the result. Pi's stop reasons: stop | length | toolUse | error | aborted.
        if stop_reason.as_deref() != Some(STOP_REASON_TOOL_USE) {
            self.stats.last_text = Some(text);
        }
    }

    fn graph_write_failed(&mut self, what: &str, e: crate::error::AgentError) {
        tracing::warn!(task_id = %self.task_id, what, error = %e, "Pi graph write failed");
        self.stats
            .errors
            .push(format!("graph write failed ({what}): {e}"));
    }

    /// Map the process outcome to the Task finish + tool result. Called on
    /// every exit path.
    async fn finish(
        self,
        result: Result<PiRunOutcome, PiProcessError>,
    ) -> Result<ToolOutput, ToolError> {
        match result {
            Ok(outcome) => self.finish_exited(outcome).await,
            Err(PiProcessError::Cancelled) => {
                let finish = self.failed("cancelled", "cancelled by the session", None);
                self.close(finish).await;
                Err(ToolError::Cancelled)
            }
            Err(PiProcessError::Timeout(after)) => {
                let detail = format!(
                    "pi timed out after {}s and was killed\n{}",
                    after.as_secs(),
                    self.partial_summary()
                );
                let finish = self.failed("timeout", detail, None);
                self.close(finish).await;
                Err(ToolError::Timeout(after.as_secs()))
            }
            Err(e @ (PiProcessError::NotFound(_) | PiProcessError::Spawn(_))) => {
                let detail = e.to_string();
                let finish = self.failed("spawn", detail.clone(), None);
                self.close(finish).await;
                Err(ToolError::ExecutionFailed(detail))
            }
        }
    }

    /// Pi exited on its own: a non-zero exit with no final message is a
    /// failure; anything else completes (the exit code is reported).
    async fn finish_exited(self, outcome: PiRunOutcome) -> Result<ToolOutput, ToolError> {
        if self.stats.last_text.is_none() && outcome.exit_code != Some(0) {
            let code = outcome
                .exit_code
                .map_or_else(|| "on a signal".to_string(), |c| c.to_string());
            let msg = format!(
                "pi exited {code} without a result: {}",
                outcome.stderr_tail.trim()
            );
            let mut finish = self.failed("exit", msg.clone(), outcome.exit_code);
            if let PiRunFinish::Failed {
                pipes_lingered,
                stderr_tail,
                ..
            } = &mut finish
            {
                *pipes_lingered = outcome.pipes_lingered;
                *stderr_tail = Some(outcome.stderr_tail.clone());
            }
            self.close(finish).await;
            return Err(ToolError::ExecutionFailed(msg));
        }
        let summary = self.summary(&outcome);
        let task_id = self.task_id.clone();
        let finish = PiRunFinish::Completed {
            summary: self.stats.last_text.clone().unwrap_or_default(),
            exit_code: outcome.exit_code,
            errors: self.stats.errors.clone(),
            judge_over_threshold: self.stats.judge_over_threshold,
            pipes_lingered: outcome.pipes_lingered,
            stderr_tail: Some(outcome.stderr_tail),
        };
        self.close(finish).await;
        Ok(ToolOutput::success_with_node(summary, task_id))
    }

    fn failed(
        &self,
        kind: &'static str,
        detail: impl Into<String>,
        exit_code: Option<i32>,
    ) -> PiRunFinish {
        PiRunFinish::Failed {
            kind,
            detail: detail.into(),
            exit_code,
            errors: self.stats.errors.clone(),
            judge_over_threshold: self.stats.judge_over_threshold,
            pipes_lingered: false,
            stderr_tail: None,
        }
    }

    /// `PiRun::finish` + the final `graph_changed`. A failed finish is logged;
    /// the tool result is still returned (the `Drop` net will not re-mark).
    async fn close(self, finish: PiRunFinish) {
        let Self {
            run,
            task_id,
            pi_agent_id,
            ctx,
            ..
        } = self;
        if let Err(e) = run.finish(finish).await {
            tracing::warn!(task_id = %task_id, error = %e, "finishing Pi delegation failed");
        }
        if let Some(sink) = ctx.event_sink.as_deref() {
            sink.graph_changed(&task_id, &[task_id.clone(), pi_agent_id]);
        }
    }

    /// The tool result text (design D2): status line, counts, judge line
    /// (only when a judge is configured), `Result:`, and `Warnings:`.
    fn summary(&self, outcome: &PiRunOutcome) -> String {
        let secs = outcome.duration.as_secs_f64();
        let mut s = match outcome.exit_code {
            Some(0) => format!("Pi completed (exit 0, {secs:.1}s)"),
            Some(code) => format!("Pi finished with exit {code} ({secs:.1}s)"),
            None => format!("Pi finished without an exit code (killed by a signal, {secs:.1}s)"),
        };
        s.push_str(&format!(
            "\nTool calls: {} ({} errors)",
            self.stats.tool_calls, self.stats.tool_errors
        ));
        if let Some(judge) = &self.tool.judge {
            s.push_str(&format!(
                "\nJudge: {} calls ≥ {} (observed, not gated)",
                self.stats.judge_over_threshold,
                judge.threshold()
            ));
        }
        s.push_str("\n\nResult:\n");
        match &self.stats.last_text {
            Some(text) => s.push_str(&truncate_within(text, self.tool.config.max_result_chars)),
            None => s.push_str("(Pi produced no final message)"),
        }
        let warnings = self.warnings(outcome);
        if !warnings.is_empty() {
            s.push_str("\n\nWarnings:");
            for w in warnings {
                s.push_str("\n- ");
                s.push_str(&w);
            }
        }
        s
    }

    fn warnings(&self, outcome: &PiRunOutcome) -> Vec<String> {
        let mut w: Vec<String> = self
            .stats
            .errors
            .iter()
            .take(MAX_SUMMARY_WARNINGS)
            .cloned()
            .collect();
        if self.stats.errors.len() > MAX_SUMMARY_WARNINGS {
            w.push(format!(
                "{} more errors recorded on the Task node",
                self.stats.errors.len() - MAX_SUMMARY_WARNINGS
            ));
        }
        if self.stats.retries > 0 {
            w.push(format!(
                "Pi retried {} time(s) after transient provider errors",
                self.stats.retries
            ));
        }
        if outcome.pipes_lingered {
            w.push(
                "Pi's stdout/stderr stayed open after it exited; leftover processes were killed"
                    .to_string(),
            );
        }
        if outcome.malformed_lines > 0 {
            w.push(format!(
                "{} malformed stdout lines skipped",
                outcome.malformed_lines
            ));
        }
        if outcome.oversized_lines > 0 {
            w.push(format!(
                "{} oversized stdout lines skipped",
                outcome.oversized_lines
            ));
        }
        if self.stats.unknown_events > 0 {
            w.push(format!(
                "{} unknown pi event types ignored",
                self.stats.unknown_events
            ));
        }
        w
    }

    /// What was seen before a timeout; stored as the Task's `failure_detail`.
    fn partial_summary(&self) -> String {
        let last = self
            .stats
            .last_text
            .as_deref()
            .map_or_else(|| "(none)".to_string(), |t| truncate_within(t, 500));
        format!(
            "Tool calls before the cut: {} ({} errors). Last message: {last}",
            self.stats.tool_calls, self.stats.tool_errors
        )
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::path::Path;
    use std::sync::atomic::AtomicU32;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use async_trait::async_trait;
    use graphirm_graph::edges::EdgeType;
    use graphirm_graph::nodes::{
        AgentData, GraphNode, InteractionData, NodeId, NodeType, TaskData, TaskStatus,
    };
    use graphirm_graph::{Direction, GraphStore};
    use graphirm_llm::{DecisionsClient, DecisionsTransport, LlmError};
    use graphirm_tools::registry::ToolRegistry;
    use graphirm_tools::{Tool, ToolContext, ToolError, ToolEventSink};
    use serde_json::{Value, json};
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::config::{AgentConfig, PiConfig};
    use crate::hitl_judge::{DestructiveJudge, JUDGE_ACTION_OBSERVED, JUDGE_QUESTION_ID};
    use crate::pi_delegate::PI_EXECUTOR;

    const FAKE_PI: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/pi/fake_pi.sh");

    fn fake_config(knobs: &[(&str, &str)]) -> PiConfig {
        let mut extra_args = vec!["--fake-knob".to_string(), "FAKE_PI_DELAY_MS=1".to_string()];
        for (k, v) in knobs {
            extra_args.push("--fake-knob".to_string());
            extra_args.push(format!("{k}={v}"));
        }
        PiConfig {
            enabled: true,
            binary: FAKE_PI.to_string(),
            provider: "openrouter".to_string(),
            model: "test/model".to_string(),
            timeout_seconds: 60,
            trust_project: false,
            extra_args,
            max_result_chars: 4000,
        }
    }

    /// Records every sink call as a string, in order.
    struct Recording(Mutex<Vec<String>>);

    impl Recording {
        fn events(&self) -> Vec<String> {
            self.0.lock().expect("lock").clone()
        }
        fn count(&self, prefix: &str) -> usize {
            self.events()
                .iter()
                .filter(|e| e.starts_with(prefix))
                .count()
        }
    }

    impl ToolEventSink for Recording {
        fn tool_started(&self, r: &NodeId, call_id: &str, tool_name: &str) {
            self.0
                .lock()
                .expect("lock")
                .push(format!("start:{r}:{call_id}:{tool_name}"));
        }
        fn tool_finished(&self, node_id: &NodeId, is_error: bool) {
            self.0
                .lock()
                .expect("lock")
                .push(format!("end:{node_id}:{is_error}"));
        }
        fn graph_changed(&self, anchor: &NodeId, touched: &[NodeId]) {
            self.0
                .lock()
                .expect("lock")
                .push(format!("graph:{anchor}:{}", touched.len()));
        }
    }

    fn make_ctx(dir: &Path) -> (ToolContext, Arc<Recording>) {
        let graph = Arc::new(GraphStore::open_memory().expect("memory graph"));
        let agent_id = graph
            .add_node(GraphNode::new(NodeType::Agent(AgentData {
                name: "director".to_string(),
                model: "test".to_string(),
                system_prompt: None,
                status: "active".to_string(),
            })))
            .expect("agent node");
        let interaction_id = graph
            .add_node(GraphNode::new(NodeType::Interaction(InteractionData {
                role: "assistant".to_string(),
                content: "calling delegate_pi".to_string(),
                token_count: None,
            })))
            .expect("interaction node");
        let sink = Arc::new(Recording(Mutex::new(Vec::new())));
        let ctx = ToolContext {
            graph,
            agent_id,
            interaction_id,
            working_dir: dir.to_path_buf(),
            signal: CancellationToken::new(),
            turn: 2,
            turn_pos_counter: Arc::new(AtomicU32::new(0)),
            knowledge_retriever: None,
            impact_provider: None,
            disable_bash: false,
            auto_link_write_to_planning: false,
            event_sink: Some(sink.clone() as Arc<dyn ToolEventSink>),
        };
        (ctx, sink)
    }

    fn delegated_tasks(ctx: &ToolContext) -> Vec<GraphNode> {
        ctx.graph
            .neighbors(
                &ctx.agent_id,
                Some(EdgeType::DelegatesTo),
                Direction::Outgoing,
            )
            .expect("neighbors")
    }

    fn task_data(graph: &GraphStore, id: &NodeId) -> (TaskData, Value) {
        let node = graph.get_node(id).expect("task node");
        match node.node_type {
            NodeType::Task(d) => (d, node.metadata),
            other => panic!("expected Task, got {}", other.type_name()),
        }
    }

    /// Tool / assistant nodes produced by the Pi Agent behind `task_id`.
    fn pi_nodes(graph: &GraphStore, task_id: &NodeId) -> Vec<GraphNode> {
        let agents = graph
            .neighbors(task_id, Some(EdgeType::SpawnedBy), Direction::Outgoing)
            .expect("spawned");
        assert_eq!(agents.len(), 1, "exactly one Pi Agent");
        graph
            .neighbors(&agents[0].id, Some(EdgeType::Produces), Direction::Outgoing)
            .expect("produced")
    }

    fn tool_nodes(nodes: &[GraphNode]) -> Vec<&GraphNode> {
        nodes
            .iter()
            .filter(|n| matches!(&n.node_type, NodeType::Interaction(d) if d.role == "tool"))
            .collect()
    }

    #[test]
    fn tool_name_schema_and_destructive() {
        let t = PiDelegateTool::new(PiConfig::default(), None);
        assert_eq!(t.name(), "delegate_pi");
        assert!(t.is_destructive());
        let p = t.parameters();
        assert_eq!(p["required"], json!(["task"]));
        assert_eq!(p["properties"]["task"]["type"], "string");
        assert!(p["properties"]["context_paths"].is_object());
        assert_eq!(p["properties"]["context_paths"]["type"], "array");
        assert_eq!(p["properties"]["timeout_seconds"]["type"], "integer");
        let d = t.description();
        assert!(d.contains("cannot be paused"), "{d}");
        assert!(d.contains("autonomous"), "{d}");
        assert!(d.contains("recorded"), "{d}");
        assert!(d.contains("verify"), "{d}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn happy_path_builds_graph_and_returns_summary() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let (ctx, sink) = make_ctx(dir.path());
        let tool = PiDelegateTool::new(fake_config(&[]), None);

        let out = tool
            .execute(json!({"task": "make hello.txt"}), &ctx)
            .await
            .expect("ok");

        assert!(!out.is_error);
        assert!(
            out.content.starts_with("Pi completed (exit 0, "),
            "{}",
            out.content
        );
        assert!(
            out.content.contains("Tool calls: 4 (1 errors)"),
            "{}",
            out.content
        );
        assert!(
            !out.content.contains("Judge:"),
            "no judge configured: {}",
            out.content
        );
        assert!(
            out.content.contains("Result:\nThe file content is: `hi`"),
            "{}",
            out.content
        );
        assert!(!out.content.contains("Warnings:"), "{}", out.content);

        let tasks = delegated_tasks(&ctx);
        assert_eq!(tasks.len(), 1);
        let task_id = tasks[0].id.clone();
        assert_eq!(out.node_id, Some(task_id.clone()), "node_id is the Task");

        let (task, meta) = task_data(&ctx.graph, &task_id);
        assert_eq!(task.status, TaskStatus::Completed);
        assert_eq!(task.description, "make hello.txt");
        assert_eq!(meta["executor"], PI_EXECUTOR);
        assert_eq!(meta["result"], "The file content is: `hi`");
        assert_eq!(meta["exit_code"], 0);
        assert_eq!(meta["tool_calls"], 4);
        assert!(meta.get("failure").is_none());
        assert!(meta.get("judge_over_threshold").is_none());

        let nodes = pi_nodes(&ctx.graph, &task_id);
        let tools = tool_nodes(&nodes);
        assert_eq!(tools.len(), 4);
        assert!(tools.iter().all(|n| n.metadata["executor"] == PI_EXECUTOR));
        assert!(
            tools.iter().all(|n| n.metadata.get("hitl_judge").is_none()),
            "no judge → no hitl_judge key"
        );
        assert_eq!(
            tools
                .iter()
                .filter(|n| n.metadata["is_error"] == true)
                .count(),
            1
        );
        let assistants = nodes
            .iter()
            .filter(|n| matches!(&n.node_type, NodeType::Interaction(d) if d.role == "assistant"))
            .count();
        assert!(assistants >= 1, "assistant messages recorded");

        // Pi Agent carries its session id from the `session` event.
        let agents = ctx
            .graph
            .neighbors(&task_id, Some(EdgeType::SpawnedBy), Direction::Outgoing)
            .expect("spawned");
        assert!(agents[0].metadata["pi_session_id"].is_string());
        assert_eq!(agents[0].metadata["pi_version"], "0.85.1-fake");

        // Sink: one start + one end per Pi tool call, graph_changed at begin,
        // per tool node, and at finish.
        let start_prefix = format!("start:{}:", ctx.interaction_id);
        assert_eq!(sink.count(&start_prefix), 4, "{:?}", sink.events());
        assert_eq!(sink.count("end:"), 4, "{:?}", sink.events());
        assert!(sink.count("graph:") >= 6, "{:?}", sink.events());
        let events = sink.events();
        assert_eq!(
            events.first().expect("first"),
            &format!("graph:{task_id}:2")
        );
        assert_eq!(events.last().expect("last"), &format!("graph:{task_id}:2"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn context_paths_are_appended_to_the_task() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let argv_file = dir.path().join("argv.txt");
        let (ctx, _) = make_ctx(dir.path());
        let cfg = fake_config(&[("FAKE_PI_ARGV", &argv_file.display().to_string())]);
        let tool = PiDelegateTool::new(cfg, None);

        tool.execute(
            json!({"task": "fix it", "context_paths": ["src/a.rs", "src/b.rs"]}),
            &ctx,
        )
        .await
        .expect("ok");

        let expected = "fix it\n\nRelevant files:\n- src/a.rs\n- src/b.rs";
        let argv = std::fs::read_to_string(&argv_file).expect("argv");
        assert!(argv.ends_with(&format!("{expected}\n")), "{argv}");
        let (task, _) = task_data(&ctx.graph, &delegated_tasks(&ctx)[0].id);
        assert_eq!(task.description, expected);
    }

    #[tokio::test]
    async fn missing_or_empty_task_is_invalid_arguments() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let (ctx, _) = make_ctx(dir.path());
        let tool = PiDelegateTool::new(fake_config(&[]), None);
        for args in [json!({}), json!({"task": "   "}), json!({"task": 7})] {
            match tool.execute(args, &ctx).await {
                Err(ToolError::InvalidArguments(_)) => {}
                other => panic!("expected InvalidArguments, got {other:?}"),
            }
        }
        assert!(delegated_tasks(&ctx).is_empty(), "no Task node created");
    }

    #[tokio::test]
    async fn disable_bash_refuses_before_spawn() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let (mut ctx, sink) = make_ctx(dir.path());
        ctx.disable_bash = true;
        let tool = PiDelegateTool::new(fake_config(&[]), None);
        match tool.execute(json!({"task": "t"}), &ctx).await {
            Err(ToolError::ExecutionFailed(msg)) => {
                assert!(msg.contains("delegate_pi is disabled"), "{msg}");
                assert!(msg.contains("bash"), "{msg}");
            }
            other => panic!("expected ExecutionFailed, got {other:?}"),
        }
        assert!(delegated_tasks(&ctx).is_empty(), "no Task node created");
        assert!(sink.events().is_empty());
    }

    #[tokio::test]
    async fn missing_binary_is_tool_error_and_no_task_node() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let (ctx, sink) = make_ctx(dir.path());
        let mut cfg = fake_config(&[]);
        cfg.binary = "/nonexistent/pi".to_string();
        let tool = PiDelegateTool::new(cfg, None);
        match tool.execute(json!({"task": "t"}), &ctx).await {
            Err(ToolError::ExecutionFailed(msg)) => {
                assert!(msg.contains("/nonexistent/pi"), "{msg}");
            }
            other => panic!("expected ExecutionFailed, got {other:?}"),
        }
        assert!(delegated_tasks(&ctx).is_empty(), "no Task node created");
        assert!(sink.events().is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn nonzero_exit_without_result_is_error_with_stderr_tail() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let (ctx, sink) = make_ctx(dir.path());
        let cfg = fake_config(&[
            ("FAKE_PI_NO_END", "1"),
            ("FAKE_PI_EXIT", "2"),
            ("FAKE_PI_STDERR", "1"),
        ]);
        let tool = PiDelegateTool::new(cfg, None);
        match tool.execute(json!({"task": "t"}), &ctx).await {
            Err(ToolError::ExecutionFailed(msg)) => {
                assert!(msg.contains("exited 2"), "{msg}");
                assert!(msg.contains("without a result"), "{msg}");
                assert!(msg.contains("warn"), "{msg}");
            }
            other => panic!("expected ExecutionFailed, got {other:?}"),
        }
        let tasks = delegated_tasks(&ctx);
        assert_eq!(tasks.len(), 1);
        let (task, meta) = task_data(&ctx.graph, &tasks[0].id);
        assert_eq!(task.status, TaskStatus::Failed);
        assert_eq!(meta["failure"], "exit");
        assert_eq!(meta["exit_code"], 2);
        assert!(meta["stderr_tail"].as_str().expect("tail").contains("warn"));
        // Tool calls before the cut are still recorded.
        assert_eq!(tool_nodes(&pi_nodes(&ctx.graph, &tasks[0].id)).len(), 4);
        let task_id = &tasks[0].id;
        assert_eq!(
            sink.events().last().expect("last"),
            &format!("graph:{task_id}:2")
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn nonzero_exit_with_result_is_success_with_exit_code() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let (ctx, _) = make_ctx(dir.path());
        let tool = PiDelegateTool::new(fake_config(&[("FAKE_PI_EXIT", "1")]), None);
        let out = tool.execute(json!({"task": "t"}), &ctx).await.expect("ok");
        assert!(
            out.content.starts_with("Pi finished with exit 1 ("),
            "{}",
            out.content
        );
        assert!(out.content.contains("Result:\nThe file content is: `hi`"));
        let (task, meta) = task_data(&ctx.graph, &delegated_tasks(&ctx)[0].id);
        assert_eq!(task.status, TaskStatus::Completed);
        assert_eq!(meta["exit_code"], 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_marks_task_failed_and_returns_cancelled() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let (ctx, sink) = make_ctx(dir.path());
        let tool = Arc::new(PiDelegateTool::new(
            fake_config(&[("FAKE_PI_HANG_AT", "3")]),
            None,
        ));
        let cancel = ctx.signal.clone();
        let ctx2 = ctx.clone();
        let t = tool.clone();
        let run = tokio::spawn(async move { t.execute(json!({"task": "t"}), &ctx2).await });
        tokio::time::sleep(Duration::from_millis(400)).await;
        cancel.cancel();
        let result = tokio::time::timeout(Duration::from_secs(8), run)
            .await
            .expect("execute returned after cancel")
            .expect("no panic");
        assert!(matches!(result, Err(ToolError::Cancelled)), "{result:?}");
        let tasks = delegated_tasks(&ctx);
        assert_eq!(tasks.len(), 1);
        let (task, meta) = task_data(&ctx.graph, &tasks[0].id);
        assert_eq!(task.status, TaskStatus::Failed);
        assert_eq!(meta["failure"], "cancelled");
        let task_id = &tasks[0].id;
        assert_eq!(
            sink.events().last().expect("last"),
            &format!("graph:{task_id}:2")
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn timeout_is_capped_by_config_and_marks_task_failed() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let (ctx, _) = make_ctx(dir.path());
        let mut cfg = fake_config(&[("FAKE_PI_HANG_AT", "3")]);
        cfg.timeout_seconds = 1;
        let tool = PiDelegateTool::new(cfg, None);
        let started = std::time::Instant::now();
        // The per-call argument may only lower the cap, never raise it.
        let result = tool
            .execute(json!({"task": "t", "timeout_seconds": 600}), &ctx)
            .await;
        assert!(matches!(result, Err(ToolError::Timeout(1))), "{result:?}");
        assert!(
            started.elapsed() < Duration::from_secs(8),
            "{:?}",
            started.elapsed()
        );
        let (task, meta) = task_data(&ctx.graph, &delegated_tasks(&ctx)[0].id);
        assert_eq!(task.status, TaskStatus::Failed);
        assert_eq!(meta["failure"], "timeout");
        assert!(
            meta["failure_detail"]
                .as_str()
                .expect("detail")
                .contains("timed out"),
            "{meta}"
        );
    }

    // ---- judge (observe-only) ----

    struct ReplyTransport(Value);

    #[async_trait]
    impl DecisionsTransport for ReplyTransport {
        async fn post(&self, _body: &Value) -> Result<Value, LlmError> {
            Ok(self.0.clone())
        }
    }

    struct FailingTransport;

    #[async_trait]
    impl DecisionsTransport for FailingTransport {
        async fn post(&self, _body: &Value) -> Result<Value, LlmError> {
            Err(LlmError::provider("judge down"))
        }
    }

    fn judge_with(transport: Arc<dyn DecisionsTransport>, threshold: f64) -> Arc<DestructiveJudge> {
        Arc::new(DestructiveJudge::new(
            Arc::new(DecisionsClient::with_transport(transport)),
            Duration::from_millis(500),
            threshold,
        ))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn judge_verdicts_are_observed_not_gated() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let (ctx, _) = make_ctx(dir.path());
        let reply = json!({"answers": {JUDGE_QUESTION_ID: {"type": "noul", "noul": 0.93}}});
        let judge = judge_with(Arc::new(ReplyTransport(reply)), 0.8);
        let tool = PiDelegateTool::new(fake_config(&[]), Some(judge));

        let out = tool.execute(json!({"task": "t"}), &ctx).await.expect("ok");
        assert!(
            out.content
                .contains("Judge: 4 calls ≥ 0.8 (observed, not gated)"),
            "{}",
            out.content
        );

        let task_id = delegated_tasks(&ctx)[0].id.clone();
        let (_, meta) = task_data(&ctx.graph, &task_id);
        assert_eq!(meta["judge_over_threshold"], 4);

        let nodes = pi_nodes(&ctx.graph, &task_id);
        for n in tool_nodes(&nodes) {
            let name = n.metadata["tool_name"].as_str().expect("tool_name");
            let judged = &n.metadata["hitl_judge"];
            assert!(
                matches!(name, "bash" | "write" | "edit"),
                "fixture only has bash/write: {name}"
            );
            assert_eq!(judged["action"], JUDGE_ACTION_OBSERVED, "{n:?}");
            assert_eq!(judged["version"], crate::hitl_judge::JUDGE_VERSION);
            assert!((judged["p_irreversible"].as_f64().expect("p") - 0.93).abs() < 1e-9);
            assert!((judged["threshold"].as_f64().expect("t") - 0.8).abs() < 1e-9);
            assert!(judged["latency_ms"].is_u64());
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn judge_failure_is_fail_soft() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let (ctx, _) = make_ctx(dir.path());
        let judge = judge_with(Arc::new(FailingTransport), 0.8);
        let tool = PiDelegateTool::new(fake_config(&[]), Some(judge));

        let out = tool.execute(json!({"task": "t"}), &ctx).await.expect("ok");
        assert!(
            out.content
                .contains("Judge: 0 calls ≥ 0.8 (observed, not gated)"),
            "{}",
            out.content
        );
        let task_id = delegated_tasks(&ctx)[0].id.clone();
        let (task, meta) = task_data(&ctx.graph, &task_id);
        assert_eq!(task.status, TaskStatus::Completed);
        assert!(meta.get("judge_over_threshold").is_none());
        let nodes = pi_nodes(&ctx.graph, &task_id);
        assert_eq!(tool_nodes(&nodes).len(), 4, "every call still recorded");
        assert!(
            tool_nodes(&nodes)
                .iter()
                .all(|n| n.metadata.get("hitl_judge").is_none())
        );
    }

    // ---- registration + system-prompt notice ----

    #[tokio::test]
    async fn register_only_when_enabled_and_notice_is_idempotent() {
        let mut reg = ToolRegistry::new();
        let mut cfg = AgentConfig::default();
        register_pi_delegate(&mut reg, &cfg).await;
        assert!(
            reg.get("delegate_pi").is_err(),
            "pi = None → not registered"
        );

        cfg.pi = Some(PiConfig::default());
        register_pi_delegate(&mut reg, &cfg).await;
        assert!(
            reg.get("delegate_pi").is_err(),
            "enabled = false → not registered"
        );

        cfg.pi = Some(fake_config(&[]));
        register_pi_delegate(&mut reg, &cfg).await;
        let tool = reg.get("delegate_pi").expect("registered");
        assert!(tool.is_destructive());

        // A missing binary still registers (the failure surfaces at call time).
        let mut reg2 = ToolRegistry::new();
        let mut missing = fake_config(&[]);
        missing.binary = "/nonexistent/pi".to_string();
        cfg.pi = Some(missing);
        register_pi_delegate(&mut reg2, &cfg).await;
        assert!(reg2.get("delegate_pi").is_ok());

        let mut prompt = String::from("base");
        apply_pi_delegate_system_notice(&mut prompt);
        let once = prompt.clone();
        assert!(once.starts_with("base"));
        assert!(once.contains("delegate_pi"));
        assert!(once.contains("cannot be paused"));
        assert!(once.to_lowercase().contains("verify"));
        apply_pi_delegate_system_notice(&mut prompt);
        assert_eq!(prompt, once, "second application is a no-op");
    }

    #[tokio::test]
    async fn register_skips_invalid_config() {
        for (binary, provider, model) in [
            ("", "p", "m"),
            ("pi", "p", ""),
            ("  ", "p", "m"),
            ("pi", "", "m"),
        ] {
            let mut reg = ToolRegistry::new();
            let mut pi = fake_config(&[]);
            pi.binary = binary.to_string();
            pi.provider = provider.to_string();
            pi.model = model.to_string();
            let cfg = AgentConfig {
                pi: Some(pi),
                ..AgentConfig::default()
            };
            register_pi_delegate(&mut reg, &cfg).await;
            assert!(
                reg.get("delegate_pi").is_err(),
                "binary={binary:?} provider={provider:?} model={model:?} must not register"
            );
        }
    }

    #[tokio::test]
    async fn register_replaces_zero_timeout_with_default() {
        let mut reg = ToolRegistry::new();
        let mut pi = fake_config(&[]);
        pi.timeout_seconds = 0;
        let cfg = AgentConfig {
            pi: Some(pi),
            ..AgentConfig::default()
        };
        register_pi_delegate(&mut reg, &cfg).await;
        assert!(reg.get("delegate_pi").is_ok());
        // The tool itself also guards against a zero cap.
        let t = PiDelegateTool::new(
            PiConfig {
                timeout_seconds: 0,
                ..PiConfig::default()
            },
            None,
        );
        assert_eq!(
            t.effective_timeout(None),
            Duration::from_secs(PiConfig::default().timeout_seconds)
        );
        let t = PiDelegateTool::new(
            PiConfig {
                timeout_seconds: 100,
                ..PiConfig::default()
            },
            None,
        );
        assert_eq!(t.effective_timeout(Some(30)), Duration::from_secs(30));
        assert_eq!(t.effective_timeout(Some(500)), Duration::from_secs(100));
        assert_eq!(t.effective_timeout(Some(0)), Duration::from_secs(100));
        assert_eq!(t.effective_timeout(None), Duration::from_secs(100));
    }

    #[test]
    fn config_applies_notice_only_when_pi_enabled() {
        fn prompt_after(pi: Option<PiConfig>, disable_bash: bool) -> String {
            let mut cfg = AgentConfig {
                system_prompt: "base".to_string(),
                disable_bash,
                pi,
                ..AgentConfig::default()
            };
            cfg.apply_pi_delegate_system_notice();
            cfg.system_prompt
        }
        let enabled = PiConfig {
            enabled: true,
            ..PiConfig::default()
        };
        assert_eq!(prompt_after(None, false), "base");
        assert_eq!(prompt_after(Some(PiConfig::default()), false), "base");
        assert!(prompt_after(Some(enabled.clone()), false).contains("delegate_pi"));
        // The tool is hidden under disable_bash, so the prompt must not advertise it.
        assert_eq!(prompt_after(Some(enabled.clone()), true), "base");
        // Same validation as registration: a config that would not register
        // must not produce a notice either.
        for (binary, provider, model) in [("", "p", "m"), ("pi", " ", "m"), ("pi", "p", "")] {
            let pi = PiConfig {
                binary: binary.to_string(),
                provider: provider.to_string(),
                model: model.to_string(),
                ..enabled.clone()
            };
            assert!(!pi.is_registrable());
            assert_eq!(prompt_after(Some(pi), false), "base");
        }
        assert!(enabled.is_registrable());
        assert!(!PiConfig::default().is_registrable(), "disabled by default");
    }

    #[test]
    fn hidden_under_disable_bash_covers_bash_and_delegate_pi() {
        assert!(hidden_under_disable_bash("bash"));
        assert!(hidden_under_disable_bash(PI_DELEGATE_TOOL_NAME));
        assert!(!hidden_under_disable_bash("read"));
        assert!(!hidden_under_disable_bash("write"));
    }

    /// A provider error message is capped before it reaches the errors list
    /// (and thus `Task.metadata.errors` / the `Warnings:` bullets).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn provider_error_message_is_capped() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let fixture = dir.path().join("err.jsonl");
        let huge = "e".repeat(10_000);
        std::fs::write(
            &fixture,
            format!(
                "{{\"type\":\"session\",\"id\":\"s\",\"cwd\":\"/w\"}}\n\
                 {{\"type\":\"message_end\",\"message\":{{\"role\":\"assistant\",\"content\":[],\
                 \"stopReason\":\"error\",\"errorMessage\":\"{huge}\"}}}}\n\
                 {{\"type\":\"agent_end\",\"willRetry\":false}}\n"
            ),
        )
        .expect("fixture");
        let (ctx, _) = make_ctx(dir.path());
        let cfg = fake_config(&[
            ("FAKE_PI_FIXTURE", &fixture.display().to_string()),
            ("FAKE_PI_DELAY_MS", "0"),
        ]);
        let tool = PiDelegateTool::new(cfg, None);

        let out = tool
            .execute(json!({"task": "t"}), &ctx)
            .await
            .expect("exit 0");
        // The tool result renders `stats.errors` verbatim, so the cap must be
        // applied there (graph.rs caps Task metadata entries separately).
        let bullet = out
            .content
            .lines()
            .find(|l| l.starts_with("- provider: eeee"))
            .expect("provider warning bullet");
        assert!(
            bullet.chars().count() <= "- provider: ".len() + MAX_ERROR_CHARS,
            "{}",
            bullet.chars().count()
        );
        assert!(bullet.ends_with('…'));
        let (_, meta) = task_data(&ctx.graph, &delegated_tasks(&ctx)[0].id);
        let stored = meta["errors"][0].as_str().expect("error entry");
        assert!(stored.starts_with("provider: eeee"));
    }
}
