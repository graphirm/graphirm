//! Graph writes for one Pi delegation.
//!
//! Mirrors the in-process shape produced by `multi::spawn_subagent` and the tool
//! result nodes recorded by `workflow.rs`, so the whiteboard and TUI render a Pi
//! run without changes:
//!
//! ```text
//! Director Agent      --DelegatesTo--> Task("Delegated to pi")
//! Parent Interaction  --Produces-->    Task           # the assistant turn that called delegate_pi
//! Task                --SpawnedBy-->   Pi Agent(name="pi")
//! Pi Agent            --Produces-->    tool / assistant Interaction nodes (RespondsTo-chained)
//! ```
//!
//! Every store call runs inside `tokio::task::spawn_blocking` — the store is
//! synchronous and must never block a runtime worker. Task and Agent metadata
//! never carry the Pi `binary` path (the node is returned by
//! `GET /api/sessions/{id}/graph`); the Task stores `provider`, `model` and
//! `exit_code`, the Pi Agent stores `pi_version`, `pi_session_id`, `pi_cwd`.
//!
//! **Deliberate deviation from design D2:** tool and assistant node *content*
//! is capped by the fixed [`MAX_TOOL_CONTENT_CHARS`], not by `max_result_chars`.
//! `[agent.pi].max_result_chars` is documented as the cap on Pi's *final
//! message* (`Task.metadata.result`); letting it also govern every recorded
//! tool result would let a small summary cap gut the audit trail.
//!
//! **Abandonment:** a [`PiRun`] that is dropped without [`PiRun::finish`]
//! (the caller's future was cancelled, a panic unwound, the `JoinSet` was
//! dropped) marks its Task `Failed` with `failure: "abandoned"` and its Pi
//! Agent `"failed"` from `Drop`, so the server's session restore never lists a
//! ghost `running` Pi agent. A `finish` whose store write fails enqueues the
//! same best-effort mark with `failure: "finish_failed"`. Both go through
//! `Handle::spawn_blocking` from a sync context: on a runtime that is already
//! shutting down the closure may never run, and the Task keeps its last state.
//!
//! **Input bounds:** `pi_session_id`, `pi_cwd`, `tool_call_id`, `tool_name`,
//! `stop_reason` and `usage` are stored as received. They are bounded upstream
//! by the 4 MiB JSONL line cap in `process.rs` (`MAX_LINE_BYTES`), which is
//! why this module caps only the free-text fields it composes itself.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use graphirm_graph::GraphStore;
use graphirm_graph::edges::{EdgeType, GraphEdge};
use graphirm_graph::nodes::{
    AgentData, ContentData, GraphNode, InteractionData, NodeId, NodeType, TaskData, TaskStatus,
};
use graphirm_tools::ToolContext;
use serde_json::{Map, Value, json};

use super::pieces::{HTML_INDEX_SHAPE_VERSION, Piece, PieceItem, cut_html_index};
use crate::error::AgentError;
use crate::hitl_judge::MAX_ARGS_CHARS;

/// Discriminator stored as `metadata.executor` on every node this module writes.
pub const PI_EXECUTOR: &str = "pi";
/// Title of the delegation Task node (same pattern as `spawn_subagent`).
pub const PI_TASK_TITLE: &str = "Delegated to pi";
/// `metadata.failure` value written by `Drop` when `finish` never ran.
pub const FAILURE_ABANDONED: &str = "abandoned";
/// `metadata.failure` value written when `finish` ran but its store write failed.
pub const FAILURE_FINISH_FAILED: &str = "finish_failed";
/// Task description cap; the full brief lives in Pi's stdin, not the graph.
pub const MAX_TASK_DESCRIPTION_CHARS: usize = 4000;
/// Tool / assistant node content cap (fixed; see module docs).
pub const MAX_TOOL_CONTENT_CHARS: usize = 16_000;
/// `metadata.failure_detail` cap.
pub const MAX_FAILURE_DETAIL_CHARS: usize = 2000;
/// `metadata.errors`: at most this many entries …
pub const MAX_ERRORS: usize = 20;
/// … each at most this many chars.
pub const MAX_ERROR_ENTRY_CHARS: usize = 500;
/// `metadata.stderr_tail` cap.
pub const MAX_STDERR_TAIL_CHARS: usize = 1000;

/// One Pi tool call, as reported by `tool_execution_end`.
#[derive(Debug, Clone)]
pub struct PiToolCall<'a> {
    pub tool_call_id: &'a str,
    /// Pi's real tool name (`bash`, `read`, `edit`, …).
    pub tool_name: &'a str,
    pub args: &'a Value,
    pub result_text: &'a str,
    pub is_error: bool,
    /// Judge verdict for this call, when one was produced (`hitl_judge` metadata).
    pub hitl_judge: Option<Value>,
}

/// How a Pi run ended. Build with [`PiRunFinish::completed`] /
/// [`PiRunFinish::failed`] and set the optional fields you have.
#[derive(Debug, Clone)]
pub enum PiRunFinish {
    /// Pi produced a final message; `summary` becomes `Task.metadata.result`.
    Completed {
        summary: String,
        exit_code: Option<i32>,
        /// Pi `error` / `extension_error` payloads seen during the run.
        errors: Vec<String>,
        /// Pi tool calls the judge scored at or above its threshold.
        judge_over_threshold: u32,
        /// Pi's stdout/stderr pipes stayed open after the process exited.
        pipes_lingered: bool,
        /// Last bytes of Pi's stderr, when captured.
        stderr_tail: Option<String>,
    },
    /// The run did not complete. `kind` is one of `"timeout"`, `"cancelled"`,
    /// `"exit"`, `"spawn"`.
    Failed {
        kind: &'static str,
        detail: String,
        exit_code: Option<i32>,
        errors: Vec<String>,
        judge_over_threshold: u32,
        pipes_lingered: bool,
        stderr_tail: Option<String>,
    },
}

impl PiRunFinish {
    /// `Completed` with the optional fields at their defaults.
    pub fn completed(summary: impl Into<String>, exit_code: Option<i32>) -> Self {
        Self::Completed {
            summary: summary.into(),
            exit_code,
            errors: Vec::new(),
            judge_over_threshold: 0,
            pipes_lingered: false,
            stderr_tail: None,
        }
    }

    /// `Failed` with the optional fields at their defaults.
    pub fn failed(kind: &'static str, detail: impl Into<String>, exit_code: Option<i32>) -> Self {
        Self::Failed {
            kind,
            detail: detail.into(),
            exit_code,
            errors: Vec::new(),
            judge_over_threshold: 0,
            pipes_lingered: false,
            stderr_tail: None,
        }
    }
}

/// Live graph handle for one Pi delegation. Create with [`PiRun::begin`], feed
/// it Pi's events, then call [`PiRun::finish`] exactly once (it consumes the
/// run). Dropping an unfinished run marks it abandoned (see module docs).
pub struct PiRun {
    graph: Arc<GraphStore>,
    director_agent: NodeId,
    turn: u32,
    /// Shared with the director's own tools so `interaction_{turn}_{pos}_1`
    /// labels never collide within a turn.
    turn_pos_counter: Arc<AtomicU32>,
    pub task_id: NodeId,
    pub pi_agent_id: NodeId,
    last_node: Option<NodeId>,
    tool_calls: u32,
    assistant_messages: u32,
    started: Instant,
    max_result_chars: usize,
    finished: bool,
}

impl PiRun {
    /// Creates the Task and Pi Agent nodes plus the three delegation edges.
    ///
    /// `provider` / `model` go to the Task, `pi_version` to the Pi Agent —
    /// never the binary path.
    pub async fn begin(
        ctx: &ToolContext,
        task_text: &str,
        provider: &str,
        model: &str,
        pi_version: Option<&str>,
        max_result_chars: usize,
    ) -> Result<Self, AgentError> {
        let director_agent = ctx.agent_id.clone();
        let parent_interaction = ctx.interaction_id.clone();

        let task_node = build_task_node(task_text, provider, model, &director_agent);
        let task_id = task_node.id.clone();
        let agent_node = build_pi_agent_node(model, pi_version, &director_agent, &task_id);
        let pi_agent_id = agent_node.id.clone();

        let graph = ctx.graph.clone();
        let g = graph.clone();
        let director = director_agent.clone();
        let task = task_id.clone();
        let pi_agent = pi_agent_id.clone();
        run_blocking(move || {
            g.add_node(task_node)?;
            g.add_node(agent_node)?;
            g.add_edge(GraphEdge::new(
                EdgeType::DelegatesTo,
                director,
                task.clone(),
            ))?;
            g.add_edge(GraphEdge::new(
                EdgeType::Produces,
                parent_interaction,
                task.clone(),
            ))?;
            g.add_edge(GraphEdge::new(EdgeType::SpawnedBy, task, pi_agent))?;
            Ok(())
        })
        .await?;

        tracing::info!(task_id = %task_id, pi_agent_id = %pi_agent_id, "Pi delegation started");

        Ok(Self {
            graph,
            director_agent,
            turn: ctx.turn,
            turn_pos_counter: ctx.turn_pos_counter.clone(),
            task_id,
            pi_agent_id,
            last_node: None,
            tool_calls: 0,
            assistant_messages: 0,
            started: Instant::now(),
            max_result_chars,
            finished: false,
        })
    }

    /// Stores Pi's own session id and working directory (from its `session`
    /// event) on the Pi Agent node as `pi_session_id` / `pi_cwd`.
    pub async fn record_session(
        &mut self,
        pi_session_id: Option<&str>,
        cwd: Option<&str>,
    ) -> Result<(), AgentError> {
        let mut extra = Map::new();
        if let Some(id) = pi_session_id {
            extra.insert("pi_session_id".into(), json!(id));
        }
        if let Some(cwd) = cwd {
            extra.insert("pi_cwd".into(), json!(cwd));
        }
        if extra.is_empty() {
            return Ok(());
        }

        let g = self.graph.clone();
        let pi_agent_id = self.pi_agent_id.clone();
        run_blocking(move || {
            let mut agent_node = g.get_node(&pi_agent_id)?;
            merge_metadata(&mut agent_node, extra);
            g.update_node(&pi_agent_id, agent_node)?;
            Ok(())
        })
        .await
    }

    /// Records one Pi tool call as `Interaction{role:"tool"}` under the Pi Agent.
    pub async fn record_tool_call(&mut self, call: PiToolCall<'_>) -> Result<NodeId, AgentError> {
        let mut meta = self.base_metadata();
        meta.insert("tool_call_id".into(), json!(call.tool_call_id));
        meta.insert("tool_name".into(), json!(call.tool_name));
        meta.insert("is_error".into(), json!(call.is_error));
        meta.insert("arguments".into(), json!(compact_args(call.args)));
        if let Some(judge) = call.hitl_judge {
            meta.insert("hitl_judge".into(), judge);
        }

        let mut node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "tool".to_string(),
            content: truncate_with_notice(call.result_text, MAX_TOOL_CONTENT_CHARS),
            token_count: None,
        }));
        node.metadata = Value::Object(meta);

        let id = self.persist_interaction(node).await?;
        self.tool_calls += 1;
        Ok(id)
    }

    /// Records an assistant message from Pi as `Interaction{role:"assistant"}`
    /// under the Pi Agent. `usage` (Pi's token accounting) is stored verbatim
    /// in metadata when given.
    /// `cut` is true for a turn-ending message. Narration (`toolUse`) passes false
    /// and stores the text with no piece nodes.
    pub async fn record_assistant_message(
        &mut self,
        text: &str,
        stop_reason: Option<&str>,
        usage: Option<&Value>,
        cut: bool,
    ) -> Result<NodeId, AgentError> {
        let mut meta = self.base_metadata();
        if let Some(reason) = stop_reason {
            meta.insert("stop_reason".into(), json!(reason));
        }
        if let Some(usage) = usage {
            meta.insert("usage".into(), usage.clone());
        }

        let mut node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".to_string(),
            content: truncate_with_notice(text, MAX_TOOL_CONTENT_CHARS),
            token_count: None,
        }));
        node.metadata = Value::Object(meta);

        let id = self.persist_interaction(node).await?;
        if cut {
            let owned = text.to_string();
            let g = self.graph.clone();
            let parent = id.clone();
            run_blocking(move || attach_reply_pieces(&g, &parent, &owned)).await?;
        }
        self.assistant_messages += 1;
        Ok(id)
    }

    /// Finalises the Task and Pi Agent status and metadata. Consumes the run;
    /// copy `task_id` / `pi_agent_id` first if you need them, or use the
    /// returned `Vec<NodeId>` = `[task_id, pi_agent_id]`.
    ///
    /// `finished` is flipped before the first await so a `finish` whose future
    /// is dropped mid-write is never double-marked from `Drop`. If the write
    /// itself fails, the run is therefore *not* re-marked by `Drop` either;
    /// instead a best-effort `failure: "finish_failed"` mark is enqueued on
    /// the blocking pool (same path as abandonment) and the error is returned,
    /// so the Task never stays `Pending` because of a transient store error.
    pub async fn finish(mut self, outcome: PiRunFinish) -> Result<Vec<NodeId>, AgentError> {
        // Flip before any await so a cancelled `finish` never double-marks from Drop.
        self.finished = true;
        let duration_ms = self.started.elapsed().as_millis() as u64;
        let mut extra = Map::new();
        extra.insert("tool_calls".into(), json!(self.tool_calls));
        extra.insert("duration_ms".into(), json!(duration_ms));

        let (task_status, agent_status) = self.outcome_metadata(outcome, &mut extra);

        let g = self.graph.clone();
        let task_id = self.task_id.clone();
        let pi_agent_id = self.pi_agent_id.clone();
        let written = run_blocking(move || {
            set_task_status(&g, &task_id, task_status, extra)?;
            set_agent_status(&g, &pi_agent_id, agent_status)
        })
        .await;
        if let Err(e) = written {
            tracing::error!(task_id = %self.task_id, error = %e, "Pi finish write failed; marking finish_failed");
            self.enqueue_failure_mark(FAILURE_FINISH_FAILED, duration_ms);
            return Err(e);
        }

        tracing::info!(
            task_id = %self.task_id,
            status = %task_status,
            tool_calls = self.tool_calls,
            duration_ms,
            "Pi delegation finished"
        );

        Ok(vec![self.task_id.clone(), self.pi_agent_id.clone()])
    }

    /// Writes the outcome-specific keys into `extra`; returns the statuses.
    fn outcome_metadata(
        &self,
        outcome: PiRunFinish,
        extra: &mut Map<String, Value>,
    ) -> (TaskStatus, &'static str) {
        match outcome {
            PiRunFinish::Completed {
                summary,
                exit_code,
                errors,
                judge_over_threshold,
                pipes_lingered,
                stderr_tail,
            } => {
                extra.insert(
                    "result".into(),
                    json!(truncate_within(&summary, self.max_result_chars)),
                );
                extra.insert("assistant_messages".into(), json!(self.assistant_messages));
                insert_run_facts(
                    extra,
                    exit_code,
                    &errors,
                    judge_over_threshold,
                    pipes_lingered,
                    stderr_tail,
                );
                (TaskStatus::Completed, "completed")
            }
            PiRunFinish::Failed {
                kind,
                detail,
                exit_code,
                errors,
                judge_over_threshold,
                pipes_lingered,
                stderr_tail,
            } => {
                extra.insert("failure".into(), json!(kind));
                extra.insert(
                    "failure_detail".into(),
                    json!(truncate_within(&detail, MAX_FAILURE_DETAIL_CHARS)),
                );
                insert_run_facts(
                    extra,
                    exit_code,
                    &errors,
                    judge_over_threshold,
                    pipes_lingered,
                    stderr_tail,
                );
                (TaskStatus::Failed, "failed")
            }
        }
    }

    /// Best-effort failure mark from a context that cannot `.await` (the
    /// `Drop` path) or must not re-enter the normal `finish` path (a failed
    /// `finish` write). Sync enqueue on the blocking pool; the write runs even
    /// if the calling task is being torn down. On a runtime that is shutting
    /// down the closure may never run — the Task then stays as it was; there
    /// is no further fallback.
    fn enqueue_failure_mark(&self, kind: &'static str, duration_ms: u64) {
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            tracing::error!(
                task_id = %self.task_id,
                kind,
                "no tokio runtime; Pi Task left unmarked"
            );
            return;
        };
        let g = self.graph.clone();
        let task_id = self.task_id.clone();
        let pi_agent_id = self.pi_agent_id.clone();
        handle.spawn_blocking(move || {
            if let Err(e) = mark_failed(&g, &task_id, &pi_agent_id, kind, duration_ms) {
                tracing::error!(task_id = %task_id, kind, error = %e, "failed to mark Pi Task failed");
            }
        });
    }

    /// Metadata keys common to every Interaction node written by this run.
    fn base_metadata(&self) -> Map<String, Value> {
        let mut meta = Map::new();
        meta.insert("executor".into(), json!(PI_EXECUTOR));
        meta.insert("session_id".into(), json!(self.pi_agent_id.to_string()));
        meta.insert(
            "parent_session_id".into(),
            json!(self.director_agent.to_string()),
        );
        meta
    }

    /// Labels the node with the shared turn counter, inserts it, and links
    /// `Pi Agent --Produces--> node` plus `node --RespondsTo--> last_node`.
    async fn persist_interaction(&mut self, mut node: GraphNode) -> Result<NodeId, AgentError> {
        let pos = self.turn_pos_counter.fetch_add(1, Ordering::SeqCst) + 1;
        node.set_label(format!("interaction_{}_{}_1", self.turn, pos));

        let g = self.graph.clone();
        let pi_agent_id = self.pi_agent_id.clone();
        let prev = self.last_node.clone();
        let id = run_blocking(move || {
            let id = g.add_node(node)?;
            g.add_edge(GraphEdge::new(EdgeType::Produces, pi_agent_id, id.clone()))?;
            if let Some(prev) = prev {
                g.add_edge(GraphEdge::new(EdgeType::RespondsTo, id.clone(), prev))?;
            }
            Ok(id)
        })
        .await?;
        self.last_node = Some(id.clone());
        Ok(id)
    }
}

impl Drop for PiRun {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let duration_ms = self.started.elapsed().as_millis() as u64;
        tracing::warn!(
            task_id = %self.task_id,
            pi_agent_id = %self.pi_agent_id,
            tool_calls = self.tool_calls,
            "PiRun dropped without finish(); marking Task failed (abandoned)"
        );
        self.enqueue_failure_mark(FAILURE_ABANDONED, duration_ms);
    }
}

/// Keys shared by both outcomes: `exit_code` always; `errors` (capped),
/// `judge_over_threshold`, `pipes_lingered`, `stderr_tail` (capped) only when
/// non-default.
fn insert_run_facts(
    extra: &mut Map<String, Value>,
    exit_code: Option<i32>,
    errors: &[String],
    judge_over_threshold: u32,
    pipes_lingered: bool,
    stderr_tail: Option<String>,
) {
    extra.insert("exit_code".into(), json!(exit_code));
    if !errors.is_empty() {
        let capped: Vec<String> = errors
            .iter()
            .take(MAX_ERRORS)
            .map(|e| truncate_within(e, MAX_ERROR_ENTRY_CHARS))
            .collect();
        extra.insert("errors".into(), json!(capped));
    }
    if judge_over_threshold > 0 {
        extra.insert("judge_over_threshold".into(), json!(judge_over_threshold));
    }
    if pipes_lingered {
        extra.insert("pipes_lingered".into(), json!(true));
    }
    if let Some(tail) = stderr_tail.filter(|t| !t.is_empty()) {
        extra.insert(
            "stderr_tail".into(),
            json!(truncate_within(&tail, MAX_STDERR_TAIL_CHARS)),
        );
    }
}

/// Task node for a delegation: title, capped description, `Pending`, and
/// `{executor, provider, model, parent_session_id}` metadata.
fn build_task_node(task_text: &str, provider: &str, model: &str, director: &NodeId) -> GraphNode {
    let mut node = GraphNode::new(NodeType::Task(TaskData {
        title: PI_TASK_TITLE.to_string(),
        description: truncate_within(task_text, MAX_TASK_DESCRIPTION_CHARS),
        status: TaskStatus::Pending,
        priority: None,
    }));
    let mut meta = Map::new();
    meta.insert("executor".into(), json!(PI_EXECUTOR));
    meta.insert("provider".into(), json!(provider));
    meta.insert("model".into(), json!(model));
    meta.insert("parent_session_id".into(), json!(director.to_string()));
    node.metadata = Value::Object(meta);
    node
}

/// Pi Agent node: `name = "pi"`, `status = "running"`, and
/// `{executor, parent_session_id, task_id, pi_version?}` metadata.
fn build_pi_agent_node(
    model: &str,
    pi_version: Option<&str>,
    director: &NodeId,
    task_id: &NodeId,
) -> GraphNode {
    let mut node = GraphNode::new(NodeType::Agent(AgentData {
        name: PI_EXECUTOR.to_string(),
        model: model.to_string(),
        system_prompt: None,
        status: "running".to_string(),
    }));
    let mut meta = Map::new();
    meta.insert("executor".into(), json!(PI_EXECUTOR));
    meta.insert("parent_session_id".into(), json!(director.to_string()));
    meta.insert("task_id".into(), json!(task_id.to_string()));
    if let Some(v) = pi_version {
        meta.insert("pi_version".into(), json!(v));
    }
    node.metadata = Value::Object(meta);
    node
}

/// Sync: sets the Task status and merges `extra` into its metadata.
fn set_task_status(
    g: &GraphStore,
    task_id: &NodeId,
    status: TaskStatus,
    extra: Map<String, Value>,
) -> Result<(), AgentError> {
    let mut node = g.get_node(task_id)?;
    if let NodeType::Task(ref mut data) = node.node_type {
        data.status = status;
    }
    merge_metadata(&mut node, extra);
    g.update_node(task_id, node)?;
    Ok(())
}

/// Sync: sets the Pi Agent status string.
fn set_agent_status(g: &GraphStore, agent_id: &NodeId, status: &str) -> Result<(), AgentError> {
    let mut node = g.get_node(agent_id)?;
    if let NodeType::Agent(ref mut data) = node.node_type {
        data.status = status.to_string();
    }
    g.update_node(agent_id, node)?;
    Ok(())
}

/// Sync: the fallback paths — Task `Failed` + `failure: <kind>`
/// (`"abandoned"` from `Drop`, `"finish_failed"` from a failed `finish`
/// write), Agent `"failed"`.
fn mark_failed(
    g: &GraphStore,
    task_id: &NodeId,
    pi_agent_id: &NodeId,
    kind: &str,
    duration_ms: u64,
) -> Result<(), AgentError> {
    let mut extra = Map::new();
    extra.insert("failure".into(), json!(kind));
    extra.insert("duration_ms".into(), json!(duration_ms));
    set_task_status(g, task_id, TaskStatus::Failed, extra)?;
    set_agent_status(g, pi_agent_id, "failed")
}

/// A clean HTML index becomes `reply_part` and `reply_line` Content nodes.
/// A failed cut leaves the assistant message as it was recorded.
fn attach_reply_pieces(graph: &GraphStore, parent: &NodeId, text: &str) -> Result<(), AgentError> {
    let Ok(pieces) = cut_html_index(text) else {
        return Ok(());
    };
    for piece in pieces {
        let part_id = graph.add_node(part_node(text, &piece))?;
        graph.add_edge(GraphEdge::new(
            EdgeType::Contains,
            parent.clone(),
            part_id.clone(),
        ))?;
        for item in &piece.items {
            let line_id = graph.add_node(line_node(item))?;
            graph.add_edge(GraphEdge::new(EdgeType::Contains, part_id.clone(), line_id))?;
        }
    }
    Ok(())
}

fn part_node(text: &str, piece: &Piece) -> GraphNode {
    let slice = text.get(piece.start..piece.end).unwrap_or("");
    let mut node = GraphNode::new(NodeType::Content(ContentData {
        content_type: "reply_part".to_string(),
        path: None,
        body: piece.heading.clone().unwrap_or_default(),
        language: None,
    }));
    node.metadata = json!({
        "order": piece.order,
        "id": element_id(slice).unwrap_or_default(),
        "kind": piece.kind.as_label(),
        "heading": piece.heading,
        "rel": piece.rel,
        "shape_version": HTML_INDEX_SHAPE_VERSION,
        "start": piece.start,
        "end": piece.end,
    });
    node
}

fn line_node(item: &PieceItem) -> GraphNode {
    let mut node = GraphNode::new(NodeType::Content(ContentData {
        content_type: "reply_line".to_string(),
        path: None,
        body: item.text.clone(),
        language: None,
    }));
    node.metadata = json!({
        "position": item.position,
        "src": item.src,
        "start": item.start,
        "end": item.end,
    });
    node
}

fn element_id(slice: &str) -> Option<String> {
    for (mark, quote) in [("id=\"", '"'), ("id='", '\'')] {
        let Some(at) = slice.find(mark) else {
            continue;
        };
        let rest = &slice[at + mark.len()..];
        let Some(end) = rest.find(quote) else {
            continue;
        };
        return Some(rest[..end].to_string());
    }
    None
}

/// Runs a synchronous store closure off the runtime, mapping `JoinError` the
/// same way `Session::persist_interaction` does.
async fn run_blocking<T, F>(f: F) -> Result<T, AgentError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, AgentError> + Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| AgentError::Join(e.to_string()))?
}

/// Inserts `extra` into the node's metadata object, overwriting existing keys.
fn merge_metadata(node: &mut GraphNode, extra: Map<String, Value>) {
    if !node.metadata.is_object() {
        node.metadata = Value::Object(Map::new());
    }
    if let Some(obj) = node.metadata.as_object_mut() {
        obj.extend(extra);
    }
}

/// Compact JSON of the arguments, capped at `MAX_ARGS_CHARS` chars with `…`
/// appended when cut (the HITL judge's rule, so the two never disagree).
fn compact_args(args: &Value) -> String {
    let s = args.to_string();
    if s.chars().count() > MAX_ARGS_CHARS {
        s.chars().take(MAX_ARGS_CHARS).collect::<String>() + "…"
    } else {
        s
    }
}

/// Keep at most `max` chars; when cut, end with `…` (counted within `max`).
/// `max == 0` yields an empty string. Shared with `tool.rs` (summary / error caps).
pub(super) fn truncate_within(s: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max - 1).collect();
    out.push('…');
    out
}

/// Keep at most `max` chars of content, then append a one-line notice.
fn truncate_with_notice(s: &str, max: usize) -> String {
    let total = s.chars().count();
    if total <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max).collect();
    out.push_str(&format!("\n[truncated: {} of {} chars stored]", max, total));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphirm_graph::{Direction, GraphStore};
    use std::path::PathBuf;
    use std::time::Duration;
    use tokio_util::sync::CancellationToken;

    fn make_ctx() -> ToolContext {
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
        ToolContext {
            graph,
            agent_id,
            interaction_id,
            working_dir: PathBuf::from("."),
            signal: CancellationToken::new(),
            turn: 3,
            turn_pos_counter: Arc::new(AtomicU32::new(0)),
            knowledge_retriever: None,
            impact_provider: None,
            disable_bash: false,
            auto_link_write_to_planning: false,
            event_sink: None,
        }
    }

    async fn begin(ctx: &ToolContext, max_result_chars: usize) -> PiRun {
        PiRun::begin(
            ctx,
            "Implement the thing",
            "openrouter",
            "deepseek/deepseek-v4-flash",
            Some("0.9.1"),
            max_result_chars,
        )
        .await
        .expect("begin")
    }

    fn out(graph: &GraphStore, from: &NodeId, et: EdgeType) -> Vec<GraphNode> {
        graph
            .neighbors(from, Some(et), Direction::Outgoing)
            .expect("neighbors")
    }

    fn task_data(graph: &GraphStore, id: &NodeId) -> (TaskData, Value) {
        let node = graph.get_node(id).expect("task node");
        match node.node_type {
            NodeType::Task(d) => (d, node.metadata),
            other => panic!("expected Task, got {}", other.type_name()),
        }
    }

    fn agent_data(graph: &GraphStore, id: &NodeId) -> (AgentData, Value) {
        let node = graph.get_node(id).expect("agent node");
        match node.node_type {
            NodeType::Agent(d) => (d, node.metadata),
            other => panic!("expected Agent, got {}", other.type_name()),
        }
    }

    fn interaction_data(graph: &GraphStore, id: &NodeId) -> (InteractionData, Value) {
        let node = graph.get_node(id).expect("interaction node");
        match node.node_type {
            NodeType::Interaction(d) => (d, node.metadata),
            other => panic!("expected Interaction, got {}", other.type_name()),
        }
    }

    fn bash_call<'a>(id: &'a str, args: &'a Value, judge: Option<Value>) -> PiToolCall<'a> {
        PiToolCall {
            tool_call_id: id,
            tool_name: "bash",
            args,
            result_text: "ok",
            is_error: false,
            hitl_judge: judge,
        }
    }

    #[tokio::test]
    async fn creates_task_and_pi_agent_with_delegation_edges() {
        let ctx = make_ctx();
        let run = begin(&ctx, 4000).await;
        let g = &ctx.graph;

        let delegated = out(g, &ctx.agent_id, EdgeType::DelegatesTo);
        assert_eq!(delegated.len(), 1);
        assert_eq!(delegated[0].id, run.task_id);

        let produced = out(g, &ctx.interaction_id, EdgeType::Produces);
        assert_eq!(produced.len(), 1);
        assert_eq!(produced[0].id, run.task_id);

        let spawned = out(g, &run.task_id, EdgeType::SpawnedBy);
        assert_eq!(spawned.len(), 1);
        assert_eq!(spawned[0].id, run.pi_agent_id);

        let (task, meta) = task_data(g, &run.task_id);
        assert_eq!(task.title, PI_TASK_TITLE);
        assert_eq!(task.description, "Implement the thing");
        assert_eq!(task.status, TaskStatus::Pending);
        assert_eq!(meta["executor"], "pi");
        assert_eq!(meta["provider"], "openrouter");
        assert_eq!(meta["model"], "deepseek/deepseek-v4-flash");
        assert_eq!(meta["parent_session_id"], ctx.agent_id.to_string());
        assert!(
            meta.get("pi_version").is_none(),
            "pi_version lives on the Agent"
        );
        assert!(meta.get("binary").is_none(), "binary must never be stored");

        let (agent, ameta) = agent_data(g, &run.pi_agent_id);
        assert_eq!(agent.name, "pi");
        assert_eq!(agent.status, "running");
        assert_eq!(agent.model, "deepseek/deepseek-v4-flash");
        assert_eq!(ameta["executor"], "pi");
        assert_eq!(ameta["pi_version"], "0.9.1");
        assert_eq!(ameta["task_id"], run.task_id.to_string());
        assert_eq!(ameta["parent_session_id"], ctx.agent_id.to_string());
        assert!(ameta.get("binary").is_none(), "binary must never be stored");

        run.finish(PiRunFinish::completed("", Some(0)))
            .await
            .expect("finish");
    }

    #[tokio::test]
    async fn task_description_is_capped() {
        let ctx = make_ctx();
        let long = "x".repeat(MAX_TASK_DESCRIPTION_CHARS + 500);
        let run = PiRun::begin(&ctx, &long, "p", "m", None, 100)
            .await
            .expect("begin");
        let (task, _) = task_data(&ctx.graph, &run.task_id);
        assert_eq!(task.description.chars().count(), MAX_TASK_DESCRIPTION_CHARS);
        assert!(task.description.ends_with('…'));
        let (_, ameta) = agent_data(&ctx.graph, &run.pi_agent_id);
        assert!(ameta.get("pi_version").is_none());
        run.finish(PiRunFinish::completed("", Some(0)))
            .await
            .expect("finish");
    }

    #[tokio::test]
    async fn record_session_stores_pi_session_id_and_cwd_on_agent() {
        let ctx = make_ctx();
        let mut run = begin(&ctx, 4000).await;

        run.record_session(Some("sess-42"), Some("/work/repo"))
            .await
            .expect("record_session");
        let (agent, meta) = agent_data(&ctx.graph, &run.pi_agent_id);
        assert_eq!(meta["pi_session_id"], "sess-42");
        assert_eq!(meta["pi_cwd"], "/work/repo");
        assert_eq!(agent.status, "running", "status untouched");
        assert_eq!(meta["pi_version"], "0.9.1", "existing metadata preserved");

        // Partial update keeps the other key.
        run.record_session(None, Some("/elsewhere"))
            .await
            .expect("record_session");
        let (_, meta) = agent_data(&ctx.graph, &run.pi_agent_id);
        assert_eq!(meta["pi_session_id"], "sess-42");
        assert_eq!(meta["pi_cwd"], "/elsewhere");

        run.finish(PiRunFinish::completed("", Some(0)))
            .await
            .expect("finish");
    }

    #[tokio::test]
    async fn tool_node_matches_in_process_shape() {
        let ctx = make_ctx();
        let mut run = begin(&ctx, 4000).await;
        let g = &ctx.graph;

        let big_args = json!({ "command": "a".repeat(2000) });
        let judge = json!({ "version": "v1", "p_irreversible": 0.07, "action": "observed" });
        let first = run
            .record_tool_call(bash_call("call_1", &big_args, Some(judge.clone())))
            .await
            .expect("record first");

        let (data, meta) = interaction_data(g, &first);
        assert_eq!(data.role, "tool");
        assert_eq!(data.content, "ok");
        assert_eq!(data.token_count, None);
        assert_eq!(meta["tool_call_id"], "call_1");
        assert_eq!(meta["tool_name"], "bash");
        assert_eq!(meta["is_error"], false);
        assert_eq!(meta["executor"], "pi");
        assert_eq!(meta["session_id"], run.pi_agent_id.to_string());
        assert_eq!(meta["parent_session_id"], ctx.agent_id.to_string());
        assert_eq!(meta["hitl_judge"], judge);
        let args = meta["arguments"].as_str().expect("arguments is a string");
        assert_eq!(args.chars().count(), MAX_ARGS_CHARS + 1);
        assert_eq!(MAX_ARGS_CHARS, 1500, "shared with hitl_judge");
        assert!(args.ends_with('…'));
        assert_eq!(meta["label"], "interaction_3_1_1");

        let produced = out(g, &run.pi_agent_id, EdgeType::Produces);
        assert_eq!(produced.len(), 1);
        assert_eq!(produced[0].id, first);
        // First Pi node has nothing to respond to.
        assert!(out(g, &first, EdgeType::RespondsTo).is_empty());

        let small_args = json!({ "path": "src/main.rs" });
        let second = run
            .record_tool_call(PiToolCall {
                tool_call_id: "call_2",
                tool_name: "read",
                args: &small_args,
                result_text: "fn main() {}",
                is_error: true,
                hitl_judge: None,
            })
            .await
            .expect("record second");

        let (data2, meta2) = interaction_data(g, &second);
        assert_eq!(data2.role, "tool");
        assert_eq!(meta2["tool_name"], "read");
        assert_eq!(meta2["is_error"], true);
        assert_eq!(meta2["arguments"], small_args.to_string());
        assert!(
            meta2.get("hitl_judge").is_none(),
            "hitl_judge only when passed"
        );
        assert_eq!(meta2["label"], "interaction_3_2_1");

        let responds = out(g, &second, EdgeType::RespondsTo);
        assert_eq!(responds.len(), 1);
        assert_eq!(responds[0].id, first);
        assert_eq!(out(g, &run.pi_agent_id, EdgeType::Produces).len(), 2);

        run.finish(PiRunFinish::completed("", Some(0)))
            .await
            .expect("finish");
    }

    #[tokio::test]
    async fn tool_result_content_is_capped_with_notice() {
        let ctx = make_ctx();
        let mut run = begin(&ctx, 4000).await;
        let huge = "y".repeat(MAX_TOOL_CONTENT_CHARS + 10);
        let args = json!({});
        let id = run
            .record_tool_call(PiToolCall {
                tool_call_id: "c",
                tool_name: "bash",
                args: &args,
                result_text: &huge,
                is_error: false,
                hitl_judge: None,
            })
            .await
            .expect("record");
        let (data, _) = interaction_data(&ctx.graph, &id);
        assert!(
            data.content
                .starts_with(&"y".repeat(MAX_TOOL_CONTENT_CHARS))
        );
        assert!(
            data.content
                .contains("[truncated: 16000 of 16010 chars stored]")
        );
        run.finish(PiRunFinish::completed("", Some(0)))
            .await
            .expect("finish");
    }

    #[tokio::test]
    async fn assistant_message_node_and_task_result() {
        let ctx = make_ctx();
        let mut run = begin(&ctx, 4000).await;
        let g = &ctx.graph;

        let args = json!({ "command": "ls" });
        let tool_node = run
            .record_tool_call(bash_call("call_1", &args, None))
            .await
            .expect("tool");
        let usage = json!({ "input": 120, "output": 45 });
        let msg = run
            .record_assistant_message("All done.", Some("stop"), Some(&usage), true)
            .await
            .expect("assistant");

        let (data, meta) = interaction_data(g, &msg);
        assert_eq!(data.role, "assistant");
        assert_eq!(data.content, "All done.");
        assert_eq!(data.token_count, None);
        assert_eq!(meta["executor"], "pi");
        assert_eq!(meta["stop_reason"], "stop");
        assert_eq!(meta["usage"], usage);
        assert_eq!(meta["session_id"], run.pi_agent_id.to_string());
        assert_eq!(meta["label"], "interaction_3_2_1");
        let responds = out(g, &msg, EdgeType::RespondsTo);
        assert_eq!(responds.len(), 1);
        assert_eq!(responds[0].id, tool_node);
        assert!(
            out(g, &run.pi_agent_id, EdgeType::Produces)
                .iter()
                .any(|n| n.id == msg)
        );

        // Without usage / stop_reason the keys are absent.
        let bare = run
            .record_assistant_message("more", None, None, true)
            .await
            .expect("assistant");
        let (_, bare_meta) = interaction_data(g, &bare);
        assert!(bare_meta.get("usage").is_none());
        assert!(bare_meta.get("stop_reason").is_none());

        let task_id = run.task_id.clone();
        let pi_agent_id = run.pi_agent_id.clone();
        let touched = run
            .finish(PiRunFinish::completed("All done.", Some(0)))
            .await
            .expect("finish");
        assert_eq!(touched, vec![task_id.clone(), pi_agent_id.clone()]);

        let (task, tmeta) = task_data(g, &task_id);
        assert_eq!(task.status, TaskStatus::Completed);
        assert_eq!(tmeta["result"], "All done.");
        assert_eq!(tmeta["tool_calls"], 1);
        assert_eq!(tmeta["assistant_messages"], 2);
        assert!(tmeta["duration_ms"].is_u64());
        assert_eq!(tmeta["exit_code"], 0);
        assert!(tmeta.get("failure").is_none());
        // Defaults are not stored.
        assert!(tmeta.get("errors").is_none());
        assert!(tmeta.get("judge_over_threshold").is_none());
        assert!(tmeta.get("pipes_lingered").is_none());
        assert!(tmeta.get("stderr_tail").is_none());
        // begin-time metadata survives finish
        assert_eq!(tmeta["provider"], "openrouter");
        assert_eq!(tmeta["executor"], "pi");
        assert!(tmeta.get("binary").is_none());

        let (agent, _) = agent_data(g, &pi_agent_id);
        assert_eq!(agent.status, "completed");
    }

    #[tokio::test]
    async fn finish_failed_records_failure_kind() {
        let ctx = make_ctx();
        let run = begin(&ctx, 4000).await;
        let task_id = run.task_id.clone();
        let pi_agent_id = run.pi_agent_id.clone();
        run.finish(PiRunFinish::failed(
            "timeout",
            "z".repeat(MAX_FAILURE_DETAIL_CHARS + 50),
            None,
        ))
        .await
        .expect("finish");

        let (task, meta) = task_data(&ctx.graph, &task_id);
        assert_eq!(task.status, TaskStatus::Failed);
        assert_eq!(meta["failure"], "timeout");
        assert_eq!(
            meta["failure_detail"]
                .as_str()
                .expect("detail")
                .chars()
                .count(),
            MAX_FAILURE_DETAIL_CHARS
        );
        assert_eq!(meta["tool_calls"], 0);
        assert!(meta["duration_ms"].is_u64());
        assert!(meta["exit_code"].is_null());
        assert!(meta.get("result").is_none());

        let (agent, _) = agent_data(&ctx.graph, &pi_agent_id);
        assert_eq!(agent.status, "failed");
    }

    #[tokio::test]
    async fn finish_stores_errors_judge_pipes_and_stderr_when_non_default() {
        let ctx = make_ctx();
        let run = begin(&ctx, 4000).await;
        let task_id = run.task_id.clone();

        let errors: Vec<String> = (0..MAX_ERRORS + 5)
            .map(|i| format!("e{i}-") + &"!".repeat(MAX_ERROR_ENTRY_CHARS))
            .collect();
        run.finish(PiRunFinish::Failed {
            kind: "exit",
            detail: "pi exited 1".to_string(),
            exit_code: Some(1),
            errors,
            judge_over_threshold: 2,
            pipes_lingered: true,
            stderr_tail: Some("s".repeat(MAX_STDERR_TAIL_CHARS + 7)),
        })
        .await
        .expect("finish");

        let (_, meta) = task_data(&ctx.graph, &task_id);
        let stored = meta["errors"].as_array().expect("errors array");
        assert_eq!(stored.len(), MAX_ERRORS);
        for e in stored {
            let s = e.as_str().expect("error string");
            assert_eq!(s.chars().count(), MAX_ERROR_ENTRY_CHARS);
            assert!(s.ends_with('…'));
        }
        assert!(stored[0].as_str().expect("s").starts_with("e0-"));
        assert_eq!(meta["judge_over_threshold"], 2);
        assert_eq!(meta["pipes_lingered"], true);
        assert_eq!(
            meta["stderr_tail"].as_str().expect("tail").chars().count(),
            MAX_STDERR_TAIL_CHARS
        );
        assert_eq!(meta["exit_code"], 1);
        assert_eq!(meta["failure"], "exit");

        // Same optional keys work on Completed.
        let ctx2 = make_ctx();
        let run2 = begin(&ctx2, 4000).await;
        let task2 = run2.task_id.clone();
        run2.finish(PiRunFinish::Completed {
            summary: "ok".to_string(),
            exit_code: Some(1),
            errors: vec!["provider hiccup".to_string()],
            judge_over_threshold: 1,
            pipes_lingered: false,
            stderr_tail: Some(String::new()),
        })
        .await
        .expect("finish");
        let (task, meta2) = task_data(&ctx2.graph, &task2);
        assert_eq!(task.status, TaskStatus::Completed);
        assert_eq!(meta2["errors"], json!(["provider hiccup"]));
        assert_eq!(meta2["judge_over_threshold"], 1);
        assert!(meta2.get("pipes_lingered").is_none());
        assert!(
            meta2.get("stderr_tail").is_none(),
            "empty tail is not stored"
        );
    }

    #[tokio::test]
    async fn result_truncated_to_max_chars() {
        let ctx = make_ctx();
        let run = begin(&ctx, 100).await;
        let task_id = run.task_id.clone();
        run.finish(PiRunFinish::completed("r".repeat(500), Some(0)))
            .await
            .expect("finish");

        let (_, meta) = task_data(&ctx.graph, &task_id);
        let result = meta["result"].as_str().expect("result");
        assert_eq!(result.chars().count(), 100);
        assert!(result.ends_with('…'));
    }

    #[test]
    fn truncate_within_zero_is_empty() {
        assert_eq!(truncate_within("abc", 0), "");
        assert_eq!(truncate_within("abc", 1), "…");
        assert_eq!(truncate_within("abc", 3), "abc");
        assert_eq!(truncate_within("abcd", 3), "ab…");
    }

    #[tokio::test]
    async fn labels_do_not_collide_with_director_tool_nodes() {
        let ctx = make_ctx();
        // Director already recorded two nodes this turn.
        ctx.turn_pos_counter.fetch_add(2, Ordering::SeqCst);
        let mut run = begin(&ctx, 4000).await;
        let args = json!({});
        let pi_node = run
            .record_tool_call(bash_call("c1", &args, None))
            .await
            .expect("record");
        // Director records its own tool result after Pi's, via the shared counter.
        let director_pos = ctx.turn_pos_counter.fetch_add(1, Ordering::SeqCst) + 1;
        let director_label = format!("interaction_{}_{}_1", ctx.turn, director_pos);

        let (_, meta) = interaction_data(&ctx.graph, &pi_node);
        assert_eq!(meta["label"], "interaction_3_3_1");
        assert_eq!(director_label, "interaction_3_4_1");
        assert_ne!(meta["label"], director_label);

        run.finish(PiRunFinish::completed("", Some(0)))
            .await
            .expect("finish");
    }

    #[tokio::test]
    async fn dropped_unfinished_run_is_marked_abandoned() {
        let ctx = make_ctx();
        let mut run = begin(&ctx, 4000).await;
        let args = json!({});
        run.record_tool_call(bash_call("c1", &args, None))
            .await
            .expect("record");
        let task_id = run.task_id.clone();
        let pi_agent_id = run.pi_agent_id.clone();

        drop(run);

        // The Drop path enqueues a blocking write; poll until it lands.
        let mut marked = false;
        for _ in 0..100 {
            tokio::time::sleep(Duration::from_millis(20)).await;
            let (task, _) = task_data(&ctx.graph, &task_id);
            if task.status == TaskStatus::Failed {
                marked = true;
                break;
            }
        }
        assert!(marked, "abandoned run was not marked within 2s");

        let (task, meta) = task_data(&ctx.graph, &task_id);
        assert_eq!(task.status, TaskStatus::Failed);
        assert_eq!(meta["failure"], FAILURE_ABANDONED);
        assert!(meta["duration_ms"].is_u64());
        assert_eq!(meta["provider"], "openrouter", "begin metadata preserved");
        let (agent, _) = agent_data(&ctx.graph, &pi_agent_id);
        assert_eq!(agent.status, "failed");
    }

    #[tokio::test]
    async fn finished_run_is_not_remarked_on_drop() {
        let ctx = make_ctx();
        let run = begin(&ctx, 4000).await;
        let task_id = run.task_id.clone();
        run.finish(PiRunFinish::completed("done", Some(0)))
            .await
            .expect("finish");
        // Give a stray Drop write time to land if it were (wrongly) enqueued.
        tokio::time::sleep(Duration::from_millis(100)).await;
        let (task, meta) = task_data(&ctx.graph, &task_id);
        assert_eq!(task.status, TaskStatus::Completed);
        assert!(meta.get("failure").is_none());
    }

    #[tokio::test]
    async fn html_reply_becomes_part_and_line_nodes() {
        let ctx = make_ctx();
        let mut run = begin(&ctx, 4000).await;
        let g = &ctx.graph;
        let html = r##"<nav id="index"><ul><li><a href="#part1">1.0.0 [statement] Overview</a></li></ul></nav>
<section id="part1" class="statement"><h2>1.0.0 Overview</h2><p>1.0.1 One sentence.</p></section>"##;

        let msg = run
            .record_assistant_message(html, Some("stop"), None, true)
            .await
            .expect("assistant");
        let (data, meta) = interaction_data(g, &msg);
        assert_eq!(data.content, html);
        assert!(meta.get("pieces").is_none());

        let parts = out(g, &msg, EdgeType::Contains);
        assert_eq!(parts.len(), 1);
        let part = &parts[0];
        let NodeType::Content(ContentData {
            content_type, body, ..
        }) = &part.node_type
        else {
            panic!("expected Content, got {}", part.node_type.type_name());
        };
        assert_eq!(content_type, "reply_part");
        assert_eq!(body, "Overview");
        assert_eq!(part.metadata["order"], json!(1));
        assert_eq!(part.metadata["id"], "part1");
        assert_eq!(part.metadata["kind"], "statement");
        assert_eq!(part.metadata["heading"], "Overview");
        let start = part.metadata["start"].as_u64().expect("start") as usize;
        let end = part.metadata["end"].as_u64().expect("end") as usize;
        assert!(html[start..end].starts_with("<section id=\"part1\""));
        assert_eq!(part.metadata["shape_version"], json!("3"));

        let lines = out(g, &part.id, EdgeType::Contains);
        assert_eq!(lines.len(), 1);
        let line = &lines[0];
        let NodeType::Content(ContentData {
            content_type, body, ..
        }) = &line.node_type
        else {
            panic!("expected Content, got {}", line.node_type.type_name());
        };
        assert_eq!(content_type, "reply_line");
        assert_eq!(body, "One sentence.");
        assert_eq!(line.metadata["position"], json!(1));
        let line_start = line.metadata["start"].as_u64().expect("start") as usize;
        let line_end = line.metadata["end"].as_u64().expect("end") as usize;
        assert_eq!(&html[line_start..line_end], "<p>1.0.1 One sentence.</p>");

        let markdown = "# Hello\n\nA paragraph.\n";
        let plain = run
            .record_assistant_message(markdown, Some("stop"), None, true)
            .await
            .expect("markdown");
        assert!(out(g, &plain, EdgeType::Contains).is_empty());

        let narration = run
            .record_assistant_message(html, Some("toolUse"), None, false)
            .await
            .expect("narration");
        assert!(
            out(g, &narration, EdgeType::Contains).is_empty(),
            "toolUse is not cut"
        );

        run.finish(PiRunFinish::completed("", Some(0)))
            .await
            .expect("finish");
    }

    #[tokio::test]
    async fn second_reply_names_a_part_of_the_first() {
        let ctx = make_ctx();
        let mut run = begin(&ctx, 4000).await;
        let g = &ctx.graph;
        let first_html = r##"<nav id="index"><ul><li><a href="#deploy">1.0.0 [steps] Deploy</a></li></ul></nav>
<section id="deploy" class="steps"><h2>1.0.0 Deploy</h2><ul><li>1.0.1 Build</li><li>1.0.2 Ship</li></ul></section>"##;
        let second_html = r##"<nav id="index"><ul><li><a href="#use">1.0.0 [statement] Follow-up</a></li></ul></nav>
<section id="use" class="statement"><h2>1.0.0 Follow-up</h2><p>1.0.1 Use step 2 of Deploy from the previous reply.</p></section>"##;
        let first = run
            .record_assistant_message(first_html, Some("stop"), None, true)
            .await
            .expect("first");
        let second = run
            .record_assistant_message(second_html, Some("stop"), None, true)
            .await
            .expect("second");

        let first_ids = contains_tree(g, &first);
        let second_ids = contains_tree(g, &second);
        assert!(first_ids.is_disjoint(&second_ids));

        let mut crossing = Vec::new();
        for id in &second_ids {
            for edge in g.edges_for_node(id).expect("edges") {
                if second_ids.contains(&edge.source) && first_ids.contains(&edge.target) {
                    crossing.push(edge);
                }
            }
        }
        assert_eq!(crossing.len(), 1, "{crossing:?}");
        assert_eq!(crossing[0].edge_type, EdgeType::RespondsTo);
        assert_eq!(crossing[0].source, second);
        assert_eq!(crossing[0].target, first);

        let second_lines = line_bodies(g, &second_ids);
        assert!(
            second_lines
                .iter()
                .any(|body| body.contains("Deploy") && body.contains("step 2")),
            "{second_lines:?}"
        );
        let ship = g
            .get_node(
                first_ids
                    .iter()
                    .find(|id| line_body(g, id).as_deref() == Some("Ship"))
                    .expect("Ship line"),
            )
            .expect("ship");
        assert!(
            crossing.iter().all(|edge| edge.target != ship.id),
            "the words do not select the Ship node"
        );

        for id in first_ids.union(&second_ids) {
            for edge in g.edges_for_node(id).expect("edges") {
                let name = edge.edge_type.as_str();
                assert!(
                    !matches!(
                        name,
                        "applies_to" | "answered_by" | "executed_by" | "skipped" | "implements"
                    ),
                    "{name}"
                );
            }
        }

        run.finish(PiRunFinish::completed("", Some(0)))
            .await
            .expect("finish");
    }

    fn contains_tree(graph: &GraphStore, root: &NodeId) -> std::collections::HashSet<NodeId> {
        let mut ids = std::collections::HashSet::new();
        let mut stack = vec![root.clone()];
        while let Some(id) = stack.pop() {
            if !ids.insert(id.clone()) {
                continue;
            }
            for child in out(graph, &id, EdgeType::Contains) {
                stack.push(child.id);
            }
        }
        ids
    }

    fn line_bodies(graph: &GraphStore, ids: &std::collections::HashSet<NodeId>) -> Vec<String> {
        ids.iter().filter_map(|id| line_body(graph, id)).collect()
    }

    fn line_body(graph: &GraphStore, id: &NodeId) -> Option<String> {
        let node = graph.get_node(id).expect("node");
        match node.node_type {
            NodeType::Content(ContentData {
                content_type, body, ..
            }) if content_type == "reply_line" => Some(body),
            _ => None,
        }
    }
}
