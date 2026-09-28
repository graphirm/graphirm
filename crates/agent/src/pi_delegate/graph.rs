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
//! `GET /api/sessions/{id}/graph`); only `provider`, `model`, `pi_version` and
//! `exit_code` are stored.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use graphirm_graph::GraphStore;
use graphirm_graph::edges::{EdgeType, GraphEdge};
use graphirm_graph::nodes::{
    AgentData, GraphNode, InteractionData, NodeId, NodeType, TaskData, TaskStatus,
};
use graphirm_tools::ToolContext;
use serde_json::{Map, Value, json};

use crate::error::AgentError;

/// Discriminator stored as `metadata.executor` on every node this module writes.
pub const PI_EXECUTOR: &str = "pi";
/// Title of the delegation Task node (same pattern as `spawn_subagent`).
pub const PI_TASK_TITLE: &str = "Delegated to pi";
/// `metadata.arguments` on a Pi tool node keeps at most this many chars of
/// compact JSON, then `…` is appended (same rule as the HITL judge).
pub const MAX_ARGS_CHARS: usize = 1500;
/// Task description cap; the full brief lives in Pi's stdin, not the graph.
pub const MAX_TASK_DESCRIPTION_CHARS: usize = 4000;
/// Tool result content cap on a Pi tool node.
pub const MAX_TOOL_CONTENT_CHARS: usize = 16_000;
/// `metadata.failure_detail` cap.
pub const MAX_FAILURE_DETAIL_CHARS: usize = 2000;

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

/// How a Pi run ended.
#[derive(Debug, Clone)]
pub enum PiRunFinish {
    /// Pi produced a final message; `summary` becomes `Task.metadata.result`.
    Completed {
        summary: String,
        exit_code: Option<i32>,
    },
    /// The run did not complete. `kind` is one of `"timeout"`, `"cancelled"`,
    /// `"exit"`, `"spawn"`.
    Failed {
        kind: &'static str,
        detail: String,
        exit_code: Option<i32>,
    },
}

/// Live graph handle for one Pi delegation. Create with [`PiRun::begin`], feed
/// it Pi's events, then call [`PiRun::finish`] exactly once.
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
}

impl PiRun {
    /// Creates the Task and Pi Agent nodes plus the three delegation edges.
    ///
    /// `provider`, `model` and `pi_version` go to metadata — never the binary path.
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

        let mut task_node = GraphNode::new(NodeType::Task(TaskData {
            title: PI_TASK_TITLE.to_string(),
            description: truncate_within(task_text, MAX_TASK_DESCRIPTION_CHARS),
            status: TaskStatus::Pending,
            priority: None,
        }));
        let mut task_meta = Map::new();
        task_meta.insert("executor".into(), json!(PI_EXECUTOR));
        task_meta.insert("provider".into(), json!(provider));
        task_meta.insert("model".into(), json!(model));
        if let Some(v) = pi_version {
            task_meta.insert("pi_version".into(), json!(v));
        }
        task_meta.insert(
            "parent_session_id".into(),
            json!(director_agent.to_string()),
        );
        task_node.metadata = Value::Object(task_meta);
        let task_id = task_node.id.clone();

        let mut agent_node = GraphNode::new(NodeType::Agent(AgentData {
            name: PI_EXECUTOR.to_string(),
            model: model.to_string(),
            system_prompt: None,
            status: "running".to_string(),
        }));
        let mut agent_meta = Map::new();
        agent_meta.insert("executor".into(), json!(PI_EXECUTOR));
        agent_meta.insert(
            "parent_session_id".into(),
            json!(director_agent.to_string()),
        );
        agent_meta.insert("task_id".into(), json!(task_id.to_string()));
        if let Some(v) = pi_version {
            agent_meta.insert("pi_version".into(), json!(v));
        }
        agent_node.metadata = Value::Object(agent_meta);
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
        })
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
    /// under the Pi Agent.
    pub async fn record_assistant_message(
        &mut self,
        text: &str,
        stop_reason: Option<&str>,
    ) -> Result<NodeId, AgentError> {
        let mut meta = self.base_metadata();
        if let Some(reason) = stop_reason {
            meta.insert("stop_reason".into(), json!(reason));
        }

        let mut node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".to_string(),
            content: truncate_with_notice(text, MAX_TOOL_CONTENT_CHARS),
            token_count: None,
        }));
        node.metadata = Value::Object(meta);

        let id = self.persist_interaction(node).await?;
        self.assistant_messages += 1;
        Ok(id)
    }

    /// Finalises the Task and Pi Agent status and metadata.
    /// Returns `[task_id, pi_agent_id]`.
    pub async fn finish(&mut self, outcome: PiRunFinish) -> Result<Vec<NodeId>, AgentError> {
        let duration_ms = self.started.elapsed().as_millis() as u64;
        let mut extra = Map::new();
        extra.insert("tool_calls".into(), json!(self.tool_calls));
        extra.insert("duration_ms".into(), json!(duration_ms));

        let (task_status, agent_status) = match outcome {
            PiRunFinish::Completed { summary, exit_code } => {
                extra.insert(
                    "result".into(),
                    json!(truncate_within(&summary, self.max_result_chars)),
                );
                extra.insert("assistant_messages".into(), json!(self.assistant_messages));
                extra.insert("exit_code".into(), json!(exit_code));
                (TaskStatus::Completed, "completed")
            }
            PiRunFinish::Failed {
                kind,
                detail,
                exit_code,
            } => {
                extra.insert("failure".into(), json!(kind));
                extra.insert(
                    "failure_detail".into(),
                    json!(truncate_within(&detail, MAX_FAILURE_DETAIL_CHARS)),
                );
                extra.insert("exit_code".into(), json!(exit_code));
                (TaskStatus::Failed, "failed")
            }
        };

        let g = self.graph.clone();
        let task_id = self.task_id.clone();
        let pi_agent_id = self.pi_agent_id.clone();
        run_blocking(move || {
            let mut task_node = g.get_node(&task_id)?;
            if let NodeType::Task(ref mut data) = task_node.node_type {
                data.status = task_status;
            }
            merge_metadata(&mut task_node, extra);
            g.update_node(&task_id, task_node)?;

            let mut agent_node = g.get_node(&pi_agent_id)?;
            if let NodeType::Agent(ref mut data) = agent_node.node_type {
                data.status = agent_status.to_string();
            }
            g.update_node(&pi_agent_id, agent_node)?;
            Ok(())
        })
        .await?;

        tracing::info!(
            task_id = %self.task_id,
            status = %task_status,
            tool_calls = self.tool_calls,
            duration_ms,
            "Pi delegation finished"
        );

        Ok(vec![self.task_id.clone(), self.pi_agent_id.clone()])
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

/// Compact JSON of the arguments, capped at [`MAX_ARGS_CHARS`] chars with `…`
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
fn truncate_within(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
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
        assert_eq!(meta["pi_version"], "0.9.1");
        assert_eq!(meta["parent_session_id"], ctx.agent_id.to_string());
        assert!(meta.get("binary").is_none(), "binary must never be stored");

        let (agent, ameta) = agent_data(g, &run.pi_agent_id);
        assert_eq!(agent.name, "pi");
        assert_eq!(agent.status, "running");
        assert_eq!(agent.model, "deepseek/deepseek-v4-flash");
        assert_eq!(ameta["executor"], "pi");
        assert_eq!(ameta["task_id"], run.task_id.to_string());
        assert_eq!(ameta["parent_session_id"], ctx.agent_id.to_string());
        assert!(ameta.get("binary").is_none(), "binary must never be stored");
    }

    #[tokio::test]
    async fn task_description_is_capped() {
        let ctx = make_ctx();
        let long = "x".repeat(MAX_TASK_DESCRIPTION_CHARS + 500);
        let run = PiRun::begin(&ctx, &long, "p", "m", None, 100)
            .await
            .expect("begin");
        let (task, meta) = task_data(&ctx.graph, &run.task_id);
        assert_eq!(task.description.chars().count(), MAX_TASK_DESCRIPTION_CHARS);
        assert!(task.description.ends_with('…'));
        assert!(meta.get("pi_version").is_none());
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
        let msg = run
            .record_assistant_message("All done.", Some("stop"))
            .await
            .expect("assistant");

        let (data, meta) = interaction_data(g, &msg);
        assert_eq!(data.role, "assistant");
        assert_eq!(data.content, "All done.");
        assert_eq!(data.token_count, None);
        assert_eq!(meta["executor"], "pi");
        assert_eq!(meta["stop_reason"], "stop");
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

        let touched = run
            .finish(PiRunFinish::Completed {
                summary: "All done.".to_string(),
                exit_code: Some(0),
            })
            .await
            .expect("finish");
        assert_eq!(touched, vec![run.task_id.clone(), run.pi_agent_id.clone()]);

        let (task, tmeta) = task_data(g, &run.task_id);
        assert_eq!(task.status, TaskStatus::Completed);
        assert_eq!(tmeta["result"], "All done.");
        assert_eq!(tmeta["tool_calls"], 1);
        assert_eq!(tmeta["assistant_messages"], 1);
        assert!(tmeta["duration_ms"].is_u64());
        assert_eq!(tmeta["exit_code"], 0);
        assert!(tmeta.get("failure").is_none());
        // begin-time metadata survives finish
        assert_eq!(tmeta["provider"], "openrouter");
        assert_eq!(tmeta["executor"], "pi");
        assert!(tmeta.get("binary").is_none());

        let (agent, _) = agent_data(g, &run.pi_agent_id);
        assert_eq!(agent.status, "completed");
    }

    #[tokio::test]
    async fn finish_failed_records_failure_kind() {
        let ctx = make_ctx();
        let mut run = begin(&ctx, 4000).await;
        run.finish(PiRunFinish::Failed {
            kind: "timeout",
            detail: "z".repeat(MAX_FAILURE_DETAIL_CHARS + 50),
            exit_code: None,
        })
        .await
        .expect("finish");

        let (task, meta) = task_data(&ctx.graph, &run.task_id);
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

        let (agent, _) = agent_data(&ctx.graph, &run.pi_agent_id);
        assert_eq!(agent.status, "failed");
    }

    #[tokio::test]
    async fn result_truncated_to_max_chars() {
        let ctx = make_ctx();
        let mut run = begin(&ctx, 100).await;
        run.finish(PiRunFinish::Completed {
            summary: "r".repeat(500),
            exit_code: Some(0),
        })
        .await
        .expect("finish");

        let (_, meta) = task_data(&ctx.graph, &run.task_id);
        let result = meta["result"].as_str().expect("result");
        assert_eq!(result.chars().count(), 100);
        assert!(result.ends_with('…'));
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
    }
}
