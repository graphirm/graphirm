//! Observe-only reply judge for assistant Interaction nodes.
//!
//! After an assistant message is stored, ask Jev the five tagset-v4 questions
//! from jev-hooks (`move`, `claims_completion`, `cites_evidence`,
//! `presents_decision`, `presents_as_options`). Raw scores are written to
//! `metadata.reply_judge`. Thresholds live in the client. Every failure path
//! warns and leaves the key unset; the turn is never failed from here.

use std::sync::Arc;
use std::time::Duration;

use graphirm_graph::GraphStore;
use graphirm_graph::nodes::{GraphNode, NodeId, NodeType};
use graphirm_llm::{Answer, Decision, DecisionsClient, LlmError};
use serde_json::{Value, json};

use crate::config::AgentConfig;
use crate::hitl_judge::resolve_decisions_config;

/// Tagset version recorded with every verdict. Matches jev-hooks `TAGSET_VERSION`.
pub const REPLY_JUDGE_VERSION: &str = "v4";

/// Assistant `message` cap, matching jev-hooks `MAX_STATE_CHARS`.
const MAX_MESSAGE_CHARS: usize = 20_000;

/// Preceding turns kept, matching jev-hooks `MAX_PREV_TURNS`.
const MAX_PREV_TURNS: usize = 3;

/// Total characters across `previous_message`, matching jev-hooks `MAX_PREV_CHARS`.
const MAX_PREV_CHARS: usize = 4_000;

/// Used when `[agent.hitl_judge]` is absent. Matches jev-hooks `HTTP_TIMEOUT_S`.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(8);

/// Sentence appended to every assistant question. Do not reword.
const ONLY_MESSAGE: &str =
    "Judge only the text in the `message` field. Ignore `previous_message` entirely.";

const MOVE_BODY: &str = "What is the assistant primarily doing in its text in the `message` field?";

const CLAIMS_COMPLETION_BODY: &str = "Does the assistant's text in the `message` field claim that work is finished, fixed, passing, or complete?";

const CITES_EVIDENCE_BODY: &str = "Does the assistant's text in the `message` field show concrete evidence such as command output, test results, or file contents to support its claims?";

const PRESENTS_DECISION_BODY: &str = "Does the assistant's text in the `message` field ask the human to make a decision: choose between alternatives, approve or reject a proposed action, or answer a question that determines what happens next? Typical forms: a 'Decision' block with options and a recommendation, 'say go / merge if you want it', 'want me to ...?'. The request may come at the end of a long report. A message that only reports, explains, or offers unsolicited suggestions without asking for a reply does not count.";

const PRESENTS_AS_OPTIONS_BODY: &str = "In the assistant's text in the `message` field, are two or more alternatives for the human to choose between presented as an explicit enumerated list with labels or ids (1./2., (a)/(b), or names), anywhere in the message including after a report?";

fn with_only_message(body: &str) -> String {
    format!("{body} {ONLY_MESSAGE}")
}

const Q_MOVE: &str = "move";
const Q_CLAIMS: &str = "claims_completion";
const Q_EVIDENCE: &str = "cites_evidence";
const Q_DECISION: &str = "presents_decision";
const Q_OPTIONS: &str = "presents_as_options";

/// One earlier turn sent as `state.previous_message`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrevTurn {
    /// Interaction role as stored (`user`, `assistant`, `tool`).
    pub role: String,
    /// Interaction content.
    pub text: String,
}

/// Scores one assistant reply. Failures are `Err`; callers leave metadata unset.
pub struct ReplyJudge {
    client: Arc<DecisionsClient>,
    timeout: Duration,
}

impl ReplyJudge {
    pub fn new(client: Arc<DecisionsClient>, timeout: Duration) -> Self {
        Self { client, timeout }
    }

    /// Per-call deadline passed to the Decisions client.
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// State shape from jev-hooks `build_body`. `previous_message` is omitted when empty.
    pub fn build_state(message: &str, previous: &[PrevTurn]) -> Value {
        let mut state = json!({
            "role": "assistant",
            "message": take_chars(message, MAX_MESSAGE_CHARS),
        });
        let trimmed = trim_turns(previous);
        if !trimmed.is_empty() {
            let turns: Vec<Value> = trimmed
                .iter()
                .map(|turn| json!({"role": turn.role, "text": turn.text}))
                .collect();
            state["previous_message"] = Value::Array(turns);
        }
        state
    }

    /// The five assistant questions, wording copied from tagset v4.
    pub fn build_questions() -> Value {
        json!({
            Q_MOVE: {
                "type": "choice",
                "instructions": with_only_message(MOVE_BODY),
                "criteria": {
                    "plan": "Describes what it intends to do before doing it",
                    "report_progress": "Reports work done so far without claiming the task is complete",
                    "claim_done": "States that the task or a requested change is complete",
                    "ask_user": "Asks the human a question or requests a decision",
                    "retry_same": "Repeats an approach already tried without a new hypothesis",
                    "explain": "Explains a concept or answers a question without changing anything",
                },
            },
            Q_CLAIMS: {
                "type": "noul",
                "instructions": with_only_message(CLAIMS_COMPLETION_BODY),
                "criteria": {
                    "true": "The `message` field asserts that a task, fix, test run, or deliverable is done or passing",
                    "false": "No completion claim; work is described as ongoing, planned, or not attempted",
                },
            },
            Q_EVIDENCE: {
                "type": "noul",
                "instructions": with_only_message(CITES_EVIDENCE_BODY),
                "criteria": {
                    "true": "Verifiable output or observation is quoted or described specifically in the `message` field",
                    "false": "Claims are asserted without shown evidence",
                },
            },
            Q_DECISION: {
                "type": "noul",
                "instructions": with_only_message(PRESENTS_DECISION_BODY),
                "criteria": {
                    "true": "The `message` field explicitly requests the human's choice or approval: options offered, a recommendation to confirm, or a direct question whose answer decides the next step",
                    "false": "Reporting, explanation, or suggestions with no request for the human's decision",
                },
            },
            Q_OPTIONS: {
                "type": "noul",
                "instructions": with_only_message(PRESENTS_AS_OPTIONS_BODY),
                "criteria": {
                    "true": "A labelled list of alternative choices for the human appears in the `message` field",
                    "false": "No alternatives are offered to the human, or they appear only in prose",
                },
            },
        })
    }

    /// Ask Jev. `Err` on transport failure, timeout, or an unusable answer.
    /// Scores are not clamped.
    pub async fn judge(&self, message: &str, previous: &[PrevTurn]) -> Result<Value, LlmError> {
        let decision = self
            .client
            .decide(
                Self::build_state(message, previous),
                Self::build_questions(),
                self.timeout,
            )
            .await?;
        metadata_from_decision(&decision)
    }
}

/// Build a client from `[agent.adaptive_routing.jev]` and the environment.
/// `None` (no warning) when endpoint/key/model cannot be resolved.
/// Does not require `hitl_judge.enabled`.
pub fn build_reply_judge(config: &AgentConfig) -> Option<Arc<ReplyJudge>> {
    build_reply_judge_with(config, |name| std::env::var(name).ok())
}

fn build_reply_judge_with(
    config: &AgentConfig,
    lookup: impl Fn(&str) -> Option<String>,
) -> Option<Arc<ReplyJudge>> {
    let jev = config
        .adaptive_routing
        .as_ref()
        .and_then(|routing| routing.jev.clone())
        .unwrap_or_default();
    let decisions = resolve_decisions_config(&jev, lookup).ok()?;
    Some(Arc::new(ReplyJudge::new(
        Arc::new(DecisionsClient::from_config(decisions)),
        reply_timeout(config),
    )))
}

/// Hitl-judge timeout when that section is present; otherwise 8 seconds.
fn reply_timeout(config: &AgentConfig) -> Duration {
    match &config.hitl_judge {
        Some(cfg) => Duration::from_millis(cfg.timeout_ms),
        None => DEFAULT_TIMEOUT,
    }
}

/// After the assistant Interaction is stored: read prior turns, ask Jev, write
/// `reply_judge`. Never returns an error. Skips silently when no client can be built.
pub async fn observe_assistant_reply(
    graph: Arc<GraphStore>,
    session_id: &str,
    node_id: &NodeId,
    config: &AgentConfig,
    message: &str,
) {
    let Some(judge) = build_reply_judge(config) else {
        return;
    };
    let previous = load_previous(graph.clone(), session_id, node_id).await;
    judge_and_store(graph, node_id.clone(), &judge, message, &previous).await;
}

/// Ask `judge` and write the metadata object. On failure, warn and leave the key unset.
pub async fn judge_and_store(
    graph: Arc<GraphStore>,
    node_id: NodeId,
    judge: &ReplyJudge,
    message: &str,
    previous: &[PrevTurn],
) {
    let metadata = match judge.judge(message, previous).await {
        Ok(value) => value,
        Err(e) => {
            tracing::warn!(error = %e, "reply_judge failed; leaving reply_judge unset");
            return;
        }
    };
    let write =
        tokio::task::spawn_blocking(move || write_reply_judge(&graph, &node_id, metadata)).await;
    match write {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            tracing::warn!(error = %e, "reply_judge failed to update node; leaving reply_judge unset");
        }
        Err(e) => {
            tracing::warn!(error = %e, "reply_judge update panicked; leaving reply_judge unset");
        }
    }
}

fn write_reply_judge(
    graph: &GraphStore,
    node_id: &NodeId,
    metadata: Value,
) -> Result<(), graphirm_graph::GraphError> {
    let mut node = graph.get_node(node_id)?;
    if let Some(map) = node.metadata.as_object_mut() {
        map.insert("reply_judge".to_string(), metadata);
    } else {
        node.metadata = json!({ "reply_judge": metadata });
    }
    graph.update_node(node_id, node)
}

async fn load_previous(
    graph: Arc<GraphStore>,
    session_id: &str,
    node_id: &NodeId,
) -> Vec<PrevTurn> {
    let session_id = session_id.to_string();
    let node_id = node_id.clone();
    match tokio::task::spawn_blocking(move || match graph.get_session_chain(&session_id) {
        Ok(chain) => previous_from_chain(&chain, &node_id),
        Err(e) => {
            tracing::warn!(error = %e, "reply_judge could not read prior turns");
            Vec::new()
        }
    })
    .await
    {
        Ok(turns) => turns,
        Err(e) => {
            tracing::warn!(error = %e, "reply_judge prior-turn read panicked");
            Vec::new()
        }
    }
}

/// Non-empty Interaction nodes before `current`, oldest first.
/// The current node is excluded. Role strings are stored as-is.
fn previous_from_chain(chain: &[GraphNode], current: &NodeId) -> Vec<PrevTurn> {
    let prior = match chain.iter().position(|node| &node.id == current) {
        Some(index) => &chain[..index],
        None => chain,
    };
    prior.iter().filter_map(turn_of).collect()
}

fn turn_of(node: &GraphNode) -> Option<PrevTurn> {
    let NodeType::Interaction(data) = &node.node_type else {
        return None;
    };
    if data.content.is_empty() {
        return None;
    }
    Some(PrevTurn {
        role: data.role.clone(),
        text: data.content.clone(),
    })
}

/// Last [`MAX_PREV_TURNS`] turns within [`MAX_PREV_CHARS`], oldest first.
/// Budget is spent newest-first; older turns are truncated, then dropped.
fn trim_turns(turns: &[PrevTurn]) -> Vec<PrevTurn> {
    let window: Vec<&PrevTurn> = turns.iter().rev().take(MAX_PREV_TURNS).collect();
    let mut kept = Vec::new();
    let mut budget = MAX_PREV_CHARS;
    for turn in window {
        if budget == 0 {
            break;
        }
        let text = take_chars(&turn.text, budget);
        if text.is_empty() {
            continue;
        }
        budget -= text.chars().count();
        kept.push(PrevTurn {
            role: turn.role.clone(),
            text,
        });
    }
    kept.reverse();
    kept
}

fn take_chars(text: &str, max: usize) -> String {
    text.chars().take(max).collect()
}

fn metadata_from_decision(decision: &Decision) -> Result<Value, LlmError> {
    let move_score = match decision.answer(Q_MOVE) {
        Some(Answer::Choice { confidence, .. }) => json_number(*confidence)?,
        Some(other) => {
            return Err(LlmError::provider(format!(
                "{Q_MOVE} answered with {other:?}, expected choice"
            )));
        }
        None => {
            return Err(LlmError::provider(format!("no {Q_MOVE} answer in reply")));
        }
    };
    Ok(json!({
        "version": REPLY_JUDGE_VERSION,
        "scores": {
            Q_MOVE: move_score,
            Q_CLAIMS: json_number(require_noul(decision, Q_CLAIMS)?)?,
            Q_EVIDENCE: json_number(require_noul(decision, Q_EVIDENCE)?)?,
            Q_DECISION: json_number(require_noul(decision, Q_DECISION)?)?,
            Q_OPTIONS: json_number(require_noul(decision, Q_OPTIONS)?)?,
        },
        "latency_ms": decision.latency.as_millis() as u64,
    }))
}

fn require_noul(decision: &Decision, id: &str) -> Result<f64, LlmError> {
    match decision.answer(id) {
        Some(Answer::Noul { p_yes }) => Ok(*p_yes),
        Some(other) => Err(LlmError::provider(format!(
            "{id} answered with {other:?}, expected noul"
        ))),
        None => Err(LlmError::provider(format!("no {id} answer in reply"))),
    }
}

/// JSON numbers only. Non-finite values cannot be stored; that is an unusable answer.
fn json_number(value: f64) -> Result<Value, LlmError> {
    serde_json::Number::from_f64(value)
        .map(Value::Number)
        .ok_or_else(|| LlmError::provider(format!("non-finite score: {value}")))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use graphirm_graph::GraphStore;
    use graphirm_graph::nodes::{GraphNode, InteractionData, NodeType};
    use graphirm_llm::DecisionsTransport;
    use serde_json::json;

    use super::*;
    use crate::config::{AgentConfig, HitlJudgeConfig, JevRouterConfig};
    use crate::hitl_judge::test_support::{HangingTransport, ReplyTransport};

    fn reply_judge(transport: Arc<dyn DecisionsTransport>, timeout: Duration) -> Arc<ReplyJudge> {
        Arc::new(ReplyJudge::new(
            Arc::new(graphirm_llm::DecisionsClient::with_transport(transport)),
            timeout,
        ))
    }

    fn assistant_node(graph: &GraphStore, content: &str) -> NodeId {
        let mut node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".to_string(),
            content: content.to_string(),
            token_count: None,
        }));
        node.metadata["session_id"] = json!("sess");
        let id = node.id.clone();
        graph.add_node(node).expect("insert");
        id
    }

    fn scored_reply(
        move_confidence: f64,
        claims: f64,
        cites: f64,
        decision: f64,
        options: f64,
    ) -> Value {
        json!({
            "answers": {
                "move": {
                    "type": "choice",
                    "choice": "explain",
                    "probabilities": {"explain": move_confidence},
                    "confidence": move_confidence
                },
                "claims_completion": {"type": "noul", "noul": claims},
                "cites_evidence": {"type": "noul", "noul": cites},
                "presents_decision": {"type": "noul", "noul": decision},
                "presents_as_options": {"type": "noul", "noul": options}
            }
        })
    }

    fn local_routing() -> crate::config::AdaptiveRoutingConfig {
        crate::config::AdaptiveRoutingConfig {
            strategy: "rules".into(),
            objective: None,
            experiment: None,
            prompt: None,
            jev: Some(JevRouterConfig {
                endpoint: Some("http://127.0.0.1:9/v1/systemone".into()),
                ..Default::default()
            }),
            candidates: Vec::new(),
        }
    }

    fn approx(value: &Value, expected: f64) {
        let got = value.as_f64().unwrap_or(f64::NAN);
        assert!(
            (got - expected).abs() < 1e-9,
            "got {got} expected {expected}"
        );
    }

    #[tokio::test]
    async fn happy_path_stores_five_scores_and_version() {
        let transport = ReplyTransport::new(scored_reply(0.5, 0.25, 0.75, 0.125, 1.0));
        let judge = reply_judge(transport.clone(), Duration::from_secs(2));
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let node_id = assistant_node(&graph, "all tests passed");
        judge_and_store(
            graph.clone(),
            node_id.clone(),
            &judge,
            "all tests passed",
            &[],
        )
        .await;

        let node = graph.get_node(&node_id).unwrap();
        let meta = &node.metadata["reply_judge"];
        assert_eq!(meta["version"], "v4");
        assert!(meta["latency_ms"].is_number());
        let scores = &meta["scores"];
        approx(&scores["move"], 0.5);
        approx(&scores["claims_completion"], 0.25);
        approx(&scores["cites_evidence"], 0.75);
        approx(&scores["presents_decision"], 0.125);
        approx(&scores["presents_as_options"], 1.0);

        let body = transport.seen().expect("request");
        assert_eq!(body["state"]["role"], "assistant");
        assert_eq!(body["state"]["message"], "all tests passed");
        assert!(body["state"].get("previous_message").is_none());
        assert_eq!(
            body["questions"]["move"]["instructions"],
            with_only_message(MOVE_BODY)
        );
        assert!(ONLY_MESSAGE.ends_with("Ignore `previous_message` entirely."));
    }

    #[tokio::test(start_paused = true)]
    async fn hanging_transport_does_not_fail_or_write() {
        let judge = reply_judge(Arc::new(HangingTransport), Duration::from_secs(8));
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let node_id = assistant_node(&graph, "still working");
        judge_and_store(graph.clone(), node_id.clone(), &judge, "still working", &[]).await;
        let node = graph.get_node(&node_id).unwrap();
        assert!(
            node.metadata.get("reply_judge").is_none(),
            "hanging judge must not write reply_judge"
        );
    }

    #[tokio::test]
    async fn out_of_range_score_is_stored_as_is() {
        let judge = reply_judge(
            ReplyTransport::new(scored_reply(1.5, -0.25, 2.5, 0.0, 3.0)),
            Duration::from_secs(2),
        );
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let node_id = assistant_node(&graph, "done");
        judge_and_store(graph.clone(), node_id.clone(), &judge, "done", &[]).await;
        let node = graph.get_node(&node_id).unwrap();
        let scores = &node.metadata["reply_judge"]["scores"];
        approx(&scores["move"], 1.5);
        approx(&scores["claims_completion"], -0.25);
        approx(&scores["cites_evidence"], 2.5);
        approx(&scores["presents_as_options"], 3.0);
    }

    #[tokio::test]
    async fn missing_answer_writes_nothing() {
        let reply = json!({"answers": {
            "move": {"type": "choice", "choice": "explain", "confidence": 0.4},
            "claims_completion": {"type": "choice", "choice": "yes", "confidence": 0.9}
        }});
        let judge = reply_judge(ReplyTransport::new(reply), Duration::from_secs(2));
        let graph = Arc::new(GraphStore::open_memory().unwrap());
        let node_id = assistant_node(&graph, "maybe");
        judge_and_store(graph.clone(), node_id.clone(), &judge, "maybe", &[]).await;
        let node = graph.get_node(&node_id).unwrap();
        assert!(node.metadata.get("reply_judge").is_none());
    }

    #[test]
    fn questions_copy_tagset_v4() {
        let q = ReplyJudge::build_questions();
        assert_eq!(q["move"]["type"], "choice");
        assert_eq!(q["move"]["instructions"], with_only_message(MOVE_BODY));
        assert_eq!(
            q["move"]["criteria"]["plan"],
            "Describes what it intends to do before doing it"
        );
        assert_eq!(
            q["move"]["criteria"]["report_progress"],
            "Reports work done so far without claiming the task is complete"
        );
        assert_eq!(
            q["move"]["criteria"]["claim_done"],
            "States that the task or a requested change is complete"
        );
        assert_eq!(
            q["move"]["criteria"]["ask_user"],
            "Asks the human a question or requests a decision"
        );
        assert_eq!(
            q["move"]["criteria"]["retry_same"],
            "Repeats an approach already tried without a new hypothesis"
        );
        assert_eq!(
            q["move"]["criteria"]["explain"],
            "Explains a concept or answers a question without changing anything"
        );
        assert_eq!(q["claims_completion"]["type"], "noul");
        assert_eq!(
            q["claims_completion"]["instructions"],
            with_only_message(CLAIMS_COMPLETION_BODY)
        );
        assert!(
            q["claims_completion"]["instructions"]
                .as_str()
                .unwrap()
                .starts_with("Does the assistant's text in the `message` field claim that work is finished, fixed, passing, or complete?")
        );
        assert_eq!(q["cites_evidence"]["type"], "noul");
        assert!(
            q["cites_evidence"]["instructions"]
                .as_str()
                .unwrap()
                .starts_with("Does the assistant's text in the `message` field show concrete evidence such as command output, test results, or file contents to support its claims?")
        );
        assert_eq!(
            q["presents_decision"]["instructions"],
            with_only_message(PRESENTS_DECISION_BODY)
        );
        assert_eq!(
            q["presents_decision"]["criteria"]["true"],
            "The `message` field explicitly requests the human's choice or approval: options offered, a recommendation to confirm, or a direct question whose answer decides the next step"
        );
        assert_eq!(
            q["presents_decision"]["criteria"]["false"],
            "Reporting, explanation, or suggestions with no request for the human's decision"
        );
        assert_eq!(
            q["presents_as_options"]["instructions"],
            with_only_message(PRESENTS_AS_OPTIONS_BODY)
        );
        assert_eq!(
            q["presents_as_options"]["criteria"]["true"],
            "A labelled list of alternative choices for the human appears in the `message` field"
        );
        assert_eq!(
            q["presents_as_options"]["criteria"]["false"],
            "No alternatives are offered to the human, or they appear only in prose"
        );
        for key in [
            "move",
            "claims_completion",
            "cites_evidence",
            "presents_decision",
            "presents_as_options",
        ] {
            let text = q[key]["instructions"].as_str().unwrap();
            assert!(
                text.ends_with(ONLY_MESSAGE),
                "{key} instructions must end with the only-message sentence"
            );
        }
    }

    #[test]
    fn previous_message_omitted_when_empty_and_trimmed_from_the_oldest_side() {
        let empty = ReplyJudge::build_state("hello", &[]);
        assert!(empty.get("previous_message").is_none());
        assert_eq!(empty["message"], "hello");

        let turns = vec![
            PrevTurn {
                role: "user".into(),
                text: "oldest-dropped".into(),
            },
            PrevTurn {
                role: "assistant".into(),
                text: "a".repeat(5_000),
            },
            PrevTurn {
                role: "tool".into(),
                text: "m".repeat(100),
            },
            PrevTurn {
                role: "user".into(),
                text: "newest".into(),
            },
        ];
        let state = ReplyJudge::build_state("reply", &turns);
        let prev = state["previous_message"].as_array().unwrap();
        assert_eq!(prev.len(), 3);
        assert_eq!(prev[2]["text"], "newest");
        assert_eq!(prev[2]["role"], "user");
        assert_eq!(prev[1]["text"], "m".repeat(100));
        assert_eq!(prev[1]["role"], "tool");
        let oldest = prev[0]["text"].as_str().unwrap();
        assert!(oldest.chars().all(|c| c == 'a'));
        assert!(oldest.chars().count() < 5_000);
        assert_eq!(prev[0]["role"], "assistant");
        let total: usize = prev
            .iter()
            .map(|turn| turn["text"].as_str().unwrap().chars().count())
            .sum();
        assert_eq!(total, MAX_PREV_CHARS);
        assert_eq!(state["message"], "reply");

        let huge = "y".repeat(MAX_MESSAGE_CHARS + 50);
        let capped = ReplyJudge::build_state(&huge, &[]);
        assert_eq!(
            capped["message"].as_str().unwrap().chars().count(),
            MAX_MESSAGE_CHARS
        );
    }

    #[test]
    fn build_reply_judge_without_key_is_none_and_timeout_follows_hitl() {
        assert!(build_reply_judge_with(&AgentConfig::default(), |_| None).is_none());

        let mut with_hitl = AgentConfig::default();
        with_hitl.hitl_judge = Some(HitlJudgeConfig {
            enabled: false,
            timeout_ms: 900,
            ..Default::default()
        });
        with_hitl.adaptive_routing = Some(local_routing());
        let judge = build_reply_judge_with(&with_hitl, |_| None).expect("local endpoint");
        assert_eq!(judge.timeout(), Duration::from_millis(900));

        let mut no_hitl = AgentConfig::default();
        no_hitl.adaptive_routing = Some(local_routing());
        let judge = build_reply_judge_with(&no_hitl, |_| None).expect("local endpoint");
        assert_eq!(judge.timeout(), Duration::from_secs(8));
    }
}
