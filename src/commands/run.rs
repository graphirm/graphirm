//! One-shot, non-interactive agent run. Used by the SWE-bench adapter.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use graphirm_llm::MockProvider;
use graphirm_llm::mock::MockResponse;
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use crate::error::GraphirmError;

use super::unknown_agent_keys;

pub const EXIT_FINISHED: i32 = 0;
pub const EXIT_BUDGET: i32 = 2;
pub const EXIT_PROVIDER: i32 = 3;
pub const EXIT_INTERNAL: i32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scoring {
    Default,
    Recency,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplyFormat {
    /// Product default used by `graphirm serve` with `config/default.toml`.
    Segments,
    /// Pi HTML index contract. Not a benchmark arm.
    Html,
    /// No segment JSON and no HTML contract.
    Md,
}

pub struct RunArgs {
    pub task: PathBuf,
    pub cwd: PathBuf,
    pub graph: PathBuf,
    pub config: PathBuf,
    pub model: Option<String>,
    pub reasoning_effort: Option<String>,
    pub scoring: Scoring,
    pub reply_format: ReplyFormat,
    pub max_turns: Option<u32>,
    pub max_tokens: Option<u64>,
    pub max_cost_usd: Option<f64>,
    pub prices: Option<PathBuf>,
    pub summary: Option<PathBuf>,
    pub allow_routing: bool,
    pub allow_handoff: bool,
    /// True keeps the old content-first pack. False is prefix-stable and deduped.
    pub packing_legacy: bool,
}

#[derive(Debug, Serialize)]
struct RunSummary {
    exit_reason: String,
    model: String,
    reasoning_effort: Option<String>,
    scoring: String,
    reply_format: String,
    turns: u32,
    tokens_in: u64,
    tokens_out: u64,
    cost_usd: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cost_usd_reported: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cost_note: Option<String>,
    wall_seconds: u64,
    config_hash: String,
    graphirm_sha: String,
    packing: String,
}

pub async fn execute(args: RunArgs) -> i32 {
    init_stderr_logging();
    let started = Instant::now();
    let prepared = match prepare(&args) {
        Ok(prepared) => prepared,
        Err(err) => {
            tracing::error!("{err}");
            eprintln!("{err}");
            write_summary(
                &args,
                &SummaryInput {
                    exit_reason: "internal",
                    model: args.model.clone().unwrap_or_default(),
                    reasoning_effort: args.reasoning_effort.clone(),
                    turns: 0,
                    tokens_in: 0,
                    tokens_out: 0,
                    cost_usd: None,
                    cost_usd_reported: None,
                    cost_note: None,
                    wall_seconds: started.elapsed().as_secs(),
                    config_hash: String::new(),
                    graphirm_sha: git_sha(),
                },
            );
            return EXIT_INTERNAL;
        }
    };

    let outcome = match drive(&args, &prepared).await {
        Ok(outcome) => outcome,
        Err(err) => {
            tracing::error!("{err}");
            eprintln!("{err}");
            write_summary(
                &args,
                &SummaryInput {
                    exit_reason: "internal",
                    model: prepared.model_id.clone(),
                    reasoning_effort: prepared.reasoning_effort.clone(),
                    turns: 0,
                    tokens_in: 0,
                    tokens_out: 0,
                    cost_usd: None,
                    cost_usd_reported: None,
                    cost_note: None,
                    wall_seconds: started.elapsed().as_secs(),
                    config_hash: prepared.config_hash.clone(),
                    graphirm_sha: prepared.git_sha.clone(),
                },
            );
            return EXIT_INTERNAL;
        }
    };

    write_summary(
        &args,
        &SummaryInput {
            exit_reason: outcome.exit_reason,
            model: prepared.model_id.clone(),
            reasoning_effort: prepared.reasoning_effort.clone(),
            turns: outcome.turns,
            tokens_in: outcome.tokens_in,
            tokens_out: outcome.tokens_out,
            cost_usd: outcome.cost_usd,
            cost_usd_reported: outcome.cost_usd_reported,
            cost_note: outcome.cost_note,
            wall_seconds: started.elapsed().as_secs(),
            config_hash: prepared.config_hash.clone(),
            graphirm_sha: prepared.git_sha.clone(),
        },
    );
    outcome.code
}

struct Prepared {
    config: graphirm_agent::AgentConfig,
    provider: Arc<dyn graphirm_llm::LlmProvider>,
    model_id: String,
    reasoning_effort: Option<String>,
    config_hash: String,
    git_sha: String,
    task_text: String,
}

struct Outcome {
    code: i32,
    exit_reason: &'static str,
    turns: u32,
    tokens_in: u64,
    tokens_out: u64,
    cost_usd: Option<f64>,
    cost_usd_reported: Option<f64>,
    cost_note: Option<String>,
}

struct SummaryInput {
    exit_reason: &'static str,
    model: String,
    reasoning_effort: Option<String>,
    turns: u32,
    tokens_in: u64,
    tokens_out: u64,
    cost_usd: Option<f64>,
    cost_usd_reported: Option<f64>,
    cost_note: Option<String>,
    wall_seconds: u64,
    config_hash: String,
    graphirm_sha: String,
}

fn prepare(args: &RunArgs) -> Result<Prepared, GraphirmError> {
    if !args.config.is_file() {
        return Err(GraphirmError::Config(format!(
            "config file not found: {}",
            args.config.display()
        )));
    }
    let config_bytes = std::fs::read(&args.config)
        .map_err(|e| GraphirmError::Config(format!("read {}: {e}", args.config.display())))?;
    let config_text = String::from_utf8(config_bytes.clone())
        .map_err(|e| GraphirmError::Config(format!("config is not utf-8: {e}")))?;
    let unknown = unknown_agent_keys(&config_text).map_err(GraphirmError::Config)?;
    if !unknown.is_empty() {
        return Err(GraphirmError::Config(format!(
            "unknown [agent] keys: {}",
            unknown.join(", ")
        )));
    }
    let mut config = graphirm_agent::AgentConfig::from_file(&args.config)
        .map_err(|e| GraphirmError::Config(format!("load {}: {e}", args.config.display())))?;

    if !args.allow_routing && (config.model_routing.is_some() || config.adaptive_routing.is_some())
    {
        return Err(GraphirmError::Config(
            "config enables routing; pass --allow-routing or remove [agent.routing] and [agent.adaptive_routing]"
                .into(),
        ));
    }
    if !args.allow_handoff {
        config.pi = None;
    }
    config.hitl_judge = None;
    config.tool_gate_enabled = false;
    config.workspaces_root = None;
    config.working_dir = args.cwd.clone();
    config.recency_only = matches!(args.scoring, Scoring::Recency);
    if let Some(turns) = args.max_turns {
        config.max_turns = turns;
    }
    if let Some(tokens) = args.max_tokens {
        config.max_session_tokens = Some(tokens);
    }
    if let Some(cost) = args.max_cost_usd {
        config.max_session_cost_micro_usd = Some(usd_to_micro(cost));
    }
    if let Some(effort) = &args.reasoning_effort {
        config.reasoning_effort = Some(effort.clone());
    }
    apply_reply_format(&mut config, args.reply_format);

    let model_spec = args.model.clone().unwrap_or_else(|| config.model.clone());
    let (provider, model_id) = provider_for(&model_spec)?;
    config.model = model_id.clone();

    let task_text = std::fs::read_to_string(&args.task)
        .map_err(|e| GraphirmError::Config(format!("read task {}: {e}", args.task.display())))?;
    let reasoning_effort = config.reasoning_effort.clone();

    Ok(Prepared {
        config,
        provider,
        model_id,
        reasoning_effort,
        config_hash: hex_sha256(&config_bytes),
        git_sha: git_sha(),
        task_text,
    })
}

fn provider_for(
    model_spec: &str,
) -> Result<(Arc<dyn graphirm_llm::LlmProvider>, String), GraphirmError> {
    if model_spec == "mock/loop" {
        let call = MockResponse::tool_call("call_ls", "ls", json!({}));
        return Ok((Arc::new(MockProvider::new(vec![call])), "mock/loop".into()));
    }
    if model_spec == "mock/patch" {
        let write = MockResponse::tool_call(
            "call_write",
            "write",
            json!({"path": "fixed.txt", "content": "ok\n"}),
        );
        let done = MockResponse::text("done");
        return Ok((
            Arc::new(MockProvider::new(vec![write, done])),
            "mock/patch".into(),
        ));
    }
    let (provider_name, model_name) = graphirm_llm::factory::parse_model_string(model_spec)
        .map_err(|e| GraphirmError::Config(e.to_string()))?;
    let api_key = super::api_key_for_provider(provider_name)?;
    let provider = graphirm_llm::factory::create_provider(provider_name, &api_key)
        .map_err(|e| GraphirmError::Config(e.to_string()))?;
    Ok((Arc::from(provider), model_name.to_string()))
}

async fn drive(args: &RunArgs, prepared: &Prepared) -> Result<Outcome, GraphirmError> {
    if let Some(parent) = args.graph.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| GraphirmError::Config(format!("create {}: {e}", parent.display())))?;
    }
    let graph = Arc::new(
        graphirm_graph::GraphStore::open(args.graph.to_str().unwrap_or("graph.db"))
            .map_err(|e| GraphirmError::Config(e.to_string()))?,
    );
    let mut config = prepared.config.clone();
    config.reasoning_effort = prepared
        .config
        .reasoning_effort
        .clone()
        .or(args.reasoning_effort.clone());
    let mut session = graphirm_agent::Session::open(graph.clone(), config.clone())
        .map_err(|e| GraphirmError::Config(e.to_string()))?;
    session.agent_config.packing_legacy = args.packing_legacy;
    if let Some(path) = &args.prices {
        let prices = load_frozen_prices(path)?;
        session.set_frozen_prices(prices.0, prices.1, prices.2);
    }
    write_run_metadata(&session, args, prepared, &config)?;
    session
        .add_task_message(&prepared.task_text)
        .await
        .map_err(|e| GraphirmError::Config(e.to_string()))?;

    let tools = register_handoff_if_needed(args.allow_handoff, &config).await;
    let events = graphirm_agent::EventBus::new();
    let cancel = CancellationToken::new();
    let loop_result = graphirm_agent::run_agent_loop(
        &session,
        prepared.provider.clone(),
        &tools,
        &events,
        &cancel,
    )
    .await;

    let (code, exit_reason) = match &loop_result {
        Ok(()) => (EXIT_FINISHED, "finished"),
        Err(graphirm_agent::AgentError::RecursionLimit(_)) => (EXIT_BUDGET, "max_turns"),
        Err(graphirm_agent::AgentError::SessionTokenCapExceeded { .. }) => {
            (EXIT_BUDGET, "max_tokens")
        }
        Err(graphirm_agent::AgentError::SessionCostCapExceeded { .. }) => (EXIT_BUDGET, "max_cost"),
        Err(graphirm_agent::AgentError::Llm(_)) => (EXIT_PROVIDER, "provider_error"),
        Err(_) => (EXIT_INTERNAL, "internal"),
    };

    let cost_note = if args.max_cost_usd.is_some()
        && !session.frozen_prices_set()
        && session.llm_cost_reports() == 0
    {
        Some(
            "provider did not return cost; --max-cost-usd was not enforced and --max-tokens still applies"
                .to_string(),
        )
    } else {
        None
    };
    let cost_usd = if session.llm_cost_reports() == 0 {
        None
    } else {
        Some(session.llm_cost_micro_usd() as f64 / 1_000_000.0)
    };
    let cost_usd_reported = if session.llm_reported_cost_reports() == 0 {
        None
    } else {
        Some(session.llm_reported_cost_micro_usd() as f64 / 1_000_000.0)
    };

    if let Err(err) = &loop_result
        && !matches!(
            err,
            graphirm_agent::AgentError::RecursionLimit(_)
                | graphirm_agent::AgentError::SessionTokenCapExceeded { .. }
                | graphirm_agent::AgentError::SessionCostCapExceeded { .. }
        )
    {
        tracing::error!(error = %err, "agent loop stopped");
        if matches!(err, graphirm_agent::AgentError::Llm(_)) {
            eprintln!("{err}");
        } else {
            eprintln!("{err}");
        }
    }

    Ok(Outcome {
        code,
        exit_reason,
        turns: session.current_turn(),
        tokens_in: session.llm_input_tokens(),
        tokens_out: session.llm_output_tokens(),
        cost_usd,
        cost_usd_reported,
        cost_note,
    })
}

fn load_frozen_prices(path: &std::path::Path) -> Result<(f64, f64, f64), GraphirmError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| GraphirmError::Config(format!("read prices {}: {e}", path.display())))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| GraphirmError::Config(format!("parse prices {}: {e}", path.display())))?;
    let input = value
        .get("input_cost_per_token")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| {
            GraphirmError::Config("prices.lock.json needs input_cost_per_token".into())
        })?;
    let cache_read = value
        .get("input_cache_read_per_token")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| {
            GraphirmError::Config("prices.lock.json needs input_cache_read_per_token".into())
        })?;
    let output = value
        .get("output_cost_per_token")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| {
            GraphirmError::Config("prices.lock.json needs output_cost_per_token".into())
        })?;
    if input <= 0.0 || cache_read <= 0.0 || output <= 0.0 {
        return Err(GraphirmError::Config(
            "frozen prices must be greater than zero".into(),
        ));
    }
    Ok((input, cache_read, output))
}

fn write_run_metadata(
    session: &graphirm_agent::Session,
    args: &RunArgs,
    prepared: &Prepared,
    config: &graphirm_agent::AgentConfig,
) -> Result<(), GraphirmError> {
    let body = json!({
        "model": prepared.model_id,
        "reasoning_effort": config.reasoning_effort,
        "scoring": scoring_name(args.scoring),
        "reply_format": reply_name(args.reply_format),
        "max_turns": config.max_turns,
        "max_tokens": config.max_session_tokens,
        "max_cost_usd": args.max_cost_usd,
        "config_hash": prepared.config_hash,
        "graphirm_sha": prepared.git_sha,
        "packing": packing_name(args.packing_legacy),
        "started_unix": unix_now(),
    });
    let mut node = graphirm_graph::nodes::GraphNode::new(graphirm_graph::nodes::NodeType::Content(
        graphirm_graph::nodes::ContentData {
            content_type: "run_metadata".into(),
            path: None,
            body: body.to_string(),
            language: Some("json".into()),
        },
    ));
    node.metadata = body;
    node.set_label("run_metadata");
    session
        .graph
        .add_node(node)
        .map_err(|e| GraphirmError::Config(e.to_string()))?;
    Ok(())
}

fn write_summary(args: &RunArgs, input: &SummaryInput) {
    let Some(path) = &args.summary else {
        return;
    };
    let summary = RunSummary {
        exit_reason: input.exit_reason.to_string(),
        model: input.model.clone(),
        reasoning_effort: input.reasoning_effort.clone(),
        scoring: scoring_name(args.scoring).to_string(),
        reply_format: reply_name(args.reply_format).to_string(),
        turns: input.turns,
        tokens_in: input.tokens_in,
        tokens_out: input.tokens_out,
        cost_usd: input.cost_usd,
        cost_usd_reported: input.cost_usd_reported,
        cost_note: input.cost_note.clone(),
        wall_seconds: input.wall_seconds,
        config_hash: input.config_hash.clone(),
        graphirm_sha: input.graphirm_sha.clone(),
        packing: packing_name(args.packing_legacy).to_string(),
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match serde_json::to_string_pretty(&summary) {
        Ok(text) => {
            if let Err(err) = std::fs::write(path, text) {
                eprintln!("failed to write summary {}: {err}", path.display());
            }
        }
        Err(err) => eprintln!("failed to encode summary: {err}"),
    }
}

/// `segments` matches `graphirm serve` loading `config/default.toml`.
/// `md` turns segmentation off. `html` appends the Pi contract and turns segmentation off.
pub fn apply_reply_format(config: &mut graphirm_agent::AgentConfig, format: ReplyFormat) {
    match format {
        ReplyFormat::Segments => {
            let mut seg = config
                .segments
                .clone()
                .unwrap_or_else(product_segment_config);
            seg.enabled = true;
            seg.structured_output = true;
            config.segments = Some(seg);
        }
        ReplyFormat::Html => {
            let contract = graphirm_agent::pi_delegate::html_piece_index_contract();
            if !config.system_prompt.contains(contract.trim()) {
                config.system_prompt.push_str(contract);
            }
            disable_segments(config);
        }
        ReplyFormat::Md => disable_segments(config),
    }
}

fn disable_segments(config: &mut graphirm_agent::AgentConfig) {
    if let Some(seg) = config.segments.as_mut() {
        seg.enabled = false;
        seg.structured_output = false;
    }
}

/// Labels from the committed `[agent.segments]` block in `config/default.toml`.
fn product_segment_config() -> graphirm_agent::config::SegmentConfig {
    graphirm_agent::config::SegmentConfig {
        enabled: true,
        labels: vec![
            "observation".into(),
            "reasoning".into(),
            "code".into(),
            "plan".into(),
            "answer".into(),
            "caveat".into(),
            "action".into(),
        ],
        structured_output: true,
        gliner2_fallback: true,
        min_confidence: 0.5,
        label_descriptions: None,
        label_min_confidence: None,
    }
}

fn init_stderr_logging() {
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .try_init();
}

fn packing_name(legacy: bool) -> &'static str {
    if legacy { "legacy" } else { "stable" }
}

fn scoring_name(scoring: Scoring) -> &'static str {
    match scoring {
        Scoring::Default => "default",
        Scoring::Recency => "recency",
    }
}

fn reply_name(format: ReplyFormat) -> &'static str {
    match format {
        ReplyFormat::Segments => "segments",
        ReplyFormat::Html => "html",
        ReplyFormat::Md => "md",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_reply_format_matches_serve_config() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config/default.toml");
        let served = graphirm_agent::AgentConfig::from_file(&path).expect("default.toml");
        let served_segments = served
            .segments
            .as_ref()
            .expect("serve config enables segments");
        assert!(served_segments.enabled);
        assert!(served_segments.structured_output);

        let mut run_config = served.clone();
        apply_reply_format(&mut run_config, ReplyFormat::Segments);
        let run_segments = run_config.segments.as_ref().unwrap();
        assert!(run_segments.enabled);
        assert!(run_segments.structured_output);
        assert_eq!(run_segments.labels, served_segments.labels);
        assert!(!run_config.system_prompt.contains("nav id="));
        assert_eq!(reply_name(ReplyFormat::Segments), "segments");
    }

    #[test]
    fn md_and_html_drop_the_segment_contract() {
        let mut config = graphirm_agent::AgentConfig::default();
        config.segments = Some(product_segment_config());
        config.system_prompt = "base".into();
        apply_reply_format(&mut config, ReplyFormat::Md);
        assert!(!config.segments.as_ref().unwrap().structured_output);
        assert!(!config.segments.as_ref().unwrap().enabled);
        assert_eq!(config.system_prompt, "base");

        let mut html = graphirm_agent::AgentConfig::default();
        html.segments = Some(product_segment_config());
        html.system_prompt = "base".into();
        apply_reply_format(&mut html, ReplyFormat::Html);
        assert!(!html.segments.as_ref().unwrap().enabled);
        assert!(html.system_prompt.contains("nav id=\"index\""));
    }
}

fn hex_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn git_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn usd_to_micro(cost: f64) -> u64 {
    if !cost.is_finite() || cost <= 0.0 {
        return 0;
    }
    (cost * 1_000_000.0).round() as u64
}

async fn register_handoff_if_needed(
    allow: bool,
    config: &graphirm_agent::AgentConfig,
) -> graphirm_tools::ToolRegistry {
    let mut registry = super::build_tool_registry();
    if allow {
        graphirm_agent::register_pi_delegate(&mut registry, config).await;
    }
    registry
}
