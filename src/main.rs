mod error;

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use error::GraphirmError;
use graphirm_tools::registry::ToolRegistry;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[derive(Parser)]
#[command(name = "graphirm")]
#[command(version, about = "Graph-native coding agent")]
struct Cli {
    /// Path to the graph database (default: ~/.local/share/graphirm/graph.db)
    #[arg(long, global = true)]
    db: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start an interactive chat session
    Chat {
        /// Resume an existing session by ID
        #[arg(short, long)]
        session: Option<String>,

        /// Model in "provider/model" format.
        /// Cloud examples:  openrouter/qwen/qwen3-coder-next (default)
        ///                  anthropic/claude-sonnet-4-20250514
        ///                  deepseek/deepseek-chat
        ///                  openai/gpt-4o
        /// Local (Ollama):  ollama/qwen2.5:72b
        ///                  ollama/qwen3:70b
        ///                  ollama/llama3.2
        #[arg(short, long, default_value = "openrouter/qwen/qwen3-coder-next")]
        model: String,
    },

    /// Inspect the graph database
    Graph {
        #[command(subcommand)]
        action: GraphAction,
    },

    /// Manage local models (e.g. GLiNER2 for offline knowledge extraction)
    Model {
        #[command(subcommand)]
        action: ModelAction,
    },

    /// Start the HTTP API server
    Serve {
        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to listen on
        #[arg(short, long, default_value = "5555")]
        port: u16,
    },

    /// Export assistant turns to JSONL for structured-response discovery (GLiNER2).
    ///
    /// Reads the graph at --db and writes one JSON object per assistant turn
    /// (session_id, turn_index, role, text). Default output is stdout.
    ExportCorpus {
        /// Output file (default: stdout)
        #[arg(short, long)]
        out: Option<PathBuf>,
        /// Maximum number of assistant turns to export (for validation samples, e.g. 100)
        #[arg(long)]
        limit: Option<u64>,
    },

    /// Run GLiNER2 over a corpus JSONL with candidate labels and output a statistics report.
    #[cfg(feature = "local-extraction")]
    LabelExplore {
        /// Path to corpus JSONL (one CorpusTurn per line)
        #[arg(short, long)]
        corpus: PathBuf,
        /// Comma-separated label names (e.g. observation,reasoning,code,answer)
        #[arg(short, long)]
        labels: String,
        /// Minimum confidence threshold for GLiNER2 (default 0.3)
        #[arg(long, default_value = "0.3")]
        min_confidence: f64,
        /// Output path for JSON report (default: stdout)
        #[arg(short, long)]
        out: Option<PathBuf>,
    },

    /// Analyse a label-exploration report and suggest segment schema (Phase 3).
    #[cfg(feature = "local-extraction")]
    SchemaSuggest {
        /// Path to report JSON from `graphirm label-explore`
        #[arg(short, long)]
        report: PathBuf,
        /// Output path for recommendation JSON (default: stdout)
        #[arg(short, long)]
        out: Option<PathBuf>,
    },

    /// Run GLiNER2 on a corpus and output per-turn spans for Phase 4 validation.
    #[cfg(feature = "local-extraction")]
    PredictSpans {
        /// Path to corpus JSONL (one CorpusTurn per line)
        #[arg(short, long)]
        corpus: PathBuf,
        /// Comma-separated label names (e.g. observation,reasoning,code,answer)
        #[arg(short, long)]
        labels: String,
        /// Minimum confidence threshold (default 0.3)
        #[arg(long, default_value = "0.3")]
        min_confidence: f64,
        /// Output path for spans JSONL (default: stdout)
        #[arg(short, long)]
        out: Option<PathBuf>,
    },

    /// Compare human annotations to GLiNER2 spans and report agreement (Phase 4).
    #[cfg(feature = "local-extraction")]
    ValidateAgreement {
        /// Path to human annotations JSONL (session_id, turn_index, segments: [{ type, start, end }])
        #[arg(long)]
        human: PathBuf,
        /// Path to GLiNER2 spans JSONL from `graphirm predict-spans`
        #[arg(long)]
        gliner: PathBuf,
        /// Pass threshold as fraction 0–100 (default 75)
        #[arg(long, default_value = "75")]
        threshold: f64,
        /// Output path for report JSON (default: stdout)
        #[arg(short, long)]
        out: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum GraphAction {
    /// Show node and edge counts by type
    Stats,
    /// List recent nodes (newest first)
    List {
        /// Max nodes to show
        #[arg(short, long, default_value = "20")]
        limit: usize,
        /// Filter by node type (interaction, agent, content, task, knowledge)
        #[arg(short, long)]
        r#type: Option<String>,
    },
}

#[derive(Subcommand)]
enum ModelAction {
    /// Download GLiNER2 ONNX model files from HuggingFace Hub (~1.95 GB).
    ///
    /// Files are cached in ~/.cache/huggingface/hub/ (same as Python hf_hub).
    /// After downloading, set GLINER2_MODEL_DIR to the printed path and restart
    /// `graphirm serve` to use the local extraction backend.
    ///
    /// Requires the binary to be built with: --features local-extraction
    Download,
}

#[tokio::main]
async fn main() -> Result<(), GraphirmError> {
    dotenvy::dotenv().ok();
    let cli = Cli::parse();
    let db_path = resolve_db_path(cli.db)?;

    match cli.command {
        Commands::Chat { session: _, model } => {
            // TUI runs in raw/alternate-screen mode — logs must go to a file
            // so they don't corrupt the rendered UI.
            let _guard = init_file_logging();
            run_chat(model, &db_path).await?;
        }
        Commands::Graph { action } => {
            // Graph inspection commands are synchronous; no TUI, logs to stderr.
            tracing_subscriber::fmt()
                .with_writer(std::io::stderr)
                .with_env_filter("error")
                .init();
            run_graph_command(action, &db_path)?;
        }
        Commands::Model { action } => {
            tracing_subscriber::fmt()
                .with_writer(std::io::stderr)
                .with_env_filter("info")
                .init();
            run_model_command(action).await?;
        }
        Commands::ExportCorpus { out, limit } => {
            tracing_subscriber::fmt()
                .with_writer(std::io::stderr)
                .with_env_filter("warn")
                .init();
            run_export_corpus(&db_path, out, limit)?;
        }
        #[cfg(feature = "local-extraction")]
        Commands::LabelExplore {
            corpus,
            labels,
            min_confidence,
            out,
        } => {
            tracing_subscriber::fmt()
                .with_writer(std::io::stderr)
                .with_env_filter("warn")
                .init();
            run_label_explore(corpus, labels, min_confidence, out).await?;
        }
        #[cfg(feature = "local-extraction")]
        Commands::SchemaSuggest { report, out } => {
            tracing_subscriber::fmt()
                .with_writer(std::io::stderr)
                .with_env_filter("warn")
                .init();
            run_schema_suggest(report, out)?;
        }
        #[cfg(feature = "local-extraction")]
        Commands::PredictSpans {
            corpus,
            labels,
            min_confidence,
            out,
        } => {
            tracing_subscriber::fmt()
                .with_writer(std::io::stderr)
                .with_env_filter("warn")
                .init();
            run_predict_spans(corpus, labels, min_confidence, out).await?;
        }
        #[cfg(feature = "local-extraction")]
        Commands::ValidateAgreement {
            human,
            gliner,
            threshold,
            out,
        } => {
            tracing_subscriber::fmt()
                .with_writer(std::io::stderr)
                .with_env_filter("warn")
                .init();
            run_validate_agreement(human, gliner, threshold, out)?;
        }
        Commands::Serve { host, port } => {
            tracing_subscriber::fmt()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::from_default_env()
                        .add_directive(tracing::Level::INFO.into()),
                )
                .init();

            let graph = Arc::new(graphirm_graph::GraphStore::open(
                db_path.to_str().unwrap_or("graph.db"),
            )?);
            let tools = Arc::new(build_tool_registry());

            let agent_config = graphirm_agent::AgentConfig::default();

            // LLM provider requires a model spec; reads GRAPHIRM_MODEL env var
            // (set in .env). Defaults to Qwen Coder Next (OpenRouter) if not configured.
            let model_spec = std::env::var("GRAPHIRM_MODEL")
                .unwrap_or_else(|_| "openrouter/qwen/qwen3-coder-next".to_string());
            let (provider_name, model_name) =
                graphirm_llm::factory::parse_model_string(&model_spec)
                    .map_err(|e| GraphirmError::Config(e.to_string()))?;
            let api_key = api_key_for_provider(provider_name)?;
            let llm: Arc<dyn graphirm_llm::LlmProvider> = Arc::from(
                graphirm_llm::factory::create_provider(provider_name, &api_key)
                    .map_err(|e| GraphirmError::Config(e.to_string()))?,
            );

            // Use the model name from GRAPHIRM_MODEL so sessions use the correct
            // model for the configured provider (not the AgentConfig default which
            // is hardcoded to a Claude model name).
            //
            // Knowledge extraction backend selection:
            // - If GLINER2_MODEL_DIR is set and the binary was built with
            //   --features local-extraction, use the fast local ONNX backend
            //   (150-200ms per call, no API cost, no timeouts).
            // - Otherwise fall back to the LLM backend (25-35s per call).
            let extraction_backend = resolve_extraction_backend();
            let agent_config = graphirm_agent::AgentConfig {
                model: model_name.to_string(),
                extraction: Some(graphirm_agent::knowledge::extraction::ExtractionConfig {
                    enabled: true,
                    model: model_name.to_string(),
                    backend: extraction_backend,
                    ..Default::default()
                }),
                ..agent_config
            };

            // Optional embedding provider for cross-session memory.
            // Set EMBEDDING_BACKEND="fastembed/bge-small-en-v1.5" (recommended — free, 12ms, 0.334 discrimination)
            // or "mistral/codestral-embed" (API, 400ms, 0.305 discrimination, requires MISTRAL_API_KEY)
            let embedding_backend = std::env::var("EMBEDDING_BACKEND").ok();
            let memory_retriever: Option<
                std::sync::Arc<graphirm_agent::knowledge::memory::MemoryRetriever>,
            > = if let Some(spec) = embedding_backend {
                let mistral_key = std::env::var("MISTRAL_API_KEY").ok();
                match graphirm_llm::factory::create_embedding_provider(
                    &spec,
                    mistral_key.as_deref(),
                ) {
                    Ok((provider, dim)) => {
                        tracing::info!(backend = %spec, dim, "Embedding provider initialised");
                        let retriever = std::sync::Arc::new(
                            graphirm_agent::knowledge::memory::MemoryRetriever::from_store(
                                graph.clone(),
                                std::sync::Arc::from(provider),
                                dim,
                            ),
                        );
                        match retriever.hydrate_from_graph().await {
                            Ok(n) => tracing::info!(count = n, "Restored embeddings from graph store"),
                            Err(e) => tracing::warn!(error = %e, "HNSW hydration failed (non-fatal); starting fresh"),
                        }
                        Some(retriever)
                    }
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "Embedding provider failed to init; memory disabled"
                        );
                        None
                    }
                }
            } else {
                tracing::info!("EMBEDDING_BACKEND not set; cross-session memory disabled");
                None
            };

            let server_config = graphirm_server::ServerConfig { host, port };
            graphirm_server::start_server(graph, llm, tools, agent_config, server_config, memory_retriever)
                .await
                .map_err(|e| GraphirmError::Config(e.to_string()))?;
        }
    }

    Ok(())
}

/// Resolve the graph DB path, creating parent directories as needed.
fn resolve_db_path(override_path: Option<PathBuf>) -> Result<PathBuf, GraphirmError> {
    let path = override_path.unwrap_or_else(|| {
        dirs_next::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("graphirm")
            .join("graph.db")
    });
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            GraphirmError::Config(format!(
                "Cannot create DB directory {}: {e}",
                parent.display()
            ))
        })?;
    }
    Ok(path)
}

fn run_graph_command(action: GraphAction, db_path: &PathBuf) -> Result<(), GraphirmError> {
    let graph = graphirm_graph::GraphStore::open(db_path.to_str().unwrap_or("graph.db"))?;

    match action {
        GraphAction::Stats => {
            let nodes = graph.node_count_db()?;
            let edges = graph.edge_count_db()?;
            let by_type = graph.node_counts_by_type()?;

            println!("Graph: {}", db_path.display());
            println!("  Nodes : {nodes}");
            println!("  Edges : {edges}");
            if !by_type.is_empty() {
                println!("  By type:");
                for (t, c) in by_type {
                    println!("    {t:<15} {c}");
                }
            }
        }
        GraphAction::List { limit, r#type } => {
            let nodes = graph.list_recent_nodes(limit)?;
            let nodes: Vec<_> = if let Some(ref filter) = r#type {
                nodes
                    .into_iter()
                    .filter(|n| n.node_type.type_name() == filter.as_str())
                    .collect()
            } else {
                nodes
            };

            if nodes.is_empty() {
                println!("No nodes found.");
                return Ok(());
            }

            println!("{:<38}  {:<12}  {}", "ID", "TYPE", "LABEL");
            println!("{}", "-".repeat(90));
            for node in nodes {
                let label = node_display_label(&node);
                println!(
                    "{:<38}  {:<12}  {}",
                    &node.id.to_string()[..36.min(node.id.to_string().len())],
                    node.node_type.type_name(),
                    label
                );
            }
        }
    }
    Ok(())
}

fn run_export_corpus(
    db_path: &PathBuf,
    out: Option<PathBuf>,
    limit: Option<u64>,
) -> Result<(), GraphirmError> {
    let graph = graphirm_graph::GraphStore::open(db_path.to_str().unwrap_or("graph.db"))?;
    let count = if let Some(path) = out {
        let mut f = std::fs::File::create(path)?;
        graphirm_graph::export_corpus_to_jsonl(&graph, &mut f, true, limit)?
    } else {
        let mut stdout = std::io::stdout();
        graphirm_graph::export_corpus_to_jsonl(&graph, &mut stdout, true, limit)?
    };
    eprintln!("Exported {} assistant turns.", count);
    Ok(())
}

#[cfg(feature = "local-extraction")]
async fn run_label_explore(
    corpus_path: PathBuf,
    labels_str: String,
    min_confidence: f64,
    out: Option<PathBuf>,
) -> Result<(), GraphirmError> {
    use std::io::BufReader;

    let model_dir = std::env::var("GLINER2_MODEL_DIR").map_err(|_| {
        GraphirmError::Config(
            "GLINER2_MODEL_DIR not set. Run `graphirm model download` and set the env var.".into(),
        )
    })?;
    let model_dir = std::path::Path::new(&model_dir);

    let file = std::fs::File::open(&corpus_path)
        .map_err(|e| GraphirmError::Config(format!("open corpus {}: {}", corpus_path.display(), e)))?;
    let turns = graphirm_agent::knowledge::label_explore::read_corpus_jsonl(BufReader::new(file))?;
    let total = turns.len();
    if total == 0 {
        eprintln!("Corpus is empty.");
        return Ok(());
    }

    let labels: Vec<String> = labels_str
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if labels.is_empty() {
        return Err(GraphirmError::Config("At least one --labels value required".into()));
    }

    let extractor = graphirm_agent::knowledge::local_extraction::OnnxExtractor::new(model_dir)
        .map_err(|e| GraphirmError::Config(format!("load GLiNER2 model: {}", e)))?;

    eprintln!("Running GLiNER2 on {} turns with {} labels...", total, labels.len());
    let report = graphirm_agent::knowledge::label_explore::run_label_exploration(
        &extractor,
        &turns,
        &labels,
        min_confidence,
    )
    .await?;

    let json = serde_json::to_string_pretty(&report).map_err(|e| GraphirmError::Config(e.to_string()))?;
    if let Some(path) = out {
        std::fs::write(&path, json).map_err(|e| GraphirmError::Io(e))?;
        eprintln!("Wrote report to {}", path.display());
    } else {
        println!("{}", json);
    }
    eprintln!(
        "Coverage: {:.1}% ({} / {} chars in {} turns)",
        report.corpus_stats.coverage_pct,
        report.corpus_stats.covered_chars,
        report.corpus_stats.total_chars,
        report.corpus_stats.turns_with_any_label
    );
    Ok(())
}

#[cfg(feature = "local-extraction")]
fn run_schema_suggest(report_path: PathBuf, out: Option<PathBuf>) -> Result<(), GraphirmError> {
    let json = std::fs::read_to_string(&report_path)
        .map_err(|e| GraphirmError::Config(format!("read report {}: {}", report_path.display(), e)))?;
    let report: graphirm_agent::knowledge::label_explore::LabelExplorationReport =
        serde_json::from_str(&json).map_err(|e| GraphirmError::Config(format!("parse report JSON: {}", e)))?;
    let rec = graphirm_agent::knowledge::schema_suggest::analyse_report(&report);
    let out_json = serde_json::to_string_pretty(&rec).map_err(|e| GraphirmError::Config(e.to_string()))?;
    if let Some(path) = out {
        std::fs::write(&path, out_json).map_err(|e| GraphirmError::Io(e))?;
        eprintln!("Wrote schema recommendation to {}", path.display());
    } else {
        println!("{}", out_json);
    }
    eprintln!(
        "Recommended segment types ({}): {}",
        rec.recommended_segment_types.len(),
        rec.recommended_segment_types.join(", ")
    );
    Ok(())
}

#[cfg(feature = "local-extraction")]
async fn run_predict_spans(
    corpus_path: PathBuf,
    labels_str: String,
    min_confidence: f64,
    out: Option<PathBuf>,
) -> Result<(), GraphirmError> {
    use std::io::BufReader;

    let model_dir = std::env::var("GLINER2_MODEL_DIR").map_err(|_| {
        GraphirmError::Config(
            "GLINER2_MODEL_DIR not set. Run `graphirm model download` and set the env var.".into(),
        )
    })?;
    let model_dir = std::path::Path::new(&model_dir);

    let file = std::fs::File::open(&corpus_path)
        .map_err(|e| GraphirmError::Config(format!("open corpus {}: {}", corpus_path.display(), e)))?;
    let turns = graphirm_agent::knowledge::label_explore::read_corpus_jsonl(BufReader::new(file))?;
    if turns.is_empty() {
        eprintln!("Corpus is empty.");
        return Ok(());
    }

    let labels: Vec<String> = labels_str
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if labels.is_empty() {
        return Err(GraphirmError::Config("At least one --labels value required".into()));
    }

    let extractor = graphirm_agent::knowledge::local_extraction::OnnxExtractor::new(model_dir)
        .map_err(|e| GraphirmError::Config(format!("load GLiNER2 model: {}", e)))?;

    let rows = graphirm_agent::knowledge::predict_spans::run_predict_spans(
        &extractor,
        &turns,
        &labels,
        min_confidence,
    )
    .await?;

    let mut writer: Box<dyn std::io::Write> = if let Some(path) = &out {
        Box::new(
            std::fs::File::create(path)
                .map_err(|e| GraphirmError::Io(e))?,
        )
    } else {
        Box::new(std::io::stdout())
    };
    for row in &rows {
        let line = serde_json::to_string(row).map_err(|e| GraphirmError::Config(e.to_string()))?;
        writeln!(writer, "{line}").map_err(GraphirmError::Io)?;
    }
    if out.is_some() {
        eprintln!("Wrote {} turn spans to output.", rows.len());
    }
    Ok(())
}

#[cfg(feature = "local-extraction")]
fn run_validate_agreement(
    human_path: PathBuf,
    gliner_path: PathBuf,
    threshold: f64,
    out: Option<PathBuf>,
) -> Result<(), GraphirmError> {
    use std::io::BufReader;

    let human_file = std::fs::File::open(&human_path)
        .map_err(|e| GraphirmError::Config(format!("open human annotations {}: {}", human_path.display(), e)))?;
    let human = graphirm_agent::knowledge::validate_agreement::read_annotations_jsonl(BufReader::new(human_file))?;

    let gliner_file = std::fs::File::open(&gliner_path)
        .map_err(|e| GraphirmError::Config(format!("open gliner spans {}: {}", gliner_path.display(), e)))?;
    let gliner = graphirm_agent::knowledge::predict_spans::read_spans_jsonl(BufReader::new(gliner_file))?;

    const OVERLAP_RATIO_MIN: f64 = 0.5;
    let report = graphirm_agent::knowledge::validate_agreement::validate_agreement(
        &human,
        &gliner,
        threshold,
        OVERLAP_RATIO_MIN,
    );

    let out_json = serde_json::to_string_pretty(&report).map_err(|e| GraphirmError::Config(e.to_string()))?;
    if let Some(path) = out {
        std::fs::write(&path, out_json).map_err(GraphirmError::Io)?;
        eprintln!("Wrote agreement report to {}", path.display());
    } else {
        println!("{}", out_json);
    }
    eprintln!(
        "Agreement: {:.1}% ({} / {} segments) — {}",
        report.agreement_pct,
        report.matched_segments,
        report.total_human_segments,
        if report.pass { "PASS" } else { "FAIL" }
    );
    Ok(())
}

fn node_display_label(node: &graphirm_graph::nodes::GraphNode) -> String {
    use graphirm_graph::nodes::NodeType;
    if let Some(label) = node.label() {
        return label.to_string();
    }
    match &node.node_type {
        NodeType::Interaction(d) => {
            let preview: String = d.content.chars().take(60).collect();
            let ellipsis = if d.content.len() > 60 { "…" } else { "" };
            format!("[{}] {}{}", d.role, preview, ellipsis)
        }
        NodeType::Agent(d) => format!("[agent] {} ({})", d.name, d.status),
        NodeType::Content(d) => {
            let name = d.path.as_deref().unwrap_or(&d.content_type);
            format!("[content] {}", name)
        }
        NodeType::Task(d) => format!("[task] {} — {}", d.title, d.status),
        NodeType::Knowledge(d) => format!("[{}] {}", d.entity_type, d.entity),
    }
}

#[cfg(test)]
mod tests {
    use super::node_display_label;
    use graphirm_graph::nodes::{GraphNode, InteractionData, NodeType};

    #[test]
    fn node_display_label_prefers_metadata_label() {
        let mut node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".to_string(),
            content: "Fallback preview".to_string(),
            token_count: None,
        }));
        node.set_label("interaction_1_2_1");

        assert_eq!(node_display_label(&node), "interaction_1_2_1");
    }
}

/// Initialise a rolling daily log file at `~/.local/share/graphirm/graphirm.log`.
/// Returns the non-blocking guard — **keep it alive** for the program's lifetime
/// or buffered log lines will be dropped on exit.
fn init_file_logging() -> tracing_appender::non_blocking::WorkerGuard {
    let log_dir = dirs_next::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("graphirm");

    std::fs::create_dir_all(&log_dir).unwrap_or_default();

    let file_appender = tracing_appender::rolling::daily(&log_dir, "graphirm.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    tracing_subscriber::fmt()
        .with_writer(non_blocking)
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    guard
}

/// Select the knowledge extraction backend.
///
/// If `GLINER2_MODEL_DIR` is set (or a cached model is found at the standard
/// HuggingFace path) **and** the binary was compiled with the
/// `local-extraction` feature, use the fast ONNX backend.
/// Otherwise fall back to the LLM backend.
fn resolve_extraction_backend(
) -> graphirm_agent::knowledge::extraction::ExtractionBackend {
    use graphirm_agent::knowledge::extraction::ExtractionBackend;

    // Explicit override always wins.
    if let Ok(dir) = std::env::var("GLINER2_MODEL_DIR") {
        let path = std::path::PathBuf::from(&dir);
        if path.join("gliner2_config.json").exists() {
            tracing::info!(model_dir = %dir, "Using Local ONNX extraction backend (GLINER2_MODEL_DIR)");
            return ExtractionBackend::Local { model_dir: dir };
        }
        tracing::warn!(
            model_dir = %dir,
            "GLINER2_MODEL_DIR is set but gliner2_config.json not found; falling back to LLM"
        );
    }

    // Auto-detect standard HuggingFace cache location.
    let hf_cache = dirs_next::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("~/.cache"))
        .join("huggingface")
        .join("hub")
        .join("models--lmo3--gliner2-large-v1-onnx")
        .join("snapshots");

    if let Ok(mut entries) = std::fs::read_dir(&hf_cache) {
        // Take the most recent snapshot (lexicographically last hash dir).
        if let Some(Ok(entry)) = entries.next() {
            let snapshot_dir = entry.path();
            if snapshot_dir.join("gliner2_config.json").exists() {
                let dir_str = snapshot_dir.to_string_lossy().to_string();
                tracing::info!(model_dir = %dir_str, "Auto-detected GLiNER2 model; using Local ONNX backend");
                return ExtractionBackend::Local { model_dir: dir_str };
            }
        }
    }

    tracing::info!("No GLiNER2 model found; using LLM extraction backend. Run `graphirm model download` to enable fast local extraction.");
    ExtractionBackend::Llm
}

/// Handle `graphirm model <action>` subcommands.
async fn run_model_command(action: ModelAction) -> Result<(), GraphirmError> {
    match action {
        ModelAction::Download => run_model_download().await,
    }
}

#[cfg(feature = "local-extraction")]
async fn run_model_download() -> Result<(), GraphirmError> {
    println!("Downloading GLiNER2-large-v1 ONNX model (~1.95 GB)...");
    println!("Files will be cached in ~/.cache/huggingface/hub/");
    println!();
    let model_dir = graphirm_agent::knowledge::local_extraction::download_model()
        .await
        .map_err(|e| GraphirmError::Config(e.to_string()))?;
    println!("Download complete.");
    println!();
    println!("Model directory: {}", model_dir.display());
    println!();
    println!("To use the local ONNX extraction backend, set:");
    println!("  export GLINER2_MODEL_DIR=\"{}\"", model_dir.display());
    println!();
    println!("Then restart `graphirm serve`. Extraction will run at");
    println!("~150-200ms per call instead of 25-35s via the LLM API.");
    Ok(())
}

#[cfg(not(feature = "local-extraction"))]
async fn run_model_download() -> Result<(), GraphirmError> {
    eprintln!("Error: this binary was not built with local extraction support.");
    eprintln!();
    eprintln!("Rebuild with:");
    eprintln!("  cargo build --release --features local-extraction");
    std::process::exit(1);
}

fn build_tool_registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(graphirm_tools::bash::BashTool));
    registry.register(Arc::new(graphirm_tools::read::ReadTool));
    registry.register(Arc::new(graphirm_tools::write::WriteTool));
    registry.register(Arc::new(graphirm_tools::edit::EditTool));
    registry.register(Arc::new(graphirm_tools::grep::GrepTool));
    registry.register(Arc::new(graphirm_tools::find::FindTool));
    registry.register(Arc::new(graphirm_tools::ls::LsTool));
    registry
}

/// Return the API key env var name for a given provider.
/// Ollama needs no key — returns an empty string.
fn api_key_for_provider(provider_name: &str) -> Result<String, GraphirmError> {
    match provider_name {
        "anthropic" => std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| GraphirmError::Config("ANTHROPIC_API_KEY not set".into())),
        "openai" => std::env::var("OPENAI_API_KEY")
            .map_err(|_| GraphirmError::Config("OPENAI_API_KEY not set".into())),
        "deepseek" => std::env::var("DEEPSEEK_API_KEY")
            .map_err(|_| GraphirmError::Config("DEEPSEEK_API_KEY not set".into())),
        "ollama" => Ok(String::new()),
        "openrouter" => std::env::var("OPENROUTER_API_KEY")
            .map_err(|_| GraphirmError::Config("OPENROUTER_API_KEY not set".into())),
        unknown => Err(GraphirmError::Config(format!(
            "Unknown provider '{unknown}'. Supported: anthropic, deepseek, ollama, openrouter"
        ))),
    }
}

async fn run_chat(model: String, db_path: &PathBuf) -> Result<(), GraphirmError> {
    let (provider_name, model_name) = graphirm_llm::factory::parse_model_string(&model)
        .map_err(|e| GraphirmError::Config(e.to_string()))?;

    let api_key = api_key_for_provider(provider_name)?;
    let provider = graphirm_llm::factory::create_provider(provider_name, &api_key)
        .map_err(|e| GraphirmError::Config(e.to_string()))?;
    let provider = Arc::new(provider);

    let graph = Arc::new(graphirm_graph::GraphStore::open(
        db_path.to_str().unwrap_or("graph.db"),
    )?);

    let config = graphirm_agent::AgentConfig {
        model: model_name.to_string(),
        ..graphirm_agent::AgentConfig::default()
    };
    let session = Arc::new(graphirm_agent::Session::new(graph.clone(), config)?);

    let mut event_bus = graphirm_agent::EventBus::new();
    let event_rx = event_bus.subscribe();
    let event_bus = Arc::new(event_bus);

    let cancel = CancellationToken::new();
    let app = graphirm_tui::app::App::new(event_rx, model_name.to_string());

    // Channel: TUI sends () each time the user submits a message; the agent
    // task receives it and kicks off a run_agent_loop call.
    let (trigger_tx, mut trigger_rx) = mpsc::unbounded_channel::<()>();

    let tools = Arc::new(build_tool_registry());

    let session_agent = session.clone();
    let event_bus_agent = event_bus.clone();
    let cancel_agent = cancel.clone();

    // Spawn the agent loop task. It sleeps until triggered, then runs one
    // full turn (possibly many tool-call iterations) per trigger.
    tokio::spawn(async move {
        while trigger_rx.recv().await.is_some() {
            if cancel_agent.is_cancelled() {
                break;
            }
            if let Err(e) = graphirm_agent::run_agent_loop(
                &session_agent,
                provider.as_ref().as_ref(),
                &tools,
                &event_bus_agent,
                &cancel_agent,
            )
            .await
            {
                tracing::error!("Agent loop error: {e}");
            }
        }
    });

    let session_for_submit = session.clone();

    // `App::run` is a blocking crossterm loop. Run it on the blocking thread
    // pool so it doesn't starve other tokio tasks (e.g. the agent loop).
    // `add_user_message` is async — use Handle::block_on to call it from
    // within the sync TUI callback running on a spawn_blocking thread.
    let handle = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || {
        app.run(move |msg| {
            if let Err(e) = handle.block_on(session_for_submit.add_user_message(&msg)) {
                tracing::error!("Failed to add user message: {e}");
                return;
            }
            // Kick the agent loop for this turn.
            let _ = trigger_tx.send(());
            tracing::info!(message = %msg, "User submitted message");
        })
    })
    .await
    .map_err(|e| std::io::Error::other(format!("TUI thread panicked: {e}")))??;

    cancel.cancel();

    Ok(())
}
