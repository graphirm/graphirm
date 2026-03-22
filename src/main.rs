mod commands;
mod error;

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use error::GraphirmError;

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

    /// Manage pinned Knowledge nodes (conventions, rules)
    Knowledge {
        #[command(subcommand)]
        action: KnowledgeAction,
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

    /// Export turns to JSONL for structured-response discovery (GLiNER2).
    ///
    /// Reads the graph at --db and writes one JSON object per turn
    /// (session_id, turn_index, role, text). Default: assistant turns only.
    ExportCorpus {
        /// Output file (default: stdout)
        #[arg(short, long)]
        out: Option<PathBuf>,
        /// Maximum number of turns to export (for validation samples, e.g. 100)
        #[arg(long)]
        limit: Option<u64>,
        /// Include user prompts as well as assistant turns (limit then applies to total turns)
        #[arg(long)]
        all_roles: bool,
    },

    /// Import Cursor agent transcript file(s) into the graph.
    ///
    /// Accepts a single .txt transcript file or a directory of .txt files.
    /// Idempotent — re-importing the same file is a no-op.
    ImportCursor {
        /// Path to a .txt transcript file or a directory containing .txt files
        path: PathBuf,
        /// Print what would be imported without writing to the graph
        #[arg(long)]
        dry_run: bool,
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
        /// Process corpus in batches of N turns to limit memory (default: all at once)
        #[arg(long)]
        batch_size: Option<u64>,
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
enum KnowledgeAction {
    /// List all pinned Knowledge nodes
    List {
        /// Max nodes to show
        #[arg(short, long, default_value = "50")]
        limit: usize,
    },
    /// Create a new pinned Knowledge node (convention/rule)
    Pin {
        /// Entity name (kebab-case identifier, e.g. "no-unwrap-rule")
        entity: String,
        /// Summary text of the rule/convention
        summary: String,
        /// Entity type (default: "convention")
        #[arg(long, default_value = "convention")]
        entity_type: String,
    },
    /// Remove the pinned flag from a Knowledge node by ID
    Unpin {
        /// Node ID (UUID) of the Knowledge node to unpin
        id: String,
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
    let db_path = commands::resolve_db_path(cli.db)?;

    match cli.command {
        Commands::Chat { session: _, model } => {
            let _guard = commands::init_file_logging();
            commands::chat::run(model, &db_path).await?;
        }
        Commands::Graph { action } => {
            tracing_subscriber::fmt()
                .with_writer(std::io::stderr)
                .with_env_filter("error")
                .init();
            commands::graph::run(action, &db_path)?;
        }
        Commands::Model { action } => {
            tracing_subscriber::fmt()
                .with_writer(std::io::stderr)
                .with_env_filter("info")
                .init();
            commands::model::run(action).await?;
        }
        Commands::Knowledge { action } => {
            tracing_subscriber::fmt()
                .with_writer(std::io::stderr)
                .with_env_filter("warn")
                .init();
            commands::knowledge::run(action, &db_path)?;
        }
        Commands::ExportCorpus {
            out,
            limit,
            all_roles,
        } => {
            tracing_subscriber::fmt()
                .with_writer(std::io::stderr)
                .with_env_filter("warn")
                .init();
            commands::export::run(&db_path, out, limit, all_roles)?;
        }
        Commands::ImportCursor { path, dry_run } => {
            tracing_subscriber::fmt()
                .with_writer(std::io::stderr)
                .with_env_filter("warn")
                .init();
            commands::import::run(path, dry_run, &db_path)?;
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
            commands::gliner::run_label_explore(corpus, labels, min_confidence, out).await?;
        }
        #[cfg(feature = "local-extraction")]
        Commands::SchemaSuggest { report, out } => {
            tracing_subscriber::fmt()
                .with_writer(std::io::stderr)
                .with_env_filter("warn")
                .init();
            commands::gliner::run_schema_suggest(report, out)?;
        }
        #[cfg(feature = "local-extraction")]
        Commands::PredictSpans {
            corpus,
            labels,
            min_confidence,
            out,
            batch_size,
        } => {
            tracing_subscriber::fmt()
                .with_writer(std::io::stderr)
                .with_env_filter("warn")
                .init();
            commands::gliner::run_predict_spans(corpus, labels, min_confidence, out, batch_size)
                .await?;
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
            commands::gliner::run_validate_agreement(human, gliner, threshold, out)?;
        }
        Commands::Serve { host, port } => {
            commands::serve::run(&db_path, host, port).await?;
        }
    }

    Ok(())
}
