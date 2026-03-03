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
        /// Cloud examples:  anthropic/claude-sonnet-4-20250514
        ///                  deepseek/deepseek-chat
        ///                  openai/gpt-4o
        /// Local (Ollama):  ollama/qwen2.5:72b
        ///                  ollama/qwen3:70b
        ///                  ollama/llama3.2
        #[arg(short, long, default_value = "anthropic/claude-sonnet-4-20250514")]
        model: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), GraphirmError> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Chat { session: _, model } => {
            // TUI runs in raw/alternate-screen mode — logs must go to a file
            // so they don't corrupt the rendered UI.
            let _guard = init_file_logging();
            run_chat(model).await?;
        }
    }

    Ok(())
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
        unknown => Err(GraphirmError::Config(format!(
            "Unknown provider '{unknown}'. Supported: anthropic, openai, deepseek, ollama"
        ))),
    }
}

async fn run_chat(model: String) -> Result<(), GraphirmError> {
    let (provider_name, model_name) = graphirm_llm::factory::parse_model_string(&model)
        .map_err(|e| GraphirmError::Config(e.to_string()))?;

    let api_key = api_key_for_provider(provider_name)?;
    let provider =
        graphirm_llm::factory::create_provider(provider_name, &api_key)
            .map_err(|e| GraphirmError::Config(e.to_string()))?;
    let provider = Arc::new(provider);

    let graph = Arc::new(graphirm_graph::GraphStore::open_memory()?);

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
    tokio::task::spawn_blocking(move || {
        app.run(move |msg| {
            if let Err(e) = session_for_submit.add_user_message(&msg) {
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
