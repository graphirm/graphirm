mod error;

use std::sync::Arc;

use clap::{Parser, Subcommand};
use error::GraphirmError;
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

        /// Model to use (e.g., "claude-sonnet-4-20250514")
        #[arg(short, long, default_value = "claude-sonnet-4-20250514")]
        model: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), GraphirmError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Chat { session: _, model } => {
            run_chat(model).await?;
        }
    }

    Ok(())
}

async fn run_chat(model: String) -> Result<(), GraphirmError> {
    let graph = Arc::new(graphirm_graph::GraphStore::open_memory()?);

    let config = graphirm_agent::AgentConfig {
        model: model.clone(),
        ..graphirm_agent::AgentConfig::default()
    };
    let session = Arc::new(graphirm_agent::Session::new(graph.clone(), config)?);

    let mut event_bus = graphirm_agent::EventBus::new();
    let event_rx = event_bus.subscribe();

    let cancel = CancellationToken::new();

    let app = graphirm_tui::app::App::new(event_rx, model);

    let session_for_submit = session.clone();
    app.run(move |msg| {
        if let Err(e) = session_for_submit.add_user_message(&msg) {
            tracing::error!("Failed to add user message: {}", e);
        }
        tracing::info!(message = %msg, "User submitted message");
    })?;

    cancel.cancel();

    Ok(())
}
