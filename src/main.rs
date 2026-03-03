mod error;

use clap::{Parser, Subcommand};
use error::GraphirmError;

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
        #[arg(short, long)]
        model: Option<String>,
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
        Commands::Chat { session, model } => {
            tracing::info!(
                session = session.as_deref().unwrap_or("new"),
                model = model.as_deref().unwrap_or("default"),
                "Starting chat session"
            );
            println!("graphirm chat — not yet implemented");
        }
    }

    Ok(())
}
