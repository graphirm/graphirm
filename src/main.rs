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
        /// Cloud examples:  anthropic/claude-sonnet-4-20250514
        ///                  deepseek/deepseek-chat
        ///                  openai/gpt-4o
        /// Local (Ollama):  ollama/qwen2.5:72b
        ///                  ollama/qwen3:70b
        ///                  ollama/llama3.2
        #[arg(short, long, default_value = "deepseek/deepseek-chat")]
        model: String,
    },

    /// Inspect the graph database
    Graph {
        #[command(subcommand)]
        action: GraphAction,
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
            // (set in .env). Defaults to DeepSeek if not configured.
            let model_spec = std::env::var("GRAPHIRM_MODEL")
                .unwrap_or_else(|_| "deepseek/deepseek-chat".to_string());
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
            // Enable post-turn knowledge extraction using the same model so the
            // /api/graph/{session_id}/knowledge endpoint returns real nodes.
            let agent_config = graphirm_agent::AgentConfig {
                model: model_name.to_string(),
                extraction: Some(graphirm_agent::knowledge::extraction::ExtractionConfig {
                    enabled: true,
                    model: model_name.to_string(),
                    ..Default::default()
                }),
                ..agent_config
            };

            // Optional embedding provider for cross-session memory.
            // Set EMBEDDING_BACKEND="mistral/codestral-embed" or "fastembed/nomic-embed-text-v1"
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
