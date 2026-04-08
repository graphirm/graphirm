pub mod chat;
pub mod export;
#[cfg(feature = "local-extraction")]
pub mod gliner;
pub mod graph;
pub mod import;
pub mod knowledge;
pub mod model;
pub mod serve;
pub mod trace_analysis;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use graphirm_tools::Tool;
use graphirm_tools::registry::ToolRegistry;

use crate::error::GraphirmError;

/// Resolve the graph DB path, creating parent directories as needed.
pub fn resolve_db_path(override_path: Option<PathBuf>) -> Result<PathBuf, GraphirmError> {
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

/// Return the API key env var value for a given provider.
pub fn api_key_for_provider(provider_name: &str) -> Result<String, GraphirmError> {
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

pub fn build_tool_registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(graphirm_tools::bash::BashTool));
    registry.register(Arc::new(graphirm_tools::read::ReadTool));
    registry.register(Arc::new(graphirm_tools::write::WriteTool));
    registry.register(Arc::new(graphirm_tools::edit::EditTool));
    registry.register(Arc::new(graphirm_tools::fetch_url::FetchUrlTool::new()));
    registry.register(Arc::new(graphirm_tools::grep::GrepTool));
    registry.register(Arc::new(graphirm_tools::find::FindTool));
    registry.register(Arc::new(graphirm_tools::ls::LsTool));
    registry.register(Arc::new(graphirm_tools::graph_query::GraphQueryTool));
    registry.register(Arc::new(graphirm_tools::diff::DiffTool::new()));
    registry.register(Arc::new(graphirm_tools::read_many::ReadManyTool::new()));
    registry.register(Arc::new(
        graphirm_tools::repo_briefing::RepoBriefingTool::new(),
    ));
    registry.register(Arc::new(graphirm_tools::graph_diff::GraphDiffTool::new()));
    registry.register(Arc::new(
        graphirm_tools::session_trace::SessionTraceTool::new(),
    ));
    registry.register(Arc::new(graphirm_tools::cargo_check::CargoCheckTool::new()));
    registry.register(Arc::new(graphirm_tools::submit::SubmitTool::new()));
    registry.register(Arc::new(
        graphirm_tools::context_report::ContextReportTool::new(),
    ));
    registry.register(Arc::new(graphirm_agent::TraceAnalysisTool::new()));

    let plugins_dir = std::env::var("GRAPHIRM_PLUGINS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            dirs_next::home_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("."))
                .join(".graphirm/plugins")
        });

    if plugins_dir.is_dir() {
        match std::fs::read_dir(&plugins_dir) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if !path.is_dir() {
                        continue;
                    }
                    match graphirm_tools::script::ScriptTool::from_dir(&path) {
                        Ok(tool) => {
                            tracing::info!(
                                name = tool.name(),
                                destructive = tool.is_destructive(),
                                dir = %path.display(),
                                "Loaded plugin tool"
                            );
                            registry.register(Arc::new(tool));
                        }
                        Err(e) => {
                            tracing::warn!(
                                dir = %path.display(),
                                error = %e,
                                "Skipping invalid plugin"
                            );
                        }
                    }
                }
            }
            Err(e) => {
                tracing::warn!(
                    dir = %plugins_dir.display(),
                    error = %e,
                    "Failed to read plugins directory"
                );
            }
        }
    }

    registry
}

/// Initialise a rolling daily log file at `~/.local/share/graphirm/graphirm.log`.
/// Returns the non-blocking guard — **keep it alive** for the program's lifetime.
pub fn init_file_logging() -> tracing_appender::non_blocking::WorkerGuard {
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

/// Open a `GraphStore` from a path, converting the error.
pub fn open_store(db_path: &Path) -> Result<graphirm_graph::GraphStore, GraphirmError> {
    Ok(graphirm_graph::GraphStore::open(
        db_path.to_str().unwrap_or("graph.db"),
    )?)
}
