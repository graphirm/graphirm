use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::GraphirmError;

pub async fn run(db_path: &Path, host: String, port: u16) -> Result<(), GraphirmError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let graph = Arc::new(graphirm_graph::GraphStore::open(
        db_path.to_str().unwrap_or("graph.db"),
    )?);
    let tools = Arc::new(super::build_tool_registry());

    let config_path = std::path::Path::new("config/default.toml");
    let agent_config = if config_path.exists() {
        graphirm_agent::AgentConfig::from_file(config_path).unwrap_or_else(|e| {
            tracing::warn!(
                "Failed to load {}: {e}; using defaults",
                config_path.display()
            );
            graphirm_agent::AgentConfig::default()
        })
    } else {
        tracing::warn!("config/default.toml not found; using AgentConfig defaults");
        graphirm_agent::AgentConfig::default()
    };

    let model_spec = std::env::var("GRAPHIRM_MODEL")
        .unwrap_or_else(|_| "openrouter/qwen/qwen3-coder:free".to_string());
    let (provider_name, model_name) = graphirm_llm::factory::parse_model_string(&model_spec)
        .map_err(|e| GraphirmError::Config(e.to_string()))?;
    let api_key = super::api_key_for_provider(provider_name)?;
    let llm: Arc<dyn graphirm_llm::LlmProvider> = Arc::from(
        graphirm_llm::factory::create_provider(provider_name, &api_key)
            .map_err(|e| GraphirmError::Config(e.to_string()))?,
    );

    let extraction_enabled = std::env::var("GRAPHIRM_EXTRACTION")
        .map(|v| v != "0" && v != "false")
        .unwrap_or(true);
    let extraction_backend = resolve_extraction_backend();
    let agent_config = graphirm_agent::AgentConfig {
        model: model_name.to_string(),
        extraction: Some(graphirm_agent::knowledge::extraction::ExtractionConfig {
            enabled: extraction_enabled,
            model: model_name.to_string(),
            backend: extraction_backend,
            ..Default::default()
        }),
        ..agent_config
    };
    if !extraction_enabled {
        tracing::info!("Knowledge extraction disabled (GRAPHIRM_EXTRACTION=false)");
    }

    let memory_retriever = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        setup_memory_retriever(&graph),
    )
    .await
    .map_err(|_| {
        GraphirmError::Config("Embedding provider initialization timed out after 30s".into())
    })?;

    let web_dir = find_web_dir();
    if let Some(ref dir) = web_dir {
        tracing::info!("Web UI found at {}", dir.display());
    } else {
        tracing::info!("No web/ directory found — web UI disabled");
    }

    let server_config = graphirm_server::ServerConfig { host, port };
    graphirm_server::start_server(
        graph,
        llm,
        tools,
        agent_config,
        server_config,
        memory_retriever?,
        web_dir,
    )
    .await
    .map_err(|e| GraphirmError::Config(e.to_string()))?;

    Ok(())
}

/// Locate the web UI static files directory.
///
/// Checks `web-app/dist/` (React build) first, then `web/` (vanilla JS fallback).
fn find_web_dir() -> Option<PathBuf> {
    let candidates = ["web-app/dist", "web"];

    if let Ok(exe) = std::env::current_exe() {
        let exe_dir = exe.parent().unwrap_or(Path::new("."));
        for subdir in &candidates {
            let dir = exe_dir.join(subdir);
            if dir.join("index.html").exists() {
                return Some(dir);
            }
        }
    }

    for subdir in &candidates {
        let dir = PathBuf::from(subdir);
        if dir.join("index.html").exists() {
            return Some(dir);
        }
    }

    None
}

/// Select the knowledge extraction backend.
fn resolve_extraction_backend() -> graphirm_agent::knowledge::extraction::ExtractionBackend {
    use graphirm_agent::knowledge::extraction::ExtractionBackend;

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

    let hf_cache = dirs_next::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("~/.cache"))
        .join("huggingface")
        .join("hub")
        .join("models--lmo3--gliner2-large-v1-onnx")
        .join("snapshots");

    if let Ok(mut entries) = std::fs::read_dir(&hf_cache)
        && let Some(Ok(entry)) = entries.next()
    {
        let snapshot_dir = entry.path();
        if snapshot_dir.join("gliner2_config.json").exists() {
            let dir_str = snapshot_dir.to_string_lossy().to_string();
            tracing::info!(model_dir = %dir_str, "Auto-detected GLiNER2 model; using Local ONNX backend");
            return ExtractionBackend::Local { model_dir: dir_str };
        }
    }

    tracing::info!(
        "No GLiNER2 model found; using LLM extraction backend. Run `graphirm model download` to enable fast local extraction."
    );
    ExtractionBackend::Llm
}

/// Initialize the optional embedding provider for cross-session memory.
async fn setup_memory_retriever(
    graph: &Arc<graphirm_graph::GraphStore>,
) -> Result<Option<std::sync::Arc<graphirm_agent::knowledge::memory::MemoryRetriever>>, GraphirmError>
{
    let embedding_backend = std::env::var("EMBEDDING_BACKEND").ok();
    if let Some(spec) = embedding_backend {
        let mistral_key = std::env::var("MISTRAL_API_KEY").ok();
        match graphirm_llm::factory::create_embedding_provider(&spec, mistral_key.as_deref()) {
            Ok((provider, dim)) => {
                tracing::info!(backend = %spec, dim, "Embedding provider initialised");
                let retriever = std::sync::Arc::new(
                    graphirm_agent::knowledge::memory::MemoryRetriever::from_store(
                        graph.clone(),
                        std::sync::Arc::from(provider),
                        dim,
                    ),
                );
                match tokio::time::timeout(
                    std::time::Duration::from_secs(60),
                    retriever.hydrate_from_graph(),
                )
                .await
                {
                    Ok(Ok(n)) => tracing::info!(count = n, "Restored embeddings from graph store"),
                    Ok(Err(e)) => {
                        tracing::warn!(error = %e, "HNSW hydration failed (non-fatal); starting fresh")
                    }
                    Err(_) => tracing::warn!(
                        "HNSW hydration timed out after 60s (non-fatal); starting fresh"
                    ),
                }
                Ok(Some(retriever))
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "Embedding provider failed to init; memory disabled"
                );
                Ok(None)
            }
        }
    } else {
        tracing::info!("EMBEDDING_BACKEND not set; cross-session memory disabled");
        Ok(None)
    }
}
