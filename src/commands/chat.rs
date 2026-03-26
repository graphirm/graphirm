use std::path::Path;
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::error::GraphirmError;

pub async fn run(model: String, db_path: &Path) -> Result<(), GraphirmError> {
    let (provider_name, model_name) = graphirm_llm::factory::parse_model_string(&model)
        .map_err(|e| GraphirmError::Config(e.to_string()))?;

    let api_key = super::api_key_for_provider(provider_name)?;
    let provider = graphirm_llm::factory::create_provider(provider_name, &api_key)
        .map_err(|e| GraphirmError::Config(e.to_string()))?;
    let provider: Arc<dyn graphirm_llm::LlmProvider> = Arc::from(provider);

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

    let (trigger_tx, mut trigger_rx) = mpsc::unbounded_channel::<()>();

    let tools = Arc::new(super::build_tool_registry());

    let session_agent = session.clone();
    let event_bus_agent = event_bus.clone();
    let cancel_agent = cancel.clone();
    let llm = provider.clone();

    tokio::spawn(async move {
        while trigger_rx.recv().await.is_some() {
            if cancel_agent.is_cancelled() {
                break;
            }
            if let Err(e) = graphirm_agent::run_agent_loop(
                &session_agent,
                llm.clone(),
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

    let handle = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || {
        app.run(move |msg| {
            if let Err(e) = handle.block_on(session_for_submit.add_user_message(&msg)) {
                tracing::error!("Failed to add user message: {e}");
                return;
            }
            let _ = trigger_tx.send(());
            tracing::info!(message = %msg, "User submitted message");
        })
    })
    .await
    .map_err(|e| std::io::Error::other(format!("TUI thread panicked: {e}")))??;

    cancel.cancel();

    Ok(())
}
