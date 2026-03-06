use thiserror::Error;

#[derive(Debug, Error)]
pub enum GraphirmError {
    #[error("Graph error: {0}")]
    Graph(#[from] graphirm_graph::GraphError),

    #[error("LLM error: {0}")]
    Llm(#[from] graphirm_llm::LlmError),

    #[error("Tool error: {0}")]
    Tool(#[from] graphirm_tools::ToolError),

    #[error("Agent error: {0}")]
    Agent(#[from] graphirm_agent::AgentError),

    #[error("Config error: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
