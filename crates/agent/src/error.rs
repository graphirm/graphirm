use thiserror::Error;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("Graph error: {0}")]
    Graph(#[from] graphirm_graph::GraphError),

    #[error("LLM error: {0}")]
    Llm(#[from] graphirm_llm::LlmError),

    #[error("Tool error: {0}")]
    Tool(#[from] graphirm_tools::ToolError),

    #[error("Session not found: {0}")]
    SessionNotFound(String),

    #[error("Workflow error: {0}")]
    Workflow(String),

    #[error("Context build failed: {0}")]
    Context(String),

    #[error("Recursion limit reached: {0} turns")]
    RecursionLimit(u32),

    #[error("Agent loop cancelled")]
    Cancelled,

    #[error("Task join error: {0}")]
    Join(String),
}
