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

    /// Session cap hit. `assistant_node_id` is `Some` when the assistant turn was
    /// recorded before stopping; `None` when the cap blocked starting an LLM call.
    #[error("Session LLM token cap exceeded (used {used}, cap {cap})")]
    SessionTokenCapExceeded {
        used: u64,
        cap: u64,
        assistant_node_id: Option<graphirm_graph::nodes::NodeId>,
    },

    #[error("Task join error: {0}")]
    Join(String),

    #[error("Subagent '{name}' failed: {reason}")]
    SubagentFailed { name: String, reason: String },

    #[error("Agent not found in registry: {0}")]
    AgentNotFound(String),
}
