pub mod anthropic;
pub mod deepseek;
pub mod embedded_tool_adapter;
pub mod error;
pub mod factory;
pub mod mistral_embed;
pub mod mock;
pub mod ollama;
pub mod openai;
pub mod openrouter;
pub mod provider;
pub mod stream;
pub mod tool;

#[cfg(feature = "local-embed")]
pub mod fastembed_provider;
#[cfg(feature = "local-embed")]
pub use fastembed_provider::FastEmbedProvider;

pub use embedded_tool_adapter::augment_embedded_tool_calls;
pub use error::LlmError;
pub use mistral_embed::{MistralEmbedModel, MistralEmbeddingProvider};
pub use mock::{MockProvider, MockResponse};
pub use provider::{
    CompletionConfig, ContentPart, EmbeddingProvider, LlmMessage, LlmProvider, LlmResponse, Role,
    StopReason,
};
pub use stream::{StreamEvent, TokenUsage};
pub use tool::ToolDefinition;
