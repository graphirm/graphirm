# graphirm-llm

LLM provider abstraction. Defines the `LlmProvider` trait and implements it for Anthropic, OpenAI,
DeepSeek, Ollama, and OpenRouter via `rig-core`. Also provides streaming, tool call types, and
embedding providers (Mistral API + optional local `fastembed`).

---

## Key Components

| File | What |
|------|------|
| `provider.rs` | `LlmProvider` trait, `LlmMessage`, `LlmResponse`, `Role`, `ContentPart`, `CompletionConfig` |
| `factory.rs` | `build_provider()` — constructs provider from model string (e.g. `"deepseek/deepseek-chat"`) |
| `stream.rs` | `StreamEvent`, `TokenUsage` — SSE streaming types |
| `tool.rs` | `ToolDefinition`, `ToolCall` — LLM-side tool call types |
| `anthropic.rs` | Anthropic Claude provider |
| `openai.rs` | OpenAI provider |
| `deepseek.rs` | DeepSeek provider (default for graphirm) |
| `ollama.rs` | Ollama (local) provider |
| `openrouter.rs` | OpenRouter provider |
| `mistral_embed.rs` | `MistralEmbeddingProvider` — remote embeddings |
| `fastembed_provider.rs` | `FastEmbedProvider` — local embeddings (feature: `local-embed`) |
| `mock.rs` | `MockProvider`, `MockResponse` — deterministic responses for tests |
| `error.rs` | `LlmError` enum |

---

## Integration Points

**Used by:** `graphirm-agent` (workflow loop, knowledge extraction prompts)

**Depends on:** `rig-core`, `reqwest`, `tokio-stream`, `futures`

**API keys (env vars):**
- `DEEPSEEK_API_KEY` — default provider
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `OPENROUTER_API_KEY`
- Ollama needs no key (local)

**Adding a provider:** Implement `LlmProvider`, add a match arm in `factory.rs`.

---

## How to Test

```bash
# Mock-based tests (no API key needed)
cargo test -p graphirm-llm

# Integration tests (requires DEEPSEEK_API_KEY or other key)
DEEPSEEK_API_KEY=sk-... cargo test -p graphirm-llm --test integration
```
