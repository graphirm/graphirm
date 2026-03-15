# graphirm-agent

The brain. Core agent loop, context engine, multi-agent coordination, knowledge extraction, HITL gate,
and session management. Everything in graphirm that involves deciding what to do next lives here.

---

## Key Components

| File | What |
|------|------|
| `workflow.rs` | `run_agent_loop` — async state machine: receive prompt → build context → call LLM → execute tools → loop until done |
| `context.rs` | `build_context`, `score_node` — relevance scoring (recency + edge weight + BFS distance + PageRank), greedy token budget fill |
| `compact.rs` | `compact_context` — summarizes old turns when context exceeds budget |
| `session.rs` | `Session`, `SessionMetadata` — persistence, resume by session ID |
| `coordinator.rs` | `Coordinator` — orchestrates subagent spawning, result collection |
| `delegate.rs` | `SubagentTool` — the `delegate` tool; spawns a subagent as a tool call |
| `multi.rs` | `spawn_subagent`, `collect_subagent_results`, `wait_for_subagents` |
| `escalation.rs` | Soft escalation — detects repeated identical tool calls, prompts synthesis |
| `hitl.rs` | `HitlGate`, `HitlDecision` — blocks on destructive tools for human approval |
| `config.rs` | `AgentConfig`, `AgentMode`, `Permission` — loaded from `config/default.toml` |
| `event.rs` | `AgentEvent`, `EventBus` — SSE streaming from agent loop to server/TUI |
| `error.rs` | `AgentError` enum |
| `knowledge/extraction.rs` | Entity extraction — LLM, GLiNER2 ONNX, or hybrid; `ExtractionBackend` enum |
| `knowledge/segments.rs` | Structured response segmentation — parse JSON segments or GLiNER2 fallback (`try_gliner2_fallback`), persist as child `Content` nodes via `Contains` edges |
| `knowledge/memory.rs` | Cross-session memory — HNSW vector search, inject past knowledge into context |
| `knowledge/local_extraction.rs` | `OnnxExtractor` — GLiNER2 ONNX tokenisation + inference (feature-gated) |

**Context scoring weights:** recency 0.3 · edge weight 0.2 · BFS distance 0.3 · PageRank 0.2

---

## Integration Points

**Used by:** `graphirm-server` (HTTP sessions), `graphirm-tui` (chat panel), `src/main.rs` (CLI)

**Depends on:** `graphirm-graph`, `graphirm-llm`, `graphirm-tools`

**Feature `local-extraction`:** Enables GLiNER2 ONNX knowledge extraction; needs `GLINER2_MODEL_DIR`.

---

## How to Test

```bash
# Standard tests
cargo test -p graphirm-agent

# With GLiNER2 local extraction
GLINER2_MODEL_DIR=/path/to/gliner2 cargo test -p graphirm-agent --features local-extraction

# Key test files
tests/test_session_metadata.rs         # session persist/resume
tests/multi_agent_integration.rs       # coordinator + subagent spawning
tests/test_escalation_integration.rs
tests/knowledge_integration.rs
tests/test_segments_integration.rs     # segment parse → persist → graph round-trip
```
