# graphirm-tui

Terminal UI for interactive chat sessions. Built on `ratatui` + `crossterm`. Two panels: chat
(message history + streaming response) and graph explorer (node/edge visualization).

---

## Key Components

| File | What |
|------|------|
| `app.rs` | `App` — top-level application state, drives the render + event loop |
| `chat.rs` | Chat panel — renders message history, handles streaming token appends |
| `input.rs` | Input bar — user text entry, keybindings (`Enter` to send, `Ctrl-C` to quit) |
| `graph.rs` | Graph explorer panel — node list, edge visualization |
| `ui.rs` | Layout composition — splits terminal into panels, arranges widgets |
| `status.rs` | Status bar — current model, session ID, token usage |
| `events.rs` | Terminal event handling — key, mouse, resize |
| `types.rs` | Shared TUI types (message structs, panel state) |

---

## Integration Points

**Used by:** `src/main.rs` — the `chat` subcommand creates an `App` and calls the render loop

**Depends on:** `graphirm-graph`, `graphirm-agent`, `graphirm-llm`, `ratatui`, `crossterm`, `tokio`

---

## How to Test

No automated tests. Manual verification:

```bash
DEEPSEEK_API_KEY=sk-... cargo run -- chat
# Or with a specific model:
cargo run -- chat --model ollama/qwen2.5:72b
```
