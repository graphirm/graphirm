# graphirm-server

HTTP API server with SSE streaming. Exposes graph and agent functionality over REST + Server-Sent Events.
Powers the VS Code/Cursor extension and the `graphirm-eval` harness. Built on `axum`.

---

## Key Components

| File | What |
|------|------|
| `routes.rs` | Route handlers — session CRUD, prompt submission, graph queries, HITL decisions |
| `sse.rs` | SSE streaming — converts `AgentEvent` stream to `text/event-stream` response |
| `state.rs` | `AppState` — shared server state (graph store, active sessions, config) |
| `session.rs` | `SessionHandle` — per-session state (agent config, event bus, HITL gate) |
| `types.rs` | Request/response types: `CreateSessionRequest`, `PromptRequest`, `SessionResponse`, `GraphResponse`, `SseEvent`, `SubgraphQuery` |
| `sdk.rs` | `GraphirmClient` — HTTP client SDK for programmatic access |
| `middleware.rs` | Request logging, CORS headers |
| `request_log.rs` | Structured request log for analysis |
| `error.rs` | `ServerError` enum |

**Key endpoints:** `POST /sessions`, `POST /sessions/{id}/prompt` (SSE), `GET /sessions/{id}/graph`,
`GET /graph/nodes`, `POST /sessions/{id}/hitl` (approve/reject/modify tool call)

---

## Integration Points

**Used by:** `src/main.rs` (the `serve` subcommand), `graphirm-eval` (HTTP only), VS Code extension

**Depends on:** `graphirm-graph`, `graphirm-agent`, `graphirm-llm`, `graphirm-tools`, `axum`, `tower`, `tower-http`

**Default port:** 3000 (`config/default.toml` `[server]` section). Override with `--port`.

---

## How to Test

```bash
cargo test -p graphirm-server

# Key test files
tests/integration.rs                  # route handler tests
tests/scenarios.rs                    # end-to-end scenario tests
tests/test_session_restore_api.rs     # session persistence via API
tests/test_e2e_session_restore.rs     # full e2e with graph DB
```
