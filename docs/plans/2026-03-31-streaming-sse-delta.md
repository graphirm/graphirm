# Streaming SSE Message Deltas

> **For Claude:** When implementing, use superpowers:executing-plans (worktree, batch tasks, verification).

**Goal:** Stream LLM response text to the web-app incrementally via SSE `message_delta` events, removing the current delay where chat text only appears after a full `getMessages()` round-trip on `message_end`.

**Status:** Phase A ✅ done. Phase B ✅ done. Phase C (real SSE streaming) ✅ done.

---

## The Problem (original)

`stream_and_record` in `workflow.rs` calls `llm.complete()` (blocking, full response), persists the `Interaction` node, then emits a synthetic `MessageStart` → single `MessageDelta` (entire text) → `MessageEnd` sequence. But `agent_event_to_sse` maps `MessageStart` and `MessageDelta` to `Heartbeat` (catch-all `_` branch), so the web-app never sees them. The chat panel only updates when `message_end` fires → `getMessages()` API call → full round-trip.

Meanwhile:
- `SseEventType` already has `MessageStart`, `MessageDelta`, `MessageEnd` variants.
- `sse.ts` already subscribes to `message_start` and `message_delta` event names.
- All LLM providers implement `stream()` — they call `complete()` internally but split the text into 10-byte chunks as `StreamEvent::TextDelta`s.
- The TUI already handles `MessageDelta` directly from `AgentEvent` (not via SSE).

**Phase A+B fixed the SSE plumbing and frontend rendering, but streaming was still fake — providers called `complete()` then split the finished response into 10-byte chunks. Phase C fixes this at the provider level.**

## Phase A — SSE plumbing + streaming chat

### A1. Backend: Map `MessageStart`/`MessageDelta` in `agent_event_to_sse`

**File:** `crates/server/src/routes.rs`

Add match arms for `AgentEvent::MessageStart` and `AgentEvent::MessageDelta`:

```rust
AgentEvent::MessageStart { node_id } => (
    SseEventType::MessageStart,
    serde_json::json!({ "node_id": node_id.to_string() }),
),
AgentEvent::MessageDelta { node_id, delta } => {
    let text = match delta {
        graphirm_llm::StreamEvent::TextDelta(t) => t.clone(),
        _ => String::new(),
    };
    (
        SseEventType::MessageDelta,
        serde_json::json!({ "node_id": node_id.to_string(), "text": text }),
    )
},
```

### A2. Backend: Switch `stream_and_record` from `complete()` to `stream()`

**File:** `crates/agent/src/workflow.rs`

Replace the `llm.complete()` call with `llm.stream()`. Accumulate `StreamEvent`s into an `LlmResponse`-equivalent. Emit `MessageDelta` per chunk as tokens arrive. Persist the `Interaction` node after `Done`. Then emit `MessageEnd`.

Key changes:
- Emit `MessageStart` with a placeholder `NodeId` (generated before streaming starts).
- On each `TextDelta`: emit `AgentEvent::MessageDelta` with the chunk.
- On `Done`: build `LlmResponse` from accumulated text + tool calls + usage, persist the `Interaction` node, emit `MessageEnd`.
- Fallback chain: if a retryable error occurs during streaming, fall back to the next model in the tier array (same as today but wrapping the stream instead of `complete()`).
- Segment extraction, knowledge extraction, metadata stamping — all happen after persistence, unchanged.

**Risk:** The `node_id` in `MessageStart`/`MessageDelta` is emitted before the node is persisted. The web-app uses it only as a correlation key for the streaming message — it doesn't query the graph for it until `message_end`. This is safe.

### A3. Frontend: Handle streaming messages in `useSession`

**File:** `web-app/src/hooks/useSession.ts`

New state: `streamingMessage: Message | null`.

```typescript
// message_start: create placeholder
} else if (ev.event === 'message_start') {
  const payload = ev.data?.data ?? ev.data;
  setStreamingMessage({
    id: payload.node_id,
    role: 'assistant',
    content: '',
    created_at: new Date().toISOString(),
  });
// message_delta: append text
} else if (ev.event === 'message_delta') {
  const payload = ev.data?.data ?? ev.data;
  const text = payload?.text ?? '';
  setStreamingMessage(prev =>
    prev ? { ...prev, content: prev.content + text } : prev
  );
}
```

On `message_end`: clear `streamingMessage` and refresh messages (as today). Export `streamingMessage` from the hook.

### A4. Frontend: Render streaming message in ChatPane

**File:** `web-app/src/components/ChatPane.tsx`

If `streamingMessage` is truthy, append it to the rendered messages list. It will render with the same `MarkdownBody` component as other assistant messages. When `message_end` fires, the streaming message is replaced by the persisted version from `getMessages()`.

### A5. Frontend: Pass `streamingMessage` through App

**File:** `web-app/src/App.tsx`

Thread `streamingMessage` from `useSession()` through to `ChatPane` (same pattern as `isThinking`, `pendingApproval`).

---

## Phase B — Pretext sizing on partial text

Once Phase A is live, the graph canvas can show a provisional node that grows as tokens stream in, with Pretext-predicted dimensions so the layout doesn't jump when the node finalizes.

### B1. Provisional streaming node in `useGraphData`

**File:** `web-app/src/hooks/useGraphData.ts`

When `streamingMessage` is truthy (Phase A exports it), inject a provisional `Node` into the React Flow node array:
- `id`: the `node_id` from `message_start` (same ID the persisted node will have)
- `type`: `'interaction'`
- `data`: synthetic `GraphNode` with `role: 'assistant'`, `content: streamingMessage.content`
- `position`: placed relative to the last user node (same `positionNewNodes` helper used for SSE patches)

When `streamingMessage` clears (on `message_end`), the provisional node is removed — the real node arrives via `graph_update` SSE patch and takes its place with the same ID, preserving position.

### B2. Pretext sizing on partial text during streaming

**File:** `web-app/src/layout/pretextDimensions.ts`

`buildPretextSizeMap` already computes sizes from collapsed preview text. The provisional node's preview text changes as `message_delta` appends content. Two options:

**Option 1 — Re-run on every delta (simple, may be fast enough):**
Call `estimateSizeFromPreview(preview)` on the provisional node's current content each time `streamingMessage` updates. Stamp `width`/`height` on the provisional node. Pretext's `prepare()` + `layout()` is arithmetic-only on cached segment widths — fast enough for per-chunk updates if chunks are small (10 bytes from current providers).

**Option 2 — Debounced re-run (safer for large chunks):**
Debounce the size computation (e.g. 100ms). The provisional node starts with a default size; Pretext sizes catch up within one debounce window. Avoids layout thrashing if a provider sends large chunks.

Recommendation: Start with Option 1. If profiling shows jank, switch to Option 2.

### B3. Stable dagre during streaming

**File:** `web-app/src/hooks/useGraphData.ts`

The `isPatchUpdate` flag (Phase 15) already skips full dagre re-runs on SSE patches, preserving existing positions. The provisional node should be placed once (on `message_start`) and only resized (not repositioned) as text grows. On `message_end`, the real node replaces it at the same position — `patchGraphData` merges by ID.

Key constraint: do **not** re-run dagre while streaming. Only resize the provisional node's `style.width` / `style.height` and top-level `width` / `height` in place.

### B4. Thread `streamingMessage` into `useGraphData`

**File:** `web-app/src/hooks/useGraphData.ts`

`useGraphData` currently takes `(graphData, sessionId, layoutMode, filter)`. Add optional 5th parameter `streamingMessage?: Message | null`. When non-null, inject the provisional node. When null, remove it.

**File:** `web-app/src/components/GraphCanvas.tsx`

Pass `streamingMessage` from props into `useGraphData()`.

---

---

## Testing

### Phase A
- **Rust unit tests:** Verify `agent_event_to_sse` maps `MessageStart` and `MessageDelta` to correct `SseEventType` (not `Heartbeat`).
- **TypeScript build:** `tsc -b && vite build` must pass.
- **Manual verification:** Run server + web-app, send a prompt, confirm text streams into chat before `message_end`.

### Phase B
- **TypeScript build:** `tsc -b && vite build` must pass.
- **Manual verification:** Send a prompt, confirm provisional node appears on canvas during streaming, resizes as text grows, and is cleanly replaced by the persisted node on `message_end` without position jump.
- **Performance:** DevTools profile with a long response (~2000 tokens). Confirm no layout thrashing or dropped frames during streaming.

---

## Files touched

### Phase A

| File | Change |
|------|--------|
| `crates/server/src/routes.rs` | `agent_event_to_sse` match arms for `MessageStart`, `MessageDelta` |
| `crates/agent/src/workflow.rs` | `stream_and_record`: `stream()` instead of `complete()`, incremental `MessageDelta` emissions |
| `web-app/src/hooks/useSession.ts` | `streamingMessage` state, `message_start`/`message_delta` handlers |
| `web-app/src/components/ChatPane.tsx` | Render `streamingMessage` inline |
| `web-app/src/App.tsx` | Thread `streamingMessage` prop |

### Phase B

| File | Change |
|------|--------|
| `web-app/src/hooks/useGraphData.ts` | Provisional streaming node injection, `streamingMessage` param |
| `web-app/src/layout/pretextDimensions.ts` | Per-delta Pretext sizing on provisional node |
| `web-app/src/components/GraphCanvas.tsx` | Pass `streamingMessage` into `useGraphData` |

### Both phases

| File | Change |
|------|--------|
| `docs/backlog.md` | Update status |
| `docs/completion-log.md` | Entry when done |

---

## Phase C — Real SSE streaming from LLM provider

Phases A+B wired the SSE plumbing end-to-end but the underlying `LlmProvider::stream()` implementations were fake — they called `complete()` first (full blocking response), then split the result into 10-byte chunks via `stream::iter()`. Console timestamps proved all `message_delta` events arrived within 2ms.

### C1. Real SSE streaming in `OpenRouterProvider::stream()`

**File:** `crates/llm/src/openrouter.rs`

Replaced fake streaming with a direct `reqwest` POST to OpenRouter's `/chat/completions` endpoint with `"stream": true` + `"stream_options": {"include_usage": true}`.

Key changes:
- `OpenRouterProvider` now stores `http: reqwest::Client` and `api_key: String` alongside the rig `CompletionsClient` (rig still used for `complete()`)
- `build_openai_body()` converts `LlmMessage`/`ToolDefinition`/`CompletionConfig` to the OpenAI-compatible JSON format (system/user/assistant/tool roles, function tools)
- `stream()` POSTs with reqwest, spawns a tokio task that reads the chunked HTTP body via `response.chunk()`, buffers lines, parses `data: {...}` SSE events
- `SseChunk`/`SseChoice`/`SseDelta`/`SseToolCallDelta`/`SseFunctionDelta`/`SseUsage` — deserialization structs for OpenAI streaming response format
- `process_sse_chunk()` maps each parsed chunk to `StreamEvent`s sent through an `mpsc::channel(128)`, returned as `ReceiverStream`
- Tool call lifecycle: `ToolCallStart` on first chunk with `id`+`name`, `ToolCallDelta` on argument fragments, `ToolCallEnd` on `finish_reason == "tool_calls" | "stop"`
- `data: [DONE]` emits `StreamEvent::Done` with accumulated usage; graceful fallback if stream ends without `[DONE]`
- SSE comments (`: OPENROUTER PROCESSING`) and empty lines silently skipped
- Unparseable data lines logged at `tracing::debug` level and skipped (non-fatal)

### C2. Verification

Console timestamps from deployed `app.graphirm.ai` confirm real streaming:

**Before (fake):** all deltas within 8ms
```
message_start  t=...4281
message_delta  t=...4283  len=10  "I'll write"
message_delta  t=...4285  len=3   "ent"
message_end    t=...4289          ← 8ms total
```

**After (real SSE):** deltas over ~1 second with natural inter-token gaps
```
message_start  t=...1017
message_delta  t=...1026  len=3   "The"          +9ms
message_delta  t=...1173  len=6   " ocean"       +147ms
message_delta  t=...1305  len=5   " blue"        +132ms
message_delta  t=...1494  len=5   "Waves"        +189ms
message_delta  t=...1937  len=9   "'s thread"    +72ms
message_end    t=...2049                          ← 1,032ms total
```

### C3. Tests

8 new unit tests in `openrouter.rs`:
- `test_build_openai_body_minimal` — message + model conversion
- `test_build_openai_body_with_system_and_tools` — system preamble, tools, max_tokens, temperature
- `test_build_openai_body_tool_result` — assistant tool_calls + tool result round-trip
- `test_parse_sse_text_chunk` — text delta deserialization
- `test_parse_sse_tool_call_chunk` — tool call delta deserialization
- `test_parse_sse_usage_chunk` — usage deserialization
- `test_process_sse_chunk_text` — async channel event emission
- `test_process_sse_chunk_tool_lifecycle` — full start→delta→end tool call flow

### Phase C files

| File | Change |
|------|--------|
| `crates/llm/src/openrouter.rs` | Real SSE streaming, `build_openai_body`, SSE chunk parsing, 8 tests |
