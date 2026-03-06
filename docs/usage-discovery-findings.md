# Phase 10: Usage Discovery Findings

**Date:** 2026-03-05  
**Sources:**  
- 5 scenarios × MockProvider, 52 total API calls (baseline)  
- 4-turn real dogfood session × DeepSeek `deepseek-chat`, 197 total API calls (real-LLM validation)

---

## Real-LLM Dogfood Session (DeepSeek) — Key Findings

### Bug found and fixed during dogfooding

**Root cause:** The context builder (`build_context`) sorted all nodes — including `Content` nodes created by tools like `read` — chronologically into the conversation thread. A `Content` node created during tool execution (e.g., reading `src/main.rs`) would be inserted between the assistant's tool-call message and the tool-result message. DeepSeek (and all OpenAI-compatible APIs) reject this: an assistant message with `tool_calls` must be immediately followed by matching tool-result messages, with no intervening user/content messages.

**Fix:** `Content` and `Knowledge` nodes are now placed before the conversation thread as context documents, not interleaved with interaction nodes.

### Real-LLM turn latencies (DeepSeek `deepseek-chat`, Hetzner spoke)

| Turn | Task | Status | Time |
|------|------|--------|------|
| 1 | Read `src/main.rs`, summarize | completed | 15s |
| 2 | `grep run_agent_loop`, explain files | completed | 40s |
| 3 | Read `workflow.rs`, 5-bullet description | completed | 30s |
| 4 | Write file + read it back to verify | completed | 20s |

**Average turn latency: 26s.** Longest turn was grep + multi-file reasoning (40s).

**UI implication:** The previous MockProvider "wait" assumption of <1s is off by 3 orders of magnitude. Every prompt submit must show a persistent loading indicator for the full agent turn duration. The polling interval in dogfood.py (5s) produced 101 `GET /api/sessions/:id` calls just for status polling — SSE streaming would reduce this to 0.

### Ripgrep dependency

The grep tool requires `rg` (ripgrep) on the system PATH. It is not installed by default on Hetzner Ubuntu spokes. Added to infrastructure runbook.

---

## Raw Numbers

| Metric | Value |
|--------|-------|
| Total API calls | 52 |
| Unique sessions | 7 |
| Avg calls per session | 6.1 |
| Scenarios run | 5 (A–E) |

---

## 1. Top Endpoints by Frequency

| Rank | Endpoint | Count | % |
|------|----------|-------|---|
| 1 | `GET /api/graph/:id` | 10 | 19.2% |
| 2 | `GET /api/sessions/:id/messages` | 9 | 17.3% |
| 3 | `POST /api/sessions/:id/prompt` | 8 | 15.4% |
| 4 | `POST /api/sessions` | 7 | 13.5% |
| 5 | `GET /api/sessions/:id` | 5 | 9.6% |
| 6 | `GET /api/graph/:id/node/:id` | 4 | 7.7% |
| 7 | `GET /api/graph/:id/subgraph/:id` | 3 | 5.8% |

**UI implication:** The graph view (`GET /api/graph/:id`) and message list (`GET /api/sessions/:id/messages`) together account for 36.5% of all calls — these are the two panes that dominate every session. They must load instantly and auto-refresh after every prompt completes.

---

## 2. Endpoint Group Distribution

| Group | Count | % |
|-------|-------|---|
| sessions | 33 | 63.5% |
| graph | 19 | 36.5% |

**UI implication:** There are no "events" calls in these scenarios (SSE is implicit in real UIs). Sessions traffic is dominant. The primary layout should be a session panel on the left, with the graph explorer always visible in a secondary pane — not hidden behind a tab.

---

## 3. Method Distribution

| Method | Count | % |
|--------|-------|---|
| GET | 35 | 67.3% |
| POST | 16 | 30.8% |
| DELETE | 1 | 1.9% |

**UI implication:** The UI is overwhelmingly read-heavy. Mutations (POST/DELETE) are infrequent and intentional. Optimistic updates on reads are safe; writes need explicit confirmation UI.

---

## 4. Most Common Call Sequences (bigrams)

| Sequence | Count |
|----------|-------|
| `GET messages → GET graph` | 9 |
| `POST prompt → GET messages` | 6 |
| `POST sessions → POST prompt` | 4 |
| `GET graph → GET session` | 3 |
| `GET graph → POST prompt` | 3 |

**UI implication:** The dominant loop is:

```
POST prompt → GET messages → GET graph
```

This is the "turn cycle" — every agent response triggers the developer to check messages then refresh the graph. The UI must implement this as a **single auto-refresh action** triggered by SSE `agent.done` events, not three separate calls the user has to make manually.

The `GET graph → POST prompt` bigram (rank 5) reveals that developers look at the graph, then send the next prompt — the graph is the thinking surface, not just a debug tool.

---

## 5. Timing Hotspots

| Endpoint | P50 (ms) | P95 (ms) | Max (ms) |
|----------|----------|----------|---------|
| `GET /api/graph/:id` | 1.6 | 3.5 | 3.5 |
| `GET /api/graph/:id/subgraph/:id` | 1.3 | 3.7 | 3.7 |
| `GET /api/sessions/:id/messages` | 0.6 | 1.9 | 1.9 |
| `POST /api/sessions/:id/prompt` | 0.5 | 1.0 | 1.0 |

*Note: MockProvider times are near-instant; production times will be higher, especially prompt (LLM latency) and graph queries (SQLite with many nodes).*

**UI implication:** Graph traversal endpoints (`/graph`, `/subgraph`) are the slowest reads at P95 ~3.5ms on an empty DB. These will grow with graph size. Add skeleton loaders to all graph views. Prompt submission is fire-and-forget (202); the wait for agent completion is SSE-driven.

---

## 6. Session Workflow Shapes

Three distinct patterns emerged:

**Pattern A: Dialogue loop** (multi-turn session)
```
create → [prompt → messages → graph] × N → get_status
```
This is the dominant shape. 4+ turns with refresh after each.

**Pattern B: Session switch** (session management)
```
get_session → messages → graph
```
When switching sessions, developers immediately load messages + graph. Both must load in parallel, not sequentially.

**Pattern C: Graph deep-dive** (graph exploration)
```
prompt → graph → node×N → subgraph×3 → tasks → knowledge
```
After prompting, some developers inspect every node individually, then drill into subgraphs. The graph explorer must support click-to-inspect without a page reload.

**Pattern D: Abort + recover**
```
prompt → abort → get_status → prompt → messages → graph
```
Abort happens, then the developer checks status before sending a new prompt. The status check is a trust signal — the UI should show session state prominently after an abort.

---

## Phase 11 UI Design Decisions (derived from data)

1. **Two-pane layout always visible:** Left = session/messages, Right = graph explorer. No tabs hiding the graph.

2. **Auto-refresh on `agent.done` SSE event:** Trigger `GET /api/sessions/:id/messages` + `GET /api/graph/:id` in parallel. This eliminates the two most common manual call sequences.

3. **Graph is the thinking surface:** Show the graph in the right pane at all times during a conversation. Nodes should appear in real-time as the agent loop runs (via SSE).

4. **Session state badge:** Always visible indicator of session status (idle / running / aborted). The `GET /api/sessions/:id` pattern after abort shows developers need this signal.

5. **Click-to-inspect nodes:** In-place node detail panel (not a separate route) — avoids the round-trip cost of navigating away.

6. **Subgraph depth slider:** The 3-depth exploration pattern appears in scenario D. Expose depth as a slider in the subgraph view.

7. **Prompt/abort is the only mutation path:** DELETE is rare (1.9%). Don't waste prominent UI real estate on session deletion — put it in a context menu.
