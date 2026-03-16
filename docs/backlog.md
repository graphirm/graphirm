# Graphirm Backlog

Items captured here are validated ideas not yet scheduled into a numbered phase. Each item includes the rationale for deferral and a suggested phase target.

---

## Bug — HITL card shows "Agent wants to run: undefined"

**What:** When the HITL gate triggers in the browser web UI, the approval card renders "Agent wants to run: **undefined**" instead of the actual tool name (e.g. "bash", "read", "write").

**Root cause:** `renderApprovalCard()` in `web/chat.js` is reading the tool name from the wrong field in the `awaiting_approval` SSE payload. The server sends the tool name under a specific key that doesn't match what the JS is accessing.

**Fix:** Check the exact shape of the `awaiting_approval` SSE event in `crates/server/src/sse.rs` or `crates/server/src/types.rs`, find the tool name field, and update `renderApprovalCard()` in `web/chat.js` to read it correctly.

**Suggested target:** Phase 12 (small fix, high visibility — HITL is a key differentiator).

---

## Phase 12 vision — Miro/n8n-style interactive whiteboard graph

**What:** Replace the current read-only d3 force graph with a fully interactive whiteboard — nodes as draggable cards, edges with visible connectors, in-place expansion, and manual annotation. Think Miro, FigJam, or n8n's workflow canvas applied to the agent interaction graph.

**Why:** The current graph is a good visualisation but passive. The "graph is the interface" philosophy implies the graph should be *the* primary surface for navigating, annotating, and steering agent work — not just a side panel. A whiteboard makes that concrete.

**Capabilities:**
- Nodes rendered as cards (show role, content preview, type badge) — pan/zoom freely
- Click to expand a card in-place (full message, tool output, file diff, knowledge summary)
- Drag to manually reposition nodes; layout persists across sessions
- Edge routing with labelled connectors (like n8n)
- Manually add `Knowledge` nodes or annotations directly on the canvas
- Select a node → steer the agent from that context point
- Group/cluster related nodes visually

**Architecture options:**
- Keep d3 but extend it significantly (high effort, fragile)
- Switch to [React Flow](https://reactflow.dev/) or [Svelte Flow](https://svelteflow.dev/) — purpose-built for node-graph UIs, handles layout/drag/connectors out of the box
- Or build on canvas directly (Konva, PixiJS) for maximum control

**Note on framework:** This is the point where vanilla JS hits its ceiling and a lightweight framework (React + React Flow, or Svelte + Svelte Flow) becomes worth the build step. The added complexity is justified by the UI complexity.

**Suggested target:** Phase 12–13, after hosted demo and HITL bug fix.

---

## graphirm.ai hosted demo

**What:** A hosted Graphirm instance at `graphirm.ai` with rate-limited trial access. Two tiers:
1. **Static demo** — serve `web/` with a pre-recorded `demo.json` session via `?demo` query param. No server needed, host on GitHub Pages / Cloudflare Pages. Visitors see the graph visualization, chat history, and knowledge nodes without an API key.
2. **Live demo** — hosted `graphirm serve` with rate limiting + GitHub OAuth. BYOK (bring your own key) for LLM. Charge for the platform (persistent graph, hosted sessions), not API token resale.

**Phase 11 shipped the foundation:** The browser UI at `web/` works with `graphirm serve`. Adding `?demo` mode (read-only, pre-recorded session) is a small follow-up — load `demo.json` instead of calling the API, hide the input bar.

**What remains for hosted:**
- Auth layer (GitHub OAuth)
- Rate limiting (per-user session/prompt limits)
- Demo mode (`?demo` loads pre-recorded session — no API key needed)
- Landing page (what it is, install instructions, GitHub link)

**Suggested target:** Phase 12.

---

## ✅ Human-in-the-Loop — backend COMPLETE, VS Code UI pending

**Backend status (shipped 2026-03-09):**
- `HitlGate` — approve / reject / modify decisions via oneshot channels per pending tool call
- `is_destructive_tool` — gates `write`, `edit`, `bash`
- Agent loop awaits the gate before executing any destructive tool
- API routes: `POST /api/graph/:session/node/:node/action`, `POST /api/sessions/:id/pause`, `POST /api/sessions/:id/resume`
- Full test coverage (approve/reject/modify flows, pause/resume, concurrent resolution)

**What remains — VS Code extension UI only:**
Per-node approve/reject/edit buttons in the node detail panel of the VS Code extension. When a tool call is pending, the node should show action buttons that POST to the existing API routes. The server wiring is complete; this is purely a UI task.

**Why it matters:** Strongest Graphirm differentiator — the only coding agent where you can intercept and change a specific agent decision. Linear agents (Cline, OpenCode, Aider) cannot do this without rebuilding their data model.

**Suggested target:** Phase 12 (small UI addition to the existing node detail panel).

---

## ✅ DAG timeline layout for the graph visualiser - COMPLETED

**Status:** Shipped and merged to main.

**Implementation:** Full DAG timeline layout with toggle button in VS Code extension.
- Timeline mode: X-axis by timestamp (left=oldest, right=newest), Y-axis by node type + group offset
- Force mode: Traditional force-directed layout (toggle between both)
- Edge colors by type (RespondsTo=white, Reads=blue, Produces=green, Modifies=orange, DependsOn=purple, SpawnedBy=red)
- Group-aware layout: interactions + tool calls + results aligned vertically
- Full zoom/pan support with drag-to-reposition nodes

**Location:** `graphirm-vscode/media/graph.js`
- Layout modes: lines 18-19
- Type positioning: lines 22-29
- Timeline assignment: lines 166-204
- Toggle button: lines 85-95

See `docs/completion-log.md` for full details.

---

## ✅ Completed Items

See `docs/completion-log.md` for detailed implementation notes on completed features.

---


## Phase 13 and Beyond

For a comprehensive list of planned features, advanced features, and strategic directions, see `docs/backlog/phase-13-advanced-features.md`. That document contains:

- **7 Major Feature Categories** with complexity estimates and dependencies
- **Strategic Insights** on graph-native differentiation
- **Sequencing recommendations** for feature prioritization
- **Success criteria** and integration points

The backlog here focuses on items in the active pipeline. Proposed features for Phase 13+ are documented separately.
