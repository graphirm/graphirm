# Graphirm Backlog

Items captured here are validated ideas not yet scheduled into a numbered phase. Each item includes the rationale for deferral and a suggested phase target.

---

## ✅ Feature — `graph_query` tool: let the agent query its own graph — COMPLETED (Phase 12)

**What:** The agent had no way to interrogate the graph it lives in. It had `bash`, `read`, `write`, `edit`, `grep`, `find`, `ls` — all filesystem tools — but nothing that exposes the graph store.

**Shipped in Phase 12:** `crates/tools/src/graph_query.rs` — a read-only `Tool` implementation with three modes:
- `bfs` — BFS traversal from a start node following outgoing edges (configurable depth up to 10, optional edge-type filter)
- `list_type` — enumerate nodes of a given type with optional session and metadata filters
- `search` — case-insensitive keyword search over `Knowledge` nodes (entity, entity_type, summary)

**Key corrections from the original backlog note:**
- `ToolContext` already carried `Arc<GraphStore>`, `agent_id`, and `interaction_id` — no plumbing was needed
- Two new `GraphStore` helpers were added: `list_nodes_by_type` and `search_knowledge` (in `store.rs`)
- Registration is in `build_tool_registry()` in `src/main.rs`, not `ToolRegistry::new()`
- Not destructive — no HITL gate applied

**Note:** `search` in Phase 12 is keyword-only. Semantic/embedding search is preserved as a future upgrade without breaking the tool interface.

---

## Bug — graph view nodes cluster when D3 simulation settles at same position

**What:** In the browser web UI graph pane, nodes often render clustered at a single point rather than spread across the canvas. The `fitToView()` function correctly computes a bounding box and applies a zoom transform, but when all nodes share the same (x, y) after the force simulation (e.g. bounding box span is 0), they all overlap.

**Root cause (hypothesis):** D3's force simulation seeds new nodes at position (0, 0) unless given initial coordinates. When the simulation runs with `alpha` too low or too briefly, nodes don't spread before `end` fires. The `fitToView()` then computes `spanX = max(0, 40) = 40` — a degenerate bounding box — and centres them all at the same point.

**Fix direction:**
- Ensure initial node positions are seeded with jitter around the SVG centre *before* `d3.forceSimulation(nodes)` is called
- Increase initial `alpha` and possibly `alphaDecay` so the simulation runs longer before settling
- Or: after `fitToView()`, if `spanX < 1 || spanY < 1` (degenerate), apply a radial spread to force nodes apart, then re-run

**Suggested target:** Phase 12 (cosmetic but impacts perceived quality of the graph UI).

---

## ✅ Bug — HITL card shows "Agent wants to run: undefined" — FIXED

**What:** When the HITL gate triggered in the browser web UI, the approval card rendered "Agent wants to run: **undefined**" instead of the actual tool name (e.g. "bash", "read", "write").

**Root cause:** The SSE handler serialises the full `SseEvent` struct as the event data payload, so the browser receives `{ session_id, event_type, data: { node_id, tool_name, arguments, is_pause } }`. `main.js` was spreading the outer envelope (`{ ...data }`) when calling `renderApprovalCard`, so `tool_name` was always `undefined` — it was one level too deep inside `data.data`.

**Fix (shipped 2026-03-16):** In `web/main.js`, extract `payload = data?.data ?? data` before spreading, so `renderApprovalCard` receives `{ node_id, tool_name, arguments, is_pause, session_id }` at the top level. Commit `2302f59`.

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

**Suggested target:** Phase 12–13, after hosted demo (HITL bug fixed 2026-03-16).

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
