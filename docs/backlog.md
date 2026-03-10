# Graphirm Backlog

Items captured here are validated ideas not yet scheduled into a numbered phase. Each item includes the rationale for deferral and a suggested phase target.

---

## graphirm.ai hosted demo

**What:** A hosted Graphirm server at `api.graphirm.ai` with rate-limited trial access — same axum server, same API, fronted with auth (GitHub OAuth) and usage limits. The VS Code extension points to `https://api.graphirm.ai` instead of `localhost:3000` for the trial tier.

**Why deferred:** Requires infrastructure decisions not yet made: hosting (Coolify vs managed), billing (Stripe), rate limiting strategy, auth layer. Each is a week of work. Deferred until the VS Code extension is validated by real daily use and there's clarity on what to charge and what to limit.

**graphirm.ai for now:** Static landing page (GitHub Pages or Cloudflare Pages) — what it is, install instructions, link to GitHub, one-liner USP. No backend needed.

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
