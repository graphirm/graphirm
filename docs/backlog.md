# Graphirm Backlog

Items captured here are validated ideas not yet scheduled into a numbered phase. Each item includes the rationale for deferral and a suggested phase target.

---

## graphirm.ai hosted demo

**What:** A hosted Graphirm server at `api.graphirm.ai` with rate-limited trial access — same axum server, same API, fronted with auth (GitHub OAuth) and usage limits. The VS Code extension points to `https://api.graphirm.ai` instead of `localhost:3000` for the trial tier.

**Why deferred:** Requires infrastructure decisions not yet made: hosting (Coolify vs managed), billing (Stripe), rate limiting strategy, auth layer. Each is a week of work. Deferred until the VS Code extension is validated by real daily use and there's clarity on what to charge and what to limit.

**graphirm.ai for now:** Static landing page (GitHub Pages or Cloudflare Pages) — what it is, install instructions, link to GitHub, one-liner USP. No backend needed.

**Suggested target:** Phase 12.

---

## Human-in-the-Loop node controls (API + agent loop)

**What:** Per-node user actions — Approve, Reject, Retry, Edit, Skip — surfaced in the VS Code extension UI and wired through the agent loop. Every action creates a typed edge (`ApprovedBy`, `RejectedBy`, `DerivedFrom`) so the intervention is part of the graph's permanent record.

**API surface:**
- `POST /api/graph/:session_id/node/:node_id/action` — body: `{ "action": "approve" | "reject" | "retry" | "edit" | "skip", "reason": "..." }`
- Agent loop checks for steering signals between turns

**Why deferred:** The graph data model already supports this — the node/edge types are expressive enough. The UI (Phase 11) will expose nodes as clickable; the action buttons are a small addition to the node detail panel. The larger work is in the agent loop: it must poll for steering signals between turns and branch/cancel accordingly.

**Why it matters:** This is Graphirm's strongest differentiator — the only coding agent where you can point at a specific agent decision and change it. Linear agents (Cline, OpenCode, Aider) cannot do this without rebuilding their data model. Connects to `adk-graph` post-MVP (checkpointing + human-in-the-loop).

**Suggested target:** Phase 12 (after Phase 11 VS Code extension ships and node detail panel exists as a UI surface).

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
