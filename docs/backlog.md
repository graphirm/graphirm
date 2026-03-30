# Graphirm Backlog

Single source of truth for planned work. Completed items are recorded in `docs/completion-log.md` and `AGENTS.md` — not here.

**Current state:** Phases 0–35 complete. See `AGENTS.md` → Current State table.

---

## How to use this file

- Items are grouped by theme, not phase number
- Each item has a **size** (S / M / L) and a **priority** (P1 high / P2 medium / P3 low)
- When work starts, create a plan in `docs/plans/YYYY-MM-DD-<topic>.md` and link it here
- Mark items ✅ and move details to `docs/completion-log.md` when shipped

---

## Deployment & Operations

### ✅ Coolify on always-on spoke — P1 · M
Done 2026-03-18. `https://app.graphirm.ai` live. Floating IP `5.75.217.23`, snapshot ID `367758410`.
Plan: `docs/plans/2026-03-16-coolify-spoke-deployment.md`

### ✅ Chunk metrics during Layer 1 (ndstrms item 23) — P2 · S
Done 2026-03-26. Agent implemented per-chunk metrics in Layer 1: `comment_ratio`, `avg_line_length`, `blank_line_ratio`. Rust struct `ChunkMetrics` with multi-language comment detection (Python #, C++ //, C /* */). Metrics extracted in orchestrator and stored in chunk nodes. Commit: `346f804` on ndstrms.

### ✅ Persist language on chunk/file in graph export (ndstrms item 24) — P2 · S
Done 2026-03-26. Canonical `language` on CHUNK with `identified_language` kept; FILE nodes get majority language from enriched chunks; `_ensure_language_attributes` backfills from `LanguageIdentifier` when there are no semantic candidates so `graph.json` always has language where paths exist. Tests: `tests/test_language_persistence.py`. Commit: `aa98893` on ndstrms.

### ✅ Cross-file-only default for TF-IDF similarity edges (ndstrms item 25) — P2 · S
Done 2026-03-27. `cross_file_only: bool = True` added to `Layer15Config` and `Layer15Orchestrator`. Same-file edges computed internally for dominance/metrics but filtered from returned `similarity_edges` by default. Parameter threaded through `run_full_graph_pipeline()` as `layer15_cross_file_only`. 85 tests pass. Commit: `9fac65a` on ndstrms.

### ✅ Higher default similarity threshold for semantic edges (ndstrms item 26) — P2 · S
Done 2026-03-27. Default raised from `0.6` → `0.75` in `add_semantic_edges()` and `run_full_graph_pipeline()`. `inspect.signature()` test added to verify default. 48 tests pass. First clean dogfood run with new scope-boundary prompt — no out-of-scope files. Commit: `a14efb1` on ndstrms.

### ✅ Store embedding model name on semantic edges (ndstrms item 27) — P3 · S
Done 2026-03-27. `model_name: str = "codestral"` param added to `add_semantic_edges()` and threaded through `run_full_graph_pipeline()` as `semantic_model_name`. Both hardcoded `"codestral"` literals replaced with the variable. Model was already stored in edge attributes via `GraphBuilder.add_semantic_edge()`. 49 tests pass. Commit: `d0bafde` on ndstrms.

### ✅ Rabin fingerprint content_hash on chunk nodes (ndstrms item 28) — P2 · M
Done 2026-03-27. `content_hash: u64` field added to `Chunk` struct in `chunking.rs`, exposed via `#[pyo3(get)]`. `rabin_fingerprint()` private fn (polynomial hash, base 131, `wrapping_mul`/`wrapping_add`). Computed in `create_chunk()`. `contentHash` surfaced in chunk node attributes in `rust_chunking.py`. 3 new tests (skipped on this machine: pre-existing FAISS/libc++ linker issue). 87 tests pass. Commit: `5c58bd1` on ndstrms.

### ✅ MinHash + LSH banding for Layer 1.5 candidate finding (ndstrms item 29) — P2 · M
Done 2026-03-27. New `src/nodestradamus/layer1_5/minhash.py`: `token_shingles` (word k-shingles), `minhash_signature` (k min-hashes over shingle set), `lsh_candidates` (band-based bucket grouping → candidate pairs), `MinHashLSH` class. `tfidf.py`: `minhash_threshold=200` param; `analyze()` guard runs MinHash+LSH pre-filter when ≥200 chunks, routes to `_compute_similarities_on_candidates`; TF-IDF only on candidates. 9 new tests all pass. 47 layer1_5 tests pass. Commit: `e0fc2ef` on ndstrms.

### ✅ Spectral clustering on similarity graph (ndstrms item 30) — P2 · M
Done 2026-03-27. New `src/nodestradamus/layer1_5/spectral.py`: `build_adjacency_matrix` (sparse `csr_matrix`, symmetric, skips out-of-scope nodes), `compute_normalized_laplacian` (D^{-1/2}(D-A)D^{-1/2}, isolated nodes zeroed), `fiedler_eigenvectors` (`eigsh` with shift-invert `sigma=0` for stability), `assign_clusters` (`scipy.cluster.vq.kmeans2` — sklearn not available), `SpectralClusterer` class. `orchestrator.py`: added `from .spectral import SpectralClusterer` + `spectral_cluster(n_clusters=2)` method. Also updated `layer1_5/__init__.py` export (structural companion). 62 layer1_5 tests pass. Commit: `cf31d87` on ndstrms.

### ✅ PQ index (IndexIVFPQ) + random projection for Layer 2 (ndstrms item 31) — P2 · S
Done 2026-03-27. New `src/nodestradamus/layer2/projection.py`: `RandomProjection` class (seeded Gaussian JL matrix, 1024→256, `project` batch + `project_one`, deterministic). `layer2/__init__.py`: exports `RandomProjection`. `src/chunking.rs`: three-tier index selection (Flat / HNSW32 / IVF100,PQ16) with training guard for IVF/PQ. 10 new projection tests all pass; Rust tests skipped (pre-existing linker issue unchanged). Agent grade D — explored 67 messages without writing; task completed manually. Commit: `e075090` on ndstrms.

### ✅ Wire `workspaces_root` on running server — P1 · S
Done 2026-03-18. Set `workspaces_root = "/data/workspaces"` in `config/default.toml` — on the Docker volume so workspaces survive redeployments.

### ✅ CI pipeline (GitHub Actions) — P2 · S
Done 2026-03-18. `cargo fmt --check`, `cargo clippy --all-features -D warnings`, `cargo build`, and `cargo test` run on every push to `main` and every PR. Fixed fmt + clippy violations across the whole codebase to get the first green run.

---

## UI (web-app)

### ✅ Markdown rendering — P1 · S
Done 2026-03-28. Chat panel now renders markdown for assistant/tool messages using
the existing `MarkdownBody` component (`marked` + `highlight.js`, 20 languages).
User messages remain plain text. Node preview (collapsed state) strips markdown
syntax (**bold**, *italic*, `code`, [links], # headers, - lists, > blockquotes)
for clean text display. Web app builds without TypeScript errors. Commit: `877f6a8`.

### Graph-First Interaction Model

The graph is the agent. The chat panel is a fallback for users who want a traditional
interface. Power users interact directly with nodes on the canvas, navigating with
keyboard like a game world. Items below are ordered as an incremental build path —
each one is useful standalone, but together they form the paradigm shift.

#### ✅ Collapsible chat panel — P1 · S

Done 2026-03-28. Dogfood session 79c59495 + manual completion. Implementation:
- `chatCollapsed` state added to App.tsx, wired to keyboard hook (C key)
- ChatPane receives `chatCollapsed` and `onToggleCollapse` props
- Toggle button (☰ hamburger) renders in ChatPane header
- CSS `.collapsed` class: width 40px, overflow hidden, messages/input hidden
- Smooth CSS transition 0.3s ease on width changes
- GraphCanvas auto-expands via flex layout to fill freed space
- Chat state preserved (no unmounting) — scroll and input draft persist
The agent got stuck during CSS/button integration but had good architecture.
Manual fixes completed the task. Commits: 081cdab, 0513abe (min-width fix),
52d576a (toggle button moved to top), 175acb9 (instant collapse, no transition).

#### ✅ Floating command input — P1 · S

Done 2026-03-28. Dogfood session a5f12a8c + bug fix in same session.
- `FloatingInput.tsx` + `FloatingInput.module.css` — new component: hint strip at bottom
  center of canvas; `/` or `Enter` (when nothing focused) expands input; `Escape` collapses
- Sends via the same `onSend`/`handleSendWithSteer` path as the chat panel
- Shows `thinking` badge when `isThinking` is true
- `GraphCanvas` receives `chatCollapsed`, `onSend`, `isThinking` optional props
- Agent bug: used `useCallback` instead of `useEffect` for keyboard listener (dead code);
  fixed in second dogfood prompt. CSS correct — all theme variables, no hardcoded colors.
Commit: a5c1df7

**Key files:**
- `web-app/src/components/GraphCanvas.tsx` — render `FloatingInput` when `chatCollapsed`
- `web-app/src/components/FloatingInput.tsx` — new component, reuses `handleSend` logic

#### ✅ Keyboard node navigation — P1 · M

Done 2026-03-28. Dogfood session 834f6952 + manual fixes.
- `useNodeNavigation(nodes, edges)` hook: refs+empty-dep-array pattern, arrow keys follow
  `produces`/`responds_to`/`contains` edges; `↑`/`↓` move between siblings by Y position;
  first arrow key focuses first visible node; `Escape` clears
- `FocusContext.tsx`: `createContext<string|null>` + `useFocusedNodeId()` hook
- `BaseCard`: `focused?` prop + `.focused` CSS class with pulsing accent ring animation
- All 5 node components consume context and pass `focused={focusedNodeId === id}` to BaseCard
- `GraphCanvas`: `FocusContext.Provider` wraps ReactFlow; `fitView` centers on focused node
Agent bugs: missing import in AgentNode (doom loop), unclosed JSX Provider tag — both fixed manually.
Commit: 9bc20b8
- Focused node gets a pulsing highlight ring (CSS animation, `--accent` color)

**Key files:**
- `web-app/src/hooks/useNodeNavigation.ts` — new hook
- `web-app/src/components/GraphCanvas.tsx` — wire hook, pass `focusedNodeId` to nodes
- `web-app/src/styles/nodes.module.css` — `.focused` ring animation

#### ✅ Inline node interaction — P2 · M

Done 2026-03-28. Dogfood sessions 060fbdd2 (implementation) + 98e2c570 (bug fixes).
- `PopoverContext.tsx` — typed context with 5 action callbacks (steer, rate, task status, pin, edit summary)
- `NodePopover.tsx` + `NodePopover.module.css` — popover component with per-type UI, fade-in animation, theme variables
- `useNodeNavigation.ts` — Enter key activates focused node via one-shot `activateNodeId` pattern
- `GraphCanvas.tsx` — popover state, open/close, double-click handler, context provider wrapping
- `client.ts` — `rateTurn` wired to existing PATCH endpoint; `updateTaskStatus`, `toggleKnowledgePin`, `editKnowledgeSummary` are console.warn stubs pending backend endpoints
- First test of Claude Opus 4.6 + DeepSeek V3.2 model fallback chain (7 fallbacks triggered)
Commits: pending

#### ✅ HITL approval on canvas — P2 · S

Done 2026-03-28. Dogfood session `2791bd3d` + follow-up (CSS module class fix).
- `HitlOverlay.tsx` — shared approve / reject / modify UI (extracted from `ChatPane.tsx`)
- `ChatPane` + `App` + `GraphCanvas` — `pendingApproval` and handlers wired; chat still shows HITL when expanded (redundancy with canvas)
- Canvas: `HitlOverlay` in `canvasWrapper` with `hitlCanvasOverlay` (bottom-center strip, scroll). **Not** a `NodeToolbar` anchored to `node_id` — future polish if desired
- Pending node: React Flow `className` + `nodes.module.css` `.pendingApproval` — warning pulse ring on the node matching `pendingApproval.node_id`

#### ✅ LOD (level-of-detail) zoom — P2 · S

Done 2026-03-28. Dogfood session 24612fe9 + CSS fix. Grade: partial.
- `ZoomContext.tsx` — `useViewport()` + 150ms debounced threshold, `isLODEnabled` flag
- `ZoomProvider` wraps `ReactFlow` inside `GraphCanvas` (required for hook to work)
- All 5 main node types: `isLODEnabled ? false : expanded` override; `onToggleExpand` no-op during LOD
- User expand preferences fully preserved: blocked during LOD, restored on zoom-in
- AnnotationNode: 1-row compact textarea during LOD
- GroupNode: collapse button hidden during LOD
- Agent bug: referenced `annotationCardLOD` / `groupLOD` CSS classes without defining them (undefined in className). Fixed manually with opacity:0.7 stub rules.
Commits: `2a6ebc5` (agent), `fix` (CSS stubs)

#### ✅ Timeline swimlane backgrounds — P3 · S

Done 2026-03-28. Dogfood session `4610758d` + coordinate-system comment.
- `timeline.ts` — `export const TYPE_Y`, `TYPE_LABELS` (redundant labels; kept for API symmetry)
- `GraphCanvas.tsx` — when `layoutMode === 'timeline'`, `swimlaneContainer` + per-type strips using `cssVar(--node-*)` and `TYPE_Y` for `top`; screen-fixed overlay (does not pan/zoom with flow) — comment in source
- `GraphCanvas.module.css` — `.swimlaneContainer`, `.swimlane` (pointer-events none, z-index 0)

#### ✅ Node quick-reply — P3 · M

Done 2026-03-29. Dogfood session `a679fbca` + doom loop bug fix.
- `NodeReplyInput.tsx` + `NodeReplyInput.module.css` — inline reply input (textarea + Send/Cancel buttons, Escape dismisses, Enter sends, auto-focus, thinking badge)
- `useNodeNavigation.ts` — R key handler: only activates for `interaction` node type, sets `replyingToNodeId` state
- `GraphCanvas.tsx` — `replyingToNodeId` + `clearReply` from hook; `useEffect` sets steer context via `onSteerFromNode`; `NodeReplyInput` rendered with node position
- Position uses graph coordinates as CSS `left`/`top` (same pre-existing pattern as `NodePopover` — decorative only, does not track pan/zoom)
- **Bug found + fixed:** Doom loop advisory re-fired every turn after threshold because `count == threshold` iterated all accumulated counts. Fixed: collect `edited_this_turn`/`read_this_turn` vecs, only check those paths against thresholds.

#### ✅ Timeline cascade layout with role-based node sizing — P1 · M

Done 2026-03-29. `buildTurns()` partitions Interaction nodes into turns by user-message boundaries; classifies each as main-row (user/final-assistant) or cascade (intermediate tool/assistant-with-tool-calls). `applyTimelineLayout()` rewritten: user + final-assistant on main row at Y=80, intermediates cascade diagonally (60px X-step, 50px Y-step). Non-Interaction nodes pushed to dynamic Y bands below cascade zone. Returns `TimelineLayoutResult { nodes, bandPositions }` — swimlane heights adapt to actual cascade depth. Compact cards (160×50px, role icon + tool name/preview) click to expand in-place. Commit: `12a0327`.

**Key files:**
- `web-app/src/layout/timeline.ts` — `buildTurns()`, cascade algorithm, `TimelineLayoutResult` return type
- `web-app/src/hooks/useGraphData.ts` — `bandPositions` state, mode-aware `setLayoutMode`
- `web-app/src/components/GraphCanvas.tsx` — dynamic swimlanes from `bandPositions`
- `web-app/src/components/nodes/InteractionNode.tsx` — compact card rendering via `data.compact`
- `web-app/src/styles/nodes.module.css` — `.compactCard`, `.roleIcon`, `.compactLabel`

#### ✅ Fix role-based color split for Interaction nodes — P2 · S

Done 2026-03-29. `ROLE_COLORS` map at module level: `user → --accent`, `assistant → --node-agent` (pink), `tool → --node-content` (teal), `system → --fg-muted`. Applied to both full cards (`BaseCard` `color` prop) and compact cascade cards (`borderLeftColor` inline style). Commit: `e77af09`.

#### ✅ Highlight file-writing tool nodes with destructive color — P1 · S

Done 2026-03-30. `tool_name` from metadata (case-insensitive) matched against `write` / `edit` /
`bash` when `role === 'tool'` → `color` is `var(--warning)` for both compact cascade cards
(`--compact-color`) and full `BaseCard` borders.

**Key files:**
- `web-app/src/components/nodes/InteractionNode.tsx` — `DESTRUCTIVE_TOOL_NAMES`, `isDestructiveTool`

#### ✅ Collapse-all cascade nodes button — P2 · S

Done 2026-03-30. `CascadeCollapseGenerationContext` holds a monotonic counter; `GraphCanvasInner`
increments it when Toolbar **⊖ Collapse all** is clicked (Timeline layout only). Each
`InteractionNode` runs `useEffect` on that value to `setLocalExpanded(false)`.

**Key files:**
- `web-app/src/context/CascadeCollapseContext.tsx` — context + `useCascadeCollapseGeneration`
- `web-app/src/components/nodes/InteractionNode.tsx` — effect on generation
- `web-app/src/components/Toolbar.tsx` — `onCollapseTimelineCascades` button
- `web-app/src/components/GraphCanvas.tsx` — provider + state + toolbar wiring

#### Node editing and annotations — P2 · M

Per-type interaction for editing, dismissal, and annotation:

- **User Interaction nodes**: editable inline. On save, the original node is marked `metadata.edited = true` with the original content backed up, and a new prompt is sent from the preceding node's context root — effectively forking the conversation. Both branches remain visible in the graph.
- **Knowledge nodes**: dismissable (sets `metadata.dismissed = true`; filtered from `build_context` and `build_repo_briefing` without deleting the node) and summary-editable (`PATCH /api/knowledge/{id}` with `{ summary }`).
- **Tool Interaction nodes**: annotatable. User adds a freeform note (e.g., "next time prefer grep over find") stored as a Knowledge node with `entity_type: "annotation"` linked to the tool node via a `RelatesTo` edge. Surfaces on the canvas as a connected Knowledge node.

Requires new backend endpoints: `PATCH /api/knowledge/{id}` (dismissed + summary), `PATCH /api/interactions/{id}/edit` (metadata marking), and extension of `POST /api/graph/{session_id}/annotate` with optional `relates_to` node ID.

Plan: `docs/plans/2026-03-29-node-editing-annotations.md`

**Key files:**
- `web-app/src/components/nodes/NodePopover.tsx` — Edit / Dismiss / Annotate actions per type
- `web-app/src/components/nodes/InteractionNode.tsx` — inline edit mode for user messages
- `web-app/src/api/client.ts` — `patchKnowledge()`, `markInteractionEdited()`, `annotateNode()`
- `crates/server/src/routes.rs` — `PATCH /api/knowledge/:id`, `PATCH /api/interactions/:id/edit`, extend annotate handler
- `crates/graph/src/store.rs` — `update_knowledge()` (dismissed flag + summary)
- `crates/agent/src/briefing.rs` + `context.rs` — filter dismissed Knowledge nodes

#### Node-as-input (canvas prompt nodes) — P2 · M

Double-clicking empty canvas creates a `PromptNode` — a temporary node with a textarea and Send/Cancel buttons. On send, the text is forwarded to the agent via the existing `POST /api/sessions/{id}/prompt`, optionally with a `context_root` set by drag-connecting from an existing node to the PromptNode's left handle. The PromptNode is removed from local state immediately on send; the SSE stream brings the real Interaction node. If the user Escapes or clicks Cancel, the node is removed with no side effects.

No backend changes required — uses existing prompt and SSE infrastructure. The current annotation-on-double-click behavior moves exclusively to the toolbar "+ Note" button. A "+ Prompt" toolbar button is added as an alternative to double-click. Double-click remains the primary gesture.

Plan: `docs/plans/2026-03-29-node-as-input.md`

**Key files:**
- `web-app/src/components/nodes/PromptNode.tsx` — new node type (editable textarea, Send/Cancel, target handle)
- `web-app/src/components/nodes/PromptNode.module.css` — dashed accent border, compact layout
- `web-app/src/components/GraphCanvas.tsx` — register node type, double-click handler, `onConnect` for context root, prompt send/cancel handlers
- `web-app/src/components/Toolbar.tsx` — "+ Prompt" button

### Pretext Integration (`@chenglou/pretext`)

Pure JS/TS library for DOM-free multiline text measurement & layout. `prepare()` does
one-time segment measurement via Canvas `measureText()`; `layout()` is arithmetic-only
on cached widths — 480× faster than DOM interleaved in Chrome, 1240× in Safari. 7680/7680
browser accuracy across Chrome/Safari/Firefox. MIT, 13.7k stars, `npm install @chenglou/pretext`.

See: https://github.com/chenglou/pretext — README, RESEARCH.md, STATUS.md.

#### Hybrid Canvas/DOM rendering — P2 · L

Every graph node is a full React DOM tree today. At 200+ nodes, React Flow struggles.
Pretext's `layoutWithLines()` gives line-by-line text data. Combined with Canvas `fillText()`,
render **all non-focused nodes on a single Canvas layer** and only mount real DOM for the
1–3 nodes the user is interacting with. The LOD zoom system (Phase 27) already collapses
nodes at low zoom — the leap: draw them to canvas instead of mounting collapsed React
components. Zero DOM cost. Pan/zoom becomes a single `drawImage()` call.

This is how professional graph tools (yFiles, Cytoscape) achieve 10k+ node performance,
but they cheat with rectangles. Pretext enables it with real, readable, accurately measured text.

**Key files:**
- `web-app/src/components/GraphCanvas.tsx` — Canvas overlay layer, LOD threshold switch
- `web-app/src/hooks/useGraphData.ts` — cache `PreparedText` per node alongside React Flow state
- `web-app/src/layout/dagre.ts` — pre-compute exact dimensions from Pretext before dagre runs

#### ✅ Shrink-wrap balanced nodes (Pretext `layout` line-count search) — P2 · M

Done 2026-03-30. `shrinkWrapInnerWidth()` binary-searches the minimum inner width where the
preview fits in ≤2 lines (same break rules as `walkLineRanges` / `layout()`). Outer width =
inner + 22px chrome, clamped 160–280 (`theme.css` card min/max). Height recomputed at that
inner width. Wired via existing `buildPretextSizeMap()` → dagre.

**Key files:**
- `web-app/src/layout/pretextDimensions.ts` — `shrinkWrapInnerWidth`, `estimateSizeFromPreview`

#### ✅ Accurate dagre first-pass layout — P1 · M

Done 2026-03-30. `buildPretextSizeMap()` runs `prepare()` + `layout()` on the same preview
strings as collapsed `BaseCard` (Interaction stripMarkdown+80 chars, Content/Task/Agent/
Knowledge parity with node components). Dagre height uses Pretext output capped at 2 preview
lines + chrome; width stays per-type. LRU-ish `prepare` cache (400 entries). Fallbacks if
Canvas/`measureText` throws. `nodeDimensions.ts` holds shared `NODE_DIMENSIONS`.

**Key files:**
- `web-app/src/layout/pretextDimensions.ts` — preview text, cache, `buildPretextSizeMap`
- `web-app/src/layout/nodeDimensions.ts` — shared fallbacks
- `web-app/src/layout/dagre.ts` — optional `pretextSizes` map
- `web-app/src/hooks/useGraphData.ts` — wires map into `applyDagreLayout`

#### Streaming pre-size during SSE — P2 · M (blocked on backend graph writes)

Assistant `Interaction` nodes are persisted once per turn with full `text_content()` in
`workflow.rs` — there is no incremental graph node during LLM token streaming. `MessageDelta`
exists in `AgentEvent` but `agent_event_to_sse` maps it to `heartbeat`, so the web app never
receives token chunks. **To implement:** emit real `message_delta` SSE (and optionally persist
or shadow a provisional node), then `useSession` can drive Pretext `prepare`/`layout` on partial
text and pass predicted sizes into `useGraphData`.

**Key files:**
- `crates/server/src/routes.rs` — `agent_event_to_sse`: `MessageStart` / `MessageDelta` / `MessageEnd`
- `crates/agent/src/workflow.rs` — if/when true streaming records partial content
- `web-app/src/hooks/useSession.ts` — handle `message_delta`, optional predicted-size context
- `web-app/src/hooks/useGraphData.ts` — merge predicted dimensions during `isThinking`

#### Text-aware edge avoidance — P3 · M (partial, 2026-03-30)

**Done this pass:** Edge labels nudged ~11px perpendicular to the chord (source→target) so
they sit slightly off the stroke; pill background + border using theme CSS variables for
contrast on light/dark. **Still open:** Pretext `walkLineRanges` / node-local anchors so
labels clear dense card text.

**Key files:**
- `web-app/src/components/edges/LabelledEdge.tsx` — nudge + pill styles
- `web-app/src/layout/pretextDimensions.ts` — future: export per-line widths for handles

#### Graph-as-image export without headless browser — P3 · M

When Pretext ships server-side support (noted as planned), generate visual graph thumbnails
entirely in Node/Rust without Puppeteer. `prepare()` + `layoutWithLines()` gives exact text
positions; draw them into an SVG or PNG with dagre coordinates. The session export (Phase 21)
currently outputs markdown — could also output a visual snapshot.

**Key files:**
- `crates/server/src/export.rs` — SVG/PNG export format option alongside markdown
- `web-app/src/layout/dagre.ts` — shared layout logic between browser and server

#### Agent-side content verification — P3 · S

The agent uses Pretext to sanity-check its own outputs before writing. Before emitting a
markdown table, verify it renders without wrapping at 80 columns. Before writing a code
block, check it fits the user's terminal width. The pre-completion verify hook (Phase 42)
could include a layout check — catching visual regressions before the user sees them.

**Key files:**
- `crates/tools/src/write.rs` — optional Pretext-based width check before file write
- `crates/agent/src/workflow.rs` — layout verification in pre-completion hook

#### Obstacle-aware text flow around expanded nodes — P2 · L

Inspired by the **Dynamic Layout** + **Editorial Engine** demos (obstacle routing with
`layoutNextLine` + slot carving). When a node is expanded (300+ px tall with full markdown),
neighboring collapsed nodes' text could flow *around* the expanded node — like a magazine
article flowing around an image. Implementation: each expanded node becomes a rectangular
obstacle; `carveTextLineSlots()` (from editorial engine pattern) subtracts obstacle intervals
from each Y band; `layoutNextLine()` emits text into remaining slots with variable widths
per line. This replaces the current "everything shifts out of the way" dagre behavior with
a dense, editorial-quality canvas where nothing moves — content reflows in place.

Even a simpler version works: when a node card is tall enough, its edge labels could use
`layoutNextLine()` with width constrained by the gap between overlapping nodes, avoiding
text-on-text overlap that happens today with SmoothStep labels.

Refs: `chenglou.me/pretext/dynamic-layout`, `chenglou.me/pretext/editorial-engine`

**Key files:**
- `web-app/src/layout/dagre.ts` — obstacle extraction from expanded node rects
- `web-app/src/components/edges/LabelledEdge.tsx` — variable-width label lines
- `web-app/src/layout/pretextDimensions.ts` — Pretext `layoutNextLine` integration

#### ✅ Masonry card layout mode — P2 · M

Done 2026-03-30. Fourth `LayoutMode`: shortest-column packing, card height from Pretext
`layout()` per node (full body text, capped 24 lines + header chrome), fixed column width.
Sorted by `created_at`. Annotations stack on the right. Toolbar + `L` cycle include Masonry.

**Key files:**
- `web-app/src/layout/masonry.ts` — `applyMasonryLayout`
- `web-app/src/hooks/useGraphData.ts` — `LayoutMode`, `applyLayout`
- `web-app/src/components/Toolbar.tsx`, `GraphCanvas.tsx`

#### ✅ Expandable card accordion (Pretext reserve height) — P2 · S (partial)

Done 2026-03-30 / extended same day. Shared `estimateExpandedPlainReserveHeight()`; per-type
caps/slack match main body widgets (`MarkdownBody` 320px, `CodeBody` 360px, etc.). Wired on
Interaction, Content, Knowledge, Task, Agent. **Not yet:** height animation from 0 (expanded
body still mounts on expand).

Ref: `chenglou.me/pretext/accordion`

**Key files:**
- `web-app/src/layout/pretextDimensions.ts` — `estimateExpandedPlainReserveHeight`, `estimateInteractionExpandedReserveHeight`
- `web-app/src/components/nodes/BaseCard.tsx` — `expandedBodyStyle`
- `web-app/src/components/nodes/InteractionNode.tsx`, `ContentNode.tsx`, `KnowledgeNode.tsx`, `TaskNode.tsx`, `AgentNode.tsx`

#### Rich inline chips in node cards — P3 · M (partial, 2026-03-30)

**Done this pass:** DOM-based collapsed previews (no Pretext `layoutNextLine` yet):
- **Interaction:** inline `` `code` `` → `.previewCode`; **`@handle`** tokens (after markdown
  strip) → `.previewMention`; remaining text markdown-stripped per segment. `BaseCard`
  `previewNode` + truncated `previewTitle`.
- **Knowledge:** `entity_type` as `.previewChip` + truncated entity string.
- **BaseCard:** optional `previewNode`, `previewTitle`; root `title` for hover.

**Still open:** Markdown link underlines in preview, Pretext `prepareWithSegments` /
`layoutNextLine` for measurement-aligned multi-run layout (rich-note demo parity).

Ref: `chenglou.me/pretext/rich-note`

**Key files:**
- `web-app/src/components/nodes/RichPreview.tsx` — `parseInteractionPreviewRuns`, previews
- `web-app/src/components/nodes/BaseCard.tsx` — `previewNode` / `previewTitle`
- `web-app/src/styles/nodes.module.css` — `.previewCode`, `.previewChip`

#### Virtualized graph with Pretext height prediction — P1 · L (partial, 2026-03-30)

**Done:** `onlyRenderVisibleElements` on `ReactFlow`; after **dagre**,
`mergePretextNodeDimensions()` stamps sizes. **Masonry** via `applyMasonryLayout`. After
**timeline**, `mergePretextNodeHeightsOnly()` stamps Pretext height + fixed timeline widths
(full + compact); top-level `Node.width` / `height` everywhere those merges run.

**Done 2026-03-30:** Top-level `Node.width` / `Node.height` set together with `style` in
`mergePretextNodeDimensions`, `mergePretextNodeHeightsOnly` (timeline: full cards
`TIMELINE_NODE_WIDTH`, compact `TIMELINE_COMPACT_*`), and masonry `applyMasonryLayout`.

**Still open:** large-session perf verification in DevTools.

Ref: `chenglou.me/pretext/masonry` (height-before-mount principle)

**Key files:**
- `web-app/src/components/GraphCanvas.tsx` — `onlyRenderVisibleElements`
- `web-app/src/hooks/useGraphData.ts` — merge after dagre / timeline
- `web-app/src/layout/pretextDimensions.ts` — merge helpers
- `web-app/src/layout/masonry.ts` — `width`/`height` on laid-out nodes
- `web-app/src/layout/timeline.ts` — exported `TIMELINE_*` sizing constants

### Smart Layout Engine

The current layout is functional but naive — dagre re-runs on every SSE event with
hardcoded `200×80` node dimensions, no animation, and no collision detection. These
items make the graph feel alive and stable during live agent sessions.

#### ✅ Layout stability on live updates — P1 · M

Done 2026-03-28. Dogfood session b2356651 + manual completion. Core implementation:
- `positionNewNodes()` helper placed new nodes relative to parents based on edge graph
- `isPatchUpdate` parameter added to useGraphData hook to detect SSE patches
- useEffect logic splits: skip full layout on patch update, use incremental positioning
- TypeScript compilation fixed (removed duplicate XYPosition interface)
- Web app builds successfully without errors
The agent got stuck in a read loop during integration but the helper function logic was solid.
Manual fixes completed the task. Commits 3d1e158, c7014f5.

#### ✅ Actual node dimensions in dagre — P1 · S

Done 2026-03-28. Dogfood session 59c384ed. Grade: pass.
- `getNodeDimensions(node)` reads `node.measured.width`/`node.measured.height` when available
- Per-type fallback estimates: Interaction 220×100, Knowledge 180×60, Content 200×80, Task 180×70, Agent 160×60
- Position centering formula updated (`pos.x - dims.width/2`) consistently
- Minimal footprint: only `dagre.ts` modified — no `useNodesInitialized` wiring needed (measured is just undefined pre-render)
- Build passed on first attempt
Commit: `422aea7`

#### ✅ Animated layout transitions — P2 · S

Done 2026-03-28. Manual (dogfood failed 3 sessions — agent looped on relative paths, never wrote).
- One line appended to `web-app/src/styles/theme.css`: `.react-flow__node { transition: transform 0.3s ease; }`
- React Flow suspends CSS transitions during drag automatically — no drag-and-drop regression
- Build passes
Commit: see next commit

#### ✅ Focus-and-context zoom — P2 · S

Done 2026-03-28. Dogfood session b2046968 + manual fixes. Grade: partial.
- `selectedNodeId` + `onNodeSelect` prop on `GraphCanvas`, wired from `App.tsx`
- `dimmedNodeIds`: `useMemo` computes 1-hop neighbor set; non-neighbors get `opacity: 0.25`
- `dimmedEdgeIds`: edges where both endpoints are dimmed also get `opacity: 0.25`
- `handlePaneClick` clears selection; Escape key listener via `useEffect`
- Agent doom-looped on GraphCanvas (5 edits); left unused `Edge` import + `handleKeyDown` defined but never attached
- Manual fixes: removed import, wired handler via `useEffect`
Commit: `6940b28`

#### ✅ Timeline collision avoidance — P3 · S

Done 2026-03-29. Two-pass layout in `applyTimelineLayout()`: pass 1 computes natural
X/Y from timestamps; pass 2 groups nodes by type band, sorts by X, and nudges any
node whose natural X would overlap the previous node rightward. Commits: `02e80f5`.

**Overhauled** same day (`ddff4e6`): collision widths raised from 180-240px to 280px
(matching CSS `--card-max-width`), gap from 16→32px, band spacing from 80→140px.
Group nodes disabled in Timeline mode (groups only make sense for dagre — in timeline
they caused parent-relative coordinate misalignment and overlapping cards). Removed
`computeNodeGroups` and depth-based Y offset entirely. Edge labels hidden when
zoomed below 0.6x to reduce visual noise. Swimlane backgrounds updated to 120px
height centered on new TYPE_Y positions.

**Key files:**
- `web-app/src/layout/timeline.ts` — two-pass layout, no groups, 280px collision widths
- `web-app/src/hooks/useGraphData.ts` — conditional `buildGroups` (dagre only)
- `web-app/src/components/edges/LabelledEdge.tsx` — zoom-based label visibility

#### ✅ Type-aware spacing in dagre — P3 · S

Done 2026-03-29. `NODE_DIMENSIONS` in `dagre.ts` already had per-type fallbacks from Phase "actual node dimensions". Updated estimates to match real rendered sizes: Interaction 220×120, Agent 240×70, Knowledge/Content 180×60. Build passes.

#### ✅ Fix blank canvas — `onNodesChange` dropping dimension updates — P1 · S

Done 2026-03-29. Graph canvas was completely blank despite `useGraphData` correctly
computing 14 nodes with dagre positions and passing them to `<ReactFlow>`. Root cause:
custom `onNodesChange` handler only applied `position` changes, silently dropping
`dimensions`, `select`, `reset`, and all other change types. React Flow v12 measures
each node via ResizeObserver and reports dimensions back through `onNodesChange` — without
applying those, React Flow couldn't finalize node rendering. Fix: replaced the hand-rolled
handler with `applyNodeChanges` from `@xyflow/react`. Minimap was already rendering
(proving React Flow knew about nodes), but actual node cards were invisible.

**Key files:**
- `web-app/src/hooks/useGraphData.ts` — import `applyNodeChanges`, replace custom handler

---

### ✅ Design system + light/dark theme — P3 · S
Done 2026-03-21. Expanded design tokens in `theme.css` (spacing scale, typography scale, surface layers, semantic colors, all edge color variables). Light/dark theme via `useTheme` hook (`localStorage` + `prefers-color-scheme`, sets `data-theme` on `<html>`). Toggle button (☀/◉) in Toolbar. Edge colors DRYed — `EDGE_COLORS` map removed from `LabelledEdge.tsx`, replaced with `getEdgeColor()` reading CSS variables via `getComputedStyle` with theme-aware cache. Clean build, 100% agent.
Plan: `docs/plans/2026-03-22-research-driven-improvements.md`

### ✅ Theme redesign + full CSS variable audit — P2 · M
Done 2026-03-27. Replaced entire dark/light palette with Linear/Raycast/Primer-inspired token system: warm off-white `#f8f7f4` light bg, near-black `#161616` dark bg, soft pastels for dark node colors, rich deep tones for light node colors. Added `--node-X-fg` foreground pairs. Fixed all hardcoded dark hex values across `chat.module.css` (message bubbles, HITL card, textarea), `nodes.module.css` (card bg, status badges, annotation), `Toolbar.module.css` (layout group, search input, type buttons), `GraphCanvas.tsx` (MiniMap bg/mask, Controls, Background dots, nodeColor map). All components now respond correctly to theme toggle. Edge colors recalibrated per theme. Commits: `0d80a92`, `f54c878`.

### ✅ Color-coded nodes and chat cards by role/type — P2 · S
Done 2026-03-27. Node cards: colored left border stripe + 12% tinted background per node type via `color-mix()` in `BaseCard.tsx`. Interaction nodes split by role: user=`--accent`, assistant=`--node-agent`. Chat panel: user messages are accent-colored pill bubbles (right-aligned); assistant messages have rose/pink tint bg + `--node-agent` left border; tool output has teal tint + `--node-content` left border. Role labels styled with sender's color. MiniMap node dots read live CSS variable values. Commits: `c5141d4`, `e906686`, `5f4aa5d`.

### ✅ Workspace selector in SessionBar — P2 · S
Done 2026-03-18. "+ New" button expands an inline form with session name + workspace inputs. Enter submits, Escape cancels, promise awaited before closing.

### ✅ Real-time graph updates via SSE — P2 · M
Done 2026-03-19. `GraphUpdate` SSE payload now includes full `nodes` + `edges` patch data. Web-app applies incremental patches to React Flow state — no full re-fetch, canvas positions preserved. `message_end` refreshes messages only. `agent_end` uses a 500 ms debounced reconciliation refresh. Also fixed `@dagrejs/dagre` 1.1.8 → 1.0.4 (broken graphlib packaging).
Plan: `docs/plans/2026-03-18-p2-sse-knowledge-plugins.md`

### ✅ Session rename — P3 · S
Done 2026-03-19. `PATCH /api/sessions/:id` with `{ "name": "new name" }` persists to Agent graph node (survives restart). `display_name: Arc<RwLock<String>>` in `SessionHandle`. `name` field added to `SessionResponse` (also fixes pre-existing bug where `Session.name` was always undefined in the UI). Inline double-click edit in `SessionBar` — Enter commits, Escape cancels, blur commits.
Plan: `docs/plans/2026-03-19-session-rename.md`

### ✅ Graph node search / filter — P3 · M
Done 2026-03-19. Search input + type filter pills (`I A C T K`) in Toolbar. `useGraphData` applies `hidden: true` to non-matching React Flow nodes via `applyFilterToNodes` helper; group nodes hidden when all children hidden; annotation nodes never hidden. `matchCount/total` counter shown when filter is active. Clear `✕` button. Ctrl+F (hover) focuses search input, Escape clears and blurs. No backend changes.
Plan: `docs/plans/2026-03-19-graph-node-search.md`

### ✅ Export session as Markdown / HTML — P3 · M
Done 2026-03-19. `GET /api/sessions/:id/export?format=markdown` returns conversation (user + assistant turns, tool interactions excluded) + extracted knowledge table as a `.md` download (`Content-Disposition: attachment`). `format=html` → 400. New `crates/server/src/export.rs` with `render_session_markdown` (5 unit tests). "↓ Export" button in `SessionBar` triggers browser download via `window.open`.
Plan: `docs/plans/2026-03-19-export-session.md`

---

## Agent Capability

### ✅ Context auto-compaction trigger — P2 · S
Done 2026-03-21. `select_nodes_for_compaction` in `compact.rs` — walks conversation thread, filters already-compacted nodes, compares token estimate to `compaction_threshold` (default 0.80), skips `guaranteed_recent_turns` newest, returns oldest eligible IDs when `>= min_nodes_to_compact`. Hook in `stream_and_record`: runs selection in `spawn_blocking` then awaits `compact_context` synchronously; non-fatal (errors logged, skipped). `compaction_threshold: f64` added to `ContextConfig`. 4 new unit tests. Enable: `enable_compaction = true` in `[context]` config.
Plan: `docs/plans/2026-03-22-research-driven-improvements.md`

### ✅ Workspace context injection at session start — P2 · S
Done 2026-03-18. `build_workspace_context` in `routes.rs` lists up to 20 sorted entries and appends `## Active Workspace` block to `config.system_prompt` at session creation time. Non-fatal, warns on errors.

### ✅ HTTP-level test for per-session workspaces — P2 · S
Done 2026-03-18. `test_workspace_creation` in `crates/server/tests/integration.rs` verifies response fields and directory existence on disk via `tempfile::tempdir()`.

### ✅ Subagent workspace isolation — P3 · M
Done 2026-03-19. `spawn_subagent` accepts `parent_working_dir: Option<PathBuf>`; when set, creates `<parent>/subagents/<agent>-<short_id>/` and sets subagent `working_dir`. Delegate passes `ctx.working_dir`. Integration test verifies subagent tool runs in workspace.
Plan: `docs/plans/2026-03-19-agent-capability-subagent-ws-multifile.md`

### ✅ Multi-file context tool (`read_many` / `diff`) — P3 · M
Done 2026-03-19. `diff` tool: file compare (`file_a`/`file_b`) and git diff (optional `ref`/`path`/`cached`), non-destructive. `read_many` tool: up to 20 paths, optional `max_lines_per_file`, concatenated output with path headers; non-destructive. Both registered in `build_tool_registry`.
Plan: `docs/plans/2026-03-19-agent-capability-subagent-ws-multifile.md`

### ✅ Semantic `graph_query` search — P3 · M
Done 2026-03-19. Added `semantic` mode to `graph_query` tool. `KnowledgeRetriever` trait defined in `graphirm-tools` (no circular deps); `MemoryRetriever` implements it via `retrieve_with_scores` (correct L2→cosine conversion: `1 - d²/2`); wired via `ToolContext.knowledge_retriever`. Returns HNSW-ranked Knowledge nodes with cosine similarity scores. Graceful `ExecutionFailed` when no embedding provider is configured.
Plan: `docs/plans/2026-03-19-semantic-graph-query.md`

---

## Harness Engineering

Systematic improvements to the agent loop infrastructure — the scaffolding around the model
that determines reliability on long-running, multi-step tasks. Inspired by LangChain's
harness engineering research (52.8% → 66.5% on Terminal Bench by changing only the harness)
and Phil Schmid's "Agent Harness as Operating System" framing.

### ✅ Pre-completion verification hook — P1 · M
Done 2026-03-27. After a text-only turn that follows tool work, the agent loop injects a user message with a 5-point verification checklist (run tests, run clippy, re-read requirements, check git diff). Fires once per session via `verify_injected: bool` guard. `pre_completion_verify: bool` config field (default true) — set false in existing unit tests that use mock providers. 2 new config tests added; all 265 agent tests pass, clippy clean. **Bug fix 2026-03-27:** Changed trigger condition from `had_tool_calls` (any tool, including reads) to `had_write_calls` (write/edit only) — prevents premature firing during read-only planning turns. `had_write_calls: bool` flag tracked alongside doom-loop path extraction.

### ✅ Doom loop detection — P1 · S
Done 2026-03-27. `file_edit_counts: HashMap<PathBuf, u32>` tracked in `run_agent_loop`; incremented on every `write`/`edit` tool call by extracting the `path` arg from tool arguments. When a file's count equals `doom_loop_threshold`, a user message advisory is injected urging the agent to step back and reconsider. Fires on each threshold crossing (not just once). `doom_loop_threshold: u32` config field (default 5, 0 disables). 2 new config tests; 267 agent tests pass, clippy clean.

### ✅ Phase-aware reasoning budget — P2 · M
Done 2026-03-27. `TaskPhase` enum (`Planning`/`Implementation`/`Verification`) added to `router.rs`; `task_phase: TaskPhase` field added to `TurnSignals`; new `RoutingRule::PhaseMatch { phase, tier }` variant (TOML: `type = "phase_match"`). Phase inferred in `infer_task_phase()` from session chain tool result metadata (`tool_name`): no write/edit calls → Planning; write/edit exist but last 5 calls are read-only → Verification; otherwise Implementation. Applied in both adaptive and legacy `spawn_blocking` router blocks. 5 new router tests; 273 agent tests pass, clippy clean.

### ✅ Token/time budget awareness — P2 · S
Done 2026-03-27. In `stream_and_record`, after `build_context_with_stats` returns, computes `usage_ratio = window.total_tokens / max_tok`. Finds the highest crossed threshold from `budget_warning_thresholds` and appends a one-line warning to `context[0]` (the system message) via `ContentPart::text`. Two tiers: <90% → "wrap up", ≥90% → "complete current step only". `budget_warning_thresholds: Vec<f64>` config field (default [0.7, 0.9]; empty list disables). 2 new tests; 269 agent tests pass, clippy clean.

### Automated trace analysis loop — P3 · L
Build an agent (or tool) that analyzes failure patterns across sessions and suggests
harness improvements. The graph already stores everything — tool calls, outcomes, errors,
context stats (Phase 37). Missing piece is the analysis loop that mines this data for
systematic failure modes (e.g. "agent skips tests in 60% of sessions", "doom loops on
Rust lifetime errors").

**Implementation:** New `trace_analyzer` tool or standalone command that queries completed
sessions, clusters failure patterns (repeated errors, long tool chains, abandoned approaches),
and outputs a structured report with suggested harness parameter changes.

**Key files:**
- `crates/tools/src/trace_analyzer.rs` — new tool
- `crates/agent/src/workflow.rs` — `TurnOutcome` metadata (Phase 36) is the data source

### ✅ Structured work loop enforcement — P3 · S
Done 2026-03-27. `enforce_work_loop: bool` config field (default true). When enabled, `create_session` in `crates/server/src/routes.rs` appends a "## Problem-Solving Framework" section to the system prompt: 4-step Plan→Build→Verify→Fix with explicit instruction to transition from Plan to Build after at most 2 messages. 2 new config tests; all agent + server tests pass, clippy clean.

---

## Infrastructure & Quality

### ✅ Cross-session knowledge extraction — P2 · L
Done 2026-03-19. Knowledge nodes now store `session_id` in metadata. After each embedding, `MemoryRetriever.find_cross_session_links` queries HNSW (min 0.7 cosine similarity, top 3) and `persist_cross_session_links` writes `RelatesTo` edges with similarity as weight. Three unit tests added.
Plan: `docs/plans/2026-03-18-p2-sse-knowledge-plugins.md`

### ✅ Custom tool plugins — P2 · L
Done 2026-03-19. `ScriptTool` loads `plugin.toml` manifests from `~/.graphirm/plugins/` (or `GRAPHIRM_PLUGINS_DIR`). Each plugin defines name, description, `command`, `destructive` flag, and JSON Schema parameters. Args passed as `GRAPHIRM_ARGS` (JSON) + `GRAPHIRM_ARG_<KEY>` env vars. `is_destructive()` added to `Tool` trait; overridden in `bash`/`write`/`edit` and respected by HITL gate alongside the built-in name list. Example plugin at `examples/plugins/hello/`.
Plan: `docs/plans/2026-03-18-p2-sse-knowledge-plugins.md`

### ✅ Agent Trace ingestion — Cursor import — P3 · M
Done 2026-03-22. `graphirm import-cursor <path>` imports Cursor `.txt` transcript files (one per conversation) into the graph as `Agent` + `Interaction` nodes with `Produces` and `RespondsTo` edges. Accepts a single file or a directory. Idempotent — re-importing the same file is a no-op (checked via `source_file` on the synthetic Agent node). Thinking blocks preserved in `metadata["thinking"]`; tool call/result blocks stripped. Parser: `crates/agent/src/import/cursor.rs` — state machine, 8 unit tests, zero new deps. 100% agent (parser + write function) + manual fix (trailing-newline edge case).
Plan: `docs/plans/2026-03-21-agent-trace-ingestion.md`

### ✅ SQLite performance indices — P3 · S
Done 2026-03-21. Four indices added to `GraphStore.init_schema()` targeting the hottest query patterns: `idx_nodes_created_at`, `idx_edges_created_at`, `idx_nodes_session_id` (`json_extract(metadata, '$.session_id')`), `idx_nodes_type_created` composite on `(node_type, created_at)`. All `CREATE INDEX IF NOT EXISTS` — safe on existing databases. 71 graph tests pass. 100% agent.

### ✅ Node-by-id TTL cache — P3 · S
Done 2026-03-21. `node_cache: Arc<RwLock<HashMap<NodeId, (GraphNode, Instant)>>>` added to `GraphStore`. 60 s TTL. `get_node` checks cache before hitting SQLite; populates on miss. `update_node` evicts the entry on write. No public API changes, no new deps. 71 tests pass, clippy clean. 100% agent.

### ✅ Performance: remaining — P3 · M
Done 2026-03-22. Two improvements shipped:
- **Phase 31** — `list_nodes_by_type` fast path: when called with no filters (session_id=None, metadata_filter=None), the SQL `LIMIT` is pushed into the query (`SELECT … LIMIT ?2`) so SQLite returns only the needed rows. Filtered path gets a `limit * 10` safety cap to avoid unbounded table scans.
- **Phase 32** — `get_agent_nodes` TTL cache: `agent_nodes_cache: Arc<RwLock<Option<(Vec<…>, Instant)>>>` added to `GraphStore`; 30 s TTL; invalidated in `add_node` and `update_node` when the node type is `"agent"`. Both `open()` and `open_memory()` initialize to `None`. 71 tests pass, clippy clean. Primarily agent (Task 1 100%, Task 2 95% — one collapsible-if clippy fix applied manually).

### ✅ Pinned Knowledge nodes + CLI — P2 · S
Done 2026-03-22. `pinned` metadata flag on Knowledge nodes; `list_pinned_knowledge(limit)` in GraphStore; `build_pinned_summary` in briefing (always surfaces regardless of recency); `POST /api/knowledge` + `GET /api/knowledge/pinned` endpoints; `graphirm knowledge list/pin/unpin` CLI subcommand. Coding conventions migrated from system prompt to graph-native pinned nodes.

### ✅ Model router — P2 · M
Done 2026-03-22. Automatic per-turn cheap/smart model selection. `ModelRouter` evaluates ordered rules (`first_turn`, `error_recovery`, `high_complexity`, `tool_only_turn`, `stuck_detection`) against graph-derived session signals. Routing decisions stored on Interaction node metadata for cost analysis. Same-provider constraint (both tiers via OpenRouter). Config: `[agent.routing]` in TOML.
Plan: `docs/plans/2026-03-22-model-router.md`

---

## Refactoring / Code Health

### ✅ Clean up stale `.worktrees/` — P3 · S
Done 2026-03-22. Removed 6 stale worktrees (detached HEADs + old feature branches).

### ✅ Extract CLI handlers from `main.rs` into `src/commands/` — P3 · S
Done 2026-03-22. Split `main.rs` from 1267 → 321 lines (75% reduction). 8 command modules: `chat`, `graph`, `knowledge`, `import`, `export`, `model`, `serve`, `gliner`. Shared utilities in `commands/mod.rs`.

### ✅ Cross-project dogfood setup (Nodestradamus100) — P2 · S
Done 2026-03-22. Deployed Graphirm binary to Nodestradamus100 always-on machine (`91.98.94.217:5555`). Created `dogfood-ndstrms` Cursor skill for delegating Nodestradamus100 tasks. Smoke test passed: agent read BACKLOG.md, ran pytest, self-corrected on timeout, produced correct summary. First cross-project dogfood validated end-to-end.

---

## Nodestradamus100 (via dogfood-ndstrms)

Cross-project work delegated to Graphirm on `91.98.94.217:5555`. Use the `dogfood-ndstrms` Cursor skill.

### ✅ Validation gates — all 8/8 implemented — P1 · L
Done (prior sessions). All 8 `BaseValidationGate` subclasses in `src/nodestradamus/validation/real_gates.py` are fully implemented: `CrossFileDuplicatePrecisionGate`, `WithinFileSectionUtilityGate`, `LanguageIdentificationAccuracyGate`, `BaselineCorrelationGate`, `WithinFileSimilarityRatioGate`, `ImportDetectionRecallGate`, `KeywordExtractionGate`, `CrossRepoStabilityGate`. 23 tests pass in `tests/test_validation/test_real_gates.py`.

### ✅ MCP batch insights server — P1 · M
Done 2026-03-26. `src/nodestradamus/mcp/` module with `FastMCP` server exposing 8 tools over stdio: `list_repos`, `get_hotspots`, `get_cycles`, `get_duplicates`, `get_dead_code`, `get_coupling`, `search_repos`, `batch_summary`. `InsightsLoader` reads from `batch_output/` (overridable via `NDSTRMS_BATCH_DIR` env). `ndstrms-mcp` CLI entry in `pyproject.toml`. 25 tests pass. Human-readable string output per tool. Commit: `8b92032 feat(mcp): batch insights MCP server with 8 tools`.

### ✅ Verify layer integration (2, 2.5, 3, 4) — P1 · M
Done 2026-03-26. Audit confirmed tests were superficial (import/instantiation only). Added 3 meaningful test classes: `TestLayer25MetadataOnGraph` (CHUNK nodes have language metadata), `TestLayer3StructuralEdges` (REFERENCE/CALLS/INHERITS edges produced), `TestLayer4CondensationDAG` (DAG has fewer nodes than input). 18/18 tests pass (was 15). `docs/layer-integration-audit.md` written with per-layer before/after status. Commit: `c5a342b test(integration): meaningful output assertions for layers 2.5, 3, 4`.

### ✅ [INFRA] FAISS shared-library linkage fix — P1 · S
Done 2026-03-27. Root cause: `nodestradamus_core` Rust extension (PyO3/maturin) was statically linking `libfaiss_c.a` which is missing the `_ZTVN5faiss14FaissExceptionE` C++ vtable, causing `ImportError: undefined symbol: _ZTVN5faiss14FaissExceptionE` at import time. The correctly-built shared `libfaiss.so` existed in `/tmp/faiss-build/` but was not on the linker path. Fix: copy both `libfaiss.so` + `libfaiss_c.so` to `/usr/local/lib/`, `patchelf --set-rpath`, `ldconfig`, then `cargo clean && LIBRARY_PATH=/usr/local/lib maturin develop --release`. Full rebuild procedure documented in ndstrms `BACKLOG.md` item 36. Triggered the addition of Python fallbacks (LRU `OrderedDict` for `EmbeddingCache`, numpy brute-force cosine similarity in layer1 orchestrator) as defensive code that now serves as graceful degradation.

### ✅ Increase test coverage to >50% — P2 · L
Done 2026-03-27. **57% coverage, 533 passing, 2 pre-existing failures** (`test_cache_wrapper` Rust serialization format). Journey: fixed FAISS import blocker → added EmbeddingCache/InsightsLoader/full_graph unit tests (90 passing, 10%) → fixed lazy scipy guard in `spectral.py` + lazy mcp server in `mcp/__init__.py` (38782ce) → installed scipy in venv → discovered validation tests were failing due to missing venv activation (not code bugs) → ran full suite including `tests/test_validation/` with venv → 57%. Key insight: 757 existing test functions across 124 files were already well-written — the blocker was import-time failures and venv inconsistency, not missing test coverage.

### ✅ Embedding batch failure fix (shiny crash) — P1 · S
Done 2026-03-28. `rstudio/shiny` (R, 2210 files, 16094 chunks) failed during batch processing with `Embedding computation failed: list index out of range`. Root cause: `_embed_one_batch` in `codestral_backend.py` had a silent `return []` that was reachable when the trim-retry `continue` exhausted `max_retries` on its last iteration, bypassing the `raise RuntimeError` guard. This made `computed_embeddings` shorter than `cache_misses`, causing `IndexError` at `computed_embeddings[i]`. Fix (two layers): (1) replaced `return []` with explicit `raise`, added `_embed_one_batch_safe` wrapper that returns `[None] * len(batch)` on any exception; (2) `compute_embeddings` in `layer2/orchestrator.py` now skips `None` slots with a warning instead of crashing. Effect: repos with extremely long files (161k+ token chunks from R docs) can now be processed — failed chunks are skipped, rest of the repo succeeds. Updated `test_codestral_backend_api_failure` to match new graceful behavior. Commit: `38c2ba2`.

### ✅ Batch output quality audit + extractor bug fixes — P1 · M
Done 2026-03-27. Audited actual output of 21 successfully-processed repos (of 95 total; 30 had zero chunks due to missing clones, 42 skipped). Found and fixed two silent bugs in `RepoInsightExtractor`:
- **Duplicate clusters always 0**: `_extract_duplicate_clusters` used `chunk_node.attributes.get("file_path", "")` — chunks store their parent as `parentFile`, file path is encoded in the node ID. Fixed by calling `_resolve_file_path()` which parses `chunk:file:/path:index`. Result: 0 → 20 clusters per repo.
- **Dead code always 0**: same root cause in `_extract_dead_code_candidates`, two call sites.
- **Hotspot noise**: 25% of hotspots were test/generated/vendor files (`_test.go`, `test.pb.go`, `testdata/` etc.) ranked by PageRank on CALLS edges into test helpers. Fixed by adding `_is_noise_path()` filter before sort+cap. pytest went from 18/18 noise to 18/18 real hotspots.
- Added regression test for the `parentFile` bug (`tests/test_insights/test_extractor.py`).
- Added `scripts/reextract_insights.py` — re-runs extraction on existing batch output without re-running the full (slow) pipeline. All 21 repos refreshed in ~2 min.
- Commits: `ec8b6c7` (extractor fix + regression test), `8ac4746` (hotspot filter), `2827893` (reextract script).
- Known remaining limitation: coupling is 0 for Go/C++/Ruby repos because import resolution is too low (Go: 6%) to produce cross-file CALLS edges — not a code bug.

### ~~Adaptive model router (A/B routing, token tracking, composite objective)~~ — ✅ done Phase 36
~~Replace the static rule-based router (Phase 34) with an adaptive routing framework.~~ Shipped: `RoutingStrategy` trait, `RuleRouter`, `PromptRouter`, `ExperimentRouter`, per-turn `TurnOutcome` metadata on Interaction nodes, `ObjectiveWeights` presets, `PATCH /rating` + `GET /routing/report` API. Phase 2 (statistical/learned router) remains as future work once data exists.
Design: `docs/plans/2026-03-26-adaptive-model-router-design.md`

---

## Refactoring / Code Health (Graphirm)

### Break TUI circular dependencies — P3 · M
All 14 cycles detected by Nodestradamus are in `crates/tui/` between `app.rs` ↔ `ui.rs` ↔ `events.rs`. Extract a shared `AppState` struct that both `ui.rs` and `events.rs` import without importing each other. TUI works fine — this is hygiene.

---

## Enterprise / Scale

### API versioning (`/api/v1/`) — P3 · M
Current REST API is unstable — breaking changes happen freely. Add a `/api/v1/` prefix, extract versioned request/response structs, and generate an OpenAPI spec. Required before any third-party tooling builds on top.

**Refactor analysis (Nodestradamus):** `routes.rs` has 1 direct dependent (`lib.rs` → `create_router`) and 55 indirect dependents (types, state, export, error, middleware). Natural clusters for splitting: session management (`create_session`, `get_session`, `list_sessions`, `rename_session` — cohesion 0.14), streaming/SSE (`agent_event_to_sse`, `prompt_session` — cohesion 0.33), and tests. Suggests `routes::v1::` sub-module with `create_router` re-exported at top.

**Useful Nodestradamus tools:**
- `get_impact file_path=crates/server/src/routes.rs refactor_mode=true` — breaking changes + clusters
- `analyze_graph algorithm=hierarchy level=module` — verify clean layering (routes → state/types → graph/agent)
- `analyze_strings mode=refs` — find all hardcoded `/api/` path strings in web-app TS + eval harness

### Multi-user support — P3 · L
No user concept exists — all sessions share one database. For teams: add OAuth2 login (GitHub), per-session ownership + sharing links, and basic permission model (owner / viewer). Foundation for a hosted SaaS tier.

**Scoping guidance (Nodestradamus):** `state.rs` (`AppState`, `SessionHandle`) is the natural place to add `user_id` ownership — it's the existing session registry. Run impact analysis on it to see the full blast radius of adding per-user fields.

**Useful Nodestradamus tools:**
- `analyze_graph algorithm=communities` — cluster modules to find natural auth boundary
- `get_impact file_path=crates/server/src/state.rs` — blast radius of adding `user_id` to `SessionHandle`
- `analyze_graph algorithm=path source=<auth_middleware> target=<SessionHandle>` — confirm dependency path is short

### graphirm.ai hosted demo — P3 · M
A `?demo` query param loads a pre-recorded session JSON instead of calling the API, hiding the input bar. Deploy to Cloudflare Pages (static, no server needed). Gives visitors a zero-friction look at the graph without an API key.

**Useful Nodestradamus tools:**
- `analyze_strings mode=refs` — find shared API URL strings + SSE event names between web-app and server (demo shim must intercept these)
- `semantic_analysis mode=search query="how does the web app fetch session data"` — find `useSession` hook and API client (the 1–2 files needing a `?demo` branch)
- `analyze_deps package=web-app` — map which React components import the API layer

---

## Graph Intelligence & Structural Awareness

Features that make Graphirm structurally intelligent — combining Nodestradamus code graph
analysis with Graphirm's persistent session memory. Inspired by what works in the ecosystem
(e.g. GitNexus), but transformed into capabilities only a graph-native agent can provide.

### ✅ Graph-Aware Tool Execution (Pre-Edit Impact Injection) — P1 · M
Done 2026-03-20. Before destructive tool calls (`write`, `edit`, `bash`), the agent loop automatically queries for structural dependents (via `rg`) and prior Knowledge notes mentioning the target file, computes a risk score (LOW/MEDIUM/HIGH), and prepends a brief to the tool output. Bash file paths extracted via tree-sitter-bash AST. Per-turn caching avoids re-analysis. Empty briefs suppressed. Non-fatal throughout. `pre_edit_impact: bool` config flag (default true).
Plan: `docs/plans/2026-03-20-graph-aware-tool-execution.md`
Design: `docs/plans/2026-03-20-graph-aware-tool-execution-design.md`

### ✅ Graph-Diff Tool (Session-Aware Blast Radius) — P1 · S
Done 2026-03-20. `graph_diff` non-destructive tool: `git` mode (resolves changed files via `git diff --name-only`) or `paths` mode (explicit list) → per-file dependent listing (rg `--files-with-matches --fixed-strings`, capped at 20), cross-session Knowledge query with stale ⚠ warnings, risk scoring (Low/Medium/High). Reuses `compute_risk` from Phase 22. 12 unit+integration tests.
Plan: `docs/plans/2026-03-20-graph-diff-tool.md`
Design: `docs/plans/2026-03-20-graph-diff-tool-design.md`

### ✅ Repo Briefing on Session Start (Structural + Memory Onboarding) — P1 · M
Done 2026-03-20. Two-tier briefing: (1) compact summary auto-injected into every session's system prompt — language breakdown (file-extension counts via async dir walk), top files by rg mention-count, recent knowledge nodes from the graph; (2) on-demand `repo_briefing` tool returning detailed files (rg --files + dir breakdown), knowledge (10 recent nodes), and git sections (`git log --oneline -10`, unstaged diff count). `repo_briefing: bool` config flag (default true). Non-fatal throughout. 10 unit+integration tests.
Plan: `docs/plans/2026-03-20-repo-briefing.md`
Design: `docs/plans/2026-03-20-repo-briefing-design.md`

### ✅ Session Flow Traces (Queryable Agent Decision History) — P2 · M
Done 2026-03-20. `session_trace` non-destructive tool: `search` (Knowledge-anchored semantic via `KnowledgeRetriever`, keyword fallback with note when no embeddings) groups by `session_id` and prints interaction traces; `replay` full chronological chain for one session; `detail` compact/full; `get_session_chain` in GraphStore. 16+ unit tests in tools, 2 in graph.
Plan: `docs/plans/2026-03-20-session-trace.md`
Design: `docs/plans/2026-03-20-session-trace-design.md`

### Any-Repo Instant Analysis (Zero-Config, Incremental, Persistent) — P2 · L

Drop Graphirm into any repo — no configuration, no prior indexing — and it automatically runs
Nodestradamus v2 analysis on first use, persists the result as graph nodes, and makes structural
intelligence available for all future sessions. On subsequent sessions, retrieves from the graph
DB instantly. If the repo has changed (new commits), runs incremental re-analysis on changed
files only — not a full re-index.

Fully automatic and persistent. Competes with tools that require explicit setup and full
re-index on every change.

**Key files:**
- `crates/agent/src/repo_analysis.rs` — `ensure_repo_analyzed(workspace, db)`; checks stored SHA
- `crates/graph/src/store.rs` — store/retrieve `RepoAnalysis` nodes with SHA + timestamp
- `crates/agent/src/workflow.rs` — call `ensure_repo_analyzed` on session init
- `crates/agent/src/config.rs` — `auto_analyze_repo: bool` (default true when Nodestradamus available)

**Useful Nodestradamus tools:**
- `codebase_health` — validate that analysis results are consistent before persisting
- `get_impact file_path=crates/graph/src/store.rs` — blast radius of adding `RepoAnalysis` node type

---

## Strategic Gaps (derived from competitive scoring 2026-03-22)

Scored Graphirm, Nodestradamus100, and Understand-Anything across 15 dimensions.
Items below target the dimensions where we score lowest relative to business impact.
See chat for full scoring table.

### Ndstrms-backed static analysis integration — P1 · L
**Score gap:** Static analysis 2/10, Analytical depth 5/10, Impact analysis 6/10
Wire Nodestradamus MCP tools as Graphirm's code understanding backend. When the agent starts
a session in a repo, call `analyze_deps` + `codebase_health` automatically, persist results
as Knowledge nodes, and surface them in `repo_briefing`, `graph_diff`, and pre-edit impact hooks.
Replaces the current `rg`-only dependent discovery with real dependency graph traversal.
Subsumes the existing "Any-Repo Instant Analysis" item below.

### Ndstrms analysis dashboard (web) — P2 · L
**Score gap:** Ndstrms Visualization 1/10
Build a read-only dashboard (could live in `web-app/` or separate) that renders Ndstrms
analysis results — dependency graph, PageRank hotspots, community clusters, cycle warnings.
This makes Ndstrms sellable as a standalone product. Could share React Flow + dagre infra
with Graphirm's existing whiteboard.

### Public demo mode — P1 · M
**Score gap:** Onboarding 3/10, Community 1/10
`graphirm.ai/?demo` loads a pre-recorded session (static JSON, no API key) so visitors
see the graph whiteboard, chat pane, and knowledge extraction without setup. Deploy to
Cloudflare Pages. Existing backlog item — promoting to P1 given competitive context.

### First-run guided experience — P2 · M
**Score gap:** Onboarding 3/10
On first `graphirm chat` or first web session, detect empty graph and offer a guided
walkthrough: "I'll analyze this repo and show you what I find." Auto-runs repo briefing,
highlights key files, creates initial Knowledge nodes. Inspired by UA's guided tours
but powered by Ndstrms analysis instead of LLM-for-everything.

### API versioning (`/api/v1/`) — P2 · M
**Score gap:** Production readiness 5/10
Already in backlog below — confirmed as prerequisite for any third-party integration
or Ndstrms dashboard consuming Graphirm's API.

### Open-source launch prep — P2 · M
**Score gap:** Community 1/10
README rewrite with screenshots/GIFs, `CONTRIBUTING.md`, issue templates, license audit,
clean commit history, GitHub topics/description. Required before any public visibility push.

---

## Completed (summary — details in `docs/completion-log.md`)

| Phase | What |
|-------|------|
| 0–9 | Scaffold → Knowledge layer (graph, LLM, tools, agent, multi-agent, context engine, TUI, HTTP, knowledge/HNSW) |
| 10 | Structured LLM response segments + GLiNER2 fallback + segment context filter |
| 11 | Browser web UI (vanilla JS, d3 force graph + chat) |
| 12 | `graph_query` tool (bfs, list_type, keyword search) |
| 13 | Interactive whiteboard (React + React Flow, node expansion, grouping, steer-from-node, annotations, keyboard shortcuts, auto-approve) |
| 14 | Per-session workspaces (`workspaces_root`, named dirs, graph persistence, restart restore) |
| 19 | Subagent workspace isolation + diff/read_many tools |
| 20–33 | Graph search/filter, session export, graph-aware tools, graph_diff, repo briefing, session traces, lessons briefing, auto-compaction, design system, read truncate, SQLite indices, node cache, cursor import, SQL fast paths, agent cache, pinned knowledge |
| 34–46 | Model router, `main.rs` extraction, adaptive router (strategies, A/B), context telemetry, agent continuity, pre-completion verify, doom loop, budget awareness, work loop, phase-aware routing |
| 47 | Model fallback chain — ordered array of models per routing tier with retry on retryable LLM errors |
