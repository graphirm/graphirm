# DAG Timeline Layout Design Document

**Date:** 2026-03-06  
**Status:** Design approved  
**Scope:** Phase 12 Task 4 refinement + MVP implementation  
**Owner:** Claude (Cursor agent)

---

## Executive Summary

The current d3-force layout in the VS Code extension's graph explorer is a "blob" — nodes repel each other randomly, making it hard to trace causality or understand temporal flow. This design introduces a **DAG-aware timeline layout** that positions nodes by time (X-axis) and semantic type (Y-axis), with intelligent visual grouping and improved edge routing.

**MVP ships Timeline Mode only.** Hierarchical Mode (topological DAG layout) is deferred to Phase 13.

---

## Design Goals

1. **Readability**: Users can immediately understand what happened when and why (causality)
2. **Clarity**: Related nodes (message → tools → results) cluster visually
3. **Interactivity**: One-click toggle between Timeline (temporal) and Force (free-form)
4. **Extensibility**: Architecture supports future Hierarchical Mode without breaking changes

---

## Data Model

### Node Attributes

Every node already has:
- `node_type` (Agent, Task, Interaction, Content, Knowledge)
- `created_at` (ISO 8601 timestamp)
- `metadata` (JSON object)
- `session_id` (for grouping)

**New computed attributes** (calculated at render time, not stored):
- `render_y` — Y-position determined by node type and logical group
- `render_group_id` — which "column" does this node belong to (session → interaction triad)
- `is_critical_path` — is this node on a chain of `DependsOn` edges? (future use)

### Edge Attributes

Every edge has:
- `edge_type` (RespondsTo, DependsOn, Produces, Reads, Modifies, SpawnedBy, etc.)
- `from` (source node)
- `to` (target node)

**Visual encoding:**
- `DependsOn` → purple (#a855f7), thicker stroke
- `Produces` → green (#4ade80), thicker stroke
- `RespondsTo` → white (#ffffff44), thin stroke
- `Reads` → blue (#3b82f6, 88 opacity)
- `Modifies` → orange (#f97316, 88 opacity)
- `SpawnedBy` → red (#ec4899, 88 opacity)

---

## Layout Algorithm

### Timeline Mode (MVP)

#### Phase 1: Position Assignment

```
1. Parse created_at timestamps across all visible nodes
2. Calculate time range: tMin, tMax, tRange = tMax - tMin
3. For each node:
   - X = padding + ((t - tMin) / tRange) * (width - 2*padding)
   - Y = TYPE_Y[node_type] + group_offset
4. Freeze positions (fx, fy) to prevent force simulation
5. Set simulation alpha = 0, stop simulation
```

#### Phase 2: Type-Based Y-Strata

Base Y-positions by node type (in render order top to bottom):

| Node Type | Base Y | Rationale |
|-----------|--------|-----------|
| Agent | 80 | Orchestrators at top |
| Task | 160 | Task DAG below agents |
| Interaction | 260 | Messages in the middle (user + assistant) |
| Content | 360 | Files/code results |
| Knowledge | 440 | Extracted entities at bottom |

#### Phase 3: Logical Grouping (Smart Y-offset)

Within each type stratum, nodes are grouped into "columns" representing related work units:

**Grouping strategy:**
```
For each Interaction node (message):
  1. Find all tool calls spawned by this message (edges: Produces → Tool Call)
  2. Find all Content nodes produced by those tool calls
  3. Assign group_id = interaction.id
  4. Vertically stack: Message at TYPE_Y[Interaction],
     then Tool Calls at TYPE_Y[Interaction] + 40px,
     then Results at TYPE_Y[Content] with slight offset
```

**Visual result:** A "conversation column" is a coherent vertical slice of related activity.

#### Phase 4: Edge Routing (Curves + Colors)

Instead of straight lines, use **Bezier curves** for better readability:

```javascript
// For each edge:
// - Color by edge_type (see color table above)
// - Stroke width: 1.5px for normal, 2.5px for DependsOn/Produces
// - Opacity: full for DependsOn/Produces/RespondsTo, 0.5 for others
// - Path: d3.linkCurve() or equivalent (curved Bezier)
// - Marker: arrowhead on target node (optional, phase 2)
```

---

## Hierarchical Mode (Post-MVP, Phase 13)

Placeholder for future implementation. When ready:

1. Topologically sort nodes by `DependsOn` edges
2. Assign X = layer (depth in DAG)
3. Assign Y = time within layer (minor sorting by created_at)
4. Use hierarchical layout algorithm (e.g., d3-hierarchy)

---

## UI Changes

### Graph Explorer Panel

**Before:** "Force" layout button in toolbar  
**After:** Toggle button shows current mode

```html
<button id="layout-toggle-btn" title="Toggle layout">
  Timeline ↔ Force
</button>
```

**On click:** Toggle between modes and re-render immediately.

**State management:**
```javascript
let _layoutMode = 'force'; // 'force' | 'timeline'
```

---

## Implementation Strategy

### File Changes

| File | Change | Type |
|------|--------|------|
| `graphirm-vscode/media/graph.js` | Add timeline layout logic, grouping, edge routing | Modify |
| `graphirm-vscode/media/index.html` | Add layout toggle button | Modify |
| `graphirm-vscode/media/styles.css` | Style toggle button | Modify (minimal) |

### Graph Store Queries (No Changes Required)

The existing `GraphStore` API is sufficient:
- `get_subgraph(node_id, depth)` → fetch related nodes
- Edge types already available in `GraphEdge.edge_type`
- Nodes already have `created_at` and `metadata`

**No new queries needed** — grouping logic lives in the extension (d3 code), not the backend.

---

## Success Criteria

- [x] Design approved by user
- [ ] Timeline layout positions nodes by time + type
- [ ] Nodes within interaction groups cluster visually
- [ ] Edges use Bezier curves and are color-coded by type
- [ ] Toggle button switches between Timeline and Force modes
- [ ] No layout shifts when toggling (smooth UX)
- [ ] Extension builds without errors
- [ ] Visually tested with a real session graph (5+ nodes)

---

## Testing Plan

### Unit Tests
None required (this is pure UI rendering).

### Integration Tests
1. Load extension with mock graph data
2. Verify Timeline mode positions nodes correctly
3. Verify Force mode releases positions and restarts simulation
4. Verify toggle button changes label and mode

### Manual Testing
1. Start server with a real session
2. Create 5+ interactions with tool calls
3. Open VS Code extension
4. Verify timeline layout is readable
5. Toggle to force, back to timeline
6. Verify no console errors

---

## Dependencies

**New d3 modules:**
- Already have: `d3`, `d3-force`
- Will use: `d3.linkCurve()` (part of d3-shape, already included in d3)

**Rust changes:** None (query API is sufficient)

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Edge overlap in dense graphs | Medium | Use opacity gradients; defer to Phase 13 edge bundling |
| Performance with 100+ nodes | Low | Grouping is O(n); simulation is already optimized |
| Toggle UX feels janky | Low | Re-render on toggle; precompute positions |

---

## Future Extensions (Phase 13+)

1. **Hierarchical Mode**: Topological DAG layout
2. **Edge bundling**: Reduce visual clutter in dense regions
3. **Critical path highlighting**: Emphasize longest dependency chain
4. **Temporal brushing**: Scrub timeline to see what happened at a specific time
5. **Export**: SVG/PNG snapshot of timeline view

---

## Approval Sign-Off

- **Designer:** Claude (AI agent)
- **User approval:** ✅ Yes (2026-03-06)
- **Ready for implementation:** ✅ Yes
