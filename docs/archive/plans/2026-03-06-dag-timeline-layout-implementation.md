# DAG Timeline Layout Implementation Plan

**Date:** 2026-03-06  
**Design Doc:** `2026-03-06-dag-timeline-layout-design.md`  
**Phase:** 12, Task 4 (refined)  
**Estimated Effort:** 2-3 hours  
**Dependencies:** Phase 11 (VS Code extension) must be complete

> **For Claude:** Use superpowers:subagent-driven-development to execute this plan task-by-task.

---

## Overview

Implement a timeline-aware DAG layout in the VS Code extension's graph explorer. Nodes are positioned by time (X-axis) and semantic type (Y-axis), with logical grouping and edge routing improvements.

**Deliverable:** One-click toggle between Timeline and Force layouts in the graph explorer panel.

---

## Architecture

### Current State
- `graphirm-vscode/media/graph.js` uses d3-force layout
- All nodes are free-floating; edges are straight lines
- No mode switching

### Target State
- Two layout modes: `force` (current) and `timeline` (new)
- Mode toggle button in toolbar
- Timeline mode: fixed positions from created_at + node type
- Intelligent grouping: related nodes (interaction → tools → results) cluster visually
- Improved edge routing: Bezier curves, color-coded by type

### Files to Modify

| File | Change | Type |
|------|--------|------|
| `graphirm-vscode/media/index.html` | Add layout toggle button | Minor |
| `graphirm-vscode/media/styles.css` | Style toggle button | Minor |
| `graphirm-vscode/media/graph.js` | Add timeline layout logic, grouping, edge routing | Major |

---

## Task Breakdown

### Task 1: Add UI Controls (20 min)

**File:** `graphirm-vscode/media/index.html`

**Changes:**
1. Find the graph toolbar (contains `#reset-zoom-btn`)
2. After `#reset-zoom-btn`, add toggle button:

```html
<button id="layout-toggle-btn" title="Toggle between timeline and force-directed layout">
  Timeline
</button>
```

**Validation:**
- Button appears in graph panel toolbar
- Button is clickable (no event listener yet)

---

### Task 2: Style Toggle Button (10 min)

**File:** `graphirm-vscode/media/styles.css`

**Changes:**
Add button styling (match existing toolbar button style):

```css
#layout-toggle-btn {
  padding: 6px 12px;
  background: var(--button-background);
  color: var(--button-foreground);
  border: 1px solid var(--button-border);
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  font-weight: 500;
  transition: background-color 0.15s;
}

#layout-toggle-btn:hover {
  background: var(--button-hovered-background);
}

#layout-toggle-btn.active {
  background: var(--accent-color);
  color: white;
}
```

**Validation:**
- Button looks consistent with other toolbar buttons
- Hover state works
- Active state (will use in Task 4)

---

### Task 3: Implement Timeline Layout Algorithm (90 min)

**File:** `graphirm-vscode/media/graph.js`

**Step 3a: Add layout mode state and constants**

At the top of the file (after imports, before existing functions):

```javascript
// Layout modes
let _layoutMode = 'force'; // 'force' | 'timeline'
let _currentData = null; // Cache for re-renders on toggle

// Y-position by node type (base position in timeline mode)
const TYPE_Y = {
  Agent: 80,
  Task: 160,
  Interaction: 260,
  Content: 360,
  Knowledge: 440,
};

// Edge colors by type
const EDGE_COLORS = {
  RespondsTo: '#ffffff44',
  Reads: '#3b82f688',
  Modifies: '#f9731688',
  Produces: '#4ade8088',
  DependsOn: '#a855f788',
  SpawnedBy: '#ec489988',
};

// Stroke widths by importance
const EDGE_STROKE_WIDTH = {
  DependsOn: 2.5,
  Produces: 2.5,
  default: 1.5,
};
```

**Step 3b: Implement grouping logic**

Add function to organize nodes into logical groups (interaction → tools → results):

```javascript
/**
 * Group nodes into logical units (interaction + its tools + results).
 * Returns a map: nodeId → { groupId, offset, depth }
 */
function computeNodeGroups(nodes, edges) {
  const groups = new Map(); // nodeId → { groupId, depth }
  const processed = new Set();
  
  // Find Interaction nodes (root of each group)
  const interactions = nodes.filter(n => n.node_type?.type === 'Interaction');
  
  interactions.forEach((interaction, idx) => {
    const groupId = interaction.id;
    const depth = 0;
    groups.set(interaction.id, { groupId, depth });
    processed.add(interaction.id);
    
    // Find tool calls produced by this interaction (Produces edges)
    const toolCalls = edges
      .filter(e => e.edge_type === 'Produces' && e.from === interaction.id)
      .map(e => nodes.find(n => n.id === e.to))
      .filter(Boolean);
    
    // Find content/results produced by tool calls (Produces edges)
    toolCalls.forEach((tool, toolIdx) => {
      groups.set(tool.id, { groupId, depth: 1 });
      processed.add(tool.id);
      
      const results = edges
        .filter(e => e.edge_type === 'Produces' && e.from === tool.id)
        .map(e => nodes.find(n => n.id === e.to))
        .filter(Boolean);
      
      results.forEach((result, resIdx) => {
        groups.set(result.id, { groupId, depth: 2 });
        processed.add(result.id);
      });
    });
  });
  
  // Assign orphan nodes (not in any group) to their own group
  nodes.forEach(n => {
    if (!processed.has(n.id)) {
      groups.set(n.id, { groupId: n.id, depth: 0 });
    }
  });
  
  return groups;
}
```

**Step 3c: Implement timeline positioning**

```javascript
/**
 * Assign X, Y positions for timeline layout mode.
 * X = time (oldest left, newest right)
 * Y = node type + group offset
 * Freezes node positions (sets fx, fy) and stops simulation.
 */
function applyTimelineLayout(nodes, edges, width, height) {
  if (nodes.length === 0) return;
  
  // Parse timestamps
  const times = nodes
    .map(n => new Date(n.created_at).getTime())
    .filter(t => !isNaN(t));
  
  if (times.length === 0) return; // No valid timestamps
  
  const tMin = Math.min(...times);
  const tMax = Math.max(...times);
  const tRange = tMax - tMin || 1;
  const padding = 60;
  
  // Compute grouping
  const groups = computeNodeGroups(nodes, edges);
  
  // Assign positions
  nodes.forEach(n => {
    const t = new Date(n.created_at).getTime();
    if (isNaN(t)) {
      n.fx = padding; // Default to left if no timestamp
      n.fy = height / 2;
      return;
    }
    
    // X: temporal position (normalized to [0, 1], then scaled to canvas)
    n.fx = padding + ((t - tMin) / tRange) * (width - padding * 2);
    
    // Y: node type base + group-based offset
    const nodeType = n.node_type?.type || 'Content';
    const baseY = TYPE_Y[nodeType] ?? 260;
    const group = groups.get(n.id) || { groupId: n.id, depth: 0 };
    const offset = group.depth * 25; // 25px between interaction/tool/result vertically
    
    n.fy = baseY + offset;
  });
}
```

**Step 3d: Implement position release (for force mode)**

```javascript
/**
 * Release node positions (set fx, fy to null) so force simulation can move them.
 */
function releaseNodePositions(nodes) {
  nodes.forEach(n => {
    n.fx = null;
    n.fy = null;
  });
}
```

**Step 3e: Update edge rendering with colors and curves**

Find the section in `renderGraph` that draws edges. Replace or update it:

```javascript
// Example: in renderGraph function, find the link rendering section
const link = g.append('g').selectAll('line')
  .data(edges)
  .join('line')
  .attr('stroke', d => {
    const color = EDGE_COLORS[d.edge_type];
    return color ?? '#ffffff22';
  })
  .attr('stroke-width', d => EDGE_STROKE_WIDTH[d.edge_type] ?? EDGE_STROKE_WIDTH.default)
  .attr('opacity', 0.7);

// For curved edges in timeline mode, use d3-path or transition to curves later
// For now, keep straight lines; Bezier curves are an enhancement
```

**Validation:**
- Timeline layout computes positions without errors
- Grouping logic groups related nodes together
- Edge colors match edge types
- Positions freeze in timeline mode

---

### Task 4: Add Toggle Handler and Mode Switching (40 min)

**File:** `graphirm-vscode/media/graph.js`

**Step 4a: Add toggle event listener**

In the `initGraph()` function or where other event listeners are set up:

```javascript
document.getElementById('layout-toggle-btn').addEventListener('click', () => {
  _layoutMode = _layoutMode === 'force' ? 'timeline' : 'force';
  const btn = document.getElementById('layout-toggle-btn');
  btn.textContent = _layoutMode === 'force' ? 'Timeline' : 'Force';
  btn.classList.toggle('active', _layoutMode === 'timeline');
  
  // Re-render with new layout
  if (_currentData) {
    renderGraph(_currentData);
  }
});
```

**Step 4b: Update handleGraphMessage to cache data**

Find the function that handles incoming graph messages. Update it to cache the data:

```javascript
// Before:
export function handleGraphMessage(msg) {
  if (msg.type === 'graph' && msg.data) {
    renderGraph(msg.data);
  }
  // ...
}

// After:
export function handleGraphMessage(msg) {
  if (msg.type === 'graph' && msg.data) {
    _currentData = msg.data; // Cache for re-render on toggle
    renderGraph(msg.data);
  }
  // ...
}
```

**Step 4c: Update renderGraph to branch on layout mode**

Find the `renderGraph` function. Before starting the force simulation, add branching logic:

```javascript
function renderGraph(data) {
  // ... existing setup code (SVG container, zoom, etc.) ...
  
  const nodes = data.nodes || [];
  const edges = data.edges || [];
  
  // Clear previous graph
  d3.select('#graph-container').selectAll('*').remove();
  
  // Create new SVG and groups
  const svg = d3.select('#graph-container').append('svg')
    .attr('width', width)
    .attr('height', height);
  
  // ... render nodes and edges ...
  
  // Branch on layout mode
  if (_layoutMode === 'timeline') {
    // Timeline mode: fixed positions
    applyTimelineLayout(nodes, edges, width, height);
    _simulation.alpha(0).stop(); // Stop force simulation
  } else {
    // Force mode: free-floating
    releaseNodePositions(nodes);
    _simulation.alpha(0.3).restart(); // Restart force simulation
  }
  
  // ... rest of render logic ...
}
```

**Validation:**
- Toggle button changes text (Timeline ↔ Force)
- Toggle changes `_layoutMode` state
- Re-render fires on toggle
- No console errors on toggle
- Positions freeze in timeline mode
- Positions release in force mode

---

### Task 5: Build and Test (20 min)

**Step 5a: Build extension**

```bash
cd graphirm-vscode
npm run build
```

**Validation:**
- No TypeScript errors
- No bundling errors
- Build completes successfully

**Step 5b: Manual testing**

1. Start the Graphirm server:
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   cargo run --release -- serve
   ```

2. Open VS Code with the built extension

3. Create a new session and ask the agent to perform a task (e.g., "What's the weather?")

4. Wait for 5+ interactions/tool calls to accumulate

5. Open the Graphirm panel (Ctrl+Shift+P → Graphirm: Open)

6. Verify:
   - Graph displays
   - Nodes are visible
   - Toggle button exists and is clickable
   - **Click toggle:** Layout switches to timeline (nodes should align vertically by type)
   - **Click toggle again:** Layout switches back to force (nodes should scatter)
   - Related nodes (interaction → tools) should group closely in timeline mode
   - No console errors during toggles

**Validation Checklist:**
- [ ] Extension builds without errors
- [ ] Graph renders initially
- [ ] Toggle button is visible and clickable
- [ ] Toggling changes layout
- [ ] Timeline mode positions nodes by time + type
- [ ] Force mode restores free-form layout
- [ ] Related nodes cluster in timeline mode
- [ ] No console errors

---

### Task 6: Code Review & Polish (20 min)

**Self-review checklist:**
- [ ] Code is readable and well-commented
- [ ] No debug logs left in place
- [ ] Edge cases handled (empty graph, no timestamps)
- [ ] Performance acceptable (100+ nodes should still render smoothly)
- [ ] Styling is consistent with existing UI

**Optional polish:**
- Add transition animation between layouts (d3 transition)
- Add keyboard shortcut for toggle (e.g., `T` key)
- Display current mode in status bar

---

### Task 7: Commit and Documentation (10 min)

**Commit:**
```bash
cd graphirm-vscode
git add .
git commit -m "feat: add DAG timeline layout toggle to graph explorer"
```

**Update Phase 12 progress:**
- Mark Task 4 as complete in `docs/plans/2026-03-05-phase-12-landing-and-polish.md`

---

## Progress Tracking

| Task | Description | Status | Time Est. |
|------|-------------|--------|-----------|
| Task 1 | Add layout toggle button HTML | ⬜ | 20 min |
| Task 2 | Style toggle button | ⬜ | 10 min |
| Task 3 | Implement timeline algorithm + grouping | ⬜ | 90 min |
| Task 4 | Add toggle handler and mode switching | ⬜ | 40 min |
| Task 5 | Build extension and manual testing | ⬜ | 20 min |
| Task 6 | Code review and polish | ⬜ | 20 min |
| Task 7 | Commit and documentation | ⬜ | 10 min |
| **Total** | | | **210 min (3.5 hrs)** |

---

## Success Criteria

✅ **Functional:**
- Timeline layout correctly positions nodes by time (X) and type (Y)
- Nodes group logically (interaction + tools + results stay close)
- Toggle switches between timeline and force layouts
- No rendering errors or visual glitches

✅ **Quality:**
- Code is readable, commented, and follows existing style
- No console errors or warnings
- Performance acceptable with 50+ nodes
- Styling matches VS Code theme variables

✅ **Testing:**
- Manual test with real session graph (5+ interactions)
- Toggle tested (forward and back)
- Edge cases tested (empty graph, single node, no timestamps)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Dense graphs have overlapping nodes | Document as known limitation; defer smart spacing to Phase 13 |
| Edge overlap hard to read | Use opacity + color coding; Bezier curves deferred to Phase 13 |
| Performance with 200+ nodes | Grouping is O(n); measure and optimize if needed |

---

## Rollback Plan

If timeline layout causes instability:
1. Disable toggle button in HTML
2. Always use force mode
3. Revert commit and rebuild

**Recovery time:** < 5 minutes

---

## Next Steps (After Approval & Completion)

1. Commit to main branch (Phase 12 Task 4 complete)
2. Update Phase 12 status in `00-execution-strategy.md`
3. Begin Phase 13 (future enhancements):
   - Hierarchical DAG layout mode
   - Edge bundling for dense graphs
   - Critical path highlighting
   - Temporal brushing

---

## Appendix: Reference Materials

- **Design Doc:** `2026-03-06-dag-timeline-layout-design.md`
- **Current code:** `graphirm-vscode/media/graph.js` (d3 rendering)
- **Phase 12 Plan:** `2026-03-05-phase-12-landing-and-polish.md` (Task 4 context)
- **D3 docs:** https://d3js.org/d3-force (force simulation)
