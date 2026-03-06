import { getCurrentSessionId } from './main.js';

let _send;
let _simulation;
let _svg;
let _zoom;
let _width = 600;
let _height = 500;

const NODE_COLORS = {
  Interaction: 'var(--node-interaction)',
  Content: 'var(--node-content)',
  Task: 'var(--node-task)',
  Knowledge: 'var(--node-knowledge)',
  Agent: 'var(--node-agent)',
};

// Layout mode tracking
let _layoutMode = 'force'; // 'force' | 'timeline'
let _currentData = null; // Cache for re-renders

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

// Stroke widths by edge importance
const EDGE_STROKE_WIDTH = {
  DependsOn: 2.5,
  Produces: 2.5,
  default: 1.5,
};

export function initGraph(send) {
  _send = send;

  const svgEl = document.getElementById('graph-svg');
  const rect = svgEl.getBoundingClientRect();
  _width = rect.width || 600;
  _height = rect.height || 500;

  _svg = d3.select('#graph-svg');
  const g = _svg.append('g').attr('class', 'graph-root');

  _zoom = d3.zoom().scaleExtent([0.1, 4]).on('zoom', (e) => {
    g.attr('transform', e.transform);
  });
  _svg.call(_zoom);

  document.getElementById('reset-zoom-btn').addEventListener('click', () => {
    _svg.transition().call(_zoom.transform, d3.zoomIdentity);
  });

  document.getElementById('node-detail-close').addEventListener('click', () => {
    document.getElementById('node-detail').hidden = true;
  });

  document.querySelector('[data-action="subgraph"]').addEventListener('click', () => {
    const detail = document.getElementById('node-detail');
    if (detail.dataset.nodeId) {
      const depth = parseInt(document.getElementById('depth-slider').value, 10);
      _send({
        type: 'get_subgraph',
        session_id: getCurrentSessionId(),
        node_id: detail.dataset.nodeId,
        depth,
      });
    }
  });

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
}

export function handleGraphMessage(msg) {
  if (msg.type === 'session_loaded' || msg.type === 'refreshed') {
    _currentData = msg.graph || { nodes: [], edges: [] };
    renderGraph(_currentData);
  } else if (msg.type === 'subgraph') {
    _currentData = msg.subgraph;
    renderGraph(_currentData);
  } else if (msg.type === 'node_detail') {
    showNodeDetail(msg.node);
  }
}

/**
 * Group nodes into logical units (interaction + its tools + results).
 * Returns a map: nodeId → { groupId, depth }
 * Depth: 0=interaction, 1=tool calls, 2=results
 */
function computeNodeGroups(nodes, edges) {
  const groups = new Map();
  const processed = new Set();
  
  // Find Interaction nodes (root of each group)
  const interactions = nodes.filter(n => n.node_type?.type === 'Interaction');
  
  interactions.forEach((interaction) => {
    const groupId = interaction.id;
    groups.set(interaction.id, { groupId, depth: 0 });
    processed.add(interaction.id);
    
    // Find tool calls produced by this interaction (Produces edges)
    const toolCalls = edges
      .filter(e => e.edge_type === 'Produces' && e.from === interaction.id)
      .map(e => nodes.find(n => n.id === e.to))
      .filter(Boolean);
    
    // Find content/results produced by tool calls
    toolCalls.forEach((tool) => {
      groups.set(tool.id, { groupId, depth: 1 });
      processed.add(tool.id);
      
      const results = edges
        .filter(e => e.edge_type === 'Produces' && e.from === tool.id)
        .map(e => nodes.find(n => n.id === e.to))
        .filter(Boolean);
      
      results.forEach((result) => {
        groups.set(result.id, { groupId, depth: 2 });
        processed.add(result.id);
      });
    });
  });
  
  // Assign orphan nodes to their own group
  nodes.forEach(n => {
    if (!processed.has(n.id)) {
      groups.set(n.id, { groupId: n.id, depth: 0 });
    }
  });
  
  return groups;
}

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
      n.fx = padding;
      n.fy = height / 2;
      return;
    }
    
    // X: temporal position (normalized and scaled)
    n.fx = padding + ((t - tMin) / tRange) * (width - padding * 2);
    
    // Y: node type base + group-based offset
    const nodeType = n.node_type?.type || 'Content';
    const baseY = TYPE_Y[nodeType] ?? 260;
    const group = groups.get(n.id) || { groupId: n.id, depth: 0 };
    const offset = group.depth * 25; // 25px vertical spacing within group
    
    n.fy = baseY + offset;
  });
}

/**
 * Release node positions (set fx, fy to null) so force simulation can move them.
 */
function releaseNodePositions(nodes) {
  nodes.forEach(n => {
    n.fx = null;
    n.fy = null;
  });
}

function renderGraph({ nodes, edges }) {
  const g = _svg.select('.graph-root');
  g.selectAll('*').remove();

  if (_simulation) _simulation.stop();

  // Arrow marker
  _svg.select('defs').remove();
  _svg.append('defs').append('marker')
    .attr('id', 'arrow')
    .attr('viewBox', '0 -5 10 10')
    .attr('refX', 20)
    .attr('markerWidth', 6)
    .attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', 'M0,-5L10,0L0,5')
    .attr('fill', '#666');

  const link = g.append('g').selectAll('line')
    .data(edges)
    .join('line')
    .attr('stroke', d => {
      const color = EDGE_COLORS[d.edge_type];
      return color ?? '#ffffff22';
    })
    .attr('stroke-width', d => EDGE_STROKE_WIDTH[d.edge_type] ?? EDGE_STROKE_WIDTH.default)
    .attr('opacity', 0.7)
    .attr('marker-end', 'url(#arrow)');

  const linkLabel = g.append('g').selectAll('text')
    .data(edges)
    .join('text')
    .attr('fill', '#888')
    .attr('font-size', 9)
    .text(d => d.edge_type);

  const node = g.append('g').selectAll('circle')
    .data(nodes)
    .join('circle')
    .attr('r', 10)
    .attr('fill', d => NODE_COLORS[d.node_type?.type] || '#999')
    .attr('stroke', '#fff')
    .attr('stroke-width', 0.5)
    .style('cursor', 'pointer')
    .on('click', (_, d) => {
      _send({
        type: 'get_node',
        session_id: getCurrentSessionId(),
        node_id: d.id,
      });
    })
    .call(d3.drag()
      .on('start', (e, d) => {
        if (!e.active) _simulation.alphaTarget(0.3).restart();
        d.fx = d.x; d.fy = d.y;
      })
      .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
      .on('end', (e, d) => {
        if (!e.active) _simulation.alphaTarget(0);
        d.fx = null; d.fy = null;
      })
    );

  const nodeLabel = g.append('g').selectAll('text')
    .data(nodes)
    .join('text')
    .attr('fill', '#ccc')
    .attr('font-size', 9)
    .attr('text-anchor', 'middle')
    .attr('dy', 20)
    .text(d => d.node_type?.type ?? '');

  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  const simEdges = edges.map(e => ({
    ...e,
    source: nodeMap.get(e.source) || e.source,
    target: nodeMap.get(e.target) || e.target,
  }));

  _simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(simEdges).id(d => d.id).distance(80))
    .force('charge', d3.forceManyBody().strength(-200))
    .force('center', d3.forceCenter(_width / 2, _height / 2));

  // Branch on layout mode
  if (_layoutMode === 'timeline') {
    // Timeline mode: fixed positions
    applyTimelineLayout(nodes, edges, _width, _height);
    _simulation.alpha(0).stop();
  } else {
    // Force mode: free-floating
    releaseNodePositions(nodes);
    _simulation.alpha(0.3).restart();
  }

  _simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);
      linkLabel
        .attr('x', d => (d.source.x + d.target.x) / 2)
        .attr('y', d => (d.source.y + d.target.y) / 2);
      node.attr('cx', d => d.x).attr('cy', d => d.y);
      nodeLabel.attr('x', d => d.x).attr('y', d => d.y);
    });
}

function showNodeDetail(node) {
  const detail = document.getElementById('node-detail');
  detail.dataset.nodeId = node.id;
  detail.hidden = false;

  const typeName = node.node_type?.type ?? 'Unknown';
  document.getElementById('node-detail-type').textContent =
    `${typeName} · ${node.id.slice(0, 8)}`;

  document.getElementById('node-detail-content').textContent =
    JSON.stringify(node.node_type, null, 2);
}
