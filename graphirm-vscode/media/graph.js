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
}

export function handleGraphMessage(msg) {
  if (msg.type === 'session_loaded' || msg.type === 'refreshed') {
    renderGraph(msg.graph || { nodes: [], edges: [] });
  } else if (msg.type === 'subgraph') {
    renderGraph(msg.subgraph);
  } else if (msg.type === 'node_detail') {
    showNodeDetail(msg.node);
  }
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
    .attr('stroke', '#555')
    .attr('stroke-width', 1)
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
    .force('center', d3.forceCenter(_width / 2, _height / 2))
    .on('tick', () => {
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
