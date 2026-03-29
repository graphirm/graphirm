import dagre from '@dagrejs/dagre';
import type { Node, Edge } from '@xyflow/react';

// Per-type fallback dimensions used before measured sizes are available (first render).
// Interaction nodes are tall (markdown content expands them); Agent nodes are wide
// (they display model/config metadata); Knowledge/Content are compact.
const NODE_DIMENSIONS: Record<string, { width: number; height: number }> = {
  interaction: { width: 220, height: 120 },
  agent:       { width: 240, height: 70  },
  content:     { width: 200, height: 80  },
  task:        { width: 180, height: 70  },
  knowledge:   { width: 180, height: 60  },
  annotation:  { width: 200, height: 80  },
  group:       { width: 400, height: 200 },
  default:     { width: 200, height: 80  },
};

function getNodeDimensions(node: Node): { width: number; height: number } {
  // Use measured dimensions if available (after first render)
  if (node.measured?.width && node.measured?.height) {
    return { width: node.measured.width, height: node.measured.height };
  }
  // Fall back to per-type estimates
  const dims = NODE_DIMENSIONS[node.type ?? 'default'];
  return dims ?? NODE_DIMENSIONS.default;
}

export function applyDagreLayout(
  nodes: Node[],
  edges: Edge[],
  direction: 'LR' | 'TB' = 'LR',
): Node[] {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: direction, nodesep: 40, ranksep: 80, marginx: 20, marginy: 20 });

  for (const node of nodes) {
    const dims = getNodeDimensions(node);
    g.setNode(node.id, { width: dims.width, height: dims.height });
  }
  for (const edge of edges) {
    g.setEdge(edge.source, edge.target);
  }

  dagre.layout(g);

  return nodes.map((node) => {
    const pos = g.node(node.id);
    const dims = getNodeDimensions(node);
    return {
      ...node,
      position: {
        x: pos.x - dims.width / 2,
        y: pos.y - dims.height / 2,
      },
    };
  });
}
