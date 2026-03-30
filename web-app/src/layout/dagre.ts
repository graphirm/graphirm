import dagre from '@dagrejs/dagre';
import type { Node, Edge } from '@xyflow/react';
import { getFallbackNodeDimensions } from './nodeDimensions';

function getNodeDimensions(
  node: Node,
  pretextSizes?: Map<string, { width: number; height: number }>,
): { width: number; height: number } {
  if (node.measured?.width && node.measured?.height) {
    return { width: node.measured.width, height: node.measured.height };
  }
  const fromPretext = pretextSizes?.get(node.id);
  if (fromPretext) {
    return fromPretext;
  }
  return getFallbackNodeDimensions(node);
}

export function applyDagreLayout(
  nodes: Node[],
  edges: Edge[],
  direction: 'LR' | 'TB' = 'LR',
  pretextSizes?: Map<string, { width: number; height: number }>,
): Node[] {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: direction, nodesep: 40, ranksep: 80, marginx: 20, marginy: 20 });

  for (const node of nodes) {
    const dims = getNodeDimensions(node, pretextSizes);
    g.setNode(node.id, { width: dims.width, height: dims.height });
  }
  for (const edge of edges) {
    g.setEdge(edge.source, edge.target);
  }

  dagre.layout(g);

  return nodes.map((node) => {
    const pos = g.node(node.id);
    const dims = getNodeDimensions(node, pretextSizes);
    return {
      ...node,
      position: {
        x: pos.x - dims.width / 2,
        y: pos.y - dims.height / 2,
      },
    };
  });
}
