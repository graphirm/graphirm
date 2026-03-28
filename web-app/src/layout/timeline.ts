import type { Node, Edge } from '@xyflow/react';
import type { GraphNode } from '../types/graph';

export const TYPE_Y: Record<string, number> = {
  Agent: 80,
  Task: 160,
  Interaction: 260,
  Content: 360,
  Knowledge: 440,
};

export const TYPE_LABELS: Record<string, string> = {
  Agent: 'Agent',
  Task: 'Task',
  Interaction: 'Interaction',
  Content: 'Content',
  Knowledge: 'Knowledge',
};

interface GroupInfo {
  groupId: string;
  depth: number;
}

function computeNodeGroups(
  nodes: GraphNode[],
  edges: Array<{ source: string; target: string; edge_type: string }>,
): Map<string, GroupInfo> {
  const groups = new Map<string, GroupInfo>();
  const processed = new Set<string>();

  const interactions = nodes.filter(n => n.node_type.type === 'Interaction');

  for (const interaction of interactions) {
    groups.set(interaction.id, { groupId: interaction.id, depth: 0 });
    processed.add(interaction.id);

    const toolCallIds = edges
      .filter(e => e.edge_type === 'produces' && e.source === interaction.id)
      .map(e => e.target);

    for (const toolId of toolCallIds) {
      groups.set(toolId, { groupId: interaction.id, depth: 1 });
      processed.add(toolId);

      const resultIds = edges
        .filter(e => e.edge_type === 'produces' && e.source === toolId)
        .map(e => e.target);

      for (const resultId of resultIds) {
        groups.set(resultId, { groupId: interaction.id, depth: 2 });
        processed.add(resultId);
      }
    }
  }

  for (const node of nodes) {
    if (!processed.has(node.id)) {
      groups.set(node.id, { groupId: node.id, depth: 0 });
    }
  }

  return groups;
}

export function applyTimelineLayout(
  nodes: Node[],
  rawNodes: GraphNode[],
  edges: Edge[],
  canvasWidth: number,
): Node[] {
  if (nodes.length === 0) return nodes;

  const rawEdges = edges.map(e => ({
    source: e.source,
    target: e.target,
    edge_type: (e.data as { edge_type?: string } | undefined)?.edge_type ?? '',
  }));

  const groups = computeNodeGroups(rawNodes, rawEdges);

  const times = rawNodes
    .map(n => new Date(n.created_at).getTime())
    .filter(t => !isNaN(t));

  if (times.length === 0) return nodes;

  const tMin = Math.min(...times);
  const tMax = Math.max(...times);
  const tRange = tMax - tMin || 1;
  const padding = 80;

  const rawById = new Map(rawNodes.map(n => [n.id, n]));

  return nodes.map((node) => {
    const raw = rawById.get(node.id);
    if (!raw) return node;

    const t = new Date(raw.created_at).getTime();
    const x = isNaN(t)
      ? padding
      : padding + ((t - tMin) / tRange) * (canvasWidth - padding * 2);

    const nodeTypeName = raw.node_type.type;
    const baseY = TYPE_Y[nodeTypeName] ?? 260;
    const group = groups.get(node.id) ?? { groupId: node.id, depth: 0 };
    const y = baseY + group.depth * 30;

    return { ...node, position: { x, y } };
  });
}
