import type { Node } from '@xyflow/react';

/** Per-type fallback dimensions before measured sizes or Pretext estimates exist. */
export const NODE_DIMENSIONS: Record<string, { width: number; height: number }> = {
  interaction: { width: 220, height: 120 },
  agent: { width: 240, height: 70 },
  content: { width: 200, height: 80 },
  task: { width: 180, height: 70 },
  knowledge: { width: 180, height: 60 },
  annotation: { width: 200, height: 80 },
  group: { width: 400, height: 200 },
  default: { width: 200, height: 80 },
};

export function getFallbackNodeDimensions(node: Node): { width: number; height: number } {
  const dims = NODE_DIMENSIONS[node.type ?? 'default'];
  return dims ?? NODE_DIMENSIONS.default;
}
