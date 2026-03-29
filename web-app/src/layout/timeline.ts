import type { Node, Edge } from '@xyflow/react';
import type { GraphNode } from '../types/graph';

// ─── Turn model ──────────────────────────────────────────────────────────────

export interface Turn {
  userNodeId: string | null;
  finalNodeId: string | null;
  intermediateIds: string[];
}

/**
 * Partition Interaction nodes into turns and classify each node's role.
 *
 * A new turn begins at each 'user' role Interaction. Within a turn:
 * - The last 'assistant' node whose metadata.tool_calls is absent/empty is finalNode.
 * - All other nodes (tool, assistant-with-tool_calls) are intermediates.
 *
 * Nodes appearing before the first user message go into a synthetic turn 0
 * with userNodeId = null.
 */
export function buildTurns(rawNodes: GraphNode[]): Turn[] {
  const interactions = [...rawNodes]
    .filter(n => n.node_type.type === 'Interaction')
    .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime());

  const turns: Turn[] = [];
  let current: Turn = { userNodeId: null, finalNodeId: null, intermediateIds: [] };

  for (const node of interactions) {
    const role = (node.node_type as { type: string; role: string }).role;
    if (role === 'user') {
      // Close previous turn if it has content
      if (current.userNodeId !== null || current.finalNodeId !== null || current.intermediateIds.length > 0) {
        turns.push(current);
      }
      current = { userNodeId: node.id, finalNodeId: null, intermediateIds: [] };
    } else if (role === 'assistant') {
      const toolCalls = (node.metadata as Record<string, unknown>)?.tool_calls;
      const hasCalls = Array.isArray(toolCalls) && (toolCalls as unknown[]).length > 0;
      if (!hasCalls) {
        // Candidate for finalNode — keep updating so we always use the last one
        if (current.finalNodeId !== null) {
          // Previous "final" was not actually final — demote to intermediate
          current.intermediateIds.push(current.finalNodeId);
        }
        current.finalNodeId = node.id;
      } else {
        current.intermediateIds.push(node.id);
      }
    } else {
      // 'tool', 'system', or any other role
      current.intermediateIds.push(node.id);
    }
  }

  // Push last open turn
  if (current.userNodeId !== null || current.finalNodeId !== null || current.intermediateIds.length > 0) {
    turns.push(current);
  }

  return turns;
}

// ─── Layout constants ─────────────────────────────────────────────────────────

const MAIN_Y = 80;           // Y of the main conversation row
const CASCADE_Y_START = 180; // Y of first intermediate in cascade
const CASCADE_STEP_X = 60;   // horizontal nudge per cascade step
const CASCADE_STEP_Y = 50;   // vertical drop per cascade step
const TURN_GAP = 80;         // gap between consecutive turns on the main row
const NODE_WIDTH = 280;      // full-width card (matches --card-max-width)
const COMPACT_WIDTH = 160;   // compact card width
const PADDING = 80;          // left padding

// ─── Return type ─────────────────────────────────────────────────────────────

export interface TimelineLayoutResult {
  nodes: Node[];
  /** Y position per node type — used by GraphCanvas for swimlane overlay rendering. */
  bandPositions: Record<string, number>;
}

/**
 * TYPE_Y kept for backward-compat import in GraphCanvas during transition.
 * @deprecated Use bandPositions from TimelineLayoutResult instead.
 */
export const TYPE_Y: Record<string, number> = {
  Agent: 0,
  Task: 140,
  Interaction: 280,
  Content: 420,
  Knowledge: 560,
};

export const TYPE_LABELS: Record<string, string> = {
  Agent: 'Agent',
  Task: 'Task',
  Interaction: 'Interaction',
  Content: 'Content',
  Knowledge: 'Knowledge',
};

// ─── Layout function ──────────────────────────────────────────────────────────

export function applyTimelineLayout(
  nodes: Node[],
  rawNodes: GraphNode[],
  _edges: Edge[],
  _canvasWidth: number,
): TimelineLayoutResult {
  if (nodes.length === 0) return { nodes, bandPositions: {} };

  const rawById = new Map(rawNodes.map(n => [n.id, n]));

  // ── Step 1: build turns ───────────────────────────────────────────────────
  const turns = buildTurns(rawNodes);

  // Classify every Interaction node
  const mainRowIds = new Set<string>();
  const cascadeMap = new Map<string, number>(); // nodeId → cascade depth index

  for (const turn of turns) {
    if (turn.userNodeId) mainRowIds.add(turn.userNodeId);
    if (turn.finalNodeId) mainRowIds.add(turn.finalNodeId);
    turn.intermediateIds.forEach((id, i) => cascadeMap.set(id, i));
  }

  // ── Step 2: compute cascade zone height ──────────────────────────────────
  const maxCascadeDepth = turns.reduce((max, t) => Math.max(max, t.intermediateIds.length), 0);
  const cascadeBottom = maxCascadeDepth > 0
    ? CASCADE_Y_START + (maxCascadeDepth - 1) * CASCADE_STEP_Y + 80
    : CASCADE_Y_START + 40;

  const dynamicBands: Record<string, number> = {
    Interaction: MAIN_Y,
    Agent: cascadeBottom + 60,
    Task: cascadeBottom + 200,
    Content: cascadeBottom + 340,
    Knowledge: cascadeBottom + 480,
  };

  // ── Step 3: position Interaction nodes turn-by-turn ───────────────────────
  const positionMap = new Map<string, { x: number; y: number; compact: boolean }>();
  let xCursor = PADDING;

  for (const turn of turns) {
    // Place user node
    if (turn.userNodeId) {
      positionMap.set(turn.userNodeId, { x: xCursor, y: MAIN_Y, compact: false });
      xCursor += NODE_WIDTH + 40;
    }

    // Place intermediates in diagonal cascade
    const count = turn.intermediateIds.length;
    for (let i = 0; i < count; i++) {
      const id = turn.intermediateIds[i];
      positionMap.set(id, {
        x: xCursor + i * CASCADE_STEP_X,
        y: CASCADE_Y_START + i * CASCADE_STEP_Y,
        compact: true,
      });
    }

    // Advance cursor past cascade
    const cascadeEndX = count > 0
      ? xCursor + (count - 1) * CASCADE_STEP_X + COMPACT_WIDTH
      : xCursor;

    // Place final node
    if (turn.finalNodeId) {
      const finalX = Math.max(xCursor, cascadeEndX + 40);
      positionMap.set(turn.finalNodeId, { x: finalX, y: MAIN_Y, compact: false });
      xCursor = finalX + NODE_WIDTH + TURN_GAP;
    } else {
      xCursor = cascadeEndX + TURN_GAP;
    }
  }

  // ── Step 4: position non-Interaction nodes with timestamp-proportional X ──
  const interactionTimes = rawNodes
    .filter(n => n.node_type.type === 'Interaction')
    .map(n => new Date(n.created_at).getTime())
    .filter(t => !isNaN(t));

  const tMin = interactionTimes.length > 0 ? Math.min(...interactionTimes) : 0;
  const tMax = interactionTimes.length > 0 ? Math.max(...interactionTimes) : 1;
  const tRange = tMax - tMin || 1;
  const nonInteractionWidth = Math.max(xCursor, 800);

  const nonInteractionByBand = new Map<string, Array<{ node: Node; x: number }>>(); 

  for (const node of nodes) {
    if (positionMap.has(node.id)) continue; // already positioned as Interaction
    const raw = rawById.get(node.id);
    if (!raw) continue;
    const t = new Date(raw.created_at).getTime();
    const x = isNaN(t)
      ? PADDING
      : PADDING + ((t - tMin) / tRange) * (nonInteractionWidth - PADDING * 2);
    const band = raw.node_type.type;
    if (!nonInteractionByBand.has(band)) nonInteractionByBand.set(band, []);
    nonInteractionByBand.get(band)!.push({ node, x });
  }

  // Collision-avoidance pass on each non-Interaction band
  const nonInteractionPositions = new Map<string, { x: number; y: number }>();
  const BAND_NODE_WIDTH = 280;
  const BAND_GAP = 32;

  for (const [band, entries] of nonInteractionByBand) {
    const y = dynamicBands[band] ?? cascadeBottom + 60;
    const sorted = [...entries].sort((a, b) => a.x - b.x);
    let nextAllowedX = -Infinity;
    for (const { node, x } of sorted) {
      const finalX = Math.max(x, nextAllowedX);
      nextAllowedX = finalX + BAND_NODE_WIDTH + BAND_GAP;
      nonInteractionPositions.set(node.id, { x: finalX, y });
    }
  }

  // ── Step 5: apply positions to React Flow nodes ───────────────────────────
  const positioned = nodes.map((node) => {
    const pos = positionMap.get(node.id);
    if (pos) {
      return {
        ...node,
        position: { x: pos.x, y: pos.y },
        data: { ...(node.data as object), compact: pos.compact },
      };
    }
    const nonPos = nonInteractionPositions.get(node.id);
    if (nonPos) {
      return { ...node, position: { x: nonPos.x, y: nonPos.y } };
    }
    return node;
  });

  return { nodes: positioned, bandPositions: dynamicBands };
}
