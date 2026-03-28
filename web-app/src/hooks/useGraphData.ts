import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Node, Edge, XYPosition } from '@xyflow/react';
import type { GraphData, GraphNode } from '../types/graph';
import { applyDagreLayout } from '../layout/dagre';
import { applyTimelineLayout } from '../layout/timeline';

export type LayoutMode = 'dagre' | 'timeline' | 'free';

export interface NodeFilter {
  query: string;
  types: Set<string>;
}

export const EMPTY_FILTER: NodeFilter = { query: '', types: new Set() };

// Layout stability: persist positions per session to avoid jumps
const STORAGE_PREFIX = 'graphirm:positions:';
const GROUP_COLOR = '#4fc3f7';

// Default spacing for incremental positioning
const INCREMENTAL_OFFSET = 220;
const INCREMENTAL_SPACING = 120;

/**
 * Incrementally position only new nodes without affecting existing ones.
 * Places new nodes relative to their parent/target nodes to maintain visual continuity.
 */
function positionNewNodes(
  newNodeIds: Set<string>,
  nodes: Node[],
  edges: Edge[],
  existingPositions: Map<string, XYPosition>,
): Node[] {
  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  const result = [...nodes];

  // Build adjacency info to find parent/target nodes for new nodes
  const producesMap = new Map<string, string[]>(); // source -> targets (produces edges)
  const targetMap = new Map<string, string[]>();   // target -> sources (incoming edges)

  for (const edge of edges) {
    const et = (edge.data as { edge_type?: string } | undefined)?.edge_type ?? '';
    if (et === 'produces') {
      const targets = producesMap.get(edge.source) ?? [];
      targets.push(edge.target);
      producesMap.set(edge.source, targets);
    }
    // Track incoming edges for all edge types
    const sources = targetMap.get(edge.target) ?? [];
    sources.push(edge.source);
    targetMap.set(edge.target, sources);
  }

  // Track where we're placing new nodes to avoid overlap
  const childYOffset = new Map<string, number>();

  for (const nodeId of newNodeIds) {
    const node = nodeMap.get(nodeId);
    if (!node) continue;

    const incoming = targetMap.get(nodeId) ?? [];
    const parentIds = incoming.filter(id => nodeMap.has(id));

    if (parentIds.length > 0) {
      // Find the first parent with a valid position
      for (const parentId of parentIds) {
        const parent = nodeMap.get(parentId);
        if (!parent) continue;
        const parentPos = existingPositions.get(parentId) ?? parent.position;

        // Calculate position relative to parent
        let offsetX = INCREMENTAL_OFFSET;
        let offsetY = 0;

        // If this node is produced by the parent, place it to the right
        if (producesMap.has(parentId) && producesMap.get(parentId)?.includes(nodeId)) {
          offsetX = INCREMENTAL_OFFSET;
          offsetY = 0;
        } else {
          // For other relationships, place below with some spacing
          offsetX = INCREMENTAL_OFFSET / 2;
          const childCount = childYOffset.get(parentId) ?? 0;
          offsetY = childCount * INCREMENTAL_SPACING + 40;
          childYOffset.set(parentId, childCount + 1);
        }

        result.find(n => n.id === nodeId)!.position = {
          x: parentPos.x + offsetX,
          y: parentPos.y + offsetY,
        };
        break;
      }
    } else {
      // No parents - place at a reasonable default offset
      const existingPositionsArray = Array.from(existingPositions.values());
      let baseX = 100;
      let baseY = 100;
      if (existingPositionsArray.length > 0) {
        const maxX = Math.max(...existingPositionsArray.map(p => p.x));
        const maxY = Math.max(...existingPositionsArray.map(p => p.y));
        baseX = maxX + INCREMENTAL_OFFSET;
        baseY = maxY < 500 ? maxY + INCREMENTAL_SPACING : 100;
      }
      result.find(n => n.id === nodeId)!.position = { x: baseX, y: baseY };
    }
  }

  return result;
}

function loadPositions(sessionId: string): Record<string, { x: number; y: number }> {
  try {
    const raw = localStorage.getItem(STORAGE_PREFIX + sessionId);
    return raw ? (JSON.parse(raw) as Record<string, { x: number; y: number }>) : {};
  } catch {
    return {};
  }
}

function savePositions(sessionId: string, nodes: Node[]): void {
  const positions: Record<string, { x: number; y: number }> = {};
  for (const n of nodes) {
    positions[n.id] = n.position;
  }
  try {
    localStorage.setItem(STORAGE_PREFIX + sessionId, JSON.stringify(positions));
  } catch {
    // Storage quota exceeded — silently ignore.
  }
}

function graphNodeToFlowNode(gn: GraphNode): Node {
  const typeMap: Record<string, string> = {
    Interaction: 'interaction',
    Agent: 'agent',
    Content: 'content',
    Task: 'task',
    Knowledge: 'knowledge',
  };
  return {
    id: gn.id,
    type: typeMap[gn.node_type.type] ?? 'interaction',
    position: { x: 0, y: 0 },
    // React Flow requires data: Record<string, unknown>; cast GraphNode to satisfy it.
    data: gn as unknown as Record<string, unknown>,
  };
}

function graphEdgeToFlowEdge(ge: {
  id: string;
  source: string;
  target: string;
  edge_type: string;
}): Edge {
  return {
    id: ge.id,
    source: ge.source,
    target: ge.target,
    type: 'labelled',
    data: { edge_type: ge.edge_type },
    markerEnd: { type: 'arrowclosed' as const, color: '#666' },
  };
}

/**
 * Build group nodes + assign parentId for interaction clusters.
 * Each Interaction node becomes the root of a group that includes
 * its directly produced Content/Knowledge nodes.
 */
function buildGroups(
  nodes: Node[],
  edges: Edge[],
): { grouped: Node[]; groupNodes: Node[] } {
  const produces = new Map<string, string[]>();
  for (const e of edges) {
    const et = (e.data as { edge_type?: string } | undefined)?.edge_type ?? '';
    if (et === 'produces') {
      const targets = produces.get(e.source) ?? [];
      targets.push(e.target);
      produces.set(e.source, targets);
    }
  }

  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  const childToGroup = new Map<string, string>();
  const groupNodes: Node[] = [];

  let groupIdx = 0;
  for (const node of nodes) {
    if (node.type !== 'interaction') continue;
    const children = produces.get(node.id) ?? [];
    if (children.length === 0) continue;

    const groupId = `__group_${groupIdx++}`;
    childToGroup.set(node.id, groupId);
    for (const cid of children) childToGroup.set(cid, groupId);

    // We'll size the group node dynamically after layout, but seed it.
    groupNodes.push({
      id: groupId,
      type: 'group',
      position: { x: 0, y: 0 },
      style: { width: 400, height: 200 },
      data: {
        label: `Turn ${groupIdx}`,
        color: GROUP_COLOR,
        collapsed: false,
        onToggle: () => {},
      } as Record<string, unknown>,
    });
  }

  const grouped = nodes.map(n => {
    const gid = childToGroup.get(n.id);
    if (!gid) return n;
    return {
      ...n,
      parentId: gid,
      extent: 'parent' as const,
    };
  });

  // Verify all referenced parents exist in nodeMap (sanity guard).
  const validGroupIds = new Set(groupNodes.map(g => g.id));
  const safe = grouped.map(n =>
    n.parentId && !validGroupIds.has(n.parentId)
      ? { ...n, parentId: undefined, extent: undefined }
      : n,
  );

  // Suppress unused var warning
  void nodeMap;

  return { grouped: safe, groupNodes };
}

function extractNodeText(gn: GraphNode): string {
  const nt = gn.node_type;
  switch (nt.type) {
    case 'Interaction': return nt.content;
    case 'Agent':       return `${nt.name} ${nt.model} ${nt.system_prompt ?? ''}`;
    case 'Content':     return `${nt.body} ${nt.path ?? ''}`;
    case 'Task':        return `${nt.title} ${nt.description}`;
    case 'Knowledge':   return `${nt.entity} ${nt.entity_type} ${nt.summary}`;
    default:            return '';
  }
}

function nodeMatchesFilter(gn: GraphNode, filter: NodeFilter): boolean {
  const { query, types } = filter;
  if (types.size > 0 && !types.has(gn.node_type.type)) return false;
  if (query.trim() === '') return true;
  return extractNodeText(gn).toLowerCase().includes(query.toLowerCase());
}

/**
 * Apply filter visibility to a flat array of React Flow nodes.
 * Group nodes are hidden only when all their children are hidden.
 * Annotation nodes are never hidden.
 */
function applyFilterToNodes(
  nodes: Node[],
  graphNodes: GraphNode[],
  filter: NodeFilter,
): { nodes: Node[]; matchCount: number } {
  const isFiltering = filter.query.trim() !== '' || filter.types.size > 0;
  if (!isFiltering) {
    return { nodes: nodes.map(n => ({ ...n, hidden: false })), matchCount: graphNodes.length };
  }
  const visibleIds = new Set(
    graphNodes.filter(gn => nodeMatchesFilter(gn, filter)).map(gn => gn.id),
  );
  const mapped = nodes.map(n => {
    if (n.type === 'group') {
      const children = nodes.filter(c => c.parentId === n.id);
      const allHidden = children.length > 0 && children.every(c => !visibleIds.has(c.id));
      return { ...n, hidden: allHidden };
    }
    if (n.type === 'annotation') return n;
    return { ...n, hidden: !visibleIds.has(n.id) };
  });
  return { nodes: mapped, matchCount: visibleIds.size };
}

interface UseGraphDataReturn {
  nodes: Node[];
  edges: Edge[];
  layoutMode: LayoutMode;
  setLayoutMode: (mode: LayoutMode) => void;
  onNodesChange: (changes: unknown) => void;
  persistPositions: () => void;
  addNode: (node: Node) => void;
  matchCount: number;
}

export function useGraphData(
  graphData: GraphData | null,
  sessionId: string | null,
  canvasWidth: number,
  filter: NodeFilter = EMPTY_FILTER,
  isPatchUpdate: boolean = false,
): UseGraphDataReturn {
  const [layoutMode, setLayoutModeState] = useState<LayoutMode>('dagre');
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [matchCount, setMatchCount] = useState<number>(0);
  const rawNodesRef = useRef<GraphNode[]>([]);

  const rawEdges = useMemo(() => {
    if (!graphData) return [];
    return graphData.edges.map(e =>
      graphEdgeToFlowEdge({
        id: e.id,
        source: e.source,
        target: e.target,
        edge_type: e.edge_type,
      }),
    );
  }, [graphData]);

  const applyLayout = useCallback(
    (
      baseNodes: Node[],
      currentEdges: Edge[],
      mode: LayoutMode,
      rawNodes: GraphNode[],
      sid: string | null,
    ): Node[] => {
      if (mode === 'dagre') {
        return applyDagreLayout(baseNodes, currentEdges, 'LR');
      }
      if (mode === 'timeline') {
        return applyTimelineLayout(baseNodes, rawNodes, currentEdges, canvasWidth);
      }
      // free mode: restore persisted positions
      if (sid) {
        const positions = loadPositions(sid);
        return baseNodes.map(n => ({
          ...n,
          position: positions[n.id] ?? n.position,
        }));
      }
      return baseNodes;
    },
    [canvasWidth],
  );

  useEffect(() => {
    if (!graphData) {
      setNodes([]);
      setEdges([]);
      return;
    }

    rawNodesRef.current = graphData.nodes;
    const baseNodes = graphData.nodes.map(graphNodeToFlowNode);
    const flowEdges = rawEdges;

    // If this is a patch update with only a few new nodes, position them incrementally
    // to avoid jarring layout shifts during agent execution.
    if (isPatchUpdate && nodes.length > 0 && baseNodes.length > nodes.length) {
      // Only new nodes were added - use incremental positioning
      const newNodeIds = new Set<string>();
      const existingIds = new Set(nodes.map(n => n.id));
      
      for (const newNode of baseNodes) {
        if (!existingIds.has(newNode.id)) {
          newNodeIds.add(newNode.id);
        }
      }

      if (newNodeIds.size > 0) {
        // Build existing positions map from current nodes
        const existingPositions = new Map<string, XYPosition>();
        for (const node of nodes) {
          existingPositions.set(node.id, node.position);
        }

        // Position new nodes incrementally relative to their parents
        const positioned = positionNewNodes(newNodeIds, baseNodes, flowEdges, existingPositions);
        
        // Apply grouping and filter as usual, but preserve positions from positionNewNodes
        const { grouped, groupNodes } = buildGroups(positioned, flowEdges);
        const { nodes: withHidden, matchCount: count } = applyFilterToNodes(
          [...groupNodes, ...grouped.filter(n => n.parentId)],
          graphData.nodes,
          filter,
        );
        setNodes(withHidden);
        setEdges(flowEdges);
        setMatchCount(count);
        return;
      }
    }

    // Full layout path (non-patch or no existing nodes)
    // Build groups, then apply layout to content nodes (group nodes get positioned separately).
    const { grouped, groupNodes } = buildGroups(baseNodes, flowEdges);
    const laid = applyLayout(grouped, flowEdges, layoutMode, graphData.nodes, sessionId);

    // Position group nodes to wrap their children.
    const PAD = 24;
    const positionedGroups = groupNodes.map(g => {
      const children = laid.filter(n => n.parentId === g.id);
      if (children.length === 0) return g;
      const xs = children.map(c => c.position.x);
      const ys = children.map(c => c.position.y);
      const minX = Math.min(...xs) - PAD;
      const minY = Math.min(...ys) - PAD;
      const maxX = Math.max(...xs) + 200 + PAD; // 200 = approx card width
      const maxY = Math.max(...ys) + 120 + PAD; // 120 = approx card height
      // Rebase children positions relative to group origin.
      return {
        ...g,
        position: { x: minX, y: minY },
        style: { width: maxX - minX, height: maxY - minY },
      };
    });

    // Adjust children positions to be relative to their group.
    const groupOrigins = new Map(positionedGroups.map(g => [g.id, g.position]));
    const rebased = laid.map(n => {
      if (!n.parentId) return n;
      const origin = groupOrigins.get(n.parentId);
      if (!origin) return n;
      return {
        ...n,
        position: {
          x: n.position.x - origin.x,
          y: n.position.y - origin.y,
        },
      };
    });

    // Apply filter: stamp hidden: true on non-matching nodes.
    const { nodes: withHidden, matchCount: count } = applyFilterToNodes(
      [...positionedGroups, ...rebased],
      graphData.nodes,
      filter,
    );
    setMatchCount(count);
    // Group nodes must come before their children in the array.
    setNodes(withHidden);
    setEdges(flowEdges);
    // filter intentionally excluded from deps: filter-only changes are handled by
    // the second useEffect below to avoid re-running the expensive layout algorithm
    // on every keystroke.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graphData, rawEdges, sessionId, isPatchUpdate, layoutMode]);

  useEffect(() => {
    if (!graphData) return;
    const isFiltering = filter.query.trim() !== '' || filter.types.size > 0;

    // Compute visibleIds and matchCount outside the state updater — updaters must be pure.
    const visibleIds = isFiltering
      ? new Set(graphData.nodes.filter(gn => nodeMatchesFilter(gn, filter)).map(gn => gn.id))
      : null;
    setMatchCount(visibleIds ? visibleIds.size : graphData.nodes.length);

    // Apply hidden flags using a pure functional updater (visibleIds is already fully computed).
    setNodes(prev => {
      if (!visibleIds) return prev.map(n => ({ ...n, hidden: false }));
      return prev.map(n => {
        if (n.type === 'group') {
          const children = prev.filter(c => c.parentId === n.id);
          const allHidden = children.length > 0 && children.every(c => !visibleIds.has(c.id));
          return { ...n, hidden: allHidden };
        }
        if (n.type === 'annotation') return n;
        return { ...n, hidden: !visibleIds.has(n.id) };
      });
    });
  }, [filter, graphData]);

  const setLayoutMode = useCallback(
    (mode: LayoutMode) => {
      setLayoutModeState(mode);
      // Re-derive from raw data rather than mutating existing nodes.
      if (!graphData) return;
      const baseNodes = graphData.nodes.map(graphNodeToFlowNode);
      const { grouped, groupNodes } = buildGroups(baseNodes, edges);
      const laid = applyLayout(grouped, edges, mode, rawNodesRef.current, sessionId);
      const PAD = 24;
      const positionedGroups = groupNodes.map(g => {
        const children = laid.filter(n => n.parentId === g.id);
        if (children.length === 0) return g;
        const xs = children.map(c => c.position.x);
        const ys = children.map(c => c.position.y);
        const minX = Math.min(...xs) - PAD;
        const minY = Math.min(...ys) - PAD;
        const maxX = Math.max(...xs) + 200 + PAD;
        const maxY = Math.max(...ys) + 120 + PAD;
        return {
          ...g,
          position: { x: minX, y: minY },
          style: { width: maxX - minX, height: maxY - minY },
        };
      });
      const groupOrigins = new Map(positionedGroups.map(g => [g.id, g.position]));
      const rebased = laid.map(n => {
        if (!n.parentId) return n;
        const origin = groupOrigins.get(n.parentId);
        if (!origin) return n;
        return { ...n, position: { x: n.position.x - origin.x, y: n.position.y - origin.y } };
      });

      const { nodes: withHidden } = applyFilterToNodes(
        [...positionedGroups, ...rebased],
        graphData.nodes,
        filter,
      );
      setNodes(withHidden);
    },
    [applyLayout, edges, graphData, sessionId, filter],
  );

  const onNodesChange = useCallback((changes: unknown) => {
    const changeArr = changes as Array<{
      type: string;
      id: string;
      position?: { x: number; y: number };
    }>;
    setNodes(prev => {
      const map = new Map(prev.map(n => [n.id, n]));
      for (const change of changeArr) {
        if (change.type === 'position' && change.position) {
          const existing = map.get(change.id);
          if (existing) {
            map.set(change.id, { ...existing, position: change.position });
          }
        }
      }
      return [...map.values()];
    });
  }, []);

  const persistPositions = useCallback(() => {
    if (sessionId) {
      setNodes(prev => {
        savePositions(sessionId, prev);
        return prev;
      });
    }
  }, [sessionId]);

  const addNode = useCallback((node: Node) => {
    setNodes(prev => [...prev, node]);
  }, []);

  return { nodes, edges, layoutMode, setLayoutMode, onNodesChange, persistPositions, addNode, matchCount };
}
