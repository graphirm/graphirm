import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Node, Edge, XYPosition, NodeChange } from '@xyflow/react';
import { applyNodeChanges } from '@xyflow/react';
import type { GraphData, GraphEdge, GraphNode, Message } from '../types/graph';
import { applyDagreLayout } from '../layout/dagre';
import { applyMasonryLayout } from '../layout/masonry';
import {
  buildPretextSizeMap,
  mergePretextNodeDimensions,
  mergePretextNodeHeightsOnly,
} from '../layout/pretextDimensions';
import { applyTimelineLayout } from '../layout/timeline';

export type LayoutMode = 'dagre' | 'timeline' | 'masonry' | 'free';

export interface NodeFilter {
  query: string;
  types: Set<string>;
  /**
   * When true, only Knowledge nodes tagged as planning (`metadata.planning`) and
   * Content nodes linked from them via `relates_to` edges carrying `artifact_link`
   * (implements / documents) are eligible; type pills and search still apply.
   */
  planGraphOnly: boolean;
}

export const EMPTY_FILTER: NodeFilter = { query: '', types: new Set(), planGraphOnly: false };

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

function syntheticAssistantFromStreaming(msg: Message): GraphNode {
  return {
    id: msg.id,
    node_type: {
      type: 'Interaction',
      role: 'assistant',
      content: msg.content,
    },
    created_at: msg.created_at,
    updated_at: msg.created_at,
    metadata: { streaming_preview: true },
  };
}

/** Latest interaction in the session graph — anchor for provisional assistant placement. */
function precedingInteractionIdForStreaming(graphNodes: GraphNode[]): string | undefined {
  const interactions = graphNodes
    .filter(g => g.node_type.type === 'Interaction')
    .sort((a, b) => a.created_at.localeCompare(b.created_at));
  return interactions[interactions.length - 1]?.id;
}

function stampStreamingNodePretext(
  nodes: Node[],
  streamingId: string,
  layoutMode: LayoutMode,
): Node[] {
  const targets = nodes.filter(n => n.id === streamingId);
  if (targets.length === 0) return nodes;
  const map = buildPretextSizeMap(targets);
  if (layoutMode === 'dagre') {
    return mergePretextNodeDimensions(nodes, map);
  }
  if (layoutMode === 'timeline') {
    return mergePretextNodeHeightsOnly(nodes, map);
  }
  return nodes;
}

/**
 * While the server has not yet persisted the assistant node, show a provisional card
 * with Pretext-sized dimensions. Same id as the final node so SSE patches merge cleanly.
 */
function appendProvisionalStreamingNode(
  finalNodes: Node[],
  flowEdges: Edge[],
  graphNodes: GraphNode[],
  streamingMessage: Message | null,
  layoutMode: LayoutMode,
): Node[] {
  if (!streamingMessage || graphNodes.some(g => g.id === streamingMessage.id)) {
    return finalNodes;
  }
  if (finalNodes.some(n => n.id === streamingMessage.id)) {
    return finalNodes;
  }
  const syn = syntheticAssistantFromStreaming(streamingMessage);
  let prov = graphNodeToFlowNode(syn);
  const pred = precedingInteractionIdForStreaming(graphNodes);
  if (pred) {
    prov = {
      ...prov,
      data: {
        ...(prov.data as Record<string, unknown>),
        precedingInteractionId: pred,
      },
    };
  }
  const existingPositions = new Map(finalNodes.map(n => [n.id, n.position]));
  const merged = positionNewNodes(
    new Set([streamingMessage.id]),
    [...finalNodes, prov],
    flowEdges,
    existingPositions,
  );
  return stampStreamingNodePretext(merged, streamingMessage.id, layoutMode);
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

/** Outgoing `responds_to` source → target (predecessor in the conversation chain). */
function enrichInteractionPredecessors(nodes: Node[], graphEdges: GraphEdge[]): Node[] {
  const pred = new Map<string, string>();
  for (const e of graphEdges) {
    if (e.edge_type === 'responds_to') pred.set(e.source, e.target);
  }
  return nodes.map(n => {
    if (n.type !== 'interaction') return n;
    const p = pred.get(n.id);
    if (!p) return n;
    return {
      ...n,
      data: {
        ...(n.data as Record<string, unknown>),
        precedingInteractionId: p,
      },
    };
  });
}

function isKnowledgeDismissed(gn: GraphNode): boolean {
  return gn.node_type.type === 'Knowledge' && gn.metadata?.dismissed === true;
}

function isPlanningKnowledgeNode(gn: GraphNode): boolean {
  if (gn.node_type.type !== 'Knowledge') return false;
  const p = gn.metadata?.planning;
  return p === true || p === 1;
}

/**
 * Planning Knowledge nodes plus Content targets of planning→content `relates_to` with `artifact_link`.
 */
function computePlanGraphAllowedIds(graphNodes: GraphNode[], edges: GraphEdge[]): Set<string> {
  const planningIds = new Set<string>();
  for (const gn of graphNodes) {
    if (isPlanningKnowledgeNode(gn)) planningIds.add(gn.id);
  }
  const allowed = new Set(planningIds);
  for (const e of edges) {
    if (e.edge_type !== 'relates_to') continue;
    if (!planningIds.has(e.source)) continue;
    const al = e.metadata?.artifact_link;
    if (typeof al !== 'string' || al.trim() === '') continue;
    allowed.add(e.target);
  }
  return allowed;
}

function nodeMatchesTypeAndQuery(gn: GraphNode, filter: NodeFilter): boolean {
  const { query, types } = filter;
  if (types.size > 0 && !types.has(gn.node_type.type)) return false;
  if (query.trim() === '') return true;
  return extractNodeText(gn).toLowerCase().includes(query.toLowerCase());
}

/** `planAllowed` is from `computePlanGraphAllowedIds` when `filter.planGraphOnly`; else `null`. */
function graphNodeMatchesFilter(
  gn: GraphNode,
  filter: NodeFilter,
  planAllowed: Set<string> | null,
): boolean {
  if (isKnowledgeDismissed(gn)) return false;
  if (planAllowed !== null && !planAllowed.has(gn.id)) return false;
  return nodeMatchesTypeAndQuery(gn, filter);
}

function graphEdgeToFlowEdge(ge: GraphEdge): Edge {
  const artifactLink =
    typeof ge.metadata?.artifact_link === 'string' ? ge.metadata.artifact_link : undefined;
  return {
    id: ge.id,
    source: ge.source,
    target: ge.target,
    type: 'labelled',
    data: { edge_type: ge.edge_type, artifact_link: artifactLink },
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

/**
 * Apply filter visibility to a flat array of React Flow nodes.
 * Group nodes are hidden only when all their children are hidden.
 * Annotation nodes are never hidden.
 */
function applyFilterToNodes(
  nodes: Node[],
  graphNodes: GraphNode[],
  filter: NodeFilter,
  graphEdges: GraphEdge[],
): { nodes: Node[]; matchCount: number } {
  const isFiltering =
    filter.query.trim() !== '' || filter.types.size > 0 || filter.planGraphOnly;
  const visibleNodeCount = graphNodes.filter(gn => !isKnowledgeDismissed(gn)).length;

  if (!isFiltering) {
    const mapped = nodes.map(n => {
      if (n.type === 'annotation' || n.type === 'prompt') return { ...n, hidden: false };
      if (n.type === 'group') return n;
      const gn = graphNodes.find(g => g.id === n.id);
      const hidden = gn ? isKnowledgeDismissed(gn) : false;
      return { ...n, hidden };
    });
    const withGroups = mapped.map(n => {
      if (n.type !== 'group') return n;
      const children = mapped.filter(c => c.parentId === n.id);
      const allHidden = children.length > 0 && children.every(c => c.hidden);
      return { ...n, hidden: allHidden };
    });
    return { nodes: withGroups, matchCount: visibleNodeCount };
  }
  const planAllowed = filter.planGraphOnly
    ? computePlanGraphAllowedIds(graphNodes, graphEdges)
    : null;
  const visibleIds = new Set(
    graphNodes
      .filter(gn => graphNodeMatchesFilter(gn, filter, planAllowed))
      .map(gn => gn.id),
  );
  const mapped = nodes.map(n => {
    if (n.type === 'group') {
      const children = nodes.filter(c => c.parentId === n.id);
      const allHidden = children.length > 0 && children.every(c => !visibleIds.has(c.id));
      return { ...n, hidden: allHidden };
    }
    if (n.type === 'annotation' || n.type === 'prompt') return n;
    const gn = graphNodes.find(g => g.id === n.id);
    if (gn && isKnowledgeDismissed(gn)) return { ...n, hidden: true };
    return { ...n, hidden: !visibleIds.has(n.id) };
  });
  return { nodes: mapped, matchCount: visibleIds.size };
}

/** Re-attach client-only `prompt` nodes after layout from server graph. */
function mergeLocalPromptNodes(laidOut: Node[], prev: Node[]): Node[] {
  const prompts = prev.filter(n => n.type === 'prompt');
  if (prompts.length === 0) return laidOut;
  const ids = new Set(laidOut.map(n => n.id));
  return [...laidOut, ...prompts.filter(p => !ids.has(p.id))];
}

interface UseGraphDataReturn {
  nodes: Node[];
  edges: Edge[];
  layoutMode: LayoutMode;
  setLayoutMode: (mode: LayoutMode) => void;
  onNodesChange: (changes: NodeChange[]) => void;
  persistPositions: () => void;
  addNode: (node: Node) => void;
  /** Imperative updates (e.g. prompt node remove / patch). */
  mutateNodes: (fn: (prev: Node[]) => Node[]) => void;
  matchCount: number;
  bandPositions: Record<string, number>;
}

export function useGraphData(
  graphData: GraphData | null,
  sessionId: string | null,
  canvasWidth: number,
  filter: NodeFilter = EMPTY_FILTER,
  isPatchUpdate: boolean = false,
  streamingMessage: Message | null = null,
): UseGraphDataReturn {
  const [layoutMode, setLayoutModeState] = useState<LayoutMode>('dagre');
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [matchCount, setMatchCount] = useState<number>(0);
  const [bandPositions, setBandPositions] = useState<Record<string, number>>({});
  const rawNodesRef = useRef<GraphNode[]>([]);
  const streamingRef = useRef<Message | null>(null);
  streamingRef.current = streamingMessage;

  const rawEdges = useMemo(() => {
    if (!graphData) return [];
    return graphData.edges.map(e => graphEdgeToFlowEdge(e));
  }, [graphData]);

  const applyLayout = useCallback(
    (
      baseNodes: Node[],
      currentEdges: Edge[],
      mode: LayoutMode,
      rawNodes: GraphNode[],
      sid: string | null,
    ): { nodes: Node[]; bandPositions: Record<string, number> } => {
      if (mode === 'dagre') {
        const pretextSizes = buildPretextSizeMap(baseNodes);
        return {
          nodes: applyDagreLayout(baseNodes, currentEdges, 'LR', pretextSizes),
          bandPositions: {},
        };
      }
      if (mode === 'timeline') {
        const result = applyTimelineLayout(baseNodes, rawNodes, currentEdges, canvasWidth);
        return result;
      }
      if (mode === 'masonry') {
        return { nodes: applyMasonryLayout(baseNodes, rawNodes, canvasWidth), bandPositions: {} };
      }
      // free mode: restore persisted positions
      if (sid) {
        const positions = loadPositions(sid);
        return {
          nodes: baseNodes.map(n => ({ ...n, position: positions[n.id] ?? n.position })),
          bandPositions: {},
        };
      }
      return { nodes: baseNodes, bandPositions: {} };
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
    const baseNodes = enrichInteractionPredecessors(
      graphData.nodes.map(graphNodeToFlowNode),
      graphData.edges,
    );
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
        let combined = [...groupNodes, ...grouped.filter(n => n.parentId)];
        combined = appendProvisionalStreamingNode(
          combined,
          flowEdges,
          graphData.nodes,
          streamingRef.current,
          layoutMode,
        );
        const { nodes: withHidden, matchCount: count } = applyFilterToNodes(
          combined,
          graphData.nodes,
          filter,
          graphData.edges,
        );
        setNodes(prev => mergeLocalPromptNodes(withHidden, prev));
        setEdges(flowEdges);
        setMatchCount(count);
        return;
      }
    }

    // Full layout path (non-patch or no existing nodes)
    // Groups only make sense for dagre — timeline positions nodes by timestamp,
    // so parent-relative coordinates would be wrong.
    const useGroups = layoutMode === 'dagre';

    const { grouped, groupNodes } = useGroups
      ? buildGroups(baseNodes, flowEdges)
      : { grouped: baseNodes, groupNodes: [] as Node[] };
    const { nodes: laid, bandPositions: newBandPositions } = applyLayout(grouped, flowEdges, layoutMode, graphData.nodes, sessionId);
    setBandPositions(newBandPositions);

    let finalNodes: Node[];
    if (useGroups && groupNodes.length > 0) {
      // Position group nodes to wrap their children.
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
      // Group nodes must come before their children in the array.
      finalNodes = [...positionedGroups, ...rebased];
    } else {
      finalNodes = laid;
    }

    if (layoutMode === 'dagre' || layoutMode === 'timeline') {
      const map = buildPretextSizeMap(
        finalNodes.filter(
          n => n.type && n.type !== 'group' && n.type !== 'annotation' && n.type !== 'prompt',
        ),
      );
      finalNodes =
        layoutMode === 'dagre'
          ? mergePretextNodeDimensions(finalNodes, map)
          : mergePretextNodeHeightsOnly(finalNodes, map);
    }

    finalNodes = appendProvisionalStreamingNode(
      finalNodes,
      flowEdges,
      graphData.nodes,
      streamingRef.current,
      layoutMode,
    );

    // Apply filter: stamp hidden: true on non-matching nodes.
    const { nodes: withHidden, matchCount: count } = applyFilterToNodes(
      finalNodes,
      graphData.nodes,
      filter,
      graphData.edges,
    );
    setMatchCount(count);
    setNodes(prev => mergeLocalPromptNodes(withHidden, prev));
    setEdges(flowEdges);
    // filter intentionally excluded from deps: filter-only changes are handled by
    // the second useEffect below to avoid re-running the expensive layout algorithm
    // on every keystroke.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graphData, rawEdges, sessionId, isPatchUpdate, layoutMode]);

  // Add or resize provisional assistant while SSE streams (no full dagre relayout).
  useEffect(() => {
    if (!streamingMessage || !graphData) return;
    if (graphData.nodes.some(g => g.id === streamingMessage.id)) return;

    setNodes(prev => {
      const i = prev.findIndex(n => n.id === streamingMessage.id);
      if (i < 0) {
        return appendProvisionalStreamingNode(
          prev,
          rawEdges,
          graphData.nodes,
          streamingMessage,
          layoutMode,
        );
      }
      const syn = syntheticAssistantFromStreaming(streamingMessage);
      let next: Node = {
        ...prev[i],
        data: syn as unknown as Record<string, unknown>,
      };
      const pred = precedingInteractionIdForStreaming(graphData.nodes);
      if (pred) {
        next = {
          ...next,
          data: { ...next.data, precedingInteractionId: pred },
        };
      }
      const replaced = prev.map((n, j) => (j === i ? next : n));
      return stampStreamingNodePretext(replaced, streamingMessage.id, layoutMode);
    });
  }, [streamingMessage, graphData, layoutMode, rawEdges]);

  useEffect(() => {
    if (!graphData) return;
    const isFiltering =
      filter.query.trim() !== '' || filter.types.size > 0 || filter.planGraphOnly;

    const dismissed = new Set(
      graphData.nodes.filter(isKnowledgeDismissed).map(g => g.id),
    );

    // Compute visibleIds and matchCount outside the state updater — updaters must be pure.
    const planAllowed = filter.planGraphOnly
      ? computePlanGraphAllowedIds(graphData.nodes, graphData.edges)
      : null;
    const visibleIds = isFiltering
      ? new Set(
          graphData.nodes
            .filter(gn => graphNodeMatchesFilter(gn, filter, planAllowed))
            .map(gn => gn.id),
        )
      : null;
    setMatchCount(
      visibleIds
        ? visibleIds.size
        : graphData.nodes.filter(gn => !isKnowledgeDismissed(gn)).length,
    );

    // Apply hidden flags using a pure functional updater (visibleIds is already fully computed).
    setNodes(prev => {
      if (!visibleIds) {
        const step = prev.map(n => {
          if (n.type === 'annotation' || n.type === 'prompt') return { ...n, hidden: false };
          if (n.type === 'group') return n;
          return { ...n, hidden: dismissed.has(n.id) };
        });
        return step.map(n => {
          if (n.type !== 'group') return n;
          const children = step.filter(c => c.parentId === n.id);
          const allHidden = children.length > 0 && children.every(c => c.hidden);
          return { ...n, hidden: allHidden };
        });
      }
      return prev.map(n => {
        if (n.type === 'group') {
          const children = prev.filter(c => c.parentId === n.id);
          const allHidden = children.length > 0 && children.every(c => !visibleIds.has(c.id));
          return { ...n, hidden: allHidden };
        }
        if (n.type === 'annotation' || n.type === 'prompt') return n;
        if (dismissed.has(n.id)) return { ...n, hidden: true };
        return { ...n, hidden: !visibleIds.has(n.id) };
      });
    });
  }, [filter, graphData]);

  const setLayoutMode = useCallback(
    (mode: LayoutMode) => {
      setLayoutModeState(mode);
      if (!graphData) return;

      const baseNodes = enrichInteractionPredecessors(
        graphData.nodes.map(graphNodeToFlowNode),
        graphData.edges,
      );
      const useGroups = mode === 'dagre';

      const { grouped, groupNodes } = useGroups
        ? buildGroups(baseNodes, edges)
        : { grouped: baseNodes, groupNodes: [] as Node[] };

      const { nodes: laid, bandPositions: newBandPositions } = applyLayout(grouped, edges, mode, rawNodesRef.current, sessionId);
      setBandPositions(newBandPositions);

      let finalNodes: Node[];
      if (useGroups && groupNodes.length > 0) {
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
          return { ...g, position: { x: minX, y: minY }, style: { width: maxX - minX, height: maxY - minY } };
        });
        const groupOrigins = new Map(positionedGroups.map(g => [g.id, g.position]));
        const rebased = laid.map(n => {
          if (!n.parentId) return n;
          const origin = groupOrigins.get(n.parentId);
          if (!origin) return n;
          return { ...n, position: { x: n.position.x - origin.x, y: n.position.y - origin.y } };
        });
        finalNodes = [...positionedGroups, ...rebased];
      } else {
        finalNodes = laid;
      }

      if (mode === 'dagre' || mode === 'timeline') {
        const map = buildPretextSizeMap(
          finalNodes.filter(
            n => n.type && n.type !== 'group' && n.type !== 'annotation' && n.type !== 'prompt',
          ),
        );
        finalNodes =
          mode === 'dagre'
            ? mergePretextNodeDimensions(finalNodes, map)
            : mergePretextNodeHeightsOnly(finalNodes, map);
      }

      finalNodes = appendProvisionalStreamingNode(
        finalNodes,
        edges,
        graphData.nodes,
        streamingRef.current,
        mode,
      );

      const { nodes: withHidden } = applyFilterToNodes(
        finalNodes,
        graphData.nodes,
        filter,
        graphData.edges,
      );
      setNodes(prev => mergeLocalPromptNodes(withHidden, prev));
    },
    [applyLayout, edges, graphData, sessionId, filter],
  );

  const onNodesChange = useCallback((changes: NodeChange[]) => {
    setNodes(prev => applyNodeChanges(changes, prev));
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

  const mutateNodes = useCallback((fn: (prev: Node[]) => Node[]) => {
    setNodes(fn);
  }, []);

  return {
    nodes,
    edges,
    layoutMode,
    setLayoutMode,
    onNodesChange,
    persistPositions,
    addNode,
    mutateNodes,
    matchCount,
    bandPositions,
  };
}
