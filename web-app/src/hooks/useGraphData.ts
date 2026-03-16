import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Node, Edge } from '@xyflow/react';
import type { GraphData, GraphNode } from '../types/graph';
import { applyDagreLayout } from '../layout/dagre';
import { applyTimelineLayout } from '../layout/timeline';

export type LayoutMode = 'dagre' | 'timeline' | 'free';

const STORAGE_PREFIX = 'graphirm:positions:';

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
  // React Flow requires data to be Record<string, unknown>.
  // We spread GraphNode fields into data so node components can cast it back.
  return {
    id: gn.id,
    type: typeMap[gn.node_type.type] ?? 'interaction',
    position: { x: 0, y: 0 },
    data: gn as unknown as Record<string, unknown>,
  };
}

function graphEdgeToFlowEdge(ge: { id: string; source: string; target: string; edge_type: string }): Edge {
  return {
    id: ge.id,
    source: ge.source,
    target: ge.target,
    type: 'labelled',
    data: { edge_type: ge.edge_type },
    markerEnd: { type: 'arrowclosed' as const, color: '#666' },
  };
}

interface UseGraphDataReturn {
  nodes: Node[];
  edges: Edge[];
  layoutMode: LayoutMode;
  setLayoutMode: (mode: LayoutMode) => void;
  onNodesChange: (changes: unknown) => void;
  persistPositions: () => void;
}

export function useGraphData(
  graphData: GraphData | null,
  sessionId: string | null,
  canvasWidth: number,
): UseGraphDataReturn {
  const [layoutMode, setLayoutModeState] = useState<LayoutMode>('dagre');
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const rawNodesRef = useRef<GraphNode[]>([]);

  const rawEdges = useMemo(() => {
    if (!graphData) return [];
    return graphData.edges.map(e => graphEdgeToFlowEdge({
      id: e.id,
      source: e.source,
      target: e.target,
      edge_type: e.edge_type,
    }));
  }, [graphData]);

  const applyLayout = useCallback((
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
  }, [canvasWidth]);

  useEffect(() => {
    if (!graphData) {
      setNodes([]);
      setEdges([]);
      return;
    }

    rawNodesRef.current = graphData.nodes;
    const baseNodes = graphData.nodes.map(graphNodeToFlowNode);
    const flowEdges = rawEdges;

    const laid = applyLayout(baseNodes, flowEdges, layoutMode, graphData.nodes, sessionId);
    setNodes(laid);
    setEdges(flowEdges);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graphData, rawEdges, sessionId]);

  const setLayoutMode = useCallback((mode: LayoutMode) => {
    setLayoutModeState(mode);
    setNodes(prev => applyLayout(prev, edges, mode, rawNodesRef.current, sessionId));
  }, [applyLayout, edges, sessionId]);

  const onNodesChange = useCallback((changes: unknown) => {
    // Only handle position changes to support manual drag in free mode.
    const changeArr = changes as Array<{ type: string; id: string; position?: { x: number; y: number } }>;
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
      setNodes(prev => { savePositions(sessionId, prev); return prev; });
    }
  }, [sessionId]);

  return { nodes, edges, layoutMode, setLayoutMode, onNodesChange, persistPositions };
}
