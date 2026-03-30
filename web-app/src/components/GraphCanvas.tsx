import { useCallback, useEffect, useRef, useState, useMemo } from 'react';
import { useTheme } from '../hooks/useTheme';
import {
  ReactFlow,
  Background,
  MiniMap,
  Controls,
  BackgroundVariant,
  useReactFlow,
  ReactFlowProvider,
  Node,
  type Connection,
  type Edge,
} from '@xyflow/react';
import type { NodeTypes, EdgeTypes } from '@xyflow/react';
import type { GraphData, GraphNode, PendingApproval } from '../types/graph';
import { useGraphData, EMPTY_FILTER } from '../hooks/useGraphData';
import type { LayoutMode, NodeFilter } from '../hooks/useGraphData';
import { InteractionNode } from './nodes/InteractionNode';
import { AgentNode } from './nodes/AgentNode';
import { ContentNode } from './nodes/ContentNode';
import { TaskNode } from './nodes/TaskNode';
import { KnowledgeNode } from './nodes/KnowledgeNode';
import { AnnotationNode } from './nodes/AnnotationNode';
import { GroupNode } from './nodes/GroupNode';
import { PromptNode } from './nodes/PromptNode';
import { NodePopover } from './nodes/NodePopover';
import { LabelledEdge } from './edges/LabelledEdge';
import { Toolbar } from './Toolbar';
import { SteerContext } from '../context/SteerContext';
import { FocusContext } from '../context/FocusContext';
import { ZoomProvider } from '../context/ZoomContext';
import { PopoverProvider } from '../context/PopoverContext';
import {
  CascadeCollapseGenerationContext,
} from '../context/CascadeCollapseContext';
import type { PopoverActions } from '../context/PopoverContext';
import {
  GraphCanvasActionsProvider,
  type GraphCanvasActionsValue,
} from '../context/GraphCanvasActionsContext';
import { FloatingInput } from './FloatingInput';
import { NodeReplyInput } from './NodeReplyInput';
import { HitlOverlay } from './HitlOverlay';
// TYPE_Y removed — swimlane positions now come from bandPositions via useGraphData
import styles from './GraphCanvas.module.css';
import nodeStyles from '../styles/nodes.module.css';
import { useNodeNavigation } from '../hooks/useNodeNavigation';
import { api } from '../api/client';

const NODE_TYPES: NodeTypes = {
  interaction: InteractionNode,
  agent: AgentNode,
  content: ContentNode,
  task: TaskNode,
  knowledge: KnowledgeNode,
  annotation: AnnotationNode,
  group: GroupNode,
  prompt: PromptNode,
};

const EDGE_TYPES: EdgeTypes = {
  labelled: LabelledEdge,
};

interface GraphCanvasProps {
  graphData: GraphData | null;
  sessionId: string | null;
  selectedNodeId: string | null;
  onNodeSelect: (nodeId: string | null) => void;
  onSteerFromNode: (nodeId: string) => void;
  onFitViewRef?: (cb: () => void) => void;
  onCycleLayoutRef?: (cb: () => void) => void;
  chatCollapsed?: boolean;
  /** Second arg: explicit graph context root (steer); omit for chat / floating input default. */
  onSend?: (content: string, contextRoot?: string) => void;
  isThinking?: boolean;
  pendingApproval?: PendingApproval | null;
  onApprove?: (nodeId: string) => void;
  onReject?: (nodeId: string, reason?: string) => void;
  onModify?: (nodeId: string, modifiedArgs: string) => void;
}

const LAYOUT_CYCLE: LayoutMode[] = ['dagre', 'timeline', 'masonry', 'free'];

function GraphCanvasInner({
  graphData,
  sessionId,
  onNodeSelect,
  onSteerFromNode,
  onFitViewRef,
  onCycleLayoutRef,
  chatCollapsed = false,
  onSend,
  isThinking = false,
  pendingApproval,
  onApprove,
  onReject,
  onModify,
}: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const canvasWidth = containerRef.current?.clientWidth ?? 800;
  const { theme } = useTheme();

  // Read CSS variables so MiniMap/Controls/Background respond to theme switches
  const cssVar = (name: string) =>
    getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  const [filter, setFilter] = useState<NodeFilter>(EMPTY_FILTER);
  const [cascadeCollapseGeneration, setCascadeCollapseGeneration] = useState(0);
  const [editingUserNodeId, setEditingUserNodeId] = useState<string | null>(null);

  // Ctrl+F (or Cmd+F) focuses the search bar when hovering the graph pane.
  // Escape clears the filter and blurs the input when it is focused.
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
        if (containerRef.current?.matches(':hover')) {
          e.preventDefault();
          searchInputRef.current?.focus();
          searchInputRef.current?.select();
        }
      }
      if (e.key === 'Escape' && document.activeElement === searchInputRef.current) {
        setFilter(EMPTY_FILTER);
        searchInputRef.current?.blur();
      }
      // Clear focus when pressing Escape (outside of search input)
      if (e.key === 'Escape' && document.activeElement !== searchInputRef.current) {
        onNodeSelect(null);
      }
    }
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
    // setFilter and EMPTY_FILTER are stable references (useState setter + module constant).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const {
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
  } = useGraphData(graphData, sessionId, canvasWidth, filter);

  const { fitView, screenToFlowPosition } = useReactFlow();

  const promptIdRef = useRef<string | null>(null);
  const promptActionsRef = useRef({
    send: (_text: string, _ctx: string | null) => {},
    cancel: () => {},
  });
  const [promptContextEdge, setPromptContextEdge] = useState<Edge | null>(null);

  const clearPromptNode = useCallback(() => {
    const id = promptIdRef.current;
    promptIdRef.current = null;
    setPromptContextEdge(null);
    if (id) {
      mutateNodes(prev => prev.filter(n => n.id !== id));
    }
  }, [mutateNodes]);

  useEffect(() => {
    promptIdRef.current = null;
    setPromptContextEdge(null);
    mutateNodes(prev => prev.filter(n => n.type !== 'prompt'));
  }, [sessionId, mutateNodes]);

  const { focusedNodeId, activateNodeId, clearActivation, replyingToNodeId, clearReply } = useNodeNavigation(nodes, edges);

  const [popoverState, setPopoverState] = useState<{
    nodeId: string;
    position: { x: number; y: number };
  } | null>(null);

  // Compute dimmed nodes: non-focused nodes that are NOT immediate 1-hop neighbors
  const dimmedNodeIds = useMemo(() => {
    if (!focusedNodeId) return new Set<string>();
    const dimmed = new Set<string>();
    const nodeMap = new Map(nodes.map(n => [n.id, n]));
    const focusedNode = nodeMap.get(focusedNodeId);
    if (!focusedNode) return new Set();
    
    // Collect neighbors (nodes connected by any edge to focused node)
    const neighbors = new Set<string>();
    for (const edge of edges) {
      if (edge.source === focusedNodeId) {
        neighbors.add(edge.target);
      } else if (edge.target === focusedNodeId) {
        neighbors.add(edge.source);
      }
    }
    
    // All non-focused, non-neighbor nodes are dimmed
    for (const node of nodes) {
      if (node.id !== focusedNodeId && !neighbors.has(node.id)) {
        dimmed.add(node.id);
      }
    }
    return dimmed;
  }, [focusedNodeId, nodes, edges]);
  
  const flowEdges = useMemo(
    () => (promptContextEdge ? [...edges, promptContextEdge] : edges),
    [edges, promptContextEdge],
  );

  // Compute dimmed edges: edges where BOTH source and target are dimmed
  const dimmedEdgeIds = useMemo(() => {
    if (dimmedNodeIds.size === 0) return new Set<string>();
    const dimmed = new Set<string>();
    for (const edge of flowEdges) {
      if (dimmedNodeIds.has(edge.source) && dimmedNodeIds.has(edge.target)) {
        dimmed.add(edge.id);
      }
    }
    return dimmed;
  }, [flowEdges, dimmedNodeIds]);

  useEffect(() => {
    if (!focusedNodeId) return;
    fitView({ nodes: [{ id: focusedNodeId }], padding: 0.4, duration: 300, maxZoom: 1.5 });
  }, [focusedNodeId, fitView]);

  // Expose fitView and layout-cycle callbacks to parent via ref callbacks.
  useEffect(() => {
    onFitViewRef?.(() => fitView({ padding: 0.12, duration: 400 }));
  }, [fitView, onFitViewRef]);

  useEffect(() => {
    onCycleLayoutRef?.(() => {
      const idx = LAYOUT_CYCLE.indexOf(layoutMode);
      const next = LAYOUT_CYCLE[(idx + 1) % LAYOUT_CYCLE.length];
      setLayoutMode(next);
      setTimeout(() => fitView({ padding: 0.12, duration: 400 }), 50);
    });
  }, [layoutMode, setLayoutMode, fitView, onCycleLayoutRef]);
  const [annotationCount, setAnnotationCount] = useState(0);

  // Store steer callback in a ref — node components access it via a shared context
  // rather than injecting a function into node data (which causes circular JSON errors
  // when React Flow internally serializes nodes).
  const steerCallbackRef = useRef(onSteerFromNode);
  steerCallbackRef.current = onSteerFromNode;

  const handleAddAnnotation = useCallback(() => {
    const count = annotationCount + 1;
    setAnnotationCount(count);
    const newNode: Node = {
      id: `annotation_${Date.now()}`,
      type: 'annotation',
      position: { x: 100 + count * 20, y: 100 + count * 20 },
      data: { text: '' } as Record<string, unknown>,
    };
    addNode(newNode);
  }, [annotationCount, addNode]);

  const handleNodeClick = useCallback(
    (_: React.MouseEvent, node: { id: string }) => {
      onNodeSelect(node.id);
    },
    [onNodeSelect],
  );

  const handleNodeDragStop = useCallback(() => {
    if (layoutMode === 'free') {
      persistPositions();
    }
  }, [layoutMode, persistPositions]);

  const openPromptAt = useCallback(
    (position: { x: number; y: number }) => {
      if (!sessionId || popoverState) return;
      const id = `prompt-${Date.now()}`;
      promptIdRef.current = id;
      setPromptContextEdge(null);
      const node: Node = {
        id,
        type: 'prompt',
        position,
        width: 300,
        height: 168,
        style: { width: 300, minHeight: 168 },
        data: {
          contextRoot: null,
          onSend: (text: string, ctx: string | null) => {
            promptActionsRef.current.send(text, ctx);
          },
          onCancel: () => {
            promptActionsRef.current.cancel();
          },
        } as Record<string, unknown>,
      };
      mutateNodes(prev => [...prev.filter(n => n.type !== 'prompt'), node]);
    },
    [sessionId, popoverState, mutateNodes],
  );

  promptActionsRef.current = {
    cancel: clearPromptNode,
    send: (text: string, ctx: string | null) => {
      clearPromptNode();
      const root = ctx && ctx.length > 0 ? ctx : undefined;
      void onSend?.(text, root);
    },
  };

  const handleConnect = useCallback(
    (c: Connection) => {
      const pid = promptIdRef.current;
      if (!pid || c.target !== pid) return;
      if (c.targetHandle != null && c.targetHandle !== 'context') return;
      const src = c.source;
      if (!src) return;
      setPromptContextEdge({
        id: `prompt-ctx-${pid}`,
        source: src,
        target: pid,
        sourceHandle: c.sourceHandle ?? undefined,
        targetHandle: c.targetHandle ?? undefined,
        selectable: false,
        focusable: false,
        style: {
          strokeDasharray: '6 4',
          stroke:
            getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() ||
            '#a78bfa',
          strokeWidth: 1.5,
        },
      });
      mutateNodes(prev =>
        prev.map(n =>
          n.id === pid
            ? {
                ...n,
                data: {
                  ...(n.data as Record<string, unknown>),
                  contextRoot: src,
                },
              }
            : n,
        ),
      );
    },
    [mutateNodes],
  );

  // Double-click empty canvas → prompt node (+ Note in toolbar still adds annotations).
  const handlePaneDoubleClick = useCallback(
    (e: React.MouseEvent) => {
      if (!sessionId || popoverState) return;
      const position = screenToFlowPosition({ x: e.clientX, y: e.clientY });
      openPromptAt(position);
    },
    [sessionId, popoverState, screenToFlowPosition, openPromptAt],
  );

  const handleAddPromptAtViewportCenter = useCallback(() => {
    const el = containerRef.current;
    if (!el || !sessionId || popoverState) return;
    const r = el.getBoundingClientRect();
    openPromptAt(screenToFlowPosition({ x: r.left + r.width / 2, y: r.top + r.height / 2 }));
  }, [sessionId, popoverState, screenToFlowPosition, openPromptAt]);

  // Clear focus when clicking empty canvas or pressing Escape
  const handlePaneClick = useCallback(() => {
    onNodeSelect(null);
    setPopoverState(null);
  }, [onNodeSelect]);

  // Find the GraphNode data for the active popover
  const popoverGraphNode = useMemo(() => {
    if (!popoverState || !graphData) return null;
    return graphData.nodes.find(n => n.id === popoverState.nodeId) ?? null;
  }, [popoverState, graphData]);

  // Open popover for a given node id — compute screen position from the React Flow node
  const openPopover = useCallback((nodeId: string) => {
    const rfNode = nodes.find(n => n.id === nodeId);
    if (!rfNode || !graphData) return;
    // Skip annotation nodes — they have inline editing already
    if (rfNode.type === 'annotation' || rfNode.type === 'group' || rfNode.type === 'prompt')
      return;
    const gn = graphData.nodes.find(n => n.id === nodeId);
    if (!gn) return;

    setPopoverState({ nodeId, position: rfNode.position });
  }, [nodes, graphData]);

  const closePopover = useCallback(() => {
    setPopoverState(null);
  }, []);

  // React to keyboard Enter activation from useNodeNavigation
  useEffect(() => {
    if (activateNodeId) {
      openPopover(activateNodeId);
      clearActivation();
    }
  }, [activateNodeId, openPopover, clearActivation]);

  // React to keyboard R activation from useNodeNavigation for quick-reply
  useEffect(() => {
    if (replyingToNodeId) {
      // Set steer context so the reply goes to the right node
      onSteerFromNode(replyingToNodeId);
    }
  }, [replyingToNodeId, onSteerFromNode]);

  // Find the node position for the reply input
  const replyingNodePosition = useMemo(() => {
    if (!replyingToNodeId) return null;
    const rfNode = nodes.find(n => n.id === replyingToNodeId);
    return rfNode?.position ?? null;
  }, [replyingToNodeId, nodes]);

  // Double-click on a node opens the popover
  const handleNodeDoubleClick = useCallback(
    (_: React.MouseEvent, node: { id: string }) => {
      openPopover(node.id);
    },
    [openPopover],
  );

  // Add pending approval styling to nodes
  const nodesWithPendingApproval = useMemo(() => {
    if (!pendingApproval?.node_id) return nodes;
    
    return nodes.map(node => {
      if (node.id === pendingApproval.node_id) {
        return {
          ...node,
          className: node.className
            ? `${node.className} ${nodeStyles.pendingApproval}`
            : nodeStyles.pendingApproval,
        };
      }
      return node;
    });
  }, [nodes, pendingApproval]);

  const graphCanvasActions = useMemo(
    (): GraphCanvasActionsValue => ({
      sendFromGraph: (content: string, contextRoot?: string) => {
        onSend?.(content, contextRoot);
      },
      editingUserNodeId,
      setEditingUserNodeId,
    }),
    [onSend, editingUserNodeId],
  );

  // Build popover actions that call the API
  const popoverActions = useMemo((): PopoverActions => ({
    sessionId,
    onSteer: (nodeId: string) => {
      onSteerFromNode(nodeId);
    },
    onUpdateTaskStatus: async (nodeId: string, status: 'completed' | 'failed') => {
      if (!sessionId) return;
      await api.updateTaskStatus(sessionId, nodeId, status);
    },
    onRateTurn: async (nodeId: string, rating: number) => {
      if (!sessionId) return;
      await api.rateTurn(sessionId, nodeId, rating);
    },
    onTogglePin: async (nodeId: string, pinned: boolean) => {
      if (!sessionId) return;
      await api.toggleKnowledgePin(sessionId, nodeId, pinned);
      mutateNodes(prev =>
        prev.map(n => {
          if (n.id !== nodeId) return n;
          const d = n.data as unknown as GraphNode;
          if (d.node_type.type !== 'Knowledge') return n;
          const metadata = { ...d.metadata };
          if (pinned) metadata.pinned = true;
          else delete metadata.pinned;
          return {
            ...n,
            data: { ...d, metadata } as unknown as Record<string, unknown>,
          };
        }),
      );
    },
    onEditSummary: async (nodeId: string, summary: string) => {
      if (!sessionId) return;
      await api.editKnowledgeSummary(sessionId, nodeId, summary);
      mutateNodes(prev =>
        prev.map(n => {
          if (n.id !== nodeId) return n;
          const d = n.data as unknown as GraphNode;
          if (d.node_type.type !== 'Knowledge') return n;
          return {
            ...n,
            data: {
              ...d,
              node_type: { ...d.node_type, summary },
            } as unknown as Record<string, unknown>,
          };
        }),
      );
    },
    onDismissKnowledge: async (nodeId: string) => {
      await api.patchKnowledge(nodeId, { dismissed: true });
      mutateNodes(prev =>
        prev.map(n => (n.id === nodeId ? { ...n, hidden: true } : n)),
      );
    },
    onStartEditUserMessage: (nodeId: string) => {
      setEditingUserNodeId(nodeId);
    },
    onAnnotateToolNode: async (toolNodeId: string, text: string) => {
      if (!sessionId) return;
      await api.createAnnotation(sessionId, 'Tool note', 'annotation', text, {
        relatesTo: toolNodeId,
      });
    },
  }), [sessionId, onSteerFromNode, mutateNodes]);

  return (
    <CascadeCollapseGenerationContext.Provider value={cascadeCollapseGeneration}>
    <div className={styles.graphPane} ref={containerRef}>
      <Toolbar
        layoutMode={layoutMode}
        onLayoutChange={mode => {
          setLayoutMode(mode);
          setTimeout(() => fitView({ padding: 0.1, duration: 400 }), 50);
        }}
        onAddAnnotation={handleAddAnnotation}
        onAddPrompt={handleAddPromptAtViewportCenter}
        onCollapseTimelineCascades={
          layoutMode === 'timeline'
            ? () => setCascadeCollapseGeneration(g => g + 1)
            : undefined
        }
        filter={filter}
        onFilterChange={setFilter}
        matchCount={matchCount}
        totalCount={graphData?.nodes.length ?? 0}
        searchInputRef={searchInputRef}
      />
      <div className={styles.canvasWrapper}>
        <FloatingInput chatCollapsed={chatCollapsed} onSend={onSend ?? (() => {})} isThinking={isThinking} />
        {/* Timeline bands: positions come from bandPositions returned by applyTimelineLayout.
            This overlay is screen-fixed (does not pan/zoom with the flow) — decorative hint only. */}
        {layoutMode === 'timeline' && Object.keys(bandPositions).length > 0 && (
          <div className={styles.swimlaneContainer}>
            {Object.entries(bandPositions).map(([type, y]) => (
              <div
                key={type}
                className={styles.swimlane}
                style={{
                  top: y - 20,
                  backgroundColor: cssVar(`--node-${type.toLowerCase()}`),
                }}
              />
            ))}
          </div>
        )}
        <GraphCanvasActionsProvider value={graphCanvasActions}>
        <PopoverProvider value={popoverActions}>
        <FocusContext.Provider value={focusedNodeId}>
        <SteerContext.Provider value={steerCallbackRef.current}>
        <ZoomProvider>
        <ReactFlow
          nodes={nodesWithPendingApproval.map(node => ({
            ...node,
            style: {
              ...node.style,
              opacity: dimmedNodeIds.has(node.id) ? 0.25 : undefined,
            },
          }))}
          edges={flowEdges.map(edge => ({
            ...edge,
            style: {
              ...edge.style,
              opacity: dimmedEdgeIds.has(edge.id) ? 0.25 : undefined,
            },
          }))}
          nodeTypes={NODE_TYPES}
          edgeTypes={EDGE_TYPES}
          onNodesChange={onNodesChange}
          onConnect={handleConnect}
          onNodeClick={handleNodeClick}
          onNodeDoubleClick={handleNodeDoubleClick}
          onNodeDragStop={handleNodeDragStop}
          onPaneClick={handlePaneClick}
          onDoubleClick={handlePaneDoubleClick}
          fitView
          fitViewOptions={{ padding: 0.12 }}
          minZoom={0.05}
          maxZoom={4}
          deleteKeyCode={null}
          onlyRenderVisibleElements
          proOptions={{ hideAttribution: true }}
        >
          <Background
            variant={BackgroundVariant.Dots}
            color={cssVar('--border-hover')}
            gap={20}
          />
          <MiniMap
            nodeColor={n => {
              const varMap: Record<string, string> = {
                interaction: '--node-interaction',
                agent:       '--node-agent',
                content:     '--node-content',
                task:        '--node-task',
                knowledge:   '--node-knowledge',
                annotation:  '--node-annotation',
                prompt:      '--accent',
              };
              const v = varMap[n.type ?? ''];
              return v ? cssVar(v) : cssVar('--fg-muted');
            }}
            maskColor={theme === 'light' ? '#f8f7f488' : '#16161688'}
            style={{
              background: cssVar('--surface-2'),
              border: `1px solid ${cssVar('--border')}`,
            }}
          />
          <Controls style={{
            background: cssVar('--surface-2'),
            border: `1px solid ${cssVar('--border')}`,
            color: cssVar('--fg'),
          }} />
        </ReactFlow>
        </ZoomProvider>
        </SteerContext.Provider>
        </FocusContext.Provider>
        {popoverState && popoverGraphNode && (
          <NodePopover
            node={popoverGraphNode}
            position={popoverState.position}
            onClose={closePopover}
          />
        )}
        {replyingToNodeId && replyingNodePosition && (
          <NodeReplyInput
            nodeId={replyingToNodeId}
            position={replyingNodePosition}
            onSend={onSend ?? (() => {})}
            onCancel={clearReply}
            isThinking={isThinking ?? false}
          />
        )}
        {pendingApproval && onApprove && onReject && onModify && (
          <HitlOverlay
            approval={pendingApproval}
            onApprove={onApprove}
            onReject={onReject}
            onModify={onModify}
            className={styles.hitlCanvasOverlay}
          />
        )}
        </PopoverProvider>
        </GraphCanvasActionsProvider>
      </div>
    </div>
    </CascadeCollapseGenerationContext.Provider>
  );
}

export function GraphCanvas(props: GraphCanvasProps) {
  return (
    <ReactFlowProvider>
      <GraphCanvasInner {...props} />
    </ReactFlowProvider>
  );
}
