import { useCallback, useEffect, useRef, useState } from 'react';
import { useTheme } from '../hooks/useTheme';
import {
  ReactFlow,
  Background,
  MiniMap,
  Controls,
  BackgroundVariant,
  useReactFlow,
  ReactFlowProvider,
} from '@xyflow/react';
import type { NodeTypes, EdgeTypes, Node } from '@xyflow/react';
import type { GraphData } from '../types/graph';
import { useGraphData, EMPTY_FILTER } from '../hooks/useGraphData';
import type { LayoutMode, NodeFilter } from '../hooks/useGraphData';
import { InteractionNode } from './nodes/InteractionNode';
import { AgentNode } from './nodes/AgentNode';
import { ContentNode } from './nodes/ContentNode';
import { TaskNode } from './nodes/TaskNode';
import { KnowledgeNode } from './nodes/KnowledgeNode';
import { AnnotationNode } from './nodes/AnnotationNode';
import { GroupNode } from './nodes/GroupNode';
import { LabelledEdge } from './edges/LabelledEdge';
import { Toolbar } from './Toolbar';
import { SteerContext } from '../context/SteerContext';
import { FloatingInput } from './FloatingInput';
import styles from './GraphCanvas.module.css';

const NODE_TYPES: NodeTypes = {
  interaction: InteractionNode,
  agent: AgentNode,
  content: ContentNode,
  task: TaskNode,
  knowledge: KnowledgeNode,
  annotation: AnnotationNode,
  group: GroupNode,
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
  onSend?: (content: string) => void;
  isThinking?: boolean;
}

const LAYOUT_CYCLE: LayoutMode[] = ['dagre', 'timeline', 'free'];

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
}: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const canvasWidth = containerRef.current?.clientWidth ?? 800;
  const { theme } = useTheme();

  // Read CSS variables so MiniMap/Controls/Background respond to theme switches
  const cssVar = (name: string) =>
    getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  const [filter, setFilter] = useState<NodeFilter>(EMPTY_FILTER);

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
    matchCount,
  } = useGraphData(graphData, sessionId, canvasWidth, filter);

  const { fitView, screenToFlowPosition } = useReactFlow();

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

  // Double-click on empty canvas area adds an annotation node.
  const handlePaneDoubleClick = useCallback(
    (e: React.MouseEvent) => {
      const position = screenToFlowPosition({ x: e.clientX, y: e.clientY });
      const newNode: Node = {
        id: `annotation_${Date.now()}`,
        type: 'annotation',
        position,
        data: { text: '' } as Record<string, unknown>,
      };
      addNode(newNode);
    },
    [screenToFlowPosition, addNode],
  );

  return (
    <div className={styles.graphPane} ref={containerRef}>
      <Toolbar
        layoutMode={layoutMode}
        onLayoutChange={mode => {
          setLayoutMode(mode);
          setTimeout(() => fitView({ padding: 0.1, duration: 400 }), 50);
        }}
        onAddAnnotation={handleAddAnnotation}
        filter={filter}
        onFilterChange={setFilter}
        matchCount={matchCount}
        totalCount={graphData?.nodes.length ?? 0}
        searchInputRef={searchInputRef}
      />
      <div className={styles.canvasWrapper}>
        <FloatingInput chatCollapsed={chatCollapsed} onSend={onSend ?? (() => {})} isThinking={isThinking} />
        <SteerContext.Provider value={steerCallbackRef.current}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={NODE_TYPES}
          edgeTypes={EDGE_TYPES}
          onNodesChange={onNodesChange}
          onNodeClick={handleNodeClick}
          onNodeDragStop={handleNodeDragStop}
          onPaneClick={() => onNodeSelect(null)}
          onDoubleClick={handlePaneDoubleClick}
          fitView
          fitViewOptions={{ padding: 0.12 }}
          minZoom={0.05}
          maxZoom={4}
          deleteKeyCode={null}
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
        </SteerContext.Provider>
      </div>
    </div>
  );
}

export function GraphCanvas(props: GraphCanvasProps) {
  return (
    <ReactFlowProvider>
      <GraphCanvasInner {...props} />
    </ReactFlowProvider>
  );
}
