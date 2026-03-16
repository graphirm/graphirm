import { useCallback, useRef, useState } from 'react';
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
import { useGraphData } from '../hooks/useGraphData';
import { InteractionNode } from './nodes/InteractionNode';
import { AgentNode } from './nodes/AgentNode';
import { ContentNode } from './nodes/ContentNode';
import { TaskNode } from './nodes/TaskNode';
import { KnowledgeNode } from './nodes/KnowledgeNode';
import { AnnotationNode } from './nodes/AnnotationNode';
import { GroupNode } from './nodes/GroupNode';
import { LabelledEdge } from './edges/LabelledEdge';
import { Toolbar } from './Toolbar';
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
}

function GraphCanvasInner({
  graphData,
  sessionId,
  onNodeSelect,
  onSteerFromNode,
}: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasWidth = containerRef.current?.clientWidth ?? 800;

  const {
    nodes,
    edges,
    layoutMode,
    setLayoutMode,
    onNodesChange,
    persistPositions,
    addNode,
  } = useGraphData(graphData, sessionId, canvasWidth);

  const { fitView, screenToFlowPosition } = useReactFlow();
  const [annotationCount, setAnnotationCount] = useState(0);

  // Inject onSteer callback into interaction node data so cards can call it.
  const nodesWithSteer = nodes.map(n =>
    n.type === 'interaction'
      ? {
          ...n,
          data: {
            ...(n.data as Record<string, unknown>),
            onSteer: onSteerFromNode,
          },
        }
      : n,
  );

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
      />
      <div className={styles.canvasWrapper}>
        <ReactFlow
          nodes={nodesWithSteer}
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
          <Background variant={BackgroundVariant.Dots} color="#333" gap={20} />
          <MiniMap
            nodeColor={n => {
              const typeColors: Record<string, string> = {
                interaction: '#4fc3f7',
                agent: '#ef9a9a',
                content: '#81c784',
                task: '#ffb74d',
                knowledge: '#ce93d8',
                annotation: '#fbbf24',
                group: '#ffffff11',
              };
              return typeColors[n.type ?? ''] ?? '#888';
            }}
            maskColor="#1e1e1e99"
            style={{ background: '#252526', border: '1px solid #333' }}
          />
          <Controls style={{ background: '#252526', border: '1px solid #333', color: '#d4d4d4' }} />
        </ReactFlow>
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
