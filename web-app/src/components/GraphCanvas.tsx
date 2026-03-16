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
import type { NodeTypes } from '@xyflow/react';
import type { GraphData } from '../types/graph';
import { useGraphData } from '../hooks/useGraphData';
import { InteractionNode } from './nodes/InteractionNode';
import { AgentNode } from './nodes/AgentNode';
import { ContentNode } from './nodes/ContentNode';
import { TaskNode } from './nodes/TaskNode';
import { KnowledgeNode } from './nodes/KnowledgeNode';
import { AnnotationNode } from './nodes/AnnotationNode';
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
};

const EDGE_TYPES = {
  labelled: LabelledEdge,
};

interface GraphCanvasProps {
  graphData: GraphData | null;
  sessionId: string | null;
  selectedNodeId: string | null;
  onNodeSelect: (nodeId: string | null) => void;
}

function GraphCanvasInner({ graphData, sessionId, onNodeSelect }: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasWidth = containerRef.current?.clientWidth ?? 800;

  const { nodes, edges, layoutMode, setLayoutMode, onNodesChange, persistPositions } =
    useGraphData(graphData, sessionId, canvasWidth);

  const { fitView } = useReactFlow();
  const [annotationCount, setAnnotationCount] = useState(0);

  const handleAddAnnotation = useCallback(() => {
    setAnnotationCount(c => c + 1);
    // Annotation nodes are added directly to canvas without server round-trip.
    // Server persistence via POST /api/graph/{id}/annotate is wired in Phase 10.
  }, []);

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

  // Suppress unused variable warning until Phase 10
  void annotationCount;

  return (
    <div className={styles.graphPane} ref={containerRef}>
      <Toolbar
        layoutMode={layoutMode}
        onLayoutChange={(mode) => {
          setLayoutMode(mode);
          setTimeout(() => fitView({ padding: 0.1, duration: 400 }), 50);
        }}
        onAddAnnotation={handleAddAnnotation}
      />
      <div className={styles.canvasWrapper}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={NODE_TYPES}
          edgeTypes={EDGE_TYPES}
          onNodesChange={onNodesChange}
          onNodeClick={handleNodeClick}
          onNodeDragStop={handleNodeDragStop}
          fitView
          fitViewOptions={{ padding: 0.1 }}
          minZoom={0.05}
          maxZoom={4}
          deleteKeyCode={null}
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} color="#333" gap={20} />
          <MiniMap
            nodeColor={(n) => {
              const typeColors: Record<string, string> = {
                interaction: '#4fc3f7',
                agent: '#ef9a9a',
                content: '#81c784',
                task: '#ffb74d',
                knowledge: '#ce93d8',
                annotation: '#fbbf24',
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
