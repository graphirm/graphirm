import { useCallback, useMemo, useRef, useState, type ComponentProps } from 'react';
import styles from '../App.module.css';
import { planStepsFromTasks } from '../chat/planCard';
import type { LayoutMode } from '../layout/layoutMode';
import type { GraphData } from '../types/graph';
import { useKeyboardShortcuts } from '../hooks/useKeyboardShortcuts';
import { ChatPane } from './ChatPane';
import { GraphCanvas } from './GraphCanvas';
import { ReviewTab } from './ReviewTab';
import { SessionBar } from './SessionBar';
import { RulesTab } from './RulesTab';
import { TabBar, type ChatTab } from './TabBar';

/** Task nodes on the loaded session graph. Description is the plan note. */
function tasksFromGraph(graphData: GraphData | null) {
  if (!graphData) return [];
  const tasks: {
    id: string;
    title?: string;
    status?: string;
    metadata?: Record<string, unknown>;
    note?: string;
  }[] = [];
  for (const node of graphData.nodes) {
    const nodeType = node.node_type;
    if (nodeType.type !== 'Task') continue;
    tasks.push({
      id: node.id,
      title: nodeType.title,
      status: nodeType.status,
      note: nodeType.description,
      metadata: node.metadata,
    });
  }
  return tasks;
}

/** Pi tasks already present on the loaded session graph. */
function piTasksFromGraph(graphData: GraphData | null) {
  if (!graphData) return [];
  const tasks: { id: string; status?: string; executor: 'pi'; title?: string }[] = [];
  for (const node of graphData.nodes) {
    const nodeType = node.node_type;
    if (nodeType.type !== 'Task') continue;
    if (node.metadata.executor !== 'pi') continue;
    tasks.push({
      id: node.id,
      status: nodeType.status,
      executor: 'pi',
      title: nodeType.title,
    });
  }
  return tasks;
}

export interface DecisionShellProps {
  layoutMode: LayoutMode;
  onLayoutMode: (mode: LayoutMode) => void;
  session: ComponentProps<typeof SessionBar>;
  chat: ComponentProps<typeof ChatPane>;
  graph: ComponentProps<typeof GraphCanvas>;
}

export function DecisionShell({ layoutMode, onLayoutMode, session, chat, graph }: DecisionShellProps) {
  const [tab, setTab] = useState<ChatTab>('chat');
  const [graphMounted, setGraphMounted] = useState(false);
  const fitViewCb = useRef<(() => void) | null>(null);
  const cycleLayoutCb = useRef<(() => void) | null>(null);
  const graphRef = useRef(graph);
  const layoutModeRef = useRef(layoutMode);
  const tabRef = useRef(tab);
  graphRef.current = graph;
  layoutModeRef.current = layoutMode;
  tabRef.current = tab;

  const graphHotkeys = layoutMode === 'legacy' || tab === 'graph';
  const planSteps = useMemo(
    () => (layoutMode === 'chat' ? planStepsFromTasks(tasksFromGraph(graph.graphData)) : []),
    [layoutMode, graph.graphData],
  );

  const handleFitViewRef = useCallback((cb: () => void) => {
    fitViewCb.current = cb;
    graphRef.current.onFitViewRef?.(cb);
  }, []);

  const handleCycleLayoutRef = useCallback((cb: () => void) => {
    cycleLayoutCb.current = cb;
    graphRef.current.onCycleLayoutRef?.(cb);
  }, []);

  useKeyboardShortcuts({
    onFitView: () => {
      if (!graphHotkeys) return;
      fitViewCb.current?.();
    },
    onToggleLayout: () => {
      if (!graphHotkeys) return;
      cycleLayoutCb.current?.();
    },
    onNewSession: () => {
      void session.onCreateSession();
    },
    onFocusChat: () => {
      chat.inputRef?.current?.focus();
    },
    onToggleChatCollapsed: () => {
      chat.onToggleCollapse?.();
    },
  });

  const selectTab = (next: ChatTab) => {
    if (next === 'graph') setGraphMounted(true);
    setTab(next);
  };

  const handleSteerFromNode = useCallback((nodeId: string) => {
    if (layoutModeRef.current === 'chat' && tabRef.current !== 'chat') setTab('chat');
    graphRef.current.onSteerFromNode(nodeId);
  }, []);

  const chatPane = (
    <ChatPane {...chat} planSteps={planSteps} onPlanResume={session.onResume} />
  );
  const graphCanvas = (
    <GraphCanvas
      {...graph}
      hotkeysEnabled={graphHotkeys}
      onSteerFromNode={handleSteerFromNode}
      onFitViewRef={handleFitViewRef}
      onCycleLayoutRef={handleCycleLayoutRef}
    />
  );

  return (
    <div className={styles.app}>
      <SessionBar {...session} />
      <button
        type="button"
        className={styles.layoutSwitch}
        onClick={() => onLayoutMode(layoutMode === 'chat' ? 'legacy' : 'chat')}
      >
        Layout: chat | legacy
      </button>
      {layoutMode === 'legacy' ? (
        <div className={styles.main}>
          {chatPane}
          {graphCanvas}
        </div>
      ) : (
        <div className={styles.columnSlot}>
          <div className={styles.column}>
            <div className={styles.columnBody}>
              {tab === 'chat' && chatPane}
              {tab === 'review' && (
                <ReviewTab
                  sessions={session.sessions}
                  pendingApproval={chat.pendingApproval}
                  tasks={piTasksFromGraph(graph.graphData)}
                  onOpenChat={() => selectTab('chat')}
                />
              )}
              {tab === 'rules' && <RulesTab />}
              {graphMounted && (
                <div
                  className={styles.graphKeepAlive}
                  hidden={tab !== 'graph'}
                  inert={tab !== 'graph'}
                >
                  {graphCanvas}
                </div>
              )}
            </div>
            <TabBar active={tab} onChange={selectTab} />
          </div>
        </div>
      )}
    </div>
  );
}
