import { useCallback, useState } from 'react';
import styles from './App.module.css';
import { SessionBar } from './components/SessionBar';
import { ChatPane } from './components/ChatPane';
import { GraphCanvas } from './components/GraphCanvas';
import { useSession } from './hooks/useSession';

export function App() {
  const {
    sessions,
    currentSession,
    messages,
    graphData,
    isThinking,
    pendingApproval,
    selectSession,
    createSession,
    sendPrompt,
    abortSession,
    approveAction,
    rejectAction,
    modifyAction,
    pauseSession,
    resumeSession,
  } = useSession();

  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  // When user clicks "Steer from here" on a node, we pre-fill the chat input
  // with a context marker and focus it.
  const [steerContext, setSteerContext] = useState<{ nodeId: string } | null>(null);

  const handleNodeSelect = useCallback((nodeId: string | null) => {
    setSelectedNodeId(nodeId);
  }, []);

  const handleSteerFromNode = useCallback((nodeId: string) => {
    setSteerContext({ nodeId });
  }, []);

  const handleSendWithSteer = useCallback(
    (content: string) => {
      if (steerContext) {
        sendPrompt(content, steerContext.nodeId);
        setSteerContext(null);
      } else {
        sendPrompt(content);
      }
    },
    [steerContext, sendPrompt],
  );

  return (
    <div className={styles.app}>
      <SessionBar
        sessions={sessions}
        currentSession={currentSession}
        onSelectSession={selectSession}
        onCreateSession={createSession}
        onPause={pauseSession}
        onResume={resumeSession}
      />
      <div className={styles.main}>
        <ChatPane
          messages={messages}
          isThinking={isThinking}
          pendingApproval={pendingApproval}
          sessionId={currentSession?.id ?? null}
          steerContext={steerContext}
          onSend={handleSendWithSteer}
          onAbort={abortSession}
          onApprove={approveAction}
          onReject={rejectAction}
          onModify={modifyAction}
          onClearSteer={() => setSteerContext(null)}
        />
        <GraphCanvas
          graphData={graphData}
          sessionId={currentSession?.id ?? null}
          selectedNodeId={selectedNodeId}
          onNodeSelect={handleNodeSelect}
          onSteerFromNode={handleSteerFromNode}
        />
      </div>
    </div>
  );
}
