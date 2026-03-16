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

  const handleNodeSelect = useCallback((nodeId: string | null) => {
    setSelectedNodeId(nodeId);
  }, []);

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
          onSend={sendPrompt}
          onAbort={abortSession}
          onApprove={approveAction}
          onReject={rejectAction}
          onModify={modifyAction}
          sessionId={currentSession?.id ?? null}
        />
        <GraphCanvas
          graphData={graphData}
          sessionId={currentSession?.id ?? null}
          selectedNodeId={selectedNodeId}
          onNodeSelect={handleNodeSelect}
        />
      </div>
    </div>
  );
}
