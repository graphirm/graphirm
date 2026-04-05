import { useCallback, useRef, useState } from 'react';
import styles from './App.module.css';
import { SessionBar } from './components/SessionBar';
import { ChatPane } from './components/ChatPane';
import { GraphCanvas } from './components/GraphCanvas';
import { useSession } from './hooks/useSession';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';

export function App() {
  const {
    sessions,
    currentSession,
    messages,
    graphData,
    streamingMessage,
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
    autoApprove,
    toggleAutoApprove,
    renameSession,
  } = useSession();

  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [steerContext, setSteerContext] = useState<{ nodeId: string } | null>(null);
  const [outlineSteer, setOutlineSteer] = useState<{ outlineNodeId: string; interactionId: string } | null>(null);
  const [chatCollapsed, setChatCollapsed] = useState(false);

  // Ref callbacks let GraphCanvasInner register its handlers after mount.
  const fitViewCb = useRef<(() => void) | null>(null);
  const cycleLayoutCb = useRef<(() => void) | null>(null);
  const chatInputRef = useRef<HTMLTextAreaElement>(null);

  const handleNodeSelect = useCallback((nodeId: string | null) => {
    setSelectedNodeId(nodeId);
  }, []);

  const handleSteerFromNode = useCallback((nodeId: string) => {
    setOutlineSteer(null);
    setSteerContext({ nodeId });
    // Focus chat input so user can type their steer message immediately.
    setTimeout(() => chatInputRef.current?.focus(), 50);
  }, []);

  const handleOutlineSteer = useCallback((outlineNodeId: string, interactionId: string) => {
    setSteerContext(null);
    setOutlineSteer({ outlineNodeId, interactionId });
    setTimeout(() => chatInputRef.current?.focus(), 50);
  }, []);

  const handleSendWithSteer = useCallback(
    (content: string) => {
      if (steerContext) {
        sendPrompt(content, steerContext.nodeId);
        setSteerContext(null);
      } else if (outlineSteer) {
        sendPrompt(content, undefined, {
          outline_node_id: outlineSteer.outlineNodeId,
          interaction_id: outlineSteer.interactionId,
        });
        setOutlineSteer(null);
      } else {
        sendPrompt(content);
      }
    },
    [steerContext, outlineSteer, sendPrompt],
  );

  useKeyboardShortcuts({
    onFitView: () => fitViewCb.current?.(),
    onToggleLayout: () => cycleLayoutCb.current?.(),
    onNewSession: createSession,
    onFocusChat: () => chatInputRef.current?.focus(),
    onToggleChatCollapsed: () => setChatCollapsed(c => !c),
  });

  return (
    <div className={styles.app}>
      <SessionBar
        sessions={sessions}
        currentSession={currentSession}
        onSelectSession={selectSession}
        onCreateSession={createSession}
        onPause={pauseSession}
        onResume={resumeSession}
        autoApprove={autoApprove}
        onToggleAutoApprove={toggleAutoApprove}
        onRenameSession={renameSession}
      />
      <div className={styles.main}>
        <ChatPane
          messages={messages}
          streamingMessage={streamingMessage}
          isThinking={isThinking}
          pendingApproval={pendingApproval}
          sessionId={currentSession?.id ?? null}
          steerContext={steerContext}
          inputRef={chatInputRef}
          onSend={handleSendWithSteer}
          onAbort={abortSession}
          onApprove={approveAction}
          onReject={rejectAction}
          onModify={modifyAction}
          onClearSteer={() => setSteerContext(null)}
          chatCollapsed={chatCollapsed}
          onToggleCollapse={() => setChatCollapsed(c => !c)}
          outlineSteer={outlineSteer}
          onClearOutlineSteer={() => setOutlineSteer(null)}
          onOutlineSteer={handleOutlineSteer}
        />
        <GraphCanvas
          graphData={graphData}
          sessionId={currentSession?.id ?? null}
          selectedNodeId={selectedNodeId}
          onNodeSelect={handleNodeSelect}
          onSteerFromNode={handleSteerFromNode}
          onFitViewRef={cb => { fitViewCb.current = cb; }}
          onCycleLayoutRef={cb => { cycleLayoutCb.current = cb; }}
          chatCollapsed={chatCollapsed}
          onSend={(content, contextRoot) => {
            if (contextRoot !== undefined && contextRoot !== '') {
              sendPrompt(content, contextRoot);
            } else {
              handleSendWithSteer(content);
            }
          }}
          isThinking={isThinking}
          streamingMessage={streamingMessage}
          pendingApproval={pendingApproval}
          onApprove={approveAction}
          onReject={rejectAction}
          onModify={modifyAction}
        />
      </div>
    </div>
  );
}
