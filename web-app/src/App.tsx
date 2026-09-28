import { useCallback, useRef, useState } from 'react';
import { DecisionShell } from './components/DecisionShell';
import { useSession } from './hooks/useSession';
import { readLayoutMode, writeLayoutMode, type LayoutMode } from './layout/layoutMode';

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
  const [layoutMode, setLayoutMode] = useState<LayoutMode>(() => readLayoutMode(localStorage));

  const chatInputRef = useRef<HTMLTextAreaElement>(null);

  const handleNodeSelect = useCallback((nodeId: string | null) => {
    setSelectedNodeId(nodeId);
  }, []);

  const handleSteerFromNode = useCallback((nodeId: string) => {
    setOutlineSteer(null);
    setSteerContext({ nodeId });
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

  const handleLayoutMode = useCallback((mode: LayoutMode) => {
    writeLayoutMode(localStorage, mode);
    setLayoutMode(mode);
  }, []);

  return (
    <DecisionShell
      layoutMode={layoutMode}
      onLayoutMode={handleLayoutMode}
      session={{
        sessions,
        currentSession,
        onSelectSession: selectSession,
        onCreateSession: createSession,
        onPause: pauseSession,
        onResume: resumeSession,
        autoApprove,
        onToggleAutoApprove: toggleAutoApprove,
        onRenameSession: renameSession,
      }}
      chat={{
        messages,
        streamingMessage,
        isThinking,
        pendingApproval,
        sessionId: currentSession?.id ?? null,
        steerContext,
        inputRef: chatInputRef,
        onSend: handleSendWithSteer,
        onAbort: abortSession,
        onApprove: approveAction,
        onReject: rejectAction,
        onModify: modifyAction,
        onClearSteer: () => setSteerContext(null),
        chatCollapsed,
        onToggleCollapse: () => setChatCollapsed(c => !c),
        outlineSteer,
        onClearOutlineSteer: () => setOutlineSteer(null),
        onOutlineSteer: handleOutlineSteer,
      }}
      graph={{
        graphData,
        sessionId: currentSession?.id ?? null,
        selectedNodeId,
        onNodeSelect: handleNodeSelect,
        onSteerFromNode: handleSteerFromNode,
        chatCollapsed,
        onSend: (content, contextRoot) => {
          if (contextRoot !== undefined && contextRoot !== '') {
            sendPrompt(content, contextRoot);
          } else {
            handleSendWithSteer(content);
          }
        },
        isThinking,
        streamingMessage,
        pendingApproval,
        onApprove: approveAction,
        onReject: rejectAction,
        onModify: modifyAction,
      }}
    />
  );
}
