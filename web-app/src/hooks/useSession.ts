import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api/client';
import { SseClient } from '../api/sse';
import type { GraphData, Message, PendingApproval, Session } from '../types/graph';

interface UseSessionReturn {
  sessions: Session[];
  currentSession: Session | null;
  messages: Message[];
  graphData: GraphData | null;
  isThinking: boolean;
  pendingApproval: PendingApproval | null;
  selectSession: (id: string) => Promise<void>;
  createSession: (name?: string) => Promise<void>;
  sendPrompt: (content: string, contextRoot?: string) => Promise<void>;
  abortSession: () => Promise<void>;
  approveAction: (nodeId: string) => Promise<void>;
  rejectAction: (nodeId: string, reason?: string) => Promise<void>;
  modifyAction: (nodeId: string, modifiedArgs: string) => Promise<void>;
  pauseSession: () => Promise<void>;
  resumeSession: () => Promise<void>;
}

export function useSession(): UseSessionReturn {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [currentSession, setCurrentSession] = useState<Session | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [isThinking, setIsThinking] = useState(false);
  const [pendingApproval, setPendingApproval] = useState<PendingApproval | null>(null);

  const sseRef = useRef<SseClient | null>(null);

  const refresh = useCallback(async (sessionId: string) => {
    const [newMessages, newGraph] = await Promise.all([
      api.getMessages(sessionId),
      api.getGraph(sessionId),
    ]);
    setMessages(newMessages);
    setGraphData(newGraph);
  }, []);

  const subscribeSse = useCallback((sessionId: string) => {
    sseRef.current?.unsubscribe();
    const client = new SseClient((ev) => {
      if (ev.event === 'agent_start') {
        setIsThinking(true);
      } else if (ev.event === 'agent_end' || ev.event === 'error') {
        setIsThinking(false);
        refresh(sessionId).catch(console.error);
      } else if (ev.event === 'graph_update') {
        refresh(sessionId).catch(console.error);
      } else if (ev.event === 'awaiting_approval') {
        // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
        const payload = ev.data?.data ?? ev.data;
        setPendingApproval({ ...payload, session_id: sessionId } as PendingApproval);
      }
    });
    client.subscribe(sessionId);
    sseRef.current = client;
  }, [refresh]);

  // Load session list on mount
  useEffect(() => {
    api.listSessions().then(setSessions).catch(console.error);
  }, []);

  // Cleanup SSE on unmount
  useEffect(() => {
    return () => { sseRef.current?.unsubscribe(); };
  }, []);

  const selectSession = useCallback(async (id: string) => {
    const session = sessions.find(s => s.id === id) ?? { id };
    setCurrentSession(session);
    setPendingApproval(null);
    setIsThinking(false);
    await refresh(id);
    subscribeSse(id);
  }, [sessions, refresh, subscribeSse]);

  const createSession = useCallback(async (name?: string) => {
    const label = name ?? `Session ${new Date().toLocaleTimeString()}`;
    const session = await api.createSession(label);
    setSessions(prev => [session, ...prev]);
    setCurrentSession(session);
    setMessages([]);
    setGraphData(null);
    setPendingApproval(null);
    setIsThinking(false);
    subscribeSse(session.id);
  }, [subscribeSse]);

  const sendPrompt = useCallback(async (content: string, contextRoot?: string) => {
    if (!currentSession) {
      await createSession();
    }
    const id = currentSession?.id;
    if (!id) return;
    setIsThinking(true);
    if (contextRoot) {
      await api.steerFromNode(id, content, contextRoot);
    } else {
      await api.sendPrompt(id, content);
    }
  }, [currentSession, createSession]);

  const abortSession = useCallback(async () => {
    if (!currentSession) return;
    await api.abortSession(currentSession.id);
    setIsThinking(false);
  }, [currentSession]);

  const approveAction = useCallback(async (nodeId: string) => {
    if (!currentSession) return;
    await api.nodeAction(currentSession.id, nodeId, 'approve');
    setPendingApproval(null);
  }, [currentSession]);

  const rejectAction = useCallback(async (nodeId: string, reason?: string) => {
    if (!currentSession) return;
    await api.nodeAction(currentSession.id, nodeId, 'reject', reason);
    setPendingApproval(null);
    setIsThinking(false);
  }, [currentSession]);

  const modifyAction = useCallback(async (nodeId: string, modifiedArgs: string) => {
    if (!currentSession) return;
    await api.nodeAction(currentSession.id, nodeId, 'approve', undefined, modifiedArgs);
    setPendingApproval(null);
  }, [currentSession]);

  const pauseSession = useCallback(async () => {
    if (!currentSession) return;
    await api.pauseSession(currentSession.id);
  }, [currentSession]);

  const resumeSession = useCallback(async () => {
    if (!currentSession) return;
    await api.resumeSession(currentSession.id);
    setPendingApproval(null);
  }, [currentSession]);

  return {
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
  };
}
