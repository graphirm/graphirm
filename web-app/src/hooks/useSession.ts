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
  createSession: (name?: string) => Promise<Session | void>;
  sendPrompt: (content: string, contextRoot?: string) => Promise<void>;
  abortSession: () => Promise<void>;
  approveAction: (nodeId: string) => Promise<void>;
  rejectAction: (nodeId: string, reason?: string) => Promise<void>;
  modifyAction: (nodeId: string, modifiedArgs: string) => Promise<void>;
  pauseSession: () => Promise<void>;
  resumeSession: () => Promise<void>;
  autoApprove: boolean;
  toggleAutoApprove: () => Promise<void>;
}

export function useSession(): UseSessionReturn {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [currentSession, setCurrentSession] = useState<Session | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [isThinking, setIsThinking] = useState(false);
  const [pendingApproval, setPendingApproval] = useState<PendingApproval | null>(null);
  const [autoApprove, setAutoApprove] = useState(false);

  const sseRef = useRef<SseClient | null>(null);
  // Track current session ID in a ref so callbacks always see the latest value.
  const currentSessionRef = useRef<Session | null>(null);
  currentSessionRef.current = currentSession;

  const refresh = useCallback(async (sessionId: string) => {
    try {
      const [newMessages, newGraph] = await Promise.all([
        api.getMessages(sessionId),
        api.getGraph(sessionId),
      ]);
      setMessages(newMessages);
      setGraphData(newGraph);
    } catch (err) {
      console.error('Failed to refresh session data:', err);
    }
  }, []);

  const subscribeSse = useCallback((sessionId: string) => {
    sseRef.current?.unsubscribe();
    const client = new SseClient((ev) => {
      if (ev.event === 'agent_start') {
        setIsThinking(true);
      } else if (ev.event === 'agent_end' || ev.event === 'error') {
        setIsThinking(false);
        refresh(sessionId).catch(console.error);
      } else if (ev.event === 'graph_update' || ev.event === 'tool_end' || ev.event === 'message_end') {
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

  const selectSession = useCallback(async (id: string) => {
    const session = sessions.find(s => s.id === id) ?? { id } as Session;
    setCurrentSession(session);
    setPendingApproval(null);
    setIsThinking(false);
    await refresh(id);
    subscribeSse(id);
  }, [sessions, refresh, subscribeSse]);

  // Load session list on mount + auto-select the first session.
  useEffect(() => {
    api.listSessions().then((list) => {
      setSessions(list);
      if (list.length > 0) {
        const first = list[0];
        setCurrentSession(first);
        refresh(first.id).catch(console.error);
        subscribeSse(first.id);
      }
    }).catch(console.error);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Cleanup SSE on unmount
  useEffect(() => {
    return () => { sseRef.current?.unsubscribe(); };
  }, []);

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
    return session;
  }, [subscribeSse]);

  const sendPrompt = useCallback(async (content: string, contextRoot?: string) => {
    let session = currentSessionRef.current;
    if (!session) {
      const newSession = await createSession();
      session = newSession;
    }
    if (!session?.id) return;
    setIsThinking(true);
    try {
      if (contextRoot) {
        await api.steerFromNode(session.id, content, contextRoot);
      } else {
        await api.sendPrompt(session.id, content);
      }
    } catch (err) {
      console.error('Failed to send prompt:', err);
      setIsThinking(false);
    }
  }, [createSession]);

  const abortSession = useCallback(async () => {
    const session = currentSessionRef.current;
    if (!session) return;
    await api.abortSession(session.id);
    setIsThinking(false);
  }, []);

  const approveAction = useCallback(async (nodeId: string) => {
    const session = currentSessionRef.current;
    if (!session) return;
    await api.nodeAction(session.id, nodeId, 'approve');
    setPendingApproval(null);
  }, []);

  const rejectAction = useCallback(async (nodeId: string, reason?: string) => {
    const session = currentSessionRef.current;
    if (!session) return;
    await api.nodeAction(session.id, nodeId, 'reject', reason);
    setPendingApproval(null);
    setIsThinking(false);
  }, []);

  const modifyAction = useCallback(async (nodeId: string, modifiedArgs: string) => {
    const session = currentSessionRef.current;
    if (!session) return;
    await api.nodeAction(session.id, nodeId, 'approve', undefined, modifiedArgs);
    setPendingApproval(null);
  }, []);

  const pauseSession = useCallback(async () => {
    const session = currentSessionRef.current;
    if (!session) return;
    await api.pauseSession(session.id);
  }, []);

  const resumeSession = useCallback(async () => {
    const session = currentSessionRef.current;
    if (!session) return;
    await api.resumeSession(session.id);
    setPendingApproval(null);
  }, []);

  const toggleAutoApprove = useCallback(async () => {
    const session = currentSessionRef.current;
    if (!session) return;
    const next = !autoApprove;
    await api.setAutoApprove(session.id, next);
    setAutoApprove(next);
  }, [autoApprove]);

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
    autoApprove,
    toggleAutoApprove,
  };
}
