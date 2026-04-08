import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../api/client';
import { SseClient } from '../api/sse';
import type {
  GraphData,
  GraphEdge,
  GraphNode,
  Message,
  PendingApproval,
  Session,
} from '../types/graph';
import { segmentPartsForInteraction } from '../utils/chatSegments';

interface UseSessionReturn {
  sessions: Session[];
  currentSession: Session | null;
  messages: Message[];
  graphData: GraphData | null;
  streamingMessage: Message | null;
  isThinking: boolean;
  pendingApproval: PendingApproval | null;
  selectSession: (id: string) => Promise<void>;
  createSession: (name?: string, workspace?: string) => Promise<Session | void>;
  sendPrompt: (
    content: string,
    contextRoot?: string,
    steerOutline?: { outline_node_id: string; interaction_id: string },
  ) => Promise<void>;
  abortSession: () => Promise<void>;
  approveAction: (nodeId: string) => Promise<void>;
  rejectAction: (nodeId: string, reason?: string) => Promise<void>;
  modifyAction: (nodeId: string, modifiedArgs: string) => Promise<void>;
  pauseSession: () => Promise<void>;
  resumeSession: () => Promise<void>;
  autoApprove: boolean;
  toggleAutoApprove: () => Promise<void>;
  renameSession: (id: string, name: string) => Promise<void>;
}

export function useSession(): UseSessionReturn {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [currentSession, setCurrentSession] = useState<Session | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [streamingMessage, setStreamingMessage] = useState<Message | null>(null);
  const [isThinking, setIsThinking] = useState(false);
  const [pendingApproval, setPendingApproval] = useState<PendingApproval | null>(null);
  const [autoApprove, setAutoApprove] = useState(false);

  const sseRef = useRef<SseClient | null>(null);
  const refreshTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Track current session ID in a ref so callbacks always see the latest value.
  const currentSessionRef = useRef<Session | null>(null);
  currentSessionRef.current = currentSession;

  const patchGraphData = useCallback((incomingNodes: GraphNode[], incomingEdges: GraphEdge[]) => {
    setGraphData(prev => {
      if (!prev) {
        return { nodes: [...incomingNodes], edges: [...incomingEdges] };
      }
      const nodeMap = new Map(prev.nodes.map(n => [n.id, n]));
      for (const n of incomingNodes) {
        nodeMap.set(n.id, n);
      }
      const edgeMap = new Map(prev.edges.map(e => [e.id, e]));
      for (const e of incomingEdges) {
        edgeMap.set(e.id, e);
      }
      return {
        nodes: [...nodeMap.values()],
        edges: [...edgeMap.values()],
      };
    });
  }, []);

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
        setStreamingMessage(null);
        if (refreshTimerRef.current) clearTimeout(refreshTimerRef.current);
        refreshTimerRef.current = setTimeout(() => {
          refreshTimerRef.current = null;
          refresh(sessionId).catch(console.error);
        }, 500);
      } else if (ev.event === 'graph_update') {
        const root = ev.data as { data?: { nodes?: GraphNode[]; edges?: GraphEdge[] } };
        const payload = root?.data ?? (ev.data as { nodes?: GraphNode[]; edges?: GraphEdge[] });
        const nodes = payload?.nodes;
        const edges = payload?.edges;
        if (Array.isArray(nodes) && Array.isArray(edges)) {
          patchGraphData(nodes, edges);
        }
      } else if (ev.event === 'message_start') {
        const root = ev.data as { data?: { node_id?: string } };
        const payload = root?.data ?? (ev.data as { node_id?: string });
        const nodeId = typeof payload?.node_id === 'string' ? payload.node_id : '';
        console.log(`[SSE] message_start  t=${Date.now()}  node=${nodeId}`);
        if (nodeId) {
          setStreamingMessage({
            id: nodeId,
            role: 'assistant',
            content: '',
            created_at: new Date().toISOString(),
          });
        }
      } else if (ev.event === 'message_delta') {
        const root = ev.data as { data?: { text?: string } };
        const payload = root?.data ?? (ev.data as { text?: string });
        const text = typeof payload?.text === 'string' ? payload.text : '';
        console.log(`[SSE] message_delta  t=${Date.now()}  len=${text.length}  text=${JSON.stringify(text.slice(0, 40))}`);
        setStreamingMessage(prev =>
          prev ? { ...prev, content: prev.content + text } : prev,
        );
      } else if (ev.event === 'message_end') {
        console.log(`[SSE] message_end    t=${Date.now()}`);
        setStreamingMessage(null);
        api.getMessages(sessionId).then(setMessages).catch(console.error);
      } else if (ev.event === 'awaiting_approval') {
        // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
        const payload = ev.data?.data ?? ev.data;
        setPendingApproval({ ...payload, session_id: sessionId } as PendingApproval);
      }
    });
    client.subscribe(sessionId);
    sseRef.current = client;
  }, [refresh, patchGraphData]);

  const selectSession = useCallback(async (id: string) => {
    const session = sessions.find(s => s.id === id) ?? { id } as Session;
    setCurrentSession(session);
    setPendingApproval(null);
    setStreamingMessage(null);
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
    return () => {
      sseRef.current?.unsubscribe();
      if (refreshTimerRef.current) clearTimeout(refreshTimerRef.current);
    };
  }, []);

  const createSession = useCallback(async (name?: string, workspace?: string) => {
    const label = name ?? `Session ${new Date().toLocaleTimeString()}`;
    const session = await api.createSession(label, workspace);
    setSessions(prev => [session, ...prev]);
    setCurrentSession(session);
    setMessages([]);
    setGraphData(null);
    setPendingApproval(null);
    setStreamingMessage(null);
    setIsThinking(false);
    subscribeSse(session.id);
    return session;
  }, [subscribeSse]);

  const sendPrompt = useCallback(
    async (
      content: string,
      contextRoot?: string,
      steerOutline?: { outline_node_id: string; interaction_id: string },
    ): Promise<void> => {
      let session = currentSessionRef.current;
      if (!session) {
        const newSession = await createSession();
        session = newSession;
      }
      if (!session?.id) return;
      setIsThinking(true);
      try {
        const opts =
          contextRoot || steerOutline
            ? {
                ...(contextRoot ? { context_root: contextRoot } : {}),
                ...(steerOutline ? { steer_context: steerOutline } : {}),
              }
            : undefined;
        await api.sendPrompt(session.id, content, opts);
        api.getMessages(session.id).then(setMessages).catch(console.error);
      } catch (err) {
        console.error('Failed to send prompt:', err);
        setIsThinking(false);
      }
    },
    [createSession],
  );

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

  const renameSession = useCallback(async (id: string, name: string) => {
    const updated = await api.renameSession(id, name);
    setSessions(prev => prev.map(s => s.id === id ? { ...s, name: updated.name } : s));
    setCurrentSession(prev => prev?.id === id ? { ...prev, name: updated.name } : prev);
  }, []);

  const messagesWithSegments = useMemo(
    () =>
      messages.map((m) => {
        if (m.role !== 'assistant' || !m.segmented) return m;
        const segments = segmentPartsForInteraction(m.id, graphData);
        if (!segments?.length) return m;
        return { ...m, segments };
      }),
    [messages, graphData],
  );

  return {
    sessions,
    currentSession,
    messages: messagesWithSegments,
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
  };
}
