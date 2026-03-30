import { createContext, useContext, type ReactNode } from 'react';

/** Prompt send + user-message edit target (popover → InteractionNode). */
export interface GraphCanvasActionsValue {
  sendFromGraph: (content: string, contextRoot?: string) => void;
  editingUserNodeId: string | null;
  setEditingUserNodeId: (id: string | null) => void;
}

const GraphCanvasActionsContext = createContext<GraphCanvasActionsValue | null>(null);

export function GraphCanvasActionsProvider({
  children,
  value,
}: {
  children: ReactNode;
  value: GraphCanvasActionsValue;
}) {
  return (
    <GraphCanvasActionsContext.Provider value={value}>
      {children}
    </GraphCanvasActionsContext.Provider>
  );
}

export function useGraphCanvasActions(): GraphCanvasActionsValue | null {
  return useContext(GraphCanvasActionsContext);
}
