import { createContext, useContext } from 'react';

export interface PopoverActions {
  /** Session ID for API calls from the popover */
  sessionId: string | null;
  /** Trigger steer-from-here on an interaction node */
  onSteer: (nodeId: string) => void;
  /** Update a task's status */
  onUpdateTaskStatus: (nodeId: string, status: 'completed' | 'failed') => Promise<void>;
  /** Rate an assistant turn (1-5) */
  onRateTurn: (nodeId: string, rating: number) => Promise<void>;
  /** Pin or unpin a knowledge node */
  onTogglePin: (nodeId: string, pinned: boolean) => Promise<void>;
  /** Edit a knowledge node's summary */
  onEditSummary: (nodeId: string, summary: string) => Promise<void>;
  /** Soft-dismiss a knowledge node (hidden from context / briefing) */
  onDismissKnowledge?: (nodeId: string) => Promise<void>;
  /** User message: open inline edit + re-send from preceding context */
  onStartEditUserMessage?: (nodeId: string) => void;
  /** Tool interaction: persist note linked via RelatesTo */
  onAnnotateToolNode?: (toolNodeId: string, text: string) => Promise<void>;
}

const PopoverContext = createContext<PopoverActions | null>(null);

export const PopoverProvider = PopoverContext.Provider;

export function usePopoverActions(): PopoverActions | null {
  return useContext(PopoverContext);
}
