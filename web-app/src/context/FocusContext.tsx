import { createContext, useContext } from "react";

export const FocusContext = createContext<string | null>(null);

export function useFocusedNodeId(): string | null {
  return useContext(FocusContext);
}
