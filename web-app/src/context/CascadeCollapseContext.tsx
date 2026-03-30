import { createContext, useContext } from 'react';

/** Monotonic counter; increment from Toolbar to reset all timeline cascade in-place expansions. */
export const CascadeCollapseGenerationContext = createContext(0);

export function useCascadeCollapseGeneration(): number {
  return useContext(CascadeCollapseGenerationContext);
}
