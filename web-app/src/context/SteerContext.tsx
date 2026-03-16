import { createContext, useContext } from 'react';

type SteerCallback = (nodeId: string) => void;

export const SteerContext = createContext<SteerCallback | null>(null);

export function useSteer(): SteerCallback | null {
  return useContext(SteerContext);
}
