import { createContext, useContext, type ReactNode } from 'react';

const ShellChromeContext = createContext<'column' | 'legacy'>('legacy');

export function ShellChromeProvider({
  mode,
  children,
}: {
  mode: 'column' | 'legacy';
  children: ReactNode;
}) {
  return <ShellChromeContext.Provider value={mode}>{children}</ShellChromeContext.Provider>;
}

/** True when this tree is the phone column, so chat type can grow instead of using the graph card clip. */
export function useColumnChrome(): boolean {
  return useContext(ShellChromeContext) === 'column';
}
