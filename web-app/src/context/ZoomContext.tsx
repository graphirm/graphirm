import { createContext, useContext, useState, useEffect, useRef } from 'react';
import { useViewport } from '@xyflow/react';

// Zoom threshold for LOD mode (below this = collapsed)
const LOD_THRESHOLD = 0.4;

interface ZoomContextType {
  isLODEnabled: boolean;
  zoom: number;
}

const ZoomContext = createContext<ZoomContextType | undefined>(undefined);

export function ZoomProvider({ children }: { children: React.ReactNode }) {
  const { zoom } = useViewport();
  const [isLODEnabled, setIsLODEnabled] = useState(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Debounced threshold crossing
  useEffect(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    timeoutRef.current = setTimeout(() => {
      const shouldEnableLOD = zoom < LOD_THRESHOLD;
      setIsLODEnabled(shouldEnableLOD);
    }, 150); // 150ms debounce for smoother transitions

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [zoom]);

  return (
    <ZoomContext.Provider value={{ isLODEnabled, zoom }}>
      {children}
    </ZoomContext.Provider>
  );
}

export function useZoom() {
  const context = useContext(ZoomContext);
  if (!context) {
    throw new Error('useZoom must be used within a ZoomProvider');
  }
  return context;
}
