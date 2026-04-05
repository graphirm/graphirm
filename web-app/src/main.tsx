import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './styles/theme.css';
import '@xyflow/react/dist/style.css';
import { App } from './App';
import { setRuntimeApiKey } from './api/apiKey';

const rootEl = document.getElementById('root');
if (!rootEl) throw new Error('No #root element found');

async function bootstrap(): Promise<void> {
  const fromEnv = import.meta.env.VITE_API_KEY as string | undefined;
  if (fromEnv && fromEnv.length > 0) return;
  try {
    const res = await fetch('/api/client-config');
    if (!res.ok) return;
    const data = (await res.json()) as { api_key?: string };
    if (typeof data.api_key === 'string' && data.api_key.length > 0) {
      setRuntimeApiKey(data.api_key);
    }
  } catch {
    /* dev proxy / offline — useSession will surface API errors */
  }
}

void bootstrap().finally(() => {
  createRoot(rootEl).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
});
