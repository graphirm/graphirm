/** Vite build-time key, or value from GET /api/client-config after bootstrap. */
let runtimeApiKey = '';

export function setRuntimeApiKey(key: string): void {
  runtimeApiKey = key;
}

export function getApiKey(): string {
  const fromEnv = import.meta.env.VITE_API_KEY as string | undefined;
  return (fromEnv && fromEnv.length > 0 ? fromEnv : runtimeApiKey) ?? '';
}
