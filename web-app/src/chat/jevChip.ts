/** Space, middle dot (U+00B7), space. */
const SEPARATOR = ' \u00b7 ';

function textPart(value: string | undefined): string | null {
  if (typeof value !== 'string' || value.length === 0) return null;
  return value;
}

function confidencePart(value: number | undefined): string | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  return `p=${value}`;
}

/** Chip label from routing metadata. Null only when tier, strategy, and confidence are all missing. */
export function formatJevChip(meta: {
  model_tier?: string;
  routing_strategy?: string;
  routing_confidence?: number;
}): string | null {
  const parts = [
    textPart(meta.model_tier),
    textPart(meta.routing_strategy),
    confidencePart(meta.routing_confidence),
  ].filter((part): part is string => part !== null);
  if (parts.length === 0) return null;
  return parts.join(SEPARATOR);
}
