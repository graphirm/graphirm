import type { GraphData, GraphEdge, SegmentPart } from '../types/graph';

/** Flatten legacy assistant JSON envelope to readable text (old DB rows). */
export function cleanLegacyAssistantContent(content: string): string {
  const t = content.trim();
  if (!t.startsWith('{') && !t.startsWith('[')) return content;
  try {
    const parsed = JSON.parse(t) as unknown;
    if (
      typeof parsed === 'object' &&
      parsed !== null &&
      'segments' in parsed &&
      Array.isArray((parsed as { segments: unknown }).segments)
    ) {
      const segs = (parsed as { segments: Array<{ content?: string }> }).segments;
      return segs.map(s => s.content ?? '').join('\n\n');
    }
    if (Array.isArray(parsed)) {
      return parsed
        .map((s: { content?: string }) =>
          typeof s === 'object' && s !== null && 'content' in s
            ? String((s as { content?: string }).content ?? '')
            : JSON.stringify(s),
        )
        .join('\n\n');
    }
  } catch {
    /* not JSON */
  }
  return content;
}

function edgeOrder(e: GraphEdge): number {
  const m = e.metadata as { order?: unknown; outline?: unknown } | undefined;
  if (m?.outline === true) return Number.MAX_SAFE_INTEGER;
  const o = m?.order;
  return typeof o === 'number' ? o : Number(o) || 0;
}

/** Direct segment children of an assistant Interaction (excludes outline rows). */
export function segmentPartsForInteraction(
  interactionId: string,
  graph: GraphData | null,
): SegmentPart[] | undefined {
  if (!graph?.edges?.length || !graph.nodes?.length) return undefined;

  const outgoing = graph.edges.filter(
    (e): e is GraphEdge =>
      e.edge_type === 'contains' &&
      e.source === interactionId &&
      !(e.metadata as { outline?: boolean } | undefined)?.outline,
  );
  if (outgoing.length === 0) return undefined;

  outgoing.sort((a, b) => edgeOrder(a) - edgeOrder(b));

  const parts: SegmentPart[] = [];
  for (const e of outgoing) {
    const n = graph.nodes.find(node => node.id === e.target);
    if (!n || n.node_type.type !== 'Content') continue;
    const ct = n.node_type;
    parts.push({
      type: ct.content_type,
      content: ct.body ?? '',
      language: ct.language,
    });
  }
  return parts.length > 0 ? parts : undefined;
}
