import { stripLineNumber } from '../chat/pieceStream.ts';
import type { GraphData, GraphEdge, GraphNode, SegmentPart } from '../types/graph';

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
    const meta = n.metadata as {
      segment_n?: unknown;
      segment_title?: unknown;
      segment_items?: unknown;
    };
    const items = Array.isArray(meta.segment_items)
      ? meta.segment_items.filter((item): item is string => typeof item === 'string')
      : undefined;
    parts.push({
      type: ct.content_type,
      content: ct.body ?? '',
      language: ct.language,
      n: typeof meta.segment_n === 'number' ? meta.segment_n : undefined,
      title: typeof meta.segment_title === 'string' ? meta.segment_title : undefined,
      items: items && items.length > 0 ? items : undefined,
    });
  }
  return parts.length > 0 ? parts : undefined;
}

function metaNumber(meta: unknown, key: string): number {
  const v = (meta as Record<string, unknown> | undefined)?.[key];
  return typeof v === 'number' ? v : Number(v) || 0;
}

function contentChildren(parentId: string, contentType: string, graph: GraphData): GraphNode[] {
  const out: GraphNode[] = [];
  for (const e of graph.edges) {
    if (e.edge_type !== 'contains' || e.source !== parentId) continue;
    const n = graph.nodes.find(node => node.id === e.target);
    if (n?.node_type.type === 'Content' && n.node_type.content_type === contentType) out.push(n);
  }
  return out;
}

/**
 * Reply pieces of an assistant Interaction (`reply_part` → `reply_line`
 * Contains children), as blocks: kind, heading and numbered lines. A code
 * part, or a part with one line, keeps its text as the block body.
 */
export function piecePartsForInteraction(
  interactionId: string,
  graph: GraphData | null,
): SegmentPart[] | undefined {
  if (!graph?.edges?.length || !graph.nodes?.length) return undefined;
  const partNodes = contentChildren(interactionId, 'reply_part', graph);
  if (partNodes.length === 0) return undefined;
  partNodes.sort((a, b) => metaNumber(a.metadata, 'order') - metaNumber(b.metadata, 'order'));

  return partNodes.map((part, index) => {
    const meta = part.metadata as { kind?: unknown; heading?: unknown } | undefined;
    const kind = typeof meta?.kind === 'string' && meta.kind ? meta.kind : 'statement';
    const heading = typeof meta?.heading === 'string' && meta.heading.trim() ? meta.heading : undefined;
    const lines = contentChildren(part.id, 'reply_line', graph)
      .sort((a, b) => metaNumber(a.metadata, 'position') - metaNumber(b.metadata, 'position'))
      .map(line => (line.node_type.type === 'Content' ? stripLineNumber(line.node_type.body ?? '').trim() : ''))
      .filter(text => text.length > 0);
    const block: SegmentPart = { type: kind, content: '', n: index + 1, title: heading };
    if (kind === 'code' || lines.length === 1) block.content = lines.join('\n');
    else block.items = lines;
    return block;
  });
}
