import type { Node } from '@xyflow/react';
import { prepare, layout, type PreparedText } from '@chenglou/pretext';
import type { GraphNode } from '../types/graph';
import { NODE_DIMENSIONS } from './nodeDimensions';

/** Matches collapsed card: `nodes.module.css` 12px × line-height 1.4 on preview. */
const PREVIEW_FONT = '12px system-ui, sans-serif';
const PREVIEW_LINE_HEIGHT = 17;
/** Max inner text width: `--card-max-width` 280 − horizontal padding 20 − border ~2 */
const INNER_TEXT_MAX_WIDTH = 256;
/** Horizontal padding (10+10) + border (1+1) → outer card width = inner + this */
const CARD_HORIZONTAL_CHROME = 22;
/** `--card-min-width` / `--card-max-width` from theme */
const MIN_CARD_OUTER_WIDTH = 160;
const MAX_CARD_OUTER_WIDTH = 280;
/** Narrowest inner width to try when shrink-wrapping (single grapheme clusters can go smaller) */
const SHRINK_MIN_INNER_WIDTH = 32;
/**
 * Vertical chrome: top padding + header row + header margin + bottom padding.
 * Tuned to match BaseCard (badge row + preview area).
 */
const PREVIEW_CHROME_HEIGHT = 52;
const PREVIEW_MAX_LINES = 2;
const MAX_PREVIEW_CHARS = 4000;

/** LRU-ish cap so huge sessions do not grow memory without bound. */
const PREPARE_CACHE_MAX = 400;
const prepareCache = new Map<string, PreparedText>();

function getPrepared(text: string): PreparedText {
  const key = text;
  let hit = prepareCache.get(key);
  if (hit) return hit;
  hit = prepare(text, PREVIEW_FONT);
  if (prepareCache.size >= PREPARE_CACHE_MAX) {
    const first = prepareCache.keys().next().value;
    if (first !== undefined) prepareCache.delete(first);
  }
  prepareCache.set(key, hit);
  return hit;
}

/** Mirrors `InteractionNode` stripMarkdown for preview measurement. */
export function stripMarkdownForPreview(text: string): string {
  return text
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/\*(.*?)\*/g, '$1')
    .replace(/__(.*?)__/g, '$1')
    .replace(/_(.*?)_/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\[([^\]]*)\]\([^)]+\)/g, '$1')
    .replace(/^#+\s*/gm, '')
    .replace(/^-\s+/gm, '')
    .replace(/^>\s+/gm, '')
    .replace(/\n+/g, ' ')
    .trim();
}

/**
 * Same string the collapsed BaseCard preview shows (2-line clamp in CSS).
 * Keeps Pretext height aligned with first-paint DOM.
 */
export function previewTextForDagreLayout(node: Node): string {
  const data = node.data as unknown as GraphNode;
  const nt = data.node_type;
  switch (node.type) {
    case 'interaction': {
      if (nt.type !== 'Interaction') return '';
      const raw = nt.content ?? '';
      const stripped = stripMarkdownForPreview(raw).slice(0, 80);
      return stripped + (raw.length > 80 ? '…' : '');
    }
    case 'content': {
      if (nt.type !== 'Content') return '';
      const label = nt.path ?? nt.content_type;
      const body = nt.body ?? '';
      const bodyPreview = body.slice(0, 60) + (body.length > 60 ? '…' : '');
      return `${label}: ${bodyPreview}`;
    }
    case 'knowledge': {
      if (nt.type !== 'Knowledge') return '';
      return `${nt.entity} (${nt.entity_type})`;
    }
    case 'agent': {
      if (nt.type !== 'Agent') return '';
      return `${nt.name} · ${nt.model}`;
    }
    case 'task': {
      if (nt.type !== 'Task') return '';
      return nt.title;
    }
    default:
      return '';
  }
}

/**
 * Smallest inner (text) width in [minInner, maxInner] such that the paragraph uses at most
 * `maxLines` lines — same break semantics as Pretext `walkLineRanges` / `layout()`.
 * Pure arithmetic after `prepare()`.
 */
export function shrinkWrapInnerWidth(
  prepared: PreparedText,
  maxLines: number,
  minInner: number,
  maxInner: number,
): number {
  const atMax = layout(prepared, maxInner, 1).lineCount;
  if (atMax === 0) return minInner;
  if (atMax > maxLines) {
    return maxInner;
  }
  let lo = minInner;
  let hi = maxInner;
  let best = maxInner;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const lines = layout(prepared, mid, 1).lineCount;
    if (lines <= maxLines) {
      best = mid;
      hi = mid - 1;
    } else {
      lo = mid + 1;
    }
  }
  return best;
}

function estimateSizeFromPreview(preview: string): { width: number; height: number } {
  const slice = preview.slice(0, MAX_PREVIEW_CHARS);
  const prepared = getPrepared(slice);
  const innerW = shrinkWrapInnerWidth(
    prepared,
    PREVIEW_MAX_LINES,
    SHRINK_MIN_INNER_WIDTH,
    INNER_TEXT_MAX_WIDTH,
  );
  const { height: layH } = layout(prepared, innerW, PREVIEW_LINE_HEIGHT);
  const textHeight = Math.min(layH, PREVIEW_MAX_LINES * PREVIEW_LINE_HEIGHT);
  const bodyH = PREVIEW_CHROME_HEIGHT + textHeight;
  const outerW = Math.min(
    MAX_CARD_OUTER_WIDTH,
    Math.max(MIN_CARD_OUTER_WIDTH, innerW + CARD_HORIZONTAL_CHROME),
  );
  const minH = NODE_DIMENSIONS.default.height;
  return { width: outerW, height: Math.max(bodyH, minH) };
}

/**
 * Map node id → dagre box size. Width and height from Pretext on the collapsed preview:
 * width shrink-wraps between theme min/max; height uses the same inner width.
 * Omits group/annotation and empty previews.
 */
export function buildPretextSizeMap(nodes: Node[]): Map<string, { width: number; height: number }> {
  const map = new Map<string, { width: number; height: number }>();
  for (const node of nodes) {
    const t = node.type;
    if (t === 'group' || t === 'annotation' || !t) continue;
    const preview = previewTextForDagreLayout(node);
    if (!preview.trim()) continue;
    try {
      const { width, height } = estimateSizeFromPreview(preview);
      map.set(node.id, { width, height });
    } catch {
      // Canvas/measureText unavailable — omit override; dagre uses fallbacks.
    }
  }
  return map;
}
