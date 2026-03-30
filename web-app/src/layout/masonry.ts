import type { Node } from '@xyflow/react';
import { prepare, layout } from '@chenglou/pretext';
import type { GraphNode } from '../types/graph';

/** Matches node cards — same as pretextDimensions preview path. */
const FONT = '12px system-ui, sans-serif';
const LINE_HEIGHT = 17;
const GAP = 12;
const CARD_PAD = 10;
const MAX_COL_WIDTH = 320;
const SINGLE_COL_BREAKPOINT = 560;
const MAX_TEXT_CHARS = 12_000;
/** Cap visual card height (~24 lines of text + header). */
const MAX_BODY_LINES = 24;
const HEADER_CHROME = 52;

function textForMasonryCard(gn: GraphNode): string {
  const nt = gn.node_type;
  switch (nt.type) {
    case 'Interaction':
      return nt.content ?? '';
    case 'Agent':
      return `${nt.name}\n${nt.model}${nt.system_prompt ? `\n${nt.system_prompt.slice(0, 2000)}` : ''}`;
    case 'Content':
      return `${nt.path ?? nt.content_type}\n${nt.body ?? ''}`;
    case 'Task':
      return `${nt.title}\n${nt.description}`;
    case 'Knowledge':
      return `${nt.entity}\n${nt.summary}`;
    default:
      return '';
  }
}

function cardBodyHeight(text: string, textInnerWidth: number): number {
  const slice = text.slice(0, MAX_TEXT_CHARS);
  if (!slice.trim()) {
    return HEADER_CHROME + LINE_HEIGHT * 2;
  }
  try {
    const prepared = prepare(slice, FONT);
    const { lineCount } = layout(prepared, textInnerWidth, LINE_HEIGHT);
    const lines = Math.min(lineCount, MAX_BODY_LINES);
    return HEADER_CHROME + lines * LINE_HEIGHT;
  } catch {
    return HEADER_CHROME + LINE_HEIGHT * 4;
  }
}

/**
 * Column masonry (Pretext demo pattern): shortest-column packing, heights from `layout()`.
 * Skips `group` nodes. `annotation` nodes stack in a strip on the right.
 */
export function applyMasonryLayout(nodes: Node[], rawNodes: GraphNode[], canvasWidth: number): Node[] {
  const gnById = new Map(rawNodes.map(g => [g.id, g]));
  const w = Math.max(480, canvasWidth);

  let colCount: number;
  let colWidth: number;
  if (w <= SINGLE_COL_BREAKPOINT) {
    colCount = 1;
    colWidth = Math.min(MAX_COL_WIDTH, w - GAP * 2);
  } else {
    const minCol = 120 + w * 0.08;
    colCount = Math.max(2, Math.floor((w + GAP) / (minCol + GAP)));
    colWidth = Math.min(MAX_COL_WIDTH, (w - (colCount + 1) * GAP) / colCount);
  }

  const textInnerW = Math.max(80, colWidth - CARD_PAD * 2 - 4);

  const layoutable = nodes.filter(n => n.type && n.type !== 'group' && n.type !== 'annotation');
  const annotations = nodes.filter(n => n.type === 'annotation');

  const sorted = [...layoutable].sort((a, b) => {
    const ta = gnById.get(a.id)?.created_at ?? '';
    const tb = gnById.get(b.id)?.created_at ?? '';
    return ta.localeCompare(tb);
  });

  const colHeights = new Float64Array(colCount);
  for (let c = 0; c < colCount; c++) colHeights[c] = GAP;

  const updates = new Map<string, { x: number; y: number; width: number; height: number }>();

  for (const node of sorted) {
    let shortest = 0;
    for (let c = 1; c < colCount; c++) {
      if (colHeights[c]! < colHeights[shortest]!) shortest = c;
    }
    const gn = gnById.get(node.id);
    const text = gn ? textForMasonryCard(gn) : '';
    const boxH = cardBodyHeight(text, textInnerW);
    const x = GAP + shortest * (colWidth + GAP);
    const y = colHeights[shortest]!;
    colHeights[shortest] = y + boxH + GAP;
    updates.set(node.id, { x, y, width: colWidth, height: boxH });
  }

  let annY = GAP;
  const annX = Math.max(GAP, w - 220);
  for (const ann of annotations) {
    updates.set(ann.id, { x: annX, y: annY, width: 200, height: 80 });
    annY += 88;
  }

  return nodes.map(n => {
    const u = updates.get(n.id);
    if (!u) return n;
    return {
      ...n,
      position: { x: u.x, y: u.y },
      width: u.width,
      height: u.height,
      style: { ...n.style, width: u.width, height: u.height },
    };
  });
}
