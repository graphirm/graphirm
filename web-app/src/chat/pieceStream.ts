import type { SegmentPart } from '../types/graph';

/**
 * Reply pieces (`[agent.reply_pieces]`): the model replies with an HTML index
 * page — a `<nav id="index">` then one `<section class="<kind>">` per part,
 * each with an `<h2>N.0.0 heading</h2>` and lines carrying `data-n`.
 *
 * This reads that page as blocks without a DOM, so it also works on a prefix
 * while the reply streams: closed sections are `done`, the open one is
 * `streaming`, and no tag text ever reaches the screen.
 */

/** True when the reply is (the start of) an HTML page rather than markdown or JSON. */
export function looksLikeHtmlReply(content: string): boolean {
  return /^\s*(?:```html\s*)?<(?:nav|section|h[1-6]|p|ul|div|article|main)\b/i.test(content);
}

/** Leading part or line number written for people (`1.0.0`, `2.0.3`). */
const NUMBER_PREFIX = /^\s*\d+(?:\.\d+)*\.?\s+/;

export function stripLineNumber(text: string): string {
  return text.replace(NUMBER_PREFIX, '');
}

const NAMED_ENTITIES: Record<string, string> = {
  amp: '&',
  lt: '<',
  gt: '>',
  quot: '"',
  apos: "'",
  nbsp: ' ',
};

function decodeEntities(text: string): string {
  return text.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (whole, name: string) => {
    if (name[0] === '#') {
      const code = name[1] === 'x' || name[1] === 'X'
        ? parseInt(name.slice(2), 16)
        : parseInt(name.slice(1), 10);
      return Number.isFinite(code) ? String.fromCodePoint(code) : whole;
    }
    return NAMED_ENTITIES[name.toLowerCase()] ?? whole;
  });
}

/** Tag-free text; a trailing unclosed tag (`<p data-n="1`) is dropped too. */
function textOf(html: string): string {
  return decodeEntities(html.replace(/<[^>]*>?/g, ''));
}

function attr(tag: string, name: string): string | undefined {
  const m = new RegExp(`\\b${name}\\s*=\\s*(?:"([^"]*)"|'([^']*)')`, 'i').exec(tag);
  return m ? (m[1] ?? m[2]) : undefined;
}

/** Text of each `<p>` / `<li>` line in a section body; the last may be unclosed. */
function lineTexts(body: string): string[] {
  const lines: string[] = [];
  const open = /<(p|li)\b[^>]*>/gi;
  let m: RegExpExecArray | null;
  while ((m = open.exec(body)) !== null) {
    const start = m.index + m[0].length;
    const rest = body.slice(start);
    const close = rest.search(new RegExp(`</${m[1]}\\s*>|<(?:p|li)\\b`, 'i'));
    const inner = close === -1 ? rest : rest.slice(0, close);
    const text = stripLineNumber(textOf(inner)).trim();
    if (text) lines.push(text);
  }
  return lines;
}

function codeText(body: string): string {
  const start = /<code\b[^>]*>/i.exec(body) ?? /<pre\b[^>]*>/i.exec(body);
  if (!start) return '';
  const rest = body.slice(start.index + start[0].length);
  const end = rest.search(/<\/code\s*>|<\/pre\s*>/i);
  return textOf(end === -1 ? rest : rest.slice(0, end)).replace(/^\n+|\s+$/g, '');
}

/**
 * The parts of an HTML index reply, in page order. `complete` marks every
 * part done (a stored reply); otherwise an unclosed last section streams.
 */
export function parseHtmlPieces(content: string, complete: boolean): SegmentPart[] {
  const parts: SegmentPart[] = [];
  const opener = /<section\b([^>]*)>/gi;
  let m: RegExpExecArray | null;
  while ((m = opener.exec(content)) !== null) {
    const bodyStart = m.index + m[0].length;
    const closeAt = content.slice(bodyStart).search(/<\/section\s*>/i);
    const body = closeAt === -1 ? content.slice(bodyStart) : content.slice(bodyStart, bodyStart + closeAt);
    const kind = (attr(m[1], 'class') ?? '').trim().split(/\s+/)[0] || 'statement';

    const h2 = /<h2\b[^>]*>([\s\S]*?)(?:<\/h2\s*>|$)/i.exec(body);
    const headingRaw = h2 ? textOf(h2[1]).trim() : '';
    const major = /^(\d+)/.exec(headingRaw);
    const title = stripLineNumber(headingRaw).trim() || undefined;
    const lineBody = h2 ? body.slice(h2.index + h2[0].length) : body;

    const part: SegmentPart = {
      type: kind,
      content: '',
      n: major ? Number(major[1]) : parts.length + 1,
      title,
      state: complete || closeAt !== -1 ? 'done' : 'streaming',
    };
    if (kind === 'code') {
      part.content = codeText(lineBody);
    } else {
      const lines = lineTexts(lineBody);
      if (lines.length > 0) part.items = lines;
      else part.content = textOf(lineBody).trim();
    }
    parts.push(part);
  }
  return parts;
}
