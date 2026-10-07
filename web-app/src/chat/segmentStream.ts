export interface ParsedSegment {
  type: string;
  content: string;
  state: 'done' | 'streaming';
  n?: number;
  title?: string;
  items?: string[];
}

export interface SegmentPrefixParse {
  segments: ParsedSegment[];
  showRecovery: boolean;
  plainText: string | null;
}

const ESCAPES: Record<string, string> = {
  '"': '"',
  '\\': '\\',
  '/': '/',
  b: '\b',
  f: '\f',
  n: '\n',
  r: '\r',
  t: '\t',
};

type StringRead =
  | { status: 'ok'; value: string; end: number }
  | { status: 'partial'; value: string }
  | { status: 'invalid'; value: string };

type ValueSkip =
  | { status: 'ok'; end: number }
  | { status: 'partial' }
  | { status: 'invalid' };

type ObjectRead =
  | { status: 'complete'; type: string; content: string; end: number; n?: number; title?: string; items?: string[] }
  | { status: 'partial'; type: string; content: string; started: boolean; n?: number; title?: string; items?: string[] }
  | { status: 'invalid'; type: string; content: string; started: boolean; n?: number; title?: string; items?: string[] };

function waiting(): SegmentPrefixParse {
  return { segments: [], showRecovery: false, plainText: null };
}

function recovery(): SegmentPrefixParse {
  return { segments: [], showRecovery: true, plainText: null };
}

function skipWs(buffer: string, index: number): number {
  let i = index;
  while (i < buffer.length && /\s/.test(buffer[i] ?? '')) i += 1;
  return i;
}

function readString(buffer: string, index: number): StringRead {
  if (buffer[index] !== '"') return { status: 'invalid', value: '' };
  let value = '';
  let i = index + 1;
  while (i < buffer.length) {
    const char = buffer[i];
    if (char === '"') return { status: 'ok', value, end: i + 1 };
    if (char === '\\') {
      if (i + 1 >= buffer.length) return { status: 'partial', value };
      const next = buffer[i + 1] ?? '';
      if (next === 'u') {
        if (i + 6 > buffer.length) return { status: 'partial', value };
        const hex = buffer.slice(i + 2, i + 6);
        if (!/^[0-9a-fA-F]{4}$/.test(hex)) return { status: 'invalid', value };
        value += String.fromCharCode(Number.parseInt(hex, 16));
        i += 6;
        continue;
      }
      const mapped = ESCAPES[next];
      if (mapped === undefined) return { status: 'invalid', value };
      value += mapped;
      i += 2;
      continue;
    }
    if (char === '\n' || char === '\r') return { status: 'invalid', value };
    value += char;
    i += 1;
  }
  return { status: 'partial', value };
}

function skipLiteral(buffer: string, index: number): ValueSkip {
  const rest = buffer.slice(index);
  for (const word of ['true', 'false', 'null']) {
    if (rest === word || rest.startsWith(word)) {
      const end = index + word.length;
      const after = buffer[end];
      if (after !== undefined && /[A-Za-z0-9_]/.test(after)) return { status: 'invalid' };
      return { status: 'ok', end };
    }
    if (word.startsWith(rest) && rest.length > 0) return { status: 'partial' };
  }
  return { status: 'invalid' };
}

function skipNumber(buffer: string, index: number): ValueSkip {
  let i = index;
  if (buffer[i] === '-') i += 1;
  const start = i;
  while (i < buffer.length && /[0-9.eE+-]/.test(buffer[i] ?? '')) i += 1;
  if (i === start) return i >= buffer.length ? { status: 'partial' } : { status: 'invalid' };
  if (i >= buffer.length) return { status: 'partial' };
  return { status: 'ok', end: i };
}

function skipContainer(buffer: string, index: number, open: '{' | '[', close: '}' | ']'): ValueSkip {
  let depth = 0;
  let inString = false;
  let escape = false;
  for (let i = index; i < buffer.length; i += 1) {
    const char = buffer[i];
    if (inString) {
      if (escape) {
        escape = false;
        continue;
      }
      if (char === '\\') {
        escape = true;
        continue;
      }
      if (char === '"') inString = false;
      continue;
    }
    if (char === '"') {
      inString = true;
      continue;
    }
    if (char === open) depth += 1;
    else if (char === close) {
      depth -= 1;
      if (depth === 0) return { status: 'ok', end: i + 1 };
    }
  }
  return { status: 'partial' };
}

function skipValue(buffer: string, index: number): ValueSkip {
  const char = buffer[index];
  if (char === undefined) return { status: 'partial' };
  if (char === '"') {
    const read = readString(buffer, index);
    if (read.status === 'ok') return { status: 'ok', end: read.end };
    if (read.status === 'partial') return { status: 'partial' };
    return { status: 'invalid' };
  }
  if (char === '{') return skipContainer(buffer, index, '{', '}');
  if (char === '[') return skipContainer(buffer, index, '[', ']');
  if (char === '-' || (char >= '0' && char <= '9')) return skipNumber(buffer, index);
  return skipLiteral(buffer, index);
}

function readObject(buffer: string, index: number): ObjectRead {
  let i = index + 1;
  let type = '';
  let content = '';
  let started = false;

  for (;;) {
    i = skipWs(buffer, i);
    if (i >= buffer.length) return { status: 'partial', type, content, started };
    if (buffer[i] === '}') {
      if (!started) return { status: 'invalid', type, content, started: false };
      return { status: 'complete', type, content, end: i + 1 };
    }
    if (buffer[i] === ',') {
      i += 1;
      continue;
    }

    const key = readString(buffer, i);
    if (key.status === 'invalid') return { status: 'invalid', type, content, started };
    if (key.status === 'partial') return { status: 'partial', type, content, started };
    i = skipWs(buffer, key.end);
    if (i >= buffer.length) return { status: 'partial', type, content, started: true };
    if (buffer[i] !== ':') return { status: 'invalid', type, content, started };
    i = skipWs(buffer, i + 1);
    if (i >= buffer.length) return { status: 'partial', type, content, started: true };

    if (key.value === 'type' || key.value === 'content') {
      const text = readString(buffer, i);
      if (text.status === 'invalid') {
        const nextType = key.value === 'type' ? text.value : type;
        const nextContent = key.value === 'content' ? text.value : content;
        return { status: 'invalid', type: nextType, content: nextContent, started: true };
      }
      if (key.value === 'type') type = text.value;
      else content = text.value;
      started = true;
      if (text.status === 'partial') return { status: 'partial', type, content, started: true };
      i = text.end;
      continue;
    }

    const skipped = skipValue(buffer, i);
    if (skipped.status === 'invalid') return { status: 'invalid', type, content, started };
    if (skipped.status === 'partial') return { status: 'partial', type, content, started };
    i = skipped.end;
  }
}

function blockFromRecord(record: unknown): Omit<ParsedSegment, 'state'> | null {
  if (typeof record !== 'object' || record === null) return null;
  const row = record as { type?: unknown; content?: unknown; n?: unknown; title?: unknown; items?: unknown };
  if (typeof row.type !== 'string' || row.type.length === 0) return null;
  const items = Array.isArray(row.items)
    ? row.items.filter((item): item is string => typeof item === 'string')
    : [];
  const hasContent = typeof row.content === 'string';
  if (!hasContent && items.length === 0) return null;
  const title = typeof row.title === 'string' ? row.title : undefined;
  const n = typeof row.n === 'number' ? row.n : undefined;
  let content = typeof row.content === 'string' ? row.content : '';
  if (items.length > 0) {
    const lines: string[] = [];
    if (title) lines.push(title);
    items.forEach((item, index) => lines.push(`${index + 1}. ${item}`));
    content = lines.join('\n');
  }
  return {
    type: row.type,
    content,
    ...(n != null ? { n } : {}),
    ...(title ? { title } : {}),
    ...(items.length > 0 ? { items } : {}),
  };
}

function segmentsFromValue(value: unknown): Array<Omit<ParsedSegment, 'state'>> | null {
  if (typeof value !== 'object' || value === null || !('segments' in value)) return null;
  const segments = (value as { segments?: unknown }).segments;
  if (!Array.isArray(segments)) return null;
  const parsed: Array<Omit<ParsedSegment, 'state'>> = [];
  for (const item of segments) {
    const block = blockFromRecord(item);
    if (!block) return null;
    parsed.push(block);
  }
  return parsed;
}

function parseWhole(buffer: string): SegmentPrefixParse | null {
  try {
    const segments = segmentsFromValue(JSON.parse(buffer) as unknown);
    if (!segments) return null;
    return {
      segments: segments.map((segment) => ({ ...segment, state: 'done' as const })),
      showRecovery: false,
      plainText: null,
    };
  } catch {
    return null;
  }
}

function withSegments(
  segments: ParsedSegment[],
  showRecovery: boolean,
): SegmentPrefixParse {
  return { segments, showRecovery, plainText: null };
}

/** Longest readable prefix of a `{"segments":[...]}` envelope. Stateless over `buffer`. */
export function parseSegmentPrefix(buffer: string): SegmentPrefixParse {
  if (buffer === '') return waiting();
  if (!buffer.startsWith('{')) {
    return { segments: [], showRecovery: false, plainText: buffer };
  }

  const whole = parseWhole(buffer);
  if (whole) return whole;

  let i = skipWs(buffer, 1);
  if (i >= buffer.length) return waiting();

  const key = readString(buffer, i);
  if (key.status === 'invalid') return recovery();
  if (key.status === 'partial') {
    return 'segments'.startsWith(key.value) ? waiting() : recovery();
  }
  if (key.value !== 'segments') return recovery();

  i = skipWs(buffer, key.end);
  if (i >= buffer.length) return waiting();
  if (buffer[i] !== ':') return recovery();
  i = skipWs(buffer, i + 1);
  if (i >= buffer.length) return waiting();
  if (buffer[i] !== '[') return recovery();
  i += 1;

  const segments: ParsedSegment[] = [];
  for (;;) {
    i = skipWs(buffer, i);
    if (i >= buffer.length) return withSegments(segments, false);

    const char = buffer[i];
    if (char === ']') {
      i = skipWs(buffer, i + 1);
      const closed = segments.map((segment) => ({ ...segment, state: 'done' as const }));
      if (i >= buffer.length) return withSegments(closed, false);
      if (buffer[i] !== '}') return withSegments(closed, true);
      i = skipWs(buffer, i + 1);
      return withSegments(closed, i < buffer.length);
    }
    if (char === ',') {
      i += 1;
      continue;
    }
    if (char !== '{') {
      return segments.length > 0 ? withSegments(segments, true) : recovery();
    }

    const object = readObject(buffer, i);
    if (object.status === 'invalid') {
      if (object.started) {
        segments.push({ type: object.type, content: object.content, state: 'streaming' });
      }
      return segments.length > 0 ? withSegments(segments, true) : recovery();
    }
    if (object.status === 'partial') {
      if (object.started) {
        segments.push({ type: object.type, content: object.content, state: 'streaming' });
      }
      return withSegments(segments, false);
    }

    const after = skipWs(buffer, object.end);
    // The envelope is still open when the buffer ends on this object's brace,
    // so the latest block stays streaming until a later token confirms it.
    const state = after >= buffer.length ? 'streaming' : 'done';
    let block: Omit<ParsedSegment, 'state'> | null = null;
    try {
      block = blockFromRecord(JSON.parse(buffer.slice(i, object.end)) as unknown);
    } catch {
      block = null;
    }
    segments.push({
      type: block?.type || object.type,
      content: block?.content || object.content,
      ...(block?.n != null ? { n: block.n } : {}),
      ...(block?.title ? { title: block.title } : {}),
      ...(block?.items && block.items.length > 0 ? { items: block.items } : {}),
      state,
    });
    if (after >= buffer.length) return withSegments(segments, false);
    i = object.end;
  }
}
