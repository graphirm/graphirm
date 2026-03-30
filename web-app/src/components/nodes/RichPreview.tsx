import { useMemo, type ReactNode } from 'react';
import { stripMarkdownForPreview } from '../../layout/pretextDimensions';
import styles from '../../styles/nodes.module.css';

export type PreviewRun =
  | { kind: 'text'; text: string }
  | { kind: 'code'; text: string }
  | { kind: 'mention'; text: string }
  | { kind: 'link'; text: string; href: string };

/** Handles @user, @repo-2, @snake_case (common in agent transcripts). */
const MENTION_SPLIT = /(@[A-Za-z0-9_.-]+)/;

const MD_INLINE_LINK = /\[([^\]]*)\]\(([^)\s]+)\)/g;

/** Allow only safe schemes for preview links (blocks javascript:, data:, etc.). */
export function sanitizePreviewHref(raw: string): string | null {
  const t = raw.trim();
  if (!t) return null;
  const probe = t.slice(0, 16).toLowerCase();
  if (
    probe.startsWith('javascript:') ||
    probe.startsWith('data:') ||
    probe.startsWith('vbscript:')
  ) {
    return null;
  }
  if (/^https?:\/\//i.test(t)) return t;
  if (/^mailto:/i.test(t)) return t;
  return null;
}

/**
 * Split interaction content into text + inline `code` + @mention + markdown link runs;
 * strip markdown in text-only segments. Respects one character budget for visible text.
 */
export function parseInteractionPreviewRuns(content: string, maxChars: number): PreviewRun[] {
  const runs: PreviewRun[] = [];
  let budget = maxChars;
  let i = 0;

  const pushPlain = (cleaned: string) => {
    if (!cleaned || budget <= 0) return;
    const take = Math.min(cleaned.length, budget);
    let piece = cleaned.slice(0, take);
    if (take < cleaned.length) piece += '…';
    runs.push({ kind: 'text', text: piece });
    budget -= take;
  };

  const pushMention = (m: string) => {
    if (!m || budget <= 0) return;
    const take = Math.min(m.length, budget);
    let piece = m.slice(0, take);
    if (take < m.length) piece += '…';
    runs.push({ kind: 'mention', text: piece });
    budget -= take;
  };

  const pushTextWithMentions = (raw: string) => {
    if (!raw || budget <= 0) return;
    const cleaned = stripMarkdownForPreview(raw);
    if (!cleaned) return;
    const parts = cleaned.split(MENTION_SPLIT);
    for (const part of parts) {
      if (!part || budget <= 0) continue;
      if (part.startsWith('@')) pushMention(part);
      else pushPlain(part);
    }
  };

  const pushLink = (labelRaw: string, hrefRaw: string) => {
    if (budget <= 0) return;
    const href = sanitizePreviewHref(hrefRaw);
    const labelFull =
      stripMarkdownForPreview(labelRaw) || hrefRaw.trim().slice(0, 32) || 'link';
    if (!href) {
      pushTextWithMentions(`[${labelRaw}](${hrefRaw})`);
      return;
    }
    const take = Math.min(labelFull.length, budget);
    let text = labelFull.slice(0, take);
    if (take < labelFull.length) text += '…';
    runs.push({ kind: 'link', text, href });
    budget -= take;
  };

  /** Markdown links on raw substring, then mentions/plain inside each text span. */
  const pushRichTextSegment = (raw: string) => {
    if (!raw || budget <= 0) return;
    const re = new RegExp(MD_INLINE_LINK.source, 'g');
    let last = 0;
    let m: RegExpExecArray | null;
    while ((m = re.exec(raw)) !== null) {
      if (m.index > last) pushTextWithMentions(raw.slice(last, m.index));
      if (budget <= 0) return;
      pushLink(m[1] ?? '', m[2] ?? '');
      last = re.lastIndex;
    }
    if (last < raw.length) pushTextWithMentions(raw.slice(last));
  };

  const pushCode = (raw: string) => {
    if (!raw || budget <= 0) return;
    const take = Math.min(raw.length, budget);
    let piece = raw.slice(0, take);
    if (take < raw.length) piece += '…';
    runs.push({ kind: 'code', text: piece });
    budget -= take;
  };

  while (i < content.length && budget > 0) {
    const bt = content.indexOf('`', i);
    if (bt === -1) {
      pushRichTextSegment(content.slice(i));
      break;
    }
    if (bt > i) pushRichTextSegment(content.slice(i, bt));
    const bt2 = content.indexOf('`', bt + 1);
    if (bt2 === -1) {
      pushRichTextSegment(content.slice(bt));
      break;
    }
    pushCode(content.slice(bt + 1, bt2));
    i = bt2 + 1;
  }
  return runs;
}

export function RichInteractionPreview({
  content,
  maxChars = 80,
}: {
  content: string;
  maxChars?: number;
}): ReactNode {
  const runs = useMemo(() => parseInteractionPreviewRuns(content, maxChars), [content, maxChars]);
  if (runs.length === 0) {
    return <span>{stripMarkdownForPreview(content).slice(0, maxChars) || '—'}</span>;
  }
  return (
    <>
      {runs.map((r, idx) =>
        r.kind === 'code' ? (
          <code key={idx} className={styles.previewCode}>
            {r.text}
          </code>
        ) : r.kind === 'mention' ? (
          <span key={idx} className={styles.previewMention}>
            {r.text}
          </span>
        ) : r.kind === 'link' ? (
          <a
            key={idx}
            href={r.href}
            className={styles.previewLink}
            target="_blank"
            rel="noopener noreferrer"
            onClick={e => e.stopPropagation()}
            onPointerDown={e => e.stopPropagation()}
          >
            {r.text}
          </a>
        ) : (
          <span key={idx}>{r.text}</span>
        ),
      )}
    </>
  );
}

export function KnowledgeEntityPreview({
  entityType,
  entity,
  maxEntityChars = 72,
}: {
  entityType: string;
  entity: string;
  maxEntityChars?: number;
}): ReactNode {
  const rest =
    entity.length <= maxEntityChars ? entity : `${entity.slice(0, maxEntityChars)}…`;
  return (
    <>
      <span className={styles.previewChip}>{entityType}</span>
      <span>{rest}</span>
    </>
  );
}
