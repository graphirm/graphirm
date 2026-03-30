import { useMemo, type ReactNode } from 'react';
import { stripMarkdownForPreview } from '../../layout/pretextDimensions';
import styles from '../../styles/nodes.module.css';

export type PreviewRun = { kind: 'text' | 'code'; text: string };

/**
 * Split interaction content into text + inline `code` runs; strip markdown in text only.
 * Respects a total character budget (plain + code).
 */
export function parseInteractionPreviewRuns(content: string, maxChars: number): PreviewRun[] {
  const runs: PreviewRun[] = [];
  let budget = maxChars;
  let i = 0;

  const pushText = (raw: string) => {
    if (!raw || budget <= 0) return;
    const cleaned = stripMarkdownForPreview(raw);
    if (!cleaned) return;
    const take = Math.min(cleaned.length, budget);
    let piece = cleaned.slice(0, take);
    if (take < cleaned.length) piece += '…';
    runs.push({ kind: 'text', text: piece });
    budget -= take;
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
      pushText(content.slice(i));
      break;
    }
    if (bt > i) pushText(content.slice(i, bt));
    const bt2 = content.indexOf('`', bt + 1);
    if (bt2 === -1) {
      pushText(content.slice(bt));
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
