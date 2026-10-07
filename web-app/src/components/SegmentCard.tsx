import type { SegmentPart } from '../types/graph';
import { BlockView } from './BlockView';

/** Persisted segments are closed. Look lives on `BlockView`. */
export function SegmentCard({ segment }: { segment: SegmentPart }) {
  return (
    <BlockView
      kicker={segment.type}
      content={segment.content}
      state="done"
      n={segment.n}
      title={segment.title}
      items={segment.items}
    />
  );
}
