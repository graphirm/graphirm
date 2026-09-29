/** Raw `metadata.reply_judge.scores`. Fields are optional; missing scores do not fire a hint. */
export interface ReplyScores {
  move?: number;
  claims_completion?: number;
  cites_evidence?: number;
  presents_decision?: number;
  presents_as_options?: number;
}

// Display cutoffs live only here. The server stores raw scores and does not apply them.
// - claims_completion >= 0.8 and cites_evidence < 0.4 → "Completion claimed without evidence"
// - presents_decision >= 0.6 → "This reply asks for a decision"
// - presents_as_options < 0.7 → "Ask for the choices as a list"
// A missing score does not fire that hint.
export function replyHints(scores: ReplyScores | undefined): string[] {
  if (!scores) return [];
  const hints: string[] = [];
  if (
    typeof scores.claims_completion === 'number' &&
    typeof scores.cites_evidence === 'number' &&
    scores.claims_completion >= 0.8 &&
    scores.cites_evidence < 0.4
  ) {
    hints.push('Completion claimed without evidence');
  }
  if (typeof scores.presents_decision === 'number' && scores.presents_decision >= 0.6) {
    hints.push('This reply asks for a decision');
  }
  if (typeof scores.presents_as_options === 'number' && scores.presents_as_options < 0.7) {
    hints.push('Ask for the choices as a list');
  }
  return hints;
}
