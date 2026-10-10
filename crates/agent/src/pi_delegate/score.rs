//! Score hand labels against the current parser and baseline.
//!
//! A label matches a piece only when the byte range is unchanged. A range that
//! moved is not a kind error.

use std::path::Path;

use super::label::{StoredLabel, parse_stored_label, segment_text_hash};
use super::pieces::{PIECE_BASELINE_VERSION, PIECE_PARSER_VERSION, PieceKind, structure_segment};

const KINDS: [PieceKind; 11] = PieceKind::ALL;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScoreBucket {
    pub replies: u32,
    pub labeled_pieces: u32,
    pub covered_pieces: u32,
    pub kind_matches: u32,
    pub heading_matches: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScoreReport {
    pub parser_version: &'static str,
    pub baseline_version: &'static str,
    pub single: ScoreBucket,
    pub several: ScoreBucket,
    pub moved_pieces: u32,
    pub changed_text: u32,
    pub missing_source: u32,
    /// `confusion[human][baseline]`.
    pub confusion: [[u32; 11]; 11],
}

impl ScoreReport {
    pub fn empty() -> Self {
        Self {
            parser_version: PIECE_PARSER_VERSION,
            baseline_version: PIECE_BASELINE_VERSION,
            single: ScoreBucket::default(),
            several: ScoreBucket::default(),
            moved_pieces: 0,
            changed_text: 0,
            missing_source: 0,
            confusion: [[0; 11]; 11],
        }
    }
}

/// Score one labeled segment against a fresh parse of `text`.
pub fn score_text(text: &str, label: &StoredLabel) -> ScoreReport {
    let mut report = ScoreReport::empty();
    add_text(&mut report, text, label);
    report
}

fn add_text(report: &mut ScoreReport, text: &str, label: &StoredLabel) {
    if label.text_sha256 != segment_text_hash(text) {
        report.changed_text += 1;
        return;
    }
    let current = structure_segment(text, false);
    let mut covered = 0u32;
    let mut kind_matches = 0u32;
    let mut heading_matches = 0u32;
    for stored in &label.pieces {
        let Some(piece) = current
            .pieces
            .iter()
            .find(|piece| piece.start == stored.start && piece.end == stored.end)
        else {
            report.moved_pieces += 1;
            continue;
        };
        covered += 1;
        if piece.kind == stored.kind {
            kind_matches += 1;
        }
        if piece.heading == stored.heading {
            heading_matches += 1;
        }
        report.confusion[kind_index(stored.kind)][kind_index(piece.kind)] += 1;
    }
    let bucket = if label.pieces.len() <= 1 {
        &mut report.single
    } else {
        &mut report.several
    };
    bucket.replies += 1;
    bucket.labeled_pieces += label.pieces.len() as u32;
    bucket.covered_pieces += covered;
    bucket.kind_matches += kind_matches;
    bucket.heading_matches += heading_matches;
}

/// Score a label file. Each row's `source` is read again and the segment is
/// checked by index and text hash.
pub fn score_label_file(path: &Path) -> std::io::Result<ScoreReport> {
    let raw = std::fs::read_to_string(path)?;
    let mut report = ScoreReport::empty();
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let Some(label) = parse_stored_label(line) else {
            continue;
        };
        let source = Path::new(&label.source);
        let segments = match super::label::load_final_segments(source) {
            Ok(segments) => segments,
            Err(_) => {
                report.missing_source += 1;
                continue;
            }
        };
        let Some(segment) = segments
            .iter()
            .find(|segment| segment.segment_index == label.segment_index)
        else {
            report.missing_source += 1;
            continue;
        };
        add_text(&mut report, &segment.text, &label);
    }
    Ok(report)
}

pub fn format_score(report: &ScoreReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "parser {}  baseline {}\n",
        report.parser_version, report.baseline_version
    ));
    out.push_str(&format_bucket("single-piece replies", &report.single));
    out.push_str(&format_bucket("several-piece replies", &report.several));
    out.push_str(&format!(
        "moved pieces: {}\nchanged text: {}\nmissing source: {}\n",
        report.moved_pieces, report.changed_text, report.missing_source
    ));
    out.push_str("human \\ baseline");
    for kind in KINDS {
        out.push_str(&format!("  {:<12}", kind.as_label()));
    }
    out.push('\n');
    for (row, human) in KINDS.iter().enumerate() {
        out.push_str(&format!("{:<16}", human.as_label()));
        for baseline in 0..KINDS.len() {
            out.push_str(&format!("  {:<12}", report.confusion[row][baseline]));
        }
        out.push('\n');
    }
    out
}

fn format_bucket(name: &str, bucket: &ScoreBucket) -> String {
    format!(
        "{name}: {} replies, coverage {}/{}, kind {}/{}, heading {}/{}\n",
        bucket.replies,
        bucket.covered_pieces,
        bucket.labeled_pieces,
        bucket.kind_matches,
        bucket.covered_pieces,
        bucket.heading_matches,
        bucket.covered_pieces,
    )
}

fn kind_index(kind: PieceKind) -> usize {
    KINDS.iter().position(|item| *item == kind).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pi_delegate::label::LabeledPiece;
    use crate::pi_delegate::label::LabeledReply;
    use crate::pi_delegate::label::{label_line, segment_text_hash};

    fn label_from_current(text: &str) -> StoredLabel {
        let structured = structure_segment(text, false);
        StoredLabel {
            parser_version: PIECE_PARSER_VERSION.to_string(),
            baseline_version: PIECE_BASELINE_VERSION.to_string(),
            source: "run.jsonl".into(),
            segment_index: 0,
            text_sha256: segment_text_hash(text),
            pieces: structured
                .pieces
                .iter()
                .map(|piece| LabeledPiece {
                    start: piece.start,
                    end: piece.end,
                    heading: piece.heading.clone(),
                    kind: piece.kind,
                })
                .collect(),
        }
    }

    #[test]
    fn a_matching_reply_scores_coverage_kind_and_heading() {
        let text = "The server is up.\n\nWant me to restart it?";
        let label = label_from_current(text);
        assert!(label.pieces.len() > 1);
        let report = score_text(text, &label);
        assert_eq!(report.several.replies, 1);
        assert_eq!(report.single.replies, 0);
        assert_eq!(report.several.covered_pieces, report.several.labeled_pieces);
        assert_eq!(report.several.kind_matches, report.several.covered_pieces);
        assert_eq!(
            report.several.heading_matches,
            report.several.covered_pieces
        );
        assert_eq!(report.moved_pieces, 0);
    }

    #[test]
    fn a_shifted_range_is_moved_and_not_a_kind_error() {
        let text = "The server is up.\n\nWant me to restart it?";
        let mut label = label_from_current(text);
        let last = label.pieces.len() - 1;
        label.pieces[last].end -= 1;
        let report = score_text(text, &label);
        assert_eq!(report.moved_pieces, 1);
        assert_eq!(
            report.several.covered_pieces + 1,
            report.several.labeled_pieces
        );
        assert_eq!(report.several.kind_matches, report.several.covered_pieces);
    }

    #[test]
    fn a_single_piece_reply_is_scored_apart_from_several() {
        let text = "The file content is: `hi`";
        let label = label_from_current(text);
        assert_eq!(label.pieces.len(), 1);
        let report = score_text(text, &label);
        assert_eq!(report.single.replies, 1);
        assert_eq!(report.several.replies, 0);
        assert_eq!(report.single.kind_matches, 1);
    }

    #[test]
    fn a_wrong_kind_lands_off_the_confusion_diagonal() {
        let text = "The file content is: `hi`";
        let mut label = label_from_current(text);
        label.pieces[0].kind = PieceKind::Caveat;
        let report = score_text(text, &label);
        assert_eq!(report.single.kind_matches, 0);
        assert_eq!(report.single.covered_pieces, 1);
        let human = kind_index(PieceKind::Caveat);
        let baseline = kind_index(PieceKind::Statement);
        assert_eq!(report.confusion[human][baseline], 1);
        let shown = format_score(&report);
        assert!(shown.contains("single-piece replies"));
        assert!(shown.contains("several-piece replies"));
        assert!(shown.contains("caveat"));
    }

    #[test]
    fn a_changed_segment_is_not_scored_as_a_kind() {
        let text = "The file content is: `hi`";
        let mut label = label_from_current(text);
        label.text_sha256 = "nope".into();
        let report = score_text(text, &label);
        assert_eq!(report.changed_text, 1);
        assert_eq!(report.single.replies, 0);
        assert_eq!(report.moved_pieces, 0);
    }

    #[test]
    fn score_label_file_reads_the_run_named_in_the_row() {
        let dir = std::env::temp_dir().join(format!("graphirm-score-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let run = dir.join("run.jsonl");
        let text = "The file content is: `hi`";
        let line = format!(
            r#"{{"type":"message_end","message":{{"role":"assistant","content":[{{"type":"text","text":"{text}"}}],"stopReason":"stop"}}}}"#
        );
        std::fs::write(&run, line).unwrap();
        let segments = crate::pi_delegate::label::load_final_segments(&run).unwrap();
        assert_eq!(segments.len(), 1);
        let current = label_from_current(&segments[0].text);
        let reply = LabeledReply {
            source: segments[0].source.clone(),
            segment_index: 0,
            text_sha256: segment_text_hash(&segments[0].text),
            pieces: current.pieces,
        };
        let labels = dir.join("piece-labels.jsonl");
        std::fs::write(&labels, label_line(&reply)).unwrap();
        let report = score_label_file(&labels).unwrap();
        assert_eq!(report.single.kind_matches, 1);
        assert_eq!(report.missing_source, 0);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
