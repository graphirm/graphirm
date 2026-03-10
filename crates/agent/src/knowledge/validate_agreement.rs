//! Phase 4: compare human annotations to GLiNER2 predictions and report agreement.
//!
//! Human annotation format: JSONL with one object per turn:
//! `{ "session_id": "...", "turn_index": 0, "segments": [ { "type": "code", "start": 0, "end": 20 }, ... ] }`
//! All offsets are character-based (UTF-8).

use std::collections::HashMap;
use std::io::BufRead;

use serde::{Deserialize, Serialize};

use crate::error::AgentError;
use super::predict_spans::{SpanPrediction, TurnSpans};

/// One human-annotated segment (type + character offsets).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanSegment {
    #[serde(rename = "type")]
    pub segment_type: String,
    pub start: usize,
    pub end: usize,
}

/// One line of the human annotations file: turn id + segments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanTurnAnnotation {
    pub session_id: String,
    pub turn_index: u32,
    pub segments: Vec<HumanSegment>,
}

/// Agreement report for one run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgreementReport {
    /// Number of human segments that had a matching GLiNER2 span (same type, boundary overlap ≥ threshold).
    pub matched_segments: u32,
    /// Total human segments across all turns.
    pub total_human_segments: u32,
    /// Agreement fraction (matched / total), or 1.0 if total is 0.
    pub agreement_pct: f64,
    /// Whether agreement_pct >= threshold.
    pub pass: bool,
    /// Threshold used (e.g. 0.75).
    pub threshold: f64,
}

/// Minimum overlap ratio (intersection / min(human_len, gliner_len)) to count as a match.
#[cfg(test)]
const DEFAULT_OVERLAP_RATIO: f64 = 0.5;

/// Read human annotations JSONL.
pub fn read_annotations_jsonl<R: BufRead>(reader: R) -> Result<Vec<HumanTurnAnnotation>, AgentError> {
    let mut rows = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| AgentError::Workflow(format!("read line {}: {}", i + 1, e)))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row: HumanTurnAnnotation = serde_json::from_str(line)
            .map_err(|e| AgentError::Workflow(format!("parse line {}: {}", i + 1, e)))?;
        rows.push(row);
    }
    Ok(rows)
}

/// Build a map (session_id, turn_index) -> TurnSpans for GLiNER2 output.
fn gliner_by_turn(spans: &[TurnSpans]) -> HashMap<(String, u32), &TurnSpans> {
    spans
        .iter()
        .map(|t| ((t.session_id.clone(), t.turn_index), t))
        .collect()
}

/// Overlap length of two ranges [a_start, a_end) and [b_start, b_end).
fn overlap_len(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> usize {
    let start = a_start.max(b_start);
    let end = a_end.min(b_end);
    if end > start {
        end - start
    } else {
        0
    }
}

/// Check if a human segment matches a GLiNER2 span: same type and overlap ratio >= min_ratio.
fn segment_matches(
    human: &HumanSegment,
    gliner: &SpanPrediction,
    min_ratio: f64,
) -> bool {
    if human.segment_type != gliner.label {
        return false;
    }
    let ov = overlap_len(human.start, human.end, gliner.start, gliner.end);
    let human_len = human.end.saturating_sub(human.start);
    let gliner_len = gliner.end.saturating_sub(gliner.start);
    let min_len = human_len.min(gliner_len);
    if min_len == 0 {
        return human_len == 0 && gliner_len == 0;
    }
    (ov as f64 / min_len as f64) >= min_ratio
}

/// Compute agreement: for each human segment, count it as matched if there exists a GLiNER2 span
/// (same turn) with same type and overlap ratio >= min_ratio.
pub fn compute_agreement(
    human: &[HumanTurnAnnotation],
    gliner: &[TurnSpans],
    overlap_ratio_min: f64,
) -> AgreementReport {
    let by_turn = gliner_by_turn(gliner);
    let mut matched = 0u32;
    let mut total = 0u32;

    for ann in human {
        let key = (ann.session_id.clone(), ann.turn_index);
        let gliner_turn = match by_turn.get(&key) {
            Some(t) => t,
            None => {
                total += ann.segments.len() as u32;
                continue;
            }
        };

        for seg in &ann.segments {
            total += 1;
            let found = gliner_turn.spans.iter().any(|s| segment_matches(seg, s, overlap_ratio_min));
            if found {
                matched += 1;
            }
        }
    }

    let agreement_pct = if total == 0 {
        1.0
    } else {
        (matched as f64 / total as f64) * 100.0
    };

    let threshold = 75.0; // pass criterion from plan
    AgreementReport {
        matched_segments: matched,
        total_human_segments: total,
        agreement_pct,
        pass: agreement_pct >= threshold,
        threshold,
    }
}

/// Run full agreement pipeline: human annotations and GLiNER2 spans, with configurable pass threshold.
pub fn validate_agreement(
    human: &[HumanTurnAnnotation],
    gliner: &[TurnSpans],
    pass_threshold_pct: f64,
    overlap_ratio_min: f64,
) -> AgreementReport {
    let mut report = compute_agreement(human, gliner, overlap_ratio_min);
    report.pass = report.agreement_pct >= pass_threshold_pct;
    report.threshold = pass_threshold_pct;
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_annotation_roundtrip() {
        let ann = HumanTurnAnnotation {
            session_id: "s1".to_string(),
            turn_index: 0,
            segments: vec![
                HumanSegment {
                    segment_type: "code".to_string(),
                    start: 0,
                    end: 20,
                },
                HumanSegment {
                    segment_type: "reasoning".to_string(),
                    start: 25,
                    end: 100,
                },
            ],
        };
        let json = serde_json::to_string(&ann).unwrap();
        let back: HumanTurnAnnotation = serde_json::from_str(&json).unwrap();
        assert_eq!(back.segments.len(), 2);
        assert_eq!(back.segments[0].segment_type, "code");
    }

    #[test]
    fn agreement_exact_match() {
        let human = vec![HumanTurnAnnotation {
            session_id: "s1".to_string(),
            turn_index: 0,
            segments: vec![HumanSegment {
                segment_type: "code".to_string(),
                start: 0,
                end: 10,
            }],
        }];
        let gliner = vec![TurnSpans {
            session_id: "s1".to_string(),
            turn_index: 0,
            spans: vec![SpanPrediction {
                label: "code".to_string(),
                start: 0,
                end: 10,
                confidence: 0.9,
            }],
        }];
        let report = compute_agreement(&human, &gliner, DEFAULT_OVERLAP_RATIO);
        assert_eq!(report.total_human_segments, 1);
        assert_eq!(report.matched_segments, 1);
        assert!((report.agreement_pct - 100.0).abs() < 0.01);
    }

    #[test]
    fn agreement_partial_overlap_same_type() {
        let human = vec![HumanTurnAnnotation {
            session_id: "s1".to_string(),
            turn_index: 0,
            segments: vec![HumanSegment {
                segment_type: "code".to_string(),
                start: 0,
                end: 20,
            }],
        }];
        let gliner = vec![TurnSpans {
            session_id: "s1".to_string(),
            turn_index: 0,
            spans: vec![SpanPrediction {
                label: "code".to_string(),
                start: 5,
                end: 25,
                confidence: 0.8,
            }],
        }];
        // Overlap 15 chars, min_len 20 -> ratio 15/20 = 0.75 >= 0.5
        let report = compute_agreement(&human, &gliner, DEFAULT_OVERLAP_RATIO);
        assert_eq!(report.matched_segments, 1);
    }

    #[test]
    fn agreement_type_mismatch_no_match() {
        let human = vec![HumanTurnAnnotation {
            session_id: "s1".to_string(),
            turn_index: 0,
            segments: vec![HumanSegment {
                segment_type: "reasoning".to_string(),
                start: 0,
                end: 10,
            }],
        }];
        let gliner = vec![TurnSpans {
            session_id: "s1".to_string(),
            turn_index: 0,
            spans: vec![SpanPrediction {
                label: "code".to_string(),
                start: 0,
                end: 10,
                confidence: 0.9,
            }],
        }];
        let report = compute_agreement(&human, &gliner, DEFAULT_OVERLAP_RATIO);
        assert_eq!(report.matched_segments, 0);
        assert_eq!(report.agreement_pct, 0.0);
    }

    #[test]
    fn validate_agreement_pass_threshold() {
        let human = vec![HumanTurnAnnotation {
            session_id: "s1".to_string(),
            turn_index: 0,
            segments: vec![
                HumanSegment {
                    segment_type: "code".to_string(),
                    start: 0,
                    end: 10,
                },
                HumanSegment {
                    segment_type: "answer".to_string(),
                    start: 15,
                    end: 50,
                },
            ],
        }];
        let gliner = vec![TurnSpans {
            session_id: "s1".to_string(),
            turn_index: 0,
            spans: vec![
                SpanPrediction {
                    label: "code".to_string(),
                    start: 0,
                    end: 10,
                    confidence: 0.9,
                },
                SpanPrediction {
                    label: "answer".to_string(),
                    start: 15,
                    end: 50,
                    confidence: 0.85,
                },
            ],
        }];
        let report = validate_agreement(&human, &gliner, 75.0, 0.5);
        assert_eq!(report.matched_segments, 2);
        assert_eq!(report.total_human_segments, 2);
        assert!(report.pass);
    }
}
