//! Phase 3: Analyse a label-exploration report and suggest a segment schema.
//!
//! Applies the plan's decision criteria to classify labels as real, redundant, or noise.

use serde::{Deserialize, Serialize};

use super::label_explore::LabelExplorationReport;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LabelVerdict {
    Real,
    Redundant { merge_into: String },
    Noise,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelAnalysis {
    pub label: String,
    pub verdict: LabelVerdict,
    pub turn_pct: f64,
    pub avg_confidence: f64,
    pub max_overlap_pct: f64,
    pub overlap_with: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaRecommendation {
    pub recommended_segment_types: Vec<String>,
    pub label_analyses: Vec<LabelAnalysis>,
    pub merge_suggestions: Vec<MergeSuggestion>,
    pub corpus_turn_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeSuggestion {
    pub label: String,
    pub merge_into: String,
}

const MIN_CONFIDENCE_REAL: f64 = 0.6;
const MIN_TURN_PCT_REAL: f64 = 0.30;
const MAX_OVERLAP_REAL: f64 = 0.40;
const REDUNDANT_OVERLAP_PCT: f64 = 0.60;

pub fn analyse_report(report: &LabelExplorationReport) -> SchemaRecommendation {
    let total_turns = report.corpus_stats.total_turns as f64;
    let max_overlap: std::collections::HashMap<String, (f64, Option<String>)> = report
        .label_stats
        .iter()
        .map(|s| {
            let total = s.total_chars as f64;
            let (pct, other) = report
                .overlap_matrix
                .iter()
                .filter(|e| e.label_a == s.label || e.label_b == s.label)
                .map(|e| {
                    let (other_label, overlap_chars) = if e.label_a == s.label {
                        (e.label_b.as_str(), e.overlap_chars)
                    } else {
                        (e.label_a.as_str(), e.overlap_chars)
                    };
                    let pct = if total > 0.0 {
                        (overlap_chars as f64 / total).min(1.0)
                    } else {
                        0.0
                    };
                    (pct, other_label.to_string())
                })
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0.0, String::new()));
            (
                s.label.clone(),
                (pct, if other.is_empty() { None } else { Some(other) }),
            )
        })
        .collect();

    let mut label_analyses = Vec::with_capacity(report.label_stats.len());
    let mut recommended_segment_types = Vec::new();
    let mut merge_suggestions = Vec::new();

    for stat in &report.label_stats {
        let turn_pct = if total_turns > 0.0 {
            stat.turns_with_label as f64 / total_turns
        } else {
            0.0
        };
        let (max_overlap_pct, overlap_with) = max_overlap
            .get(&stat.label)
            .cloned()
            .unwrap_or((0.0, None));

        let verdict = if max_overlap_pct >= REDUNDANT_OVERLAP_PCT {
            let merge_into = overlap_with.clone().unwrap_or_default();
            merge_suggestions.push(MergeSuggestion {
                label: stat.label.clone(),
                merge_into: merge_into.clone(),
            });
            LabelVerdict::Redundant { merge_into }
        } else if stat.avg_confidence >= MIN_CONFIDENCE_REAL
            && turn_pct >= MIN_TURN_PCT_REAL
            && max_overlap_pct < MAX_OVERLAP_REAL
        {
            recommended_segment_types.push(stat.label.clone());
            LabelVerdict::Real
        } else {
            LabelVerdict::Noise
        };

        label_analyses.push(LabelAnalysis {
            label: stat.label.clone(),
            verdict: verdict.clone(),
            turn_pct,
            avg_confidence: stat.avg_confidence,
            max_overlap_pct,
            overlap_with,
        });
    }

    SchemaRecommendation {
        recommended_segment_types,
        label_analyses,
        merge_suggestions,
        corpus_turn_count: report.corpus_stats.total_turns,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::label_explore::{CorpusStats, LabelStat, OverlapEntry};

    fn make_report(
        total_turns: u32,
        label_stats: Vec<LabelStat>,
        overlap_matrix: Vec<OverlapEntry>,
    ) -> LabelExplorationReport {
        LabelExplorationReport {
            labels: label_stats.iter().map(|s| s.label.clone()).collect(),
            min_confidence: 0.3,
            corpus_stats: CorpusStats {
                total_turns,
                total_chars: 10_000,
                covered_chars: 8_000,
                coverage_pct: 80.0,
                turns_with_any_label: total_turns,
            },
            label_stats,
            overlap_matrix,
        }
    }

    #[test]
    fn real_label_meets_criteria() {
        let report = make_report(
            10,
            vec![LabelStat {
                label: "code".into(),
                span_count: 20,
                turns_with_label: 5,
                total_chars: 2000,
                avg_confidence: 0.72,
                min_confidence: 0.5,
                max_confidence: 0.9,
                avg_span_chars: 100.0,
                min_span_chars: 10,
                max_span_chars: 500,
            }],
            vec![],
        );
        let rec = analyse_report(&report);
        assert!(rec.recommended_segment_types.contains(&"code".to_string()));
        let code_analysis = rec.label_analyses.iter().find(|a| a.label == "code").unwrap();
        assert!(matches!(code_analysis.verdict, LabelVerdict::Real));
    }

    #[test]
    fn noise_low_confidence() {
        let report = make_report(
            10,
            vec![LabelStat {
                label: "mumble".into(),
                span_count: 5,
                turns_with_label: 4,
                total_chars: 500,
                avg_confidence: 0.45,
                min_confidence: 0.2,
                max_confidence: 0.6,
                avg_span_chars: 100.0,
                min_span_chars: 20,
                max_span_chars: 200,
            }],
            vec![],
        );
        let rec = analyse_report(&report);
        assert!(!rec.recommended_segment_types.contains(&"mumble".to_string()));
        let a = rec.label_analyses.iter().find(|a| a.label == "mumble").unwrap();
        assert!(matches!(a.verdict, LabelVerdict::Noise));
    }

    #[test]
    fn redundant_high_overlap() {
        let report = make_report(
            10,
            vec![
                LabelStat {
                    label: "reasoning".into(),
                    span_count: 30,
                    turns_with_label: 8,
                    total_chars: 3000,
                    avg_confidence: 0.7,
                    min_confidence: 0.5,
                    max_confidence: 0.9,
                    avg_span_chars: 100.0,
                    min_span_chars: 20,
                    max_span_chars: 200,
                },
                LabelStat {
                    label: "analysis".into(),
                    span_count: 25,
                    turns_with_label: 7,
                    total_chars: 2500,
                    avg_confidence: 0.68,
                    min_confidence: 0.4,
                    max_confidence: 0.88,
                    avg_span_chars: 100.0,
                    min_span_chars: 30,
                    max_span_chars: 180,
                },
            ],
            vec![OverlapEntry {
                label_a: "analysis".into(),
                label_b: "reasoning".into(),
                overlap_chars: 1800,
                overlap_pct: 72.0,
            }],
        );
        let rec = analyse_report(&report);
        let analysis = rec.label_analyses.iter().find(|a| a.label == "analysis").unwrap();
        assert!(analysis.max_overlap_pct >= REDUNDANT_OVERLAP_PCT);
        assert!(matches!(&analysis.verdict, LabelVerdict::Redundant { .. }));
        assert!(rec.merge_suggestions.iter().any(|m| m.label == "analysis" && m.merge_into == "reasoning"));
    }
}
