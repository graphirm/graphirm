//! Label exploration for structured-LLM-response discovery.
//!
//! Reads a JSONL corpus, runs GLiNER2 with candidate labels, and produces
//! a statistics report (per-label counts, confidence, coverage, overlap).
//!
//! # Generating a real corpus
//!
//! Export assistant turns from a Graphirm graph database:
//!
//! ```bash
//! graphirm export-corpus --db path/to/graph.db -o corpus.jsonl
//! ```
//!
//! Then run label exploration (requires `GLINER2_MODEL_DIR` and `--features local-extraction`):
//!
//! ```bash
//! graphirm label-explore --corpus corpus.jsonl --labels "observation,reasoning,code,answer" -o report.json
//! ```
//!
//! A small synthetic fixture for tests lives at `crates/agent/tests/fixtures/corpus_synthetic.jsonl`.

use std::collections::HashMap;
use std::io::BufRead;

use graphirm_graph::CorpusTurn;
use serde::{Deserialize, Serialize};

use crate::error::AgentError;
use super::local_extraction::{OnnxExtractor, RawOnnxEntity};

/// Corpus-level statistics from a label-exploration run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusStats {
    pub total_turns: u32,
    pub total_chars: u64,
    pub covered_chars: u64,
    pub coverage_pct: f64,
    pub turns_with_any_label: u32,
}

/// Per-label statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelStat {
    pub label: String,
    pub span_count: u32,
    pub turns_with_label: u32,
    /// Total character count covered by this label (sum of span lengths; overlapping spans counted once per label).
    pub total_chars: u64,
    pub avg_confidence: f64,
    pub min_confidence: f64,
    pub max_confidence: f64,
    pub avg_span_chars: f64,
    pub min_span_chars: u32,
    pub max_span_chars: u32,
}

/// Overlap between two labels (only pairs with overlap > 0).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlapEntry {
    pub label_a: String,
    pub label_b: String,
    pub overlap_chars: u64,
    pub overlap_pct: f64,
}

/// Full report from a label-exploration run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelExplorationReport {
    pub labels: Vec<String>,
    pub min_confidence: f64,
    pub corpus_stats: CorpusStats,
    pub label_stats: Vec<LabelStat>,
    pub overlap_matrix: Vec<OverlapEntry>,
}

/// Read a JSONL corpus file (one `CorpusTurn` per line).
pub fn read_corpus_jsonl<R: BufRead>(reader: R) -> Result<Vec<CorpusTurn>, AgentError> {
    let mut turns = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| AgentError::Workflow(format!("read line {}: {}", i + 1, e)))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let turn: CorpusTurn = serde_json::from_str(line)
            .map_err(|e| AgentError::Workflow(format!("parse line {}: {}", i + 1, e)))?;
        turns.push(turn);
    }
    Ok(turns)
}

/// Read up to `limit` corpus turns from a reader (for batched processing to limit memory).
pub fn read_corpus_jsonl_batch<R: BufRead>(
    reader: &mut R,
    limit: usize,
) -> Result<Vec<CorpusTurn>, AgentError> {
    let mut turns = Vec::with_capacity(limit.min(1024));
    let mut line = String::new();
    let mut line_num = 0usize;
    while turns.len() < limit {
        line.clear();
        let n = reader
            .read_line(&mut line)
            .map_err(|e| AgentError::Workflow(format!("read line: {}", e)))?;
        if n == 0 {
            break;
        }
        line_num += 1;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let turn: CorpusTurn = serde_json::from_str(trimmed)
            .map_err(|e| AgentError::Workflow(format!("parse line {}: {}", line_num, e)))?;
        turns.push(turn);
    }
    Ok(turns)
}

/// Run GLiNER2 over each turn and aggregate statistics.
pub async fn run_label_exploration(
    extractor: &OnnxExtractor,
    turns: &[CorpusTurn],
    labels: &[String],
    min_confidence: f64,
) -> Result<LabelExplorationReport, AgentError> {
    let mut total_chars: u64 = 0;
    let mut covered_chars: u64 = 0;
    let mut turns_with_any_label: u32 = 0;

    // Per-label: span_count, turns_with_label, sum_conf, min_conf, max_conf, sum_span_chars, min_span_chars, max_span_chars
    let mut per_label: HashMap<String, (u32, u32, f64, f64, f64, u64, u32, u32)> = HashMap::new();
    for label in labels {
        per_label.insert(
            label.clone(),
            (0, 0, 0.0, f64::MAX, f64::MIN, 0, u32::MAX, 0),
        );
    }

    // Overlap: (label_a, label_b) -> overlap_chars (only a < b to avoid double count)
    let mut overlap_chars: HashMap<(String, String), u64> = HashMap::new();
    let mut label_total_chars: HashMap<String, u64> = HashMap::new();
    for label in labels {
        label_total_chars.insert(label.clone(), 0);
    }

    for turn in turns {
        let text_len = turn.text.chars().count() as u64;
        total_chars += text_len;

        let raw = extractor
            .extract_raw(&turn.text, labels, min_confidence)
            .await?;

        if raw.is_empty() {
            continue;
        }

        turns_with_any_label += 1;

        // Covered: union of all [start, end)
        let mut covered_in_turn: Vec<(usize, usize)> = raw
            .iter()
            .map(|e| (e.start, e.end))
            .collect();
        covered_in_turn.sort_by_key(|&(s, _)| s);
        let merged = merge_ranges(covered_in_turn);
        let turn_covered: u64 = merged.iter().map(|&(s, e)| (e - s) as u64).sum();
        covered_chars += turn_covered;

        // Group by label
        let by_label: HashMap<String, Vec<&RawOnnxEntity>> = raw.iter().fold(
            HashMap::new(),
            |mut acc, e| {
                acc.entry(e.entity_type.clone()).or_default().push(e);
                acc
            },
        );

        for (label, entities) in &by_label {
            let count = entities.len() as u32;
            let sum_conf: f64 = entities.iter().map(|e| e.confidence).sum();
            let min_conf = entities.iter().map(|e| e.confidence).fold(f64::MAX, f64::min);
            let max_conf = entities.iter().map(|e| e.confidence).fold(f64::MIN, f64::max);
            let span_chars: Vec<u32> = entities.iter().map(|e| (e.end - e.start) as u32).collect();
            let sum_chars: u64 = span_chars.iter().map(|&c| c as u64).sum();
            let min_c = *span_chars.iter().min().unwrap_or(&0);
            let max_c = *span_chars.iter().max().unwrap_or(&0);

            if let Some(entry) = per_label.get_mut(label) {
                entry.0 += count;
                entry.1 += 1; // turns_with_label
                entry.2 += sum_conf;
                entry.3 = entry.3.min(min_conf);
                entry.4 = entry.4.max(max_conf);
                entry.5 += sum_chars;
                entry.6 = entry.6.min(min_c);
                entry.7 = entry.7.max(max_c);
            }
            *label_total_chars.get_mut(label).unwrap_or(&mut 0) += sum_chars;
        }

        // Overlap between each pair of labels (in this turn); merge ranges so we don't double-count
        let label_list: Vec<&String> = by_label.keys().collect();
        for (i, &la) in label_list.iter().enumerate() {
            for &lb in label_list.iter().skip(i + 1) {
                let spans_a: Vec<(usize, usize)> =
                    by_label[la].iter().map(|e| (e.start, e.end)).collect();
                let spans_b: Vec<(usize, usize)> =
                    by_label[lb].iter().map(|e| (e.start, e.end)).collect();
                let merged_a = merge_ranges(spans_a);
                let merged_b = merge_ranges(spans_b);
                let ov = range_overlap_chars(&merged_a, &merged_b);
                if ov > 0 {
                    let key = (la.clone(), lb.clone());
                    *overlap_chars.entry(key).or_insert(0) += ov;
                }
            }
        }
    }

    let coverage_pct = if total_chars > 0 {
        (covered_chars as f64 / total_chars as f64) * 100.0
    } else {
        0.0
    };

    let label_stats: Vec<LabelStat> = labels
        .iter()
        .map(|label| {
            let (span_count, turns_with_label, sum_conf, min_conf, max_conf, sum_span_chars, min_span_chars, max_span_chars) =
                per_label.get(label).copied().unwrap_or((0, 0, 0.0, 0.0, 0.0, 0, 0, 0));
            let total_chars = label_total_chars.get(label).copied().unwrap_or(0);
            let avg_confidence = if span_count > 0 {
                sum_conf / span_count as f64
            } else {
                0.0
            };
            let avg_span_chars = if span_count > 0 {
                sum_span_chars as f64 / span_count as f64
            } else {
                0.0
            };
            let min_confidence_val = if min_conf == f64::MAX { 0.0 } else { min_conf };
            let max_confidence_val = if max_conf == f64::MIN { 0.0 } else { max_conf };
            let min_span_chars_val = if min_span_chars == u32::MAX { 0 } else { min_span_chars };
            LabelStat {
                label: label.clone(),
                span_count,
                turns_with_label,
                total_chars,
                avg_confidence,
                min_confidence: min_confidence_val,
                max_confidence: max_confidence_val,
                avg_span_chars,
                min_span_chars: min_span_chars_val,
                max_span_chars,
            }
        })
        .collect();

    let overlap_matrix: Vec<OverlapEntry> = overlap_chars
        .into_iter()
        .map(|((label_a, label_b), overlap_chars)| {
            let a_chars = label_total_chars.get(&label_a).copied().unwrap_or(0);
            let b_chars = label_total_chars.get(&label_b).copied().unwrap_or(0);
            let overlap_pct = if a_chars > 0 && b_chars > 0 {
                (overlap_chars as f64 / (a_chars.min(b_chars) as f64)) * 100.0
            } else {
                0.0
            };
            OverlapEntry {
                label_a,
                label_b,
                overlap_chars,
                overlap_pct,
            }
        })
        .collect();

    Ok(LabelExplorationReport {
        labels: labels.to_vec(),
        min_confidence,
        corpus_stats: CorpusStats {
            total_turns: turns.len() as u32,
            total_chars,
            covered_chars,
            coverage_pct,
            turns_with_any_label,
        },
        label_stats,
        overlap_matrix,
    })
}

fn merge_ranges(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    if ranges.is_empty() {
        return ranges;
    }
    ranges.sort_by_key(|&(s, _)| s);
    let mut merged = vec![ranges[0]];
    for &(s, e) in ranges.iter().skip(1) {
        let last = merged.last_mut().unwrap();
        if s <= last.1 {
            last.1 = last.1.max(e);
        } else {
            merged.push((s, e));
        }
    }
    merged
}

fn range_overlap_chars(a: &[(usize, usize)], b: &[(usize, usize)]) -> u64 {
    let mut total: u64 = 0;
    for &(s1, e1) in a {
        for &(s2, e2) in b {
            let start = s1.max(s2);
            let end = e1.min(e2);
            if end > start {
                total += (end - start) as u64;
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn report_roundtrip() {
        let report = LabelExplorationReport {
            labels: vec!["observation".into(), "code".into()],
            min_confidence: 0.3,
            corpus_stats: CorpusStats {
                total_turns: 10,
                total_chars: 5000,
                covered_chars: 4000,
                coverage_pct: 80.0,
                turns_with_any_label: 8,
            },
            label_stats: vec![
                LabelStat {
                    label: "observation".into(),
                    span_count: 20,
                    turns_with_label: 7,
                    total_chars: 910,
                    avg_confidence: 0.65,
                    min_confidence: 0.31,
                    max_confidence: 0.92,
                    avg_span_chars: 45.5,
                    min_span_chars: 5,
                    max_span_chars: 120,
                },
                LabelStat {
                    label: "code".into(),
                    span_count: 15,
                    turns_with_label: 6,
                    total_chars: 1200,
                    avg_confidence: 0.72,
                    min_confidence: 0.4,
                    max_confidence: 0.95,
                    avg_span_chars: 80.0,
                    min_span_chars: 12,
                    max_span_chars: 200,
                },
            ],
            overlap_matrix: vec![OverlapEntry {
                label_a: "observation".into(),
                label_b: "code".into(),
                overlap_chars: 50,
                overlap_pct: 2.5,
            }],
        };
        let json = serde_json::to_string(&report).unwrap();
        let back: LabelExplorationReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back.labels, report.labels);
        assert_eq!(back.corpus_stats.total_turns, report.corpus_stats.total_turns);
        assert_eq!(back.label_stats.len(), 2);
        assert_eq!(back.overlap_matrix.len(), 1);
    }

    #[test]
    fn merge_ranges_non_overlapping() {
        let r = vec![(0, 10), (20, 30)];
        assert_eq!(super::merge_ranges(r), [(0, 10), (20, 30)]);
    }

    #[test]
    fn merge_ranges_overlapping() {
        let r = vec![(0, 10), (5, 15), (20, 30)];
        assert_eq!(super::merge_ranges(r), [(0, 15), (20, 30)]);
    }

    #[test]
    fn range_overlap_chars_disjoint() {
        let a = vec![(0, 10)];
        let b = vec![(20, 30)];
        assert_eq!(super::range_overlap_chars(&a, &b), 0);
    }

    #[test]
    fn range_overlap_chars_intersect() {
        let a = vec![(0, 100)];
        let b = vec![(50, 80)];
        assert_eq!(super::range_overlap_chars(&a, &b), 30);
    }

    #[test]
    fn read_corpus_jsonl_parses_turns() {
        let input = r#"{"session_id":"s1","turn_index":0,"role":"assistant","text":"Hello world."}
{"session_id":"s1","turn_index":1,"role":"assistant","text":"Here is code:\n```rust\nfn main() {}\n```"}
"#;
        let turns = read_corpus_jsonl(Cursor::new(input)).unwrap();
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].session_id, "s1");
        assert_eq!(turns[0].text, "Hello world.");
        assert_eq!(turns[1].text, "Here is code:\n```rust\nfn main() {}\n```");
    }

    #[test]
    fn read_corpus_jsonl_skips_empty_lines() {
        let input = r#"{"session_id":"s1","turn_index":0,"role":"assistant","text":"One"}

{"session_id":"s1","turn_index":1,"role":"assistant","text":"Two"}
"#;
        let turns = read_corpus_jsonl(Cursor::new(input)).unwrap();
        assert_eq!(turns.len(), 2);
    }
}
