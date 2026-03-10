//! Per-turn GLiNER2 span predictions for Phase 4 validation.
//!
//! Reads a corpus JSONL, runs GLiNER2 on each turn, and outputs JSONL with
//! (session_id, turn_index, spans) for comparison with human annotations.

use std::io::BufRead;

use graphirm_graph::CorpusTurn;
use serde::{Deserialize, Serialize};

use crate::error::AgentError;
use super::local_extraction::{OnnxExtractor, RawOnnxEntity};

/// A single predicted span (label + character offsets + confidence).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanPrediction {
    pub label: String,
    pub start: usize,
    pub end: usize,
    pub confidence: f64,
}

/// One line of the predict-spans output: turn id + predicted spans.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnSpans {
    pub session_id: String,
    pub turn_index: u32,
    pub spans: Vec<SpanPrediction>,
}

/// Run GLiNER2 on each turn and collect per-turn spans.
pub async fn run_predict_spans(
    extractor: &OnnxExtractor,
    turns: &[CorpusTurn],
    labels: &[String],
    min_confidence: f64,
) -> Result<Vec<TurnSpans>, AgentError> {
    let mut out = Vec::with_capacity(turns.len());
    for turn in turns {
        let raw: Vec<RawOnnxEntity> = extractor
            .extract_raw(&turn.text, labels, min_confidence)
            .await?;
        let spans: Vec<SpanPrediction> = raw
            .into_iter()
            .map(|e| SpanPrediction {
                label: e.entity_type,
                start: e.start,
                end: e.end,
                confidence: e.confidence,
            })
            .collect();
        out.push(TurnSpans {
            session_id: turn.session_id.clone(),
            turn_index: turn.turn_index,
            spans,
        });
    }
    Ok(out)
}

/// Read predict-spans JSONL (one `TurnSpans` per line).
pub fn read_spans_jsonl<R: BufRead>(reader: R) -> Result<Vec<TurnSpans>, AgentError> {
    let mut rows = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| AgentError::Workflow(format!("read line {}: {}", i + 1, e)))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row: TurnSpans = serde_json::from_str(line)
            .map_err(|e| AgentError::Workflow(format!("parse line {}: {}", i + 1, e)))?;
        rows.push(row);
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turn_spans_roundtrip() {
        let row = TurnSpans {
            session_id: "s1".to_string(),
            turn_index: 2,
            spans: vec![
                SpanPrediction {
                    label: "code".to_string(),
                    start: 0,
                    end: 20,
                    confidence: 0.8,
                },
                SpanPrediction {
                    label: "reasoning".to_string(),
                    start: 25,
                    end: 100,
                    confidence: 0.65,
                },
            ],
        };
        let json = serde_json::to_string(&row).unwrap();
        let back: TurnSpans = serde_json::from_str(&json).unwrap();
        assert_eq!(back.session_id, row.session_id);
        assert_eq!(back.turn_index, row.turn_index);
        assert_eq!(back.spans.len(), 2);
        assert_eq!(back.spans[0].label, "code");
        assert_eq!(back.spans[1].confidence, 0.65);
    }

    #[test]
    fn read_spans_jsonl_parses_lines() {
        let input = r#"{"session_id":"s1","turn_index":0,"spans":[{"label":"code","start":0,"end":10,"confidence":0.9}]}
{"session_id":"s1","turn_index":1,"spans":[]}
"#;
        let rows = read_spans_jsonl(std::io::Cursor::new(input)).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].spans.len(), 1);
        assert_eq!(rows[0].spans[0].label, "code");
        assert_eq!(rows[1].spans.len(), 0);
    }
}
