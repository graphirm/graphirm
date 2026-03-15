//! Structured response segment parsing and graph persistence.

use crate::error::AgentError;
use serde::{Deserialize, Serialize};

/// A parsed segment from an LLM structured response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    /// The segment type label (e.g. "code", "reasoning", "answer").
    pub segment_type: String,
    /// The text content of this segment.
    pub content: String,
    /// Character offset where this segment starts in the concatenated raw text.
    #[serde(default)]
    pub start: usize,
    /// Character offset where this segment ends (exclusive) in the concatenated raw text.
    #[serde(default)]
    pub end: usize,
}

/// Wire format the LLM is asked to produce.
#[derive(Debug, Deserialize)]
struct StructuredResponse {
    segments: Vec<SegmentWire>,
}

#[derive(Debug, Deserialize)]
struct SegmentWire {
    #[serde(rename = "type")]
    segment_type: String,
    content: String,
}

/// Try to parse LLM response text as a structured segment JSON envelope.
///
/// Expected format: `{"segments": [{"type": "...", "content": "..."}, ...]}`.
/// Strips markdown code fences if the model wrapped the JSON.
/// Returns `Err` if the text is not valid JSON or missing the `segments` key.
pub fn parse_structured_segments(text: &str) -> Result<Vec<Segment>, AgentError> {
    let trimmed = text.trim();
    let clean = if trimmed.starts_with("```") {
        trimmed
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim()
    } else {
        trimmed
    };

    let parsed: StructuredResponse = serde_json::from_str(clean)
        .map_err(|e| AgentError::Workflow(format!("Segment JSON parse failed: {e}")))?;

    // Compute monotonically increasing character offsets from concatenation order.
    let mut offset = 0usize;
    let segments = parsed
        .segments
        .into_iter()
        .map(|w| {
            let start = offset;
            let end = start + w.content.len();
            offset = end;
            Segment {
                segment_type: w.segment_type,
                content: w.content,
                start,
                end,
            }
        })
        .collect();

    Ok(segments)
}

/// Find parent-child nesting pairs among segments based on character span containment.
///
/// Returns `(parent_index, child_index)` pairs where the child's span falls entirely
/// within the parent's span (but is not identical to it).
pub fn detect_nesting(segments: &[Segment]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (i, parent) in segments.iter().enumerate() {
        for (j, child) in segments.iter().enumerate() {
            if i == j {
                continue;
            }
            let child_inside_parent = child.start >= parent.start && child.end <= parent.end;
            let not_identical = child.start != parent.start || child.end != parent.end;
            if child_inside_parent && not_identical {
                pairs.push((i, j));
            }
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_structured_output_valid() {
        let json = r#"{"segments": [
            {"type": "reasoning", "content": "The bug is in line 42."},
            {"type": "code", "content": "fn fix() {}"}
        ]}"#;
        let result = parse_structured_segments(json);
        assert!(result.is_ok());
        let segments = result.unwrap();
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].segment_type, "reasoning");
        assert_eq!(segments[0].content, "The bug is in line 42.");
        assert_eq!(segments[1].segment_type, "code");
        assert_eq!(segments[1].content, "fn fix() {}");
    }

    #[test]
    fn test_parse_structured_output_offsets() {
        let json = r#"{"segments": [
            {"type": "reasoning", "content": "hello"},
            {"type": "code", "content": "world"}
        ]}"#;
        let segments = parse_structured_segments(json).unwrap();
        assert_eq!(segments[0].start, 0);
        assert_eq!(segments[0].end, 5); // len("hello")
        assert_eq!(segments[1].start, 5);
        assert_eq!(segments[1].end, 10); // len("world")
    }

    #[test]
    fn test_parse_structured_output_strips_code_fences() {
        let json = "```json\n{\"segments\": [{\"type\": \"answer\", \"content\": \"done\"}]}\n```";
        let result = parse_structured_segments(json);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_parse_structured_output_invalid_json() {
        let result = parse_structured_segments("This is plain text, not JSON.");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_structured_output_missing_segments_key() {
        let result = parse_structured_segments(r#"{"items": []}"#);
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_nesting_found() {
        let segments = vec![
            Segment {
                segment_type: "plan".into(),
                content: "long plan text here foo bar baz".into(),
                start: 0,
                end: 100,
            },
            Segment {
                segment_type: "code".into(),
                content: "fn f(){}".into(),
                start: 30,
                end: 60,
            },
            Segment {
                segment_type: "answer".into(),
                content: "done".into(),
                start: 110,
                end: 130,
            },
        ];
        let pairs = detect_nesting(&segments);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], (0, 1)); // plan contains code
    }

    #[test]
    fn test_detect_nesting_no_overlap() {
        let segments = vec![
            Segment {
                segment_type: "reasoning".into(),
                content: "a".into(),
                start: 0,
                end: 50,
            },
            Segment {
                segment_type: "code".into(),
                content: "b".into(),
                start: 55,
                end: 100,
            },
        ];
        let pairs = detect_nesting(&segments);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_detect_nesting_identical_spans_not_nested() {
        let segments = vec![
            Segment {
                segment_type: "plan".into(),
                content: "x".into(),
                start: 0,
                end: 10,
            },
            Segment {
                segment_type: "code".into(),
                content: "x".into(),
                start: 0,
                end: 10,
            },
        ];
        let pairs = detect_nesting(&segments);
        assert!(pairs.is_empty()); // identical span = not nested
    }
}
