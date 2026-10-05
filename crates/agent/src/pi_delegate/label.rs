//! Blind labeling of final Pi replies.
//!
//! The prompt shows the piece text and does not show the baseline kind, unless
//! `show_baseline` is set for a later comparison pass.

use std::collections::HashSet;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::events::{PiEvent, parse_line};
use super::pieces::{
    PIECE_BASELINE_VERSION, PIECE_PARSER_VERSION, Piece, PieceKind, structure_segment,
};

/// One final assistant segment inside a Pi recording.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinalSegment {
    pub source: PathBuf,
    /// Index among final replies in `source`, starting at 0. Narration is not counted.
    pub segment_index: usize,
    pub text: String,
}

/// One human label for one piece, keyed by its byte range in the segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabeledPiece {
    pub start: usize,
    pub end: usize,
    pub heading: Option<String>,
    pub kind: PieceKind,
}

/// Labels for one final segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabeledReply {
    pub source: PathBuf,
    pub segment_index: usize,
    pub text_sha256: String,
    pub pieces: Vec<LabeledPiece>,
}

/// SHA-256 of the segment text, hex encoded. Stable across machines.
pub fn segment_text_hash(text: &str) -> String {
    let digest = Sha256::digest(text.as_bytes());
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        hex.push(char::from(b"0123456789abcdef"[usize::from(byte >> 4)]));
        hex.push(char::from(b"0123456789abcdef"[usize::from(byte & 0xf)]));
    }
    hex
}

fn source_key(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Final assistant texts from a Pi `--mode json` recording.
///
/// Narration (`stopReason: toolUse`) is skipped. Thinking parts are already
/// dropped by the event parser.
pub fn final_reply_texts(jsonl: &str) -> Vec<String> {
    let mut replies = Vec::new();
    for line in jsonl.lines() {
        let Ok(event) = parse_line(line) else {
            continue;
        };
        if let PiEvent::AssistantMessage {
            text, stop_reason, ..
        } = event
        {
            if stop_reason.as_deref() == Some("toolUse") || text.trim().is_empty() {
                continue;
            }
            replies.push(text);
        }
    }
    replies
}

/// Load final segments from a JSONL file or a directory of `*.jsonl` files.
///
/// `segment_index` counts final replies inside each file. `piece-labels.jsonl`
/// is not a recording.
pub fn load_final_segments(path: &Path) -> std::io::Result<Vec<FinalSegment>> {
    let mut files = Vec::new();
    if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let file = entry.path();
            if file.extension().and_then(|ext| ext.to_str()) != Some("jsonl") {
                continue;
            }
            if file.file_name().and_then(|name| name.to_str()) == Some("piece-labels.jsonl") {
                continue;
            }
            files.push(file);
        }
        files.sort();
    } else {
        files.push(path.to_path_buf());
    }
    let mut segments = Vec::new();
    for file in files {
        let raw = std::fs::read_to_string(&file)?;
        let source = source_key(&file);
        for (segment_index, text) in final_reply_texts(&raw).into_iter().enumerate() {
            segments.push(FinalSegment {
                source: source.clone(),
                segment_index,
                text,
            });
        }
    }
    Ok(segments)
}

/// Segments whose `(source, segment_index, text hash)` is not already in `labels`.
///
/// A later run of the labeler skips those, so `quit` partway through a sitting
/// does not ask for them again.
pub fn segments_still_to_label(segments: &[FinalSegment], labels: &str) -> Vec<FinalSegment> {
    let done = labeled_keys(labels);
    segments
        .iter()
        .filter(|segment| !done.contains(&segment_key(segment)))
        .cloned()
        .collect()
}

fn segment_key(segment: &FinalSegment) -> (String, usize, String) {
    (
        segment.source.display().to_string(),
        segment.segment_index,
        segment_text_hash(&segment.text),
    )
}

fn labeled_keys(labels: &str) -> HashSet<(String, usize, String)> {
    let mut keys = HashSet::new();
    for line in labels.lines() {
        let Some(label) = parse_stored_label(line) else {
            continue;
        };
        keys.insert((label.source, label.segment_index, label.text_sha256));
    }
    keys
}

/// One label line. Identity is the run file, the segment index, the text hash,
/// and each piece's byte range. Piece order is not stored.
pub fn label_line(reply: &LabeledReply) -> String {
    let pieces: Vec<serde_json::Value> = reply
        .pieces
        .iter()
        .map(|piece| {
            serde_json::json!({
                "start": piece.start,
                "end": piece.end,
                "heading": piece.heading,
                "kind": piece.kind.as_label(),
            })
        })
        .collect();
    serde_json::json!({
        "parser_version": PIECE_PARSER_VERSION,
        "baseline_version": PIECE_BASELINE_VERSION,
        "source": reply.source.display().to_string(),
        "segment_index": reply.segment_index,
        "text_sha256": reply.text_sha256,
        "pieces": pieces,
    })
    .to_string()
}

/// A label row read back from `piece-labels.jsonl`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredLabel {
    pub parser_version: String,
    pub baseline_version: String,
    pub source: String,
    pub segment_index: usize,
    pub text_sha256: String,
    pub pieces: Vec<LabeledPiece>,
}

pub fn parse_stored_label(line: &str) -> Option<StoredLabel> {
    let value: serde_json::Value = serde_json::from_str(line).ok()?;
    let pieces = value.get("pieces")?.as_array()?;
    let mut stored = Vec::new();
    for piece in pieces {
        let kind = PieceKind::from_label(piece.get("kind")?.as_str()?)?;
        let heading = match piece.get("heading") {
            None | Some(serde_json::Value::Null) => None,
            Some(value) => Some(value.as_str()?.to_string()),
        };
        stored.push(LabeledPiece {
            start: piece.get("start")?.as_u64()? as usize,
            end: piece.get("end")?.as_u64()? as usize,
            heading,
            kind,
        });
    }
    Some(StoredLabel {
        parser_version: value.get("parser_version")?.as_str()?.to_string(),
        baseline_version: value.get("baseline_version")?.as_str()?.to_string(),
        source: value.get("source")?.as_str()?.to_string(),
        segment_index: value.get("segment_index")?.as_u64()? as usize,
        text_sha256: value.get("text_sha256")?.as_str()?.to_string(),
        pieces: stored,
    })
}

/// The text a person sees for one piece. No kind, including the baseline kind.
pub fn format_piece_blind(text: &str, piece: &Piece) -> String {
    let mut out = format!("[{}]\n", piece.order.max(1));
    match &piece.heading {
        Some(heading) => out.push_str(&format!("heading: {heading}\n")),
        None => out.push_str("heading: none\n"),
    }
    if piece.items.is_empty() {
        out.push_str(text.get(piece.start..piece.end).unwrap_or(""));
        if !out.ends_with('\n') {
            out.push('\n');
        }
    } else {
        for item in &piece.items {
            out.push_str(&format!("  {}. {}\n", item.position, item.text));
        }
    }
    out
}

/// Ask for a kind on every piece. `quit` stops and keeps replies already finished.
///
/// A reply that is abandoned mid-way is not included. `show_baseline` adds the
/// baseline kind under the piece; the default labeling pass leaves it off.
pub fn run_label_session<R: BufRead, W: Write>(
    segments: &[FinalSegment],
    show_baseline: bool,
    input: &mut R,
    output: &mut W,
) -> std::io::Result<Vec<LabeledReply>> {
    let mut labeled = Vec::new();
    for (index, segment) in segments.iter().enumerate() {
        let structured = structure_segment(&segment.text, false);
        writeln!(
            output,
            "reply {} of {} ({} pieces)",
            index + 1,
            segments.len(),
            structured.pieces.len()
        )?;
        if !structured.parsed {
            writeln!(
                output,
                "over the character cap; one block, parser was not run"
            )?;
        } else if !structured.tiled {
            writeln!(
                output,
                "coverage failed; one block covering the whole reply"
            )?;
        }
        let mut pieces = Vec::new();
        let mut quit = false;
        for piece in &structured.pieces {
            write!(output, "{}", format_piece_blind(&segment.text, piece))?;
            if show_baseline {
                writeln!(output, "baseline: {}", piece.kind.as_label())?;
            }
            writeln!(output, "kind?")?;
            output.flush()?;
            loop {
                let mut line = String::new();
                if input.read_line(&mut line)? == 0 || line.trim().eq_ignore_ascii_case("quit") {
                    quit = true;
                    break;
                }
                if let Some(kind) = PieceKind::from_label(&line) {
                    pieces.push(LabeledPiece {
                        start: piece.start,
                        end: piece.end,
                        heading: piece.heading.clone(),
                        kind,
                    });
                    break;
                }
                writeln!(
                    output,
                    "use statement, options, steps, instructions, example, caveat, code, or question"
                )?;
                writeln!(output, "kind?")?;
                output.flush()?;
            }
            if quit {
                break;
            }
        }
        if quit {
            break;
        }
        labeled.push(LabeledReply {
            source: segment.source.clone(),
            segment_index: segment.segment_index,
            text_sha256: segment_text_hash(&segment.text),
            pieces,
        });
    }
    Ok(labeled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_replies_skip_narration_and_thinking() {
        let jsonl = concat!(
            r#"{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"Let me check"}],"stopReason":"toolUse"}}"#,
            "\n",
            r#"{"type":"message_end","message":{"role":"assistant","content":[{"type":"thinking","thinking":"hmm"},{"type":"text","text":"Done."}],"stopReason":"stop"}}"#,
            "\n",
        );
        assert_eq!(final_reply_texts(jsonl), vec!["Done.".to_string()]);
    }

    #[test]
    fn blind_prompt_does_not_show_the_baseline_kind() {
        let text = "To fix it:\n1. Convert exp.\n2. Re-run the tests.\n";
        let segment = structure_segment(text, false);
        assert_eq!(segment.pieces[0].kind, PieceKind::Steps);
        let shown = format_piece_blind(text, &segment.pieces[0]);
        assert!(!shown.to_ascii_lowercase().contains("steps"));
        assert!(!shown.to_ascii_lowercase().contains("baseline"));
        assert!(shown.contains("Convert"));
        assert!(shown.contains("heading: To fix it"));
    }

    fn segment(index: usize, text: &str) -> FinalSegment {
        FinalSegment {
            source: PathBuf::from("/runs/one.jsonl"),
            segment_index: index,
            text: text.to_string(),
        }
    }

    #[test]
    fn a_blind_session_records_ranges_and_the_text_hash() {
        let replies = [segment(3, "The server is up.\n\nWant me to restart it?")];
        let script = "statement\nquestion\n";
        let mut input = std::io::Cursor::new(script);
        let mut output = Vec::new();
        let labeled = run_label_session(&replies, false, &mut input, &mut output).expect("session");
        let shown = String::from_utf8(output).unwrap();
        assert!(!shown.contains("baseline:"));
        assert_eq!(labeled.len(), 1);
        assert_eq!(labeled[0].segment_index, 3);
        assert_eq!(labeled[0].text_sha256, segment_text_hash(&replies[0].text));
        assert_eq!(
            labeled[0].pieces.iter().map(|p| p.kind).collect::<Vec<_>>(),
            vec![PieceKind::Statement, PieceKind::Question]
        );
        let line = label_line(&labeled[0]);
        assert!(line.contains("\"parser_version\":\"1\""));
        assert!(line.contains("\"baseline_version\":\"1\""));
        assert!(line.contains("\"segment_index\":3"));
        assert!(!line.contains("reply_index"));
        assert!(!line.contains("\"order\""));
        let stored = parse_stored_label(&line).expect("round trip");
        assert_eq!(stored.pieces[0].start, labeled[0].pieces[0].start);
        assert_eq!(stored.pieces[0].end, labeled[0].pieces[0].end);
        assert_eq!(stored.pieces[1].kind, PieceKind::Question);
    }

    #[test]
    fn quit_keeps_only_finished_replies() {
        let replies = [segment(0, "One sentence."), segment(1, "Another sentence.")];
        let script = "statement\nquit\n";
        let mut input = std::io::Cursor::new(script);
        let mut output = Vec::new();
        let labeled = run_label_session(&replies, false, &mut input, &mut output).expect("session");
        assert_eq!(labeled.len(), 1);
        assert_eq!(labeled[0].segment_index, 0);
    }

    #[test]
    fn a_second_sitting_skips_segments_already_labeled() {
        let replies = [segment(0, "One sentence."), segment(1, "Another sentence.")];
        let first = [replies[0].clone()];
        let mut input = std::io::Cursor::new("statement\n");
        let mut output = Vec::new();
        let labeled = run_label_session(&first, false, &mut input, &mut output).expect("first");
        let file = label_line(&labeled[0]);
        let pending = segments_still_to_label(&replies, &file);
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].segment_index, 1);
    }
}
