//! Blind labeling of final Pi replies.
//!
//! The prompt shows the piece text and does not show the baseline kind, unless
//! `show_baseline` is set for a later comparison pass.

use std::io::{BufRead, Write};
use std::path::Path;

use super::events::{PiEvent, parse_line};
use super::pieces::{Piece, PieceKind, structure_segment};

/// One human label for one piece. Offsets are into the reply text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabeledPiece {
    pub order: u32,
    pub start: usize,
    pub end: usize,
    pub kind: PieceKind,
}

/// Labels for one final reply, in piece order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabeledReply {
    pub reply_index: usize,
    pub pieces: Vec<LabeledPiece>,
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

/// Load final replies from a JSONL file or a directory of `*.jsonl` files.
pub fn load_final_replies(path: &Path) -> std::io::Result<Vec<String>> {
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
    let mut replies = Vec::new();
    for file in files {
        let raw = std::fs::read_to_string(&file)?;
        replies.extend(final_reply_texts(&raw));
    }
    Ok(replies)
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
    replies: &[String],
    show_baseline: bool,
    input: &mut R,
    output: &mut W,
) -> std::io::Result<Vec<LabeledReply>> {
    let mut labeled = Vec::new();
    for (index, reply) in replies.iter().enumerate() {
        let segment = structure_segment(reply, false);
        writeln!(
            output,
            "reply {} of {} ({} pieces)",
            index + 1,
            replies.len(),
            segment.pieces.len()
        )?;
        if !segment.parsed {
            writeln!(
                output,
                "over the character cap; one block, parser was not run"
            )?;
        } else if !segment.tiled {
            writeln!(
                output,
                "coverage failed; one block covering the whole reply"
            )?;
        }
        let mut pieces = Vec::new();
        let mut quit = false;
        for piece in &segment.pieces {
            write!(output, "{}", format_piece_blind(reply, piece))?;
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
                        order: piece.order.max(1),
                        start: piece.start,
                        end: piece.end,
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
            reply_index: index,
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

    #[test]
    fn a_blind_session_records_the_kinds_that_were_typed() {
        let replies = ["The server is up.\n\nWant me to restart it?".to_string()];
        let script = "statement\nquestion\n";
        let mut input = std::io::Cursor::new(script);
        let mut output = Vec::new();
        let labeled = run_label_session(&replies, false, &mut input, &mut output).expect("session");
        let shown = String::from_utf8(output).unwrap();
        assert!(!shown.contains("baseline:"));
        assert_eq!(labeled.len(), 1);
        assert_eq!(
            labeled[0].pieces.iter().map(|p| p.kind).collect::<Vec<_>>(),
            vec![PieceKind::Statement, PieceKind::Question]
        );
    }

    #[test]
    fn quit_keeps_only_finished_replies() {
        let replies = ["One sentence.".to_string(), "Another sentence.".to_string()];
        let script = "statement\nquit\n";
        let mut input = std::io::Cursor::new(script);
        let mut output = Vec::new();
        let labeled = run_label_session(&replies, false, &mut input, &mut output).expect("session");
        assert_eq!(labeled.len(), 1);
        assert_eq!(labeled[0].reply_index, 0);
    }
}
