//! Pure parser for Pi's `--mode json` (and `--mode rpc`) event lines.
//!
//! One JSON object per line → one [`PiEvent`]. No process or graph imports, so
//! the parser is reused unchanged when RPC mode lands. Shapes follow the
//! recorded fixture in `tests/fixtures/pi/` (see its README), not the upstream
//! docs.

use serde_json::Value;

/// One decoded Pi stdout line.
#[derive(Debug, Clone, PartialEq)]
pub enum PiEvent {
    /// First line of a run: `{"type":"session","id","cwd",...}`.
    Session {
        id: Option<String>,
        cwd: Option<String>,
    },
    /// Structural events with no user-visible payload the executor needs
    /// (`agent_start`, `agent_settled`, `turn_start`, `turn_end`, `agent_end`,
    /// `message_start`, `queue_update`, `compaction_*`, `tool_execution_update`).
    /// Carries the raw `type` string.
    Lifecycle(String),
    /// Streamed assistant text (`message_update` with `text_delta`).
    TextDelta(String),
    /// Completed assistant message (`message_end` with `role == "assistant"`).
    /// `text` is the concatenation of the `text` parts only; `thinking` and
    /// `toolCall` parts are dropped.
    AssistantMessage {
        text: String,
        stop_reason: Option<String>,
        usage: Option<Value>,
    },
    /// `tool_execution_start`.
    ToolStart {
        tool_call_id: String,
        tool_name: String,
        args: Value,
    },
    /// `tool_execution_end`. `result` is kept raw; use [`flatten_result`] for text.
    ToolEnd {
        tool_call_id: String,
        tool_name: String,
        result: Value,
        is_error: bool,
    },
    /// `error` / `extension_error`; the message, or the raw line if none was given.
    Error(String),
    /// Well-formed line the executor deliberately does not act on
    /// (non-assistant `message_end`, non-text `message_update`).
    Ignored,
    /// Well-formed line with a `type` this parser does not know (empty string
    /// when `type` is missing). Never an error: Pi may add event types.
    Unknown(String),
}

/// Parse one stdout line. Trailing `\r`/`\n` are tolerated.
///
/// Only malformed JSON is an error; unknown or unexpected shapes map to
/// [`PiEvent::Unknown`] / [`PiEvent::Ignored`] so a single surprising line
/// never aborts a run.
pub fn parse_line(line: &str) -> Result<PiEvent, serde_json::Error> {
    let v: Value = serde_json::from_str(line.trim_end_matches(['\r', '\n']))?;
    let ty = v.get("type").and_then(Value::as_str).unwrap_or("");
    Ok(match ty {
        "session" => PiEvent::Session {
            id: str_opt(&v, "id"),
            cwd: str_opt(&v, "cwd"),
        },
        "agent_start"
        | "agent_end"
        | "agent_settled"
        | "turn_start"
        | "turn_end"
        | "message_start"
        | "queue_update"
        | "compaction_start"
        | "compaction_end"
        | "tool_execution_update" => PiEvent::Lifecycle(ty.to_string()),
        "message_update" => match v
            .pointer("/assistantMessageEvent/type")
            .and_then(Value::as_str)
        {
            Some("text_delta") => PiEvent::TextDelta(
                v.pointer("/assistantMessageEvent/delta")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            ),
            _ => PiEvent::Ignored,
        },
        "message_end" => {
            let Some(msg) = v.get("message") else {
                return Ok(PiEvent::Ignored);
            };
            if msg.get("role").and_then(Value::as_str) != Some("assistant") {
                return Ok(PiEvent::Ignored);
            }
            PiEvent::AssistantMessage {
                text: flatten_content(msg.get("content")),
                stop_reason: str_opt(msg, "stopReason"),
                usage: msg.get("usage").cloned(),
            }
        }
        "tool_execution_start" => PiEvent::ToolStart {
            tool_call_id: str_field(&v, "toolCallId"),
            tool_name: str_field(&v, "toolName"),
            args: v.get("args").cloned().unwrap_or(Value::Null),
        },
        "tool_execution_end" => PiEvent::ToolEnd {
            tool_call_id: str_field(&v, "toolCallId"),
            tool_name: str_field(&v, "toolName"),
            result: v.get("result").cloned().unwrap_or(Value::Null),
            is_error: v.get("isError").and_then(Value::as_bool).unwrap_or(false),
        },
        "error" | "extension_error" => PiEvent::Error(
            v.get("message")
                .or_else(|| v.get("error"))
                .map(|e| match e {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
                .unwrap_or_else(|| line.to_string()),
        ),
        other => PiEvent::Unknown(other.to_string()),
    })
}

/// Flatten a Pi `content` field to plain text.
///
/// `content` is either a string or an array of typed parts. Only parts with
/// `type == "text"` contribute (newline-joined); `thinking` and `toolCall`
/// parts are dropped. Anything else yields `""`.
pub fn flatten_content(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter(|p| p.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Flatten a tool `result` to plain text.
///
/// The usual shape is `{"content":[{"type":"text","text"}],"details"?}` and
/// is flattened like [`flatten_content`]. A bare string is returned as-is,
/// `null` becomes `""`, and any other shape falls back to its compact JSON so
/// no information is silently lost.
pub fn flatten_result(result: &Value) -> String {
    match result {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Object(obj)
            if obj
                .get("content")
                .is_some_and(|c| c.is_array() || c.is_string()) =>
        {
            flatten_content(obj.get("content"))
        }
        other => other.to_string(),
    }
}

fn str_field(v: &Value, k: &str) -> String {
    str_opt(v, k).unwrap_or_default()
}

fn str_opt(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(Value::as_str).map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> Vec<String> {
        std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/pi/hello-run.jsonl"
        ))
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect()
    }

    fn fixture_events() -> Vec<PiEvent> {
        fixture()
            .iter()
            .map(|l| parse_line(l).expect("fixture line parses"))
            .collect()
    }

    #[test]
    fn parses_every_fixture_line_without_error() {
        for line in fixture() {
            assert!(parse_line(&line).is_ok(), "line failed: {line}");
        }
    }

    #[test]
    fn tool_execution_events_carry_ids_names_and_payloads() {
        let events = fixture_events();
        let (write_id, write_args) = events
            .iter()
            .find_map(|e| match e {
                PiEvent::ToolStart {
                    tool_call_id,
                    tool_name,
                    args,
                } if tool_name == "write" => Some((tool_call_id.clone(), args.clone())),
                _ => None,
            })
            .expect("fixture has a write ToolStart");
        assert!(
            write_args.get("path").is_some(),
            "write args carry a path: {write_args}"
        );

        let end = events
            .iter()
            .find(
                |e| matches!(e, PiEvent::ToolEnd { tool_call_id, .. } if *tool_call_id == write_id),
            )
            .expect("matching write ToolEnd");
        match end {
            PiEvent::ToolEnd {
                tool_name,
                is_error,
                result,
                ..
            } => {
                assert_eq!(tool_name, "write");
                assert!(!is_error);
                assert!(flatten_result(result).contains("Successfully wrote"));
            }
            other => panic!("expected ToolEnd, got {other:?}"),
        }
    }

    #[test]
    fn bash_error_result_is_flagged() {
        let events = fixture_events();
        let failed = events
            .iter()
            .filter_map(|e| match e {
                PiEvent::ToolEnd {
                    tool_name,
                    is_error: true,
                    result,
                    ..
                } if tool_name == "bash" => Some(flatten_result(result)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(failed.len(), 1, "exactly one failing bash call: {failed:?}");
        assert!(failed[0].contains("No such file"), "{}", failed[0]);
    }

    #[test]
    fn message_end_assistant_flattens_text_parts() {
        let events = fixture_events();
        let text = events
            .iter()
            .rev()
            .find_map(|e| match e {
                PiEvent::AssistantMessage { text, .. } => Some(text.clone()),
                _ => None,
            })
            .expect("fixture has an assistant message");
        assert!(text.to_lowercase().contains("hi"), "{text}");
        assert!(
            !text.contains("The user wants me"),
            "thinking leaked: {text}"
        );
        assert!(!text.contains("timing issue"), "thinking leaked: {text}");
    }

    #[test]
    fn assistant_message_carries_stop_reason_and_usage() {
        let events = fixture_events();
        let last = events
            .iter()
            .rev()
            .find(|e| matches!(e, PiEvent::AssistantMessage { .. }))
            .expect("fixture has an assistant message");
        match last {
            PiEvent::AssistantMessage {
                stop_reason, usage, ..
            } => {
                assert_eq!(stop_reason.as_deref(), Some("stop"));
                assert!(usage.is_some());
                assert!(usage.as_ref().and_then(|u| u.get("totalTokens")).is_some());
            }
            other => panic!("expected AssistantMessage, got {other:?}"),
        }
    }

    #[test]
    fn user_and_tool_result_message_end_are_skipped() {
        let user = r#"{"type":"message_end","message":{"role":"user","content":[{"type":"text","text":"hello"}]}}"#;
        assert_eq!(parse_line(user).unwrap(), PiEvent::Ignored);
        let tool = r#"{"type":"message_end","message":{"role":"toolResult","toolCallId":"c1","toolName":"bash","content":[{"type":"text","text":"hi"}],"isError":false}}"#;
        assert_eq!(parse_line(tool).unwrap(), PiEvent::Ignored);

        // And the fixture never produces an AssistantMessage from those roles.
        let assistant_count = fixture_events()
            .iter()
            .filter(|e| matches!(e, PiEvent::AssistantMessage { .. }))
            .count();
        assert_eq!(
            assistant_count, 4,
            "fixture has 4 assistant message_end lines"
        );
    }

    #[test]
    fn unknown_type_is_unknown_not_error() {
        assert!(matches!(
            parse_line(r#"{"type":"banana"}"#).unwrap(),
            PiEvent::Unknown(t) if t == "banana"
        ));
    }

    #[test]
    fn missing_type_is_unknown_empty() {
        assert!(matches!(
            parse_line(r#"{"foo":1}"#).unwrap(),
            PiEvent::Unknown(t) if t.is_empty()
        ));
    }

    #[test]
    fn malformed_json_is_error() {
        assert!(parse_line("{not json").is_err());
    }

    #[test]
    fn crlf_is_tolerated() {
        assert!(matches!(
            parse_line("{\"type\":\"agent_start\"}\r").unwrap(),
            PiEvent::Lifecycle(_)
        ));
        assert!(matches!(
            parse_line("{\"type\":\"agent_settled\"}\r\n").unwrap(),
            PiEvent::Lifecycle(t) if t == "agent_settled"
        ));
    }

    #[test]
    fn text_delta_extracted_from_message_update() {
        let l = r#"{"type":"message_update","usage":{},"assistantMessageEvent":{"type":"text_delta","contentIndex":0,"delta":"He"}}"#;
        assert!(matches!(parse_line(l).unwrap(), PiEvent::TextDelta(d) if d == "He"));
    }

    #[test]
    fn fixture_text_deltas_concatenate_to_final_answer() {
        let events = fixture_events();
        // Deltas after the last assistant message *start* belong to the final answer.
        let mut streamed = String::new();
        let mut final_text = None;
        for e in &events {
            match e {
                PiEvent::Lifecycle(t) if t == "message_start" => streamed.clear(),
                PiEvent::TextDelta(d) => streamed.push_str(d),
                PiEvent::AssistantMessage { text, .. } => final_text = Some(text.clone()),
                _ => {}
            }
        }
        assert_eq!(final_text.as_deref(), Some(streamed.as_str()));
    }

    #[test]
    fn thinking_delta_is_ignored() {
        let l = r#"{"type":"message_update","usage":{},"assistantMessageEvent":{"type":"thinking_delta","contentIndex":0,"delta":"hmm"}}"#;
        assert_eq!(parse_line(l).unwrap(), PiEvent::Ignored);
        let l = r#"{"type":"message_update","usage":{},"assistantMessageEvent":{"type":"toolcall_delta","contentIndex":1,"delta":"{\"pa"}}"#;
        assert_eq!(parse_line(l).unwrap(), PiEvent::Ignored);
    }

    #[test]
    fn session_event_carries_id_and_cwd() {
        let first = fixture().into_iter().next().unwrap();
        match parse_line(&first).unwrap() {
            PiEvent::Session { id, cwd } => {
                assert!(id.is_some());
                assert_eq!(cwd.as_deref(), Some("/workspace"));
            }
            other => panic!("expected Session, got {other:?}"),
        }
    }

    #[test]
    fn error_events_parse() {
        match parse_line(r#"{"type":"error","message":"boom"}"#).unwrap() {
            PiEvent::Error(s) => assert!(s.contains("boom")),
            other => panic!("expected Error, got {other:?}"),
        }
        match parse_line(r#"{"type":"extension_error","error":"x"}"#).unwrap() {
            PiEvent::Error(s) => assert_eq!(s, "x"),
            other => panic!("expected Error, got {other:?}"),
        }
        // Structured error payloads are stringified rather than dropped.
        match parse_line(r#"{"type":"error","error":{"code":7}}"#).unwrap() {
            PiEvent::Error(s) => assert!(s.contains("\"code\":7"), "{s}"),
            other => panic!("expected Error, got {other:?}"),
        }
        // No payload at all → the raw line so nothing is lost.
        match parse_line(r#"{"type":"error"}"#).unwrap() {
            PiEvent::Error(s) => assert!(s.contains("\"type\":\"error\""), "{s}"),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[test]
    fn flatten_content_handles_string_array_and_other() {
        assert_eq!(flatten_content(Some(&json!("plain"))), "plain");
        let parts = json!([
            {"type":"thinking","thinking":"secret"},
            {"type":"text","text":"a"},
            {"type":"toolCall","id":"c","name":"bash","arguments":{}},
            {"type":"text","text":"b"}
        ]);
        assert_eq!(flatten_content(Some(&parts)), "a\nb");
        assert_eq!(flatten_content(None), "");
        assert_eq!(flatten_content(Some(&json!(42))), "");
    }

    #[test]
    fn flatten_result_falls_back_to_compact_json() {
        assert_eq!(
            flatten_result(&json!({"content":[{"type":"text","text":"ok"}],"details":{}})),
            "ok"
        );
        assert_eq!(flatten_result(&json!({"foo":"bar"})), r#"{"foo":"bar"}"#);
        assert_eq!(flatten_result(&json!("raw")), "raw");
        assert_eq!(flatten_result(&Value::Null), "");
    }
}
