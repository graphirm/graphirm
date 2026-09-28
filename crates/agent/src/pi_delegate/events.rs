//! Pure parser for Pi's `--mode json` (and `--mode rpc`) event lines.
//!
//! One JSON object per line → one [`PiEvent`]. No process or graph imports, so
//! the parser is reused unchanged when RPC mode lands. Shapes follow the
//! recorded fixture in `tests/fixtures/pi/` (see its README), not the upstream
//! docs.

use serde_json::Value;

/// Upper bound on the text carried by [`PiEvent::Error`]. Longer payloads are
/// truncated on a char boundary and suffixed with `…` so a pathological
/// provider error cannot balloon a graph node.
pub const MAX_ERROR_CHARS: usize = 4096;

/// One decoded Pi stdout line.
#[derive(Debug, Clone, PartialEq)]
pub enum PiEvent {
    /// First line of a run: `{"type":"session","id","cwd",...}`.
    Session {
        id: Option<String>,
        cwd: Option<String>,
    },
    /// Structural events with no user-visible payload the executor needs
    /// (`agent_start`, `agent_settled`, `turn_start`, `turn_end`,
    /// `message_start`, `queue_update`, `compaction_*`, `auto_retry_*`,
    /// `tool_execution_update`). Carries the raw `type` string.
    Lifecycle(String),
    /// `agent_end`. Pi retries transient provider errors (3 attempts by
    /// default) and signals that with `willRetry: true`, followed by
    /// `auto_retry_start` → `agent_start` → …. A run is only finished when
    /// `will_retry` is `false`.
    AgentEnd { will_retry: bool },
    /// Streamed assistant text (`message_update` with `text_delta`).
    /// `content_index` identifies the `text` part the delta belongs to, so a
    /// consumer can pair deltas with the parts of the final message.
    TextDelta {
        delta: String,
        content_index: Option<u64>,
    },
    /// Completed assistant message (`message_end` with `role == "assistant"`).
    /// `text` is the concatenation of the `text` parts only; `thinking` and
    /// `toolCall` parts are dropped. Provider failures arrive here too, as
    /// `stop_reason` `"error"` / `"aborted"` with `error_message` set and
    /// empty `text`.
    AssistantMessage {
        text: String,
        stop_reason: Option<String>,
        error_message: Option<String>,
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
    /// `error` / `extension_error`; the message, or the raw line if none was
    /// given. Capped at [`MAX_ERROR_CHARS`] chars.
    Error(String),
    /// Well-formed line the executor deliberately does not act on
    /// (non-assistant `message_end`, non-text `message_update`), or a blank line.
    Ignored,
    /// Well-formed line with a `type` this parser does not know (empty string
    /// when `type` is missing). Never an error: Pi may add event types.
    Unknown(String),
}

/// Parse one stdout line. Trailing `\r`/`\n` are tolerated; an empty or
/// whitespace-only line is [`PiEvent::Ignored`].
///
/// Only malformed JSON is an error; unknown or unexpected shapes map to
/// [`PiEvent::Unknown`] / [`PiEvent::Ignored`] so a single surprising line
/// never aborts a run.
pub fn parse_line(line: &str) -> Result<PiEvent, serde_json::Error> {
    let trimmed = line.trim_end_matches(['\r', '\n']);
    if trimmed.trim().is_empty() {
        return Ok(PiEvent::Ignored);
    }
    let v: Value = serde_json::from_str(trimmed)?;
    let ty = v.get("type").and_then(Value::as_str).unwrap_or("");
    Ok(match ty {
        "session" => PiEvent::Session {
            id: str_opt(&v, "id"),
            cwd: str_opt(&v, "cwd"),
        },
        "agent_start"
        | "agent_settled"
        | "turn_start"
        | "turn_end"
        | "message_start"
        | "queue_update"
        | "compaction_start"
        | "compaction_end"
        | "auto_retry_start"
        | "auto_retry_end"
        | "tool_execution_update" => PiEvent::Lifecycle(ty.to_string()),
        "agent_end" => PiEvent::AgentEnd {
            will_retry: v.get("willRetry").and_then(Value::as_bool).unwrap_or(false),
        },
        "message_update" => match v
            .pointer("/assistantMessageEvent/type")
            .and_then(Value::as_str)
        {
            Some("text_delta") => PiEvent::TextDelta {
                delta: v
                    .pointer("/assistantMessageEvent/delta")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                content_index: v
                    .pointer("/assistantMessageEvent/contentIndex")
                    .and_then(Value::as_u64),
            },
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
                error_message: str_opt(msg, "errorMessage"),
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
        "error" | "extension_error" => PiEvent::Error(truncate_chars(
            v.get("message")
                .or_else(|| v.get("error"))
                .map(|e| match e {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
                .unwrap_or_else(|| trimmed.to_string()),
            MAX_ERROR_CHARS,
        )),
        other => PiEvent::Unknown(other.to_string()),
    })
}

/// Keep at most `max` chars; when cut, end with `…` (counted within `max`).
fn truncate_chars(s: String, max: usize) -> String {
    if s.chars().count() <= max {
        return s;
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
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

/// Required identifier field (`toolCallId`, `toolName`). Strings are taken
/// as-is; other scalars (e.g. a numeric id) are stringified rather than
/// dropped; missing/null/compound values yield `""`.
fn str_field(v: &Value, k: &str) -> String {
    match v.get(k) {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Number(n)) => n.to_string(),
        Some(Value::Bool(b)) => b.to_string(),
        _ => String::new(),
    }
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
        assert!(!failed.is_empty(), "at least one failing bash call");
        assert!(
            failed.iter().any(|r| r.contains("No such file")),
            "{failed:?}"
        );
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
        assert!(
            assistant_count >= 1,
            "fixture has assistant message_end lines"
        );
    }

    #[test]
    fn failed_assistant_message_carries_error_message() {
        let l = r#"{"type":"message_end","message":{"role":"assistant","content":[],"api":"openai-completions","provider":"openrouter","model":"m","usage":{"input":0,"output":0,"totalTokens":0},"stopReason":"error","errorMessage":"429 rate limited","timestamp":1}}"#;
        match parse_line(l).unwrap() {
            PiEvent::AssistantMessage {
                text,
                stop_reason,
                error_message,
                ..
            } => {
                assert_eq!(text, "");
                assert_eq!(stop_reason.as_deref(), Some("error"));
                assert_eq!(error_message.as_deref(), Some("429 rate limited"));
            }
            other => panic!("expected AssistantMessage, got {other:?}"),
        }

        // Successful messages have no errorMessage.
        let ok = fixture_events()
            .into_iter()
            .rev()
            .find(|e| matches!(e, PiEvent::AssistantMessage { .. }))
            .unwrap();
        assert!(matches!(
            ok,
            PiEvent::AssistantMessage {
                error_message: None,
                ..
            }
        ));
    }

    #[test]
    fn agent_end_carries_will_retry() {
        assert_eq!(
            parse_line(r#"{"type":"agent_end","willRetry":true,"messages":[]}"#).unwrap(),
            PiEvent::AgentEnd { will_retry: true }
        );
        assert_eq!(
            parse_line(r#"{"type":"agent_end","willRetry":false,"messages":[]}"#).unwrap(),
            PiEvent::AgentEnd { will_retry: false }
        );
        // Fixture agent_end has no willRetry key → default false.
        let ends: Vec<_> = fixture_events()
            .into_iter()
            .filter(|e| matches!(e, PiEvent::AgentEnd { .. }))
            .collect();
        assert_eq!(ends, vec![PiEvent::AgentEnd { will_retry: false }]);
    }

    #[test]
    fn auto_retry_events_are_lifecycle() {
        assert_eq!(
            parse_line(r#"{"type":"auto_retry_start","attempt":1,"maxAttempts":3,"delayMs":2000,"errorMessage":"overloaded"}"#).unwrap(),
            PiEvent::Lifecycle("auto_retry_start".into())
        );
        assert_eq!(
            parse_line(r#"{"type":"auto_retry_end","success":true,"attempt":1}"#).unwrap(),
            PiEvent::Lifecycle("auto_retry_end".into())
        );
    }

    #[test]
    fn blank_line_is_ignored() {
        assert_eq!(parse_line("").unwrap(), PiEvent::Ignored);
        assert_eq!(parse_line("   \t").unwrap(), PiEvent::Ignored);
        assert_eq!(parse_line("\r\n").unwrap(), PiEvent::Ignored);
    }

    #[test]
    fn numeric_tool_ids_are_stringified() {
        let l = r#"{"type":"tool_execution_start","toolCallId":42,"toolName":"bash","args":{}}"#;
        match parse_line(l).unwrap() {
            PiEvent::ToolStart { tool_call_id, .. } => assert_eq!(tool_call_id, "42"),
            other => panic!("expected ToolStart, got {other:?}"),
        }
        let l = r#"{"type":"tool_execution_end","toolName":"bash","result":null}"#;
        match parse_line(l).unwrap() {
            PiEvent::ToolEnd {
                tool_call_id,
                tool_name,
                ..
            } => {
                assert_eq!(tool_call_id, "");
                assert_eq!(tool_name, "bash");
            }
            other => panic!("expected ToolEnd, got {other:?}"),
        }
    }

    #[test]
    fn error_payload_is_capped() {
        let long = "x".repeat(MAX_ERROR_CHARS * 2);
        let l = format!(r#"{{"type":"error","message":"{long}"}}"#);
        match parse_line(&l).unwrap() {
            PiEvent::Error(s) => {
                assert_eq!(s.chars().count(), MAX_ERROR_CHARS);
                assert!(s.ends_with('…'));
            }
            other => panic!("expected Error, got {other:?}"),
        }
        // Exactly at the cap → untouched.
        let exact = "y".repeat(MAX_ERROR_CHARS);
        let l = format!(r#"{{"type":"error","message":"{exact}"}}"#);
        assert_eq!(parse_line(&l).unwrap(), PiEvent::Error(exact));
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
        let l = r#"{"type":"message_update","usage":{},"assistantMessageEvent":{"type":"text_delta","contentIndex":3,"delta":"He"}}"#;
        assert_eq!(
            parse_line(l).unwrap(),
            PiEvent::TextDelta {
                delta: "He".into(),
                content_index: Some(3),
            }
        );
        // contentIndex is optional.
        let l = r#"{"type":"message_update","usage":{},"assistantMessageEvent":{"type":"text_delta","delta":"y"}}"#;
        assert_eq!(
            parse_line(l).unwrap(),
            PiEvent::TextDelta {
                delta: "y".into(),
                content_index: None,
            }
        );
    }

    #[test]
    fn fixture_text_deltas_concatenate_to_final_answer() {
        // Deltas are keyed by contentIndex: each `text` part of the final
        // message streams as its own run of deltas. Concatenating *all*
        // deltas would therefore not equal `AssistantMessage::text`, which
        // newline-joins the parts. Compare per part instead: the deltas of
        // the last text part must reproduce that part exactly.
        let lines = fixture();
        let mut per_index: std::collections::BTreeMap<Option<u64>, String> = Default::default();
        let mut last_text_part = None;
        for line in &lines {
            match parse_line(line).unwrap() {
                PiEvent::Lifecycle(t) if t == "message_start" => per_index.clear(),
                PiEvent::TextDelta {
                    delta,
                    content_index,
                } => per_index.entry(content_index).or_default().push_str(&delta),
                PiEvent::AssistantMessage { .. } => {
                    let raw: Value = serde_json::from_str(line).unwrap();
                    last_text_part = raw
                        .pointer("/message/content")
                        .and_then(Value::as_array)
                        .and_then(|parts| {
                            parts
                                .iter()
                                .rev()
                                .find(|p| p.get("type").and_then(Value::as_str) == Some("text"))
                        })
                        .and_then(|p| p.get("text"))
                        .and_then(Value::as_str)
                        .map(str::to_string);
                }
                _ => {}
            }
        }
        let streamed_last = per_index.values().next_back().cloned();
        assert!(streamed_last.is_some(), "fixture streams text deltas");
        assert_eq!(last_text_part, streamed_last);
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
