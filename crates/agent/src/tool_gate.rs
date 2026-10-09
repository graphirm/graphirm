//! Heuristic tool gate: omit tool definitions for short, non-technical user turns
//! so models cannot emit tool calls on greetings and small talk.

use graphirm_llm::{ContentPart, LlmMessage, Role};

/// Returns the text of the last [`Role::Human`] message in `messages` (most recent first).
pub(crate) fn last_human_message_text(messages: &[LlmMessage]) -> Option<String> {
    for msg in messages.iter().rev() {
        if msg.role != Role::Human {
            continue;
        }
        let mut out = String::new();
        for part in &msg.content {
            if let ContentPart::Text { text } = part {
                out.push_str(text);
            }
        }
        let t = out.trim();
        if !t.is_empty() {
            return Some(t.to_string());
        }
    }
    None
}

fn tokenize_alnum(s: &str) -> impl Iterator<Item = &str> {
    s.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|w| !w.is_empty())
}

/// Opening line of the one-shot verification checklist. The tool gate leaves
/// tools on for a message that starts with this line.
pub(crate) const VERIFICATION_LEAD: &str = "Before marking this task complete, verify your work:";

/// True when `text` is the verification checklist the loop injected.
pub(crate) fn is_verification_checklist(text: &str) -> bool {
    text.trim_start().starts_with(VERIFICATION_LEAD)
}

/// True when some human message in `messages` is a task, so a later short
/// "go on" still has tools.
pub(crate) fn task_is_open(messages: &[LlmMessage]) -> bool {
    messages.iter().any(|msg| {
        if msg.role != Role::Human {
            return false;
        }
        let mut out = String::new();
        for part in &msg.content {
            if let ContentPart::Text { text } = part {
                out.push_str(text);
            }
        }
        let text = out.trim();
        !text.is_empty() && !should_omit_tools_for_user_message(text)
    })
}

/// When true, the harness should not send tool definitions for this user text.
///
/// Conservative: only returns true for short messages with no code/repo/shell signals.
pub(crate) fn should_omit_tools_for_user_message(content: &str) -> bool {
    let t = content.trim();
    if t.is_empty() {
        return false;
    }
    if t.chars().count() > 200 || t.split_whitespace().count() > 24 {
        return false;
    }
    let lower = t.to_lowercase();

    if lower.contains("::")
        || lower.contains("src/")
        || lower.contains("./")
        || lower.contains("../")
        || lower.contains(".rs")
        || lower.contains(".toml")
        || lower.contains(".lock")
        || lower.contains("http://")
        || lower.contains("https://")
        || lower.contains("graph_query")
        || lower.contains('`')
        || lower.contains('{')
        || lower.contains('}')
        || lower.contains('\\')
    {
        return false;
    }

    const WORK_WORDS: &[&str] = &[
        "cargo",
        "clippy",
        "rustfmt",
        "git",
        "bash",
        "grep",
        "rg",
        "diff",
        "patch",
        "edit",
        "write",
        "read",
        "run",
        "execute",
        "shell",
        "compile",
        "error",
        "panic",
        "failed",
        "build",
        "refactor",
        "implement",
        "bug",
        "fix",
        "tests",
        "test",
        "mod",
        "struct",
        "enum",
        "trait",
        "crate",
        "fn",
        "subagent",
        "workspace",
        "cargo_check",
        "read_many",
        "repo_briefing",
        // Repo / task phrasing (avoid gating "list files", "show me the error", …)
        "list",
        "files",
        "file",
        "directory",
        "folder",
        "show",
        "search",
        "find",
        "delete",
        "create",
        "update",
        "move",
        "copy",
        "install",
        "path",
        "line",
        "stack",
        "print",
        "add",
        "remove",
        "replace",
    ];

    for w in tokenize_alnum(&lower) {
        if WORK_WORDS.contains(&w) {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omits_for_hi_and_thanks() {
        assert!(should_omit_tools_for_user_message("hi"));
        assert!(should_omit_tools_for_user_message("Hello!"));
        assert!(should_omit_tools_for_user_message("thanks"));
    }

    #[test]
    fn never_omits_for_cargo_or_path() {
        assert!(!should_omit_tools_for_user_message("run cargo test"));
        assert!(!should_omit_tools_for_user_message("see src/main.rs"));
        assert!(!should_omit_tools_for_user_message("fix the ::foo issue"));
    }

    #[test]
    fn go_on_keeps_tools_after_a_task() {
        let msgs = vec![
            LlmMessage::human("Read selection/early.txt and write the token."),
            LlmMessage::assistant("wrote it"),
            LlmMessage::human("go on"),
        ];
        assert!(should_omit_tools_for_user_message("go on"));
        assert!(task_is_open(&msgs));
    }

    #[test]
    fn go_on_alone_is_not_an_open_task() {
        let msgs = vec![LlmMessage::human("go on")];
        assert!(!task_is_open(&msgs));
    }

    #[test]
    fn verification_lead_is_exempt_even_though_it_looks_like_chat() {
        assert!(should_omit_tools_for_user_message(VERIFICATION_LEAD));
        assert!(is_verification_checklist(VERIFICATION_LEAD));
        assert!(is_verification_checklist(&format!(
            "{VERIFICATION_LEAD}\n1. Summarize what changed and stop.\n"
        )));
    }

    #[test]
    fn action_continue_keeps_tools() {
        assert!(!should_omit_tools_for_user_message(
            "Continue. Call the read, write, or edit tool for the action you just stated."
        ));
    }

    #[test]
    fn does_not_match_latest_via_test_substring() {
        assert!(should_omit_tools_for_user_message("use the latest version"));
    }

    #[test]
    fn last_human_skips_assistant() {
        let msgs = vec![
            LlmMessage::system("sys"),
            LlmMessage::human("first"),
            LlmMessage::assistant("mid"),
            LlmMessage::human("last user"),
        ];
        assert_eq!(last_human_message_text(&msgs).as_deref(), Some("last user"));
    }
}
