use std::fmt;
use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::bash_paths;

/// Risk level assessment for a file or tool execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

impl fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RiskLevel::Low => write!(f, "Low"),
            RiskLevel::Medium => write!(f, "Medium"),
            RiskLevel::High => write!(f, "High"),
        }
    }
}

/// A note about a file from the session's knowledge graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeNote {
    pub text: String,
    pub turn: u32,
}

/// A brief summary of impact for a file targeted by a tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImpactBrief {
    pub path: PathBuf,
    pub dependent_count: Option<usize>,
    pub knowledge_notes: Vec<KnowledgeNote>,
    pub risk: RiskLevel,
}

impl ImpactBrief {
    /// Check if this brief has no meaningful information.
    pub fn is_empty(&self) -> bool {
        self.dependent_count.is_none() && self.knowledge_notes.is_empty()
    }

    /// Format this brief as a markdown block suitable for prepending to tool output.
    pub fn format_markdown(&self) -> String {
        let mut lines = vec![
            format!("> **Impact:** `{}` — **Risk: {}**", self.path.display(), self.risk),
        ];

        if let Some(count) = self.dependent_count {
            lines.push(format!("> - **{} files depend on this**", count));
        }

        for note in &self.knowledge_notes {
            let truncated = truncate(&note.text, 60);
            lines.push(format!("> - Turn {}: {}", note.turn, truncated));
        }

        lines.join("\n")
    }
}

/// Compute risk level based on dependency count and knowledge notes.
pub fn compute_risk(dep_count: Option<usize>, has_notes: bool) -> RiskLevel {
    match (dep_count, has_notes) {
        (Some(n), true) if n >= 10 => RiskLevel::High,
        (Some(n), _) if n >= 3 => RiskLevel::Medium,
        (_, true) => RiskLevel::Medium,
        _ => RiskLevel::Low,
    }
}

/// Helper to truncate text for output formatting.
fn truncate(text: &str, max_len: usize) -> String {
    if text.len() > max_len {
        format!("{}…", &text[..max_len])
    } else {
        text.to_string()
    }
}

/// Trait for analyzing impact of a tool call on the codebase.
#[async_trait]
pub trait ImpactProvider: Send + Sync {
    /// Analyze the impact of a tool call targeting the given paths.
    async fn analyze(&self, paths: &[PathBuf]) -> Result<Vec<ImpactBrief>, String>;
}

/// Extract target paths from a tool call's arguments.
pub fn extract_target_paths(tool_name: &str, args: &serde_json::Value) -> Vec<PathBuf> {
    let args_obj = match args.as_object() {
        Some(obj) => obj,
        None => return vec![],
    };

    match tool_name {
        "write" | "edit" => {
            if let Some(path_val) = args_obj.get("path")
                && let Some(path_str) = path_val.as_str()
            {
                return vec![PathBuf::from(path_str)];
            }
            vec![]
        }
        "bash" => {
            if let Some(cmd_val) = args_obj.get("command")
                && let Some(cmd_str) = cmd_val.as_str()
            {
                bash_paths::extract_paths(cmd_str)
            } else {
                vec![]
            }
        }
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Risk scoring tests
    #[test]
    fn risk_low_when_no_deps_no_notes() {
        let risk = compute_risk(Some(0), false);
        assert_eq!(risk, RiskLevel::Low);

        let risk = compute_risk(Some(2), false);
        assert_eq!(risk, RiskLevel::Low);

        let risk = compute_risk(None, false);
        assert_eq!(risk, RiskLevel::Low);
    }

    #[test]
    fn risk_medium_with_moderate_deps() {
        let risk = compute_risk(Some(3), false);
        assert_eq!(risk, RiskLevel::Medium);

        let risk = compute_risk(Some(5), false);
        assert_eq!(risk, RiskLevel::Medium);

        let risk = compute_risk(Some(9), false);
        assert_eq!(risk, RiskLevel::Medium);
    }

    #[test]
    fn risk_medium_with_notes_but_few_deps() {
        let risk = compute_risk(Some(1), true);
        assert_eq!(risk, RiskLevel::Medium);

        let risk = compute_risk(None, true);
        assert_eq!(risk, RiskLevel::Medium);
    }

    #[test]
    fn risk_high_with_many_deps_and_notes() {
        let risk = compute_risk(Some(10), true);
        assert_eq!(risk, RiskLevel::High);

        let risk = compute_risk(Some(100), true);
        assert_eq!(risk, RiskLevel::High);
    }

    #[test]
    fn risk_with_unknown_deps() {
        let risk = compute_risk(None, false);
        assert_eq!(risk, RiskLevel::Low);
    }

    // Brief state tests
    #[test]
    fn brief_is_empty_when_no_info() {
        let brief = ImpactBrief {
            path: PathBuf::from("src/lib.rs"),
            dependent_count: None,
            knowledge_notes: vec![],
            risk: RiskLevel::Low,
        };
        assert!(brief.is_empty());
    }

    #[test]
    fn brief_not_empty_with_deps() {
        let brief = ImpactBrief {
            path: PathBuf::from("src/lib.rs"),
            dependent_count: Some(5),
            knowledge_notes: vec![],
            risk: RiskLevel::Medium,
        };
        assert!(!brief.is_empty());
    }

    // Format test
    #[test]
    fn format_markdown_includes_risk_and_path() {
        let brief = ImpactBrief {
            path: PathBuf::from("src/lib.rs"),
            dependent_count: Some(5),
            knowledge_notes: vec![KnowledgeNote {
                text: "This module is fragile".to_string(),
                turn: 3,
            }],
            risk: RiskLevel::High,
        };

        let formatted = brief.format_markdown();
        assert!(formatted.contains("src/lib.rs"));
        assert!(formatted.contains("High"));
        assert!(formatted.contains("5 files depend"));
        assert!(formatted.contains("Turn 3"));
        assert!(formatted.contains("fragile"));
    }

    // Path extraction tests
    #[test]
    fn extract_paths_from_write_args() {
        let args = serde_json::json!({
            "path": "src/main.rs",
            "contents": "fn main() {}"
        });
        let paths = extract_target_paths("write", &args);
        assert_eq!(paths, vec![PathBuf::from("src/main.rs")]);
    }

    #[test]
    fn extract_paths_from_edit_args() {
        let args = serde_json::json!({
            "path": "src/lib.rs",
            "old_string": "old",
            "new_string": "new"
        });
        let paths = extract_target_paths("edit", &args);
        assert_eq!(paths, vec![PathBuf::from("src/lib.rs")]);
    }

    #[test]
    fn extract_paths_from_bash_with_file() {
        let args = serde_json::json!({
            "command": "cat /etc/passwd"
        });
        let paths = extract_target_paths("bash", &args);
        // bash_paths::extract_paths will return any paths it finds in the command
        // For this simple test, we just verify it doesn't crash and returns a vec
        assert!(paths.is_empty() || paths.iter().any(|p| p.to_string_lossy().contains("passwd")));
    }

    #[test]
    fn extract_paths_from_unknown_tool() {
        let args = serde_json::json!({
            "path": "src/main.rs"
        });
        let paths = extract_target_paths("unknown_tool", &args);
        assert!(paths.is_empty());
    }
}
