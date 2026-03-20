//! Tier-2 on-demand repo briefing tool — detailed structural + memory onboarding.

use async_trait::async_trait;
use graphirm_graph::GraphStore;
use serde_json::{Value, json};
use tokio::process::Command;

use crate::{Tool, ToolContext, ToolError, ToolOutput};

pub struct RepoBriefingTool;

impl RepoBriefingTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for RepoBriefingTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for RepoBriefingTool {
    fn name(&self) -> &str {
        "repo_briefing"
    }

    fn description(&self) -> &str {
        "Generate a detailed repository briefing covering file structure, \
         recent knowledge from prior sessions, and git activity. Use at the \
         start of a task to orient yourself to the codebase."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "enum": ["all", "files", "knowledge", "git"],
                    "description": "Which section to include. Defaults to 'all'."
                }
            },
            "required": []
        })
    }

    async fn execute(&self, params: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let section = params
            .get("section")
            .and_then(|v| v.as_str())
            .filter(|s| matches!(*s, "all" | "files" | "knowledge" | "git"))
            .unwrap_or("all");

        let root = ctx.working_dir.as_path();
        let mut parts: Vec<String> = Vec::new();

        if section == "all" || section == "files" {
            let files_section = build_files_section(root).await;
            parts.push(files_section);
        }

        if section == "all" || section == "knowledge" {
            let ks = build_knowledge_section(ctx.graph.as_ref());
            parts.push(ks);
        }

        if section == "all" || section == "git" {
            let gs = build_git_section(root).await;
            parts.push(gs);
        }

        Ok(ToolOutput::success(parts.join("\n\n")))
    }
}

// ── Files section ─────────────────────────────────────────────────────────────

/// Run `rg --files <root>` and produce a summary: total file count + top directories.
async fn build_files_section(root: &std::path::Path) -> String {
    let output = Command::new("rg")
        .args(["--files", "--no-messages", root.to_str().unwrap_or(".")])
        .output()
        .await;

    let (total, top_dirs) = match output {
        Ok(out) if out.status.success() || !out.stdout.is_empty() => {
            let text = String::from_utf8_lossy(&out.stdout);
            let files: Vec<&str> = text.lines().collect();
            let total = files.len();

            // Count files per top-level directory (relative to root)
            let mut dir_counts: std::collections::HashMap<String, usize> =
                std::collections::HashMap::new();
            let root_str = root.to_str().unwrap_or(".");
            for file in &files {
                // Strip root prefix, get first path component
                let relative = file
                    .strip_prefix(root_str)
                    .unwrap_or(file)
                    .trim_start_matches('/');
                let dir = relative.split('/').next().unwrap_or("(root)");
                *dir_counts.entry(dir.to_string()).or_insert(0) += 1;
            }

            let mut pairs: Vec<(String, usize)> = dir_counts.into_iter().collect();
            pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            pairs.truncate(8);
            let dir_list = pairs
                .iter()
                .map(|(d, c)| format!("  {}: {}", d, c))
                .collect::<Vec<_>>()
                .join("\n");
            (total, dir_list)
        }
        _ => (0, String::new()),
    };

    if total == 0 {
        return "**Files:** (no files found or rg not available)".to_string();
    }

    format!("## Files ({} total)\nTop directories:\n{}", total, top_dirs)
}

// ── Knowledge section (stub for Task 7) ──────────────────────────────────────

fn build_knowledge_section(_store: &GraphStore) -> String {
    "**Knowledge:** (coming in Task 7)".to_string()
}

// ── Git section (stub for Task 7) ────────────────────────────────────────────

async fn build_git_section(_root: &std::path::Path) -> String {
    "**Git:** (coming in Task 7)".to_string()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::make_test_context;

    #[test]
    fn tool_name_and_description() {
        let tool = RepoBriefingTool::new();
        assert_eq!(tool.name(), "repo_briefing");
        assert!(!tool.description().is_empty());
    }

    #[test]
    fn parameters_has_section_enum() {
        let tool = RepoBriefingTool::new();
        let params = tool.parameters();
        let section_enum = &params["properties"]["section"]["enum"];
        assert!(section_enum.is_array());
        let arr = section_enum.as_array().unwrap();
        assert!(arr.iter().any(|v| v.as_str() == Some("files")));
        assert!(arr.iter().any(|v| v.as_str() == Some("knowledge")));
        assert!(arr.iter().any(|v| v.as_str() == Some("git")));
        assert!(arr.iter().any(|v| v.as_str() == Some("all")));
    }

    #[tokio::test]
    async fn execute_files_section_returns_string() {
        let ctx = make_test_context();
        let tool = RepoBriefingTool::new();
        let result = tool
            .execute(serde_json::json!({"section": "files"}), &ctx)
            .await
            .expect("should not error");
        // Either "Files:" header or fallback message — just verify it's non-empty text
        assert!(!result.content.is_empty());
    }
}
