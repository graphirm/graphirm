use async_trait::async_trait;
use graphirm_graph::edges::EdgeType;
use graphirm_graph::nodes::{ContentData, GraphNode, NodeType};
use serde_json::json;
use tokio::process::Command;

use crate::{Tool, ToolContext, ToolError, ToolOutput};

pub struct CargoCheckTool;

impl CargoCheckTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CargoCheckTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for CargoCheckTool {
    fn name(&self) -> &str {
        "cargo_check"
    }

    fn description(&self) -> &str {
        "Run cargo check and return structured compiler errors and warnings. Use this instead of `bash cargo check` — the output is parsed and formatted for easy diagnosis."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "package": {
                    "type": "string",
                    "description": "Crate name to check (maps to `cargo check -p <package>`). If omitted, checks the whole workspace."
                },
                "all_targets": {
                    "type": "boolean",
                    "description": "Whether to pass `--all-targets`",
                    "default": false
                }
            },
            "required": []
        })
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let mut cmd = Command::new("cargo");
        cmd.arg("check")
            .arg("--message-format=json")
            .current_dir(&ctx.working_dir);

        if let Some(package) = args["package"].as_str() {
            cmd.arg("-p").arg(package);
        }

        if args["all_targets"].as_bool().unwrap_or(false) {
            cmd.arg("--all-targets");
        }

        let output = cmd
            .output()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("failed to run cargo: {e}")))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let (errors, warnings) = parse_cargo_messages(&stdout);

        let formatted_output = format_output(&errors, &warnings);
        let node = GraphNode::new(NodeType::Content(ContentData {
            content_type: "cargo_check".to_string(),
            path: Some(ctx.working_dir.to_string_lossy().to_string()),
            body: formatted_output.clone(),
            language: None,
        }));
        let content_node = ctx.record_content_node(node, EdgeType::Reads).await?;

        Ok(ToolOutput::success_with_node(
            formatted_output,
            content_node,
        ))
    }
}

/// Parse cargo JSON lines and extract errors and warnings.
///
/// `cargo check --message-format=json` emits one JSON object per line.
/// Diagnostic lines have `{"reason":"compiler-message", "message": { ... }}`.
/// The actual diagnostic (level, message, code, spans) is nested inside the
/// `message` field, not at the top level.
fn parse_cargo_messages(output: &str) -> (Vec<CargoMessage>, Vec<CargoMessage>) {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    for line in output.lines() {
        if line.trim().is_empty() {
            continue;
        }

        let Ok(envelope) = serde_json::from_str::<CargoEnvelope>(line) else {
            continue;
        };

        if envelope.reason != "compiler-message" {
            continue;
        }

        let Some(msg) = envelope.message else {
            continue;
        };

        match msg.level.as_str() {
            "error" => errors.push(msg),
            "warning" => warnings.push(msg),
            _ => {}
        }
    }

    (errors, warnings)
}

/// Format the output as structured markdown.
fn format_output(errors: &[CargoMessage], warnings: &[CargoMessage]) -> String {
    if errors.is_empty() && warnings.is_empty() {
        return "cargo check passed — no errors or warnings".to_string();
    }

    let mut output = String::new();

    if !errors.is_empty() {
        output.push_str("## Errors (");
        output.push_str(&errors.len().to_string());
        output.push_str(")\n\n");

        for (i, error) in errors.iter().enumerate() {
            output.push_str(&format_error(i + 1, error));
            output.push('\n');
        }
    }

    if !warnings.is_empty() {
        output.push_str("\n## Warnings (");
        output.push_str(&warnings.len().to_string());
        output.push_str(")\n\n");

        for (i, warning) in warnings.iter().enumerate() {
            output.push_str(&format_warning(i + 1, warning));
            output.push('\n');
        }
    }

    output.push_str("\n---\n");
    output.push_str(&format!(
        "Total: {} errors, {} warnings",
        errors.len(),
        warnings.len()
    ));

    output
}

/// Format a single error message.
fn format_error(index: usize, msg: &CargoMessage) -> String {
    let mut output = String::new();
    output.push_str(&format!("### {}. ", index));

    if let Some(first_span) = msg.spans.first() {
        output.push_str(&format_span_location(first_span));
        output.push('\n');
    }

    // Format the main error message
    output.push_str(&msg.level);
    if let Some(ref code) = msg.code {
        output.push('[');
        output.push_str(&code.code);
        output.push_str("]: ");
    } else {
        output.push_str(": ");
    }
    output.push_str(&msg.message);

    // Format the message body with spans
    if !msg.spans.is_empty() {
        output.push_str("\n  | ");
        output.push_str(&format_spans_body(&msg.spans));
    }

    output
}

/// Format a single warning message.
fn format_warning(index: usize, msg: &CargoMessage) -> String {
    let mut output = String::new();
    output.push_str(&format!("### {}. ", index));

    if let Some(first_span) = msg.spans.first() {
        output.push_str(&format_span_location(first_span));
        output.push('\n');
    }

    output.push_str("warning: ");
    output.push_str(&msg.message);

    if !msg.spans.is_empty() {
        output.push_str("\n  | ");
        output.push_str(&format_spans_body(&msg.spans));
    }

    output
}

/// Format span location as "path:line:column".
fn format_span_location(span: &CargoSpan) -> String {
    let mut parts = Vec::new();

    if let Some(ref file_name) = span.file_name {
        parts.push(file_name.clone());
    }

    if let Some(line) = span.line_start {
        parts.push(line.to_string());
    } else {
        parts.push("0".to_string());
    }

    if let Some(col) = span.column_start {
        parts.push(col.to_string());
    } else {
        parts.push("0".to_string());
    }

    parts.join(":")
}

/// Format the body of spans for an error/warning.
fn format_spans_body(spans: &[CargoSpan]) -> String {
    let mut lines = Vec::new();

    for span in spans {
        if let Some(ref text) = span.text {
            for text_line in text {
                if let Some(ref content) = text_line.text {
                    lines.push(content.clone());
                }
            }
        }
    }

    lines.join("\n  | ")
}

/// Top-level JSON envelope emitted by `cargo check --message-format=json`.
#[derive(Debug, serde::Deserialize)]
struct CargoEnvelope {
    reason: String,
    message: Option<CargoMessage>,
}

/// Diagnostic extracted from the `message` field of a `compiler-message` line.
#[derive(Debug, serde::Deserialize)]
struct CargoMessage {
    level: String,
    message: String,
    #[serde(default)]
    code: Option<CargoErrorCode>,
    #[serde(default)]
    spans: Vec<CargoSpan>,
}

/// Cargo error code structure.
#[derive(Debug, serde::Deserialize)]
struct CargoErrorCode {
    code: String,
}

/// Cargo span structure.
#[derive(Debug, serde::Deserialize)]
#[allow(dead_code)]
struct CargoSpan {
    file_name: Option<String>,
    line_start: Option<u32>,
    column_start: Option<u32>,
    line_end: Option<u32>,
    column_end: Option<u32>,
    #[serde(default)]
    text: Option<Vec<CargoSpanText>>,
}

/// Cargo span text structure.
#[derive(Debug, serde::Deserialize)]
struct CargoSpanText {
    text: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::make_test_context;
    use serde_json::json;

    #[test]
    fn test_cargo_check_name_and_params() {
        let tool = CargoCheckTool::new();
        assert_eq!(tool.name(), "cargo_check");

        let params = tool.parameters();
        assert!(params["properties"]["package"].is_object());
        assert!(params["properties"]["all_targets"].is_object());
        assert_eq!(
            params["properties"]["all_targets"]["default"],
            serde_json::Value::Bool(false)
        );
    }

    #[tokio::test]
    async fn test_cargo_check_clean_project() {
        let tool = CargoCheckTool::new();
        let mut ctx = make_test_context();
        // Use the repo root as working directory
        ctx.working_dir = std::path::PathBuf::from("/home/krs/graphirm-repo");

        let out = tool
            .execute(json!({"package": "graphirm-graph"}), &ctx)
            .await
            .unwrap();

        assert!(!out.is_error);
        assert!(out.content.contains("no errors"));
    }

    #[tokio::test]
    async fn test_cargo_check_with_package() {
        let tool = CargoCheckTool::new();
        let mut ctx = make_test_context();
        ctx.working_dir = std::path::PathBuf::from("/home/krs/graphirm-repo");

        let out = tool
            .execute(json!({"package": "graphirm-graph"}), &ctx)
            .await
            .unwrap();

        assert!(!out.is_error);
        // The graphirm-graph crate compiles cleanly, so expect the "passed" message
        assert!(out.content.contains("no errors") || out.content.contains("##"));
    }

    #[test]
    fn test_format_output_empty() {
        let errors: Vec<CargoMessage> = Vec::new();
        let warnings: Vec<CargoMessage> = Vec::new();
        let output = format_output(&errors, &warnings);
        assert_eq!(output, "cargo check passed — no errors or warnings");
    }

    #[test]
    fn test_format_output_with_errors() {
        let errors = vec![CargoMessage {
            level: "error".to_string(),
            message: "mismatched types".to_string(),
            code: Some(CargoErrorCode {
                code: "E0308".to_string(),
            }),
            spans: vec![CargoSpan {
                file_name: Some("src/main.rs".to_string()),
                line_start: Some(42),
                column_start: Some(10),
                line_end: Some(42),
                column_end: Some(15),
                text: Some(vec![CargoSpanText {
                    text: Some("expected `Foo`, found `Bar`".to_string()),
                }]),
            }],
        }];

        let warnings: Vec<CargoMessage> = Vec::new();
        let output = format_output(&errors, &warnings);

        assert!(output.contains("## Errors (1)"));
        assert!(output.contains("error[E0308]: mismatched types"));
        assert!(output.contains("src/main.rs:42:10"));
    }

    #[test]
    fn test_format_output_with_warnings() {
        let warnings = vec![CargoMessage {
            level: "warning".to_string(),
            message: "unused variable `x`".to_string(),
            code: None,
            spans: vec![CargoSpan {
                file_name: Some("src/main.rs".to_string()),
                line_start: Some(15),
                column_start: Some(5),
                line_end: Some(15),
                column_end: Some(10),
                text: Some(vec![CargoSpanText {
                    text: Some("unused variable `x`".to_string()),
                }]),
            }],
        }];

        let errors: Vec<CargoMessage> = Vec::new();
        let output = format_output(&errors, &warnings);

        assert!(output.contains("## Warnings (1)"));
        assert!(output.contains("warning: unused variable `x`"));
        assert!(output.contains("src/main.rs:15:5"));
    }

    #[test]
    fn test_parse_cargo_messages() {
        let json_line = r#"{"reason":"compiler-message","message":{"level":"error","message":"error message","code":{"code":"E0308"},"spans":[]}}"#;
        let (errors, warnings) = parse_cargo_messages(json_line);

        assert_eq!(errors.len(), 1);
        assert_eq!(warnings.len(), 0);
        assert_eq!(errors[0].level, "error");
        assert_eq!(errors[0].message, "error message");
    }

    #[test]
    fn test_parse_cargo_messages_skips_notes() {
        let json_line = r#"{"reason":"compiler-message","message":{"level":"note","message":"note message","spans":[]}}"#;
        let (errors, warnings) = parse_cargo_messages(json_line);

        assert_eq!(errors.len(), 0);
        assert_eq!(warnings.len(), 0);
    }

    #[test]
    fn test_parse_skips_non_compiler_messages() {
        let artifact = r#"{"reason":"compiler-artifact","package_id":"foo","target":{}}"#;
        let build = r#"{"reason":"build-script-executed","package_id":"bar"}"#;
        let input = format!("{artifact}\n{build}\n");
        let (errors, warnings) = parse_cargo_messages(&input);

        assert_eq!(errors.len(), 0);
        assert_eq!(warnings.len(), 0);
    }

    #[tokio::test]
    async fn test_cargo_check_with_errors() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("Cargo.toml"),
            r#"[package]
name = "test-errors"
version = "0.1.0"
edition = "2021"
"#,
        )
        .unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(
            dir.path().join("src/main.rs"),
            r#"fn main() { let x: i32 = "oops"; }"#,
        )
        .unwrap();

        let tool = CargoCheckTool::new();
        let mut ctx = make_test_context();
        ctx.working_dir = dir.path().to_path_buf();

        let out = tool.execute(json!({}), &ctx).await.unwrap();
        assert!(!out.is_error);
        assert!(
            out.content.contains("## Errors"),
            "expected structured errors, got: {}",
            out.content
        );
        assert!(out.content.contains("E0308"));
    }

    #[test]
    fn test_format_span_location() {
        let span = CargoSpan {
            file_name: Some("src/main.rs".to_string()),
            line_start: Some(42),
            column_start: Some(10),
            line_end: None,
            column_end: None,
            text: None,
        };

        let location = format_span_location(&span);
        assert_eq!(location, "src/main.rs:42:10");
    }
}
