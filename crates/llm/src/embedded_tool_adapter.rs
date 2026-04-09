//! When upstream models emit tool calls inside **assistant text** (e.g. DeepSeek-style DSML
//! `<invoke name="...">` blocks) instead of OpenAI `delta.tool_calls`, the SSE layer only
//! produces [`ContentPart::Text`]. This module parses those embedded blocks and converts them
//! into [`ContentPart::ToolCall`] so the agent loop can execute tools.

use crate::provider::{ContentPart, LlmResponse, StopReason};

/// If `response` has no native tool calls but the concatenated text contains embedded
/// `invoke name="tool"` + `parameter name="key">value` blocks, split them into real
/// [`ContentPart::ToolCall`] parts and optional leading prose text.
pub fn augment_embedded_tool_calls(mut response: LlmResponse) -> LlmResponse {
    if response.has_tool_calls() {
        return response;
    }
    let text = response.text_content();
    if !text.contains("invoke name=\"") {
        return response;
    }
    let extracted = extract_invokes(&text);
    if extracted.is_empty() {
        return response;
    }

    let preamble = strip_embedded_tool_markup(&text);
    let mut new_parts: Vec<ContentPart> = Vec::new();
    if !preamble.is_empty() {
        new_parts.push(ContentPart::text(preamble));
    }
    for (i, (name, args)) in extracted.into_iter().enumerate() {
        let id = format!("call_embed_{}", i + 1);
        new_parts.push(ContentPart::tool_call(id, name, args));
    }

    response.content = new_parts;
    response.stop_reason = StopReason::ToolUse;
    tracing::info!(
        count = response.tool_calls().len(),
        "Augmented LlmResponse with embedded text tool calls (DSML-style)"
    );
    response
}

/// Parse `invoke name="foo"> ... parameter name="bar">val` sequences from full assistant text.
fn extract_invokes(text: &str) -> Vec<(String, serde_json::Value)> {
    let mut out = Vec::new();
    let mut s = text;
    while let Some(idx) = s.find("invoke name=\"") {
        s = &s[idx + 13..];
        let Some(end_name) = s.find('"') else {
            break;
        };
        let name = s[..end_name].to_string();
        s = &s[end_name + 1..];
        let next_invoke = s.find("invoke name=\"");
        let block_len = next_invoke.unwrap_or(s.len());
        let block = &s[..block_len];
        let args = parse_parameters_in_block(block);
        out.push((name, args));
        s = &s[block_len..];
    }
    out
}

fn parse_parameters_in_block(block: &str) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    let mut s = block;
    while let Some(idx) = s.find("parameter name=\"") {
        s = &s[idx + 16..];
        let Some(ke) = s.find('"') else {
            break;
        };
        let key = &s[..ke];
        s = &s[ke + 1..];
        let Some(gt) = s.find('>') else {
            break;
        };
        s = &s[gt + 1..];
        let ve = s
            .find("</")
            .or_else(|| s.find("<｜DSML｜"))
            .or_else(|| s.find("<|DSML|"))
            .unwrap_or(s.len());
        let val = s[..ve].trim();
        map.insert(key.to_string(), serde_json::json!(val));
        s = &s[ve..];
    }
    serde_json::Value::Object(map)
}

/// Remove embedded tool markup so the recorded assistant message is not duplicate XML.
fn strip_embedded_tool_markup(text: &str) -> String {
    let Some(marker) = text.find("function_calls") else {
        return text.to_string();
    };
    let cut_start = text[..marker].rfind('<').unwrap_or(marker);
    text[..cut_start].trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::TokenUsage;

    #[test]
    fn augment_extracts_read_and_strips_markup() {
        let raw = r#"I'll read the file.

<｜DSML｜function_calls>
<｜DSML｜invoke name="read">
<｜DSML｜parameter name="path" string="true">crates/graph/src/graph.rs</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let response = LlmResponse {
            content: vec![ContentPart::text(raw)],
            usage: TokenUsage::new(0, 0),
            stop_reason: StopReason::EndTurn,
        };
        let out = augment_embedded_tool_calls(response);
        assert!(out.has_tool_calls());
        assert_eq!(out.tool_calls().len(), 1);
        match &out.content[0] {
            ContentPart::Text { text } => assert!(text.contains("I'll read")),
            _ => panic!("expected preamble text"),
        }
        match &out.content[1] {
            ContentPart::ToolCall { name, arguments, .. } => {
                assert_eq!(name, "read");
                assert_eq!(arguments["path"], "crates/graph/src/graph.rs");
            }
            _ => panic!("expected ToolCall"),
        }
    }

    #[test]
    fn augment_two_invokes() {
        let raw = r#"Ok.

<｜DSML｜function_calls>
<｜DSML｜invoke name="graph_query">
<｜DSML｜parameter name="query_type" string="true">keyword_search</｜DSML｜parameter>
<｜DSML｜parameter name="keyword" string="true">seggy</｜DSML｜parameter>
</｜DSML｜invoke>
<｜DSML｜invoke name="read">
<｜DSML｜parameter name="path" string="true">README.md</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let response = LlmResponse {
            content: vec![ContentPart::text(raw)],
            usage: TokenUsage::new(0, 0),
            stop_reason: StopReason::EndTurn,
        };
        let out = augment_embedded_tool_calls(response);
        assert_eq!(out.tool_calls().len(), 2);
    }

    #[test]
    fn native_tool_calls_unchanged() {
        let response = LlmResponse {
            content: vec![ContentPart::tool_call(
                "c1",
                "bash",
                serde_json::json!({"command": "ls"}),
            )],
            usage: TokenUsage::new(0, 0),
            stop_reason: StopReason::ToolUse,
        };
        let out = augment_embedded_tool_calls(response);
        assert_eq!(out.content.len(), 1);
    }
}
