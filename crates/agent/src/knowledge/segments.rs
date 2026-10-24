//! Structured response segment parsing and graph persistence.

#[cfg(feature = "local-extraction")]
use std::collections::HashMap;
use std::sync::Arc;

use graphirm_graph::{ContentData, EdgeType, GraphEdge, GraphNode, GraphStore, NodeId, NodeType};
use serde::{Deserialize, Serialize};

use crate::error::AgentError;

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
    /// 1-based block number when the reply was split into ordered parts.
    #[serde(default)]
    pub n: Option<u32>,
    /// Section title, such as "Produce".
    #[serde(default)]
    pub title: Option<String>,
    /// Numbered lines inside the block. Empty when the segment is prose.
    #[serde(default)]
    pub items: Vec<String>,
    /// Sources for each item, same order as `items`. Each entry is a list of
    /// `file:`, `tool:<node_id>`, or `web:` citations. Empty when the item cites nothing.
    #[serde(default)]
    pub item_src: Vec<Vec<String>>,
    /// Relations for each item, same order as `items`.
    #[serde(default)]
    pub item_rel: Vec<Vec<String>>,
}

/// Wire format the LLM is asked to produce.
#[derive(Debug, Deserialize)]
struct StructuredResponse {
    segments: Vec<SegmentWire>,
}

#[derive(Debug, Deserialize)]
struct SegmentWire {
    #[serde(default)]
    n: Option<u32>,
    #[serde(rename = "type", default)]
    segment_type: String,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    items: Vec<ItemWire>,
    #[serde(default)]
    content: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ItemWire {
    Text(String),
    Rich(RichItem),
}

#[derive(Debug, Deserialize)]
struct RichItem {
    text: String,
    #[serde(default)]
    src: Vec<String>,
    #[serde(default)]
    rel: Vec<String>,
}

fn split_items(items: Vec<ItemWire>) -> (Vec<String>, Vec<Vec<String>>, Vec<Vec<String>>) {
    let mut texts = Vec::with_capacity(items.len());
    let mut srcs = Vec::with_capacity(items.len());
    let mut rels = Vec::with_capacity(items.len());
    for item in items {
        match item {
            ItemWire::Text(text) => {
                texts.push(text);
                srcs.push(Vec::new());
                rels.push(Vec::new());
            }
            ItemWire::Rich(rich) => {
                texts.push(rich.text);
                srcs.push(rich.src);
                rels.push(rich.rel);
            }
        }
    }
    (texts, srcs, rels)
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
    let mut next_n = 1u32;
    let segments = parsed
        .segments
        .into_iter()
        .map(|w| {
            let (texts, item_src, item_rel) = split_items(w.items);
            let (content, items, title) = if texts.is_empty() {
                (w.content, Vec::new(), w.title)
            } else {
                let content = format_numbered_block(w.title.as_deref(), &texts);
                (content, texts, w.title)
            };
            let n = w.n.or_else(|| {
                if items.is_empty() && title.is_none() {
                    None
                } else {
                    Some(next_n)
                }
            });
            if n.is_some() {
                next_n += 1;
            }
            let segment_type = if w.segment_type.is_empty() {
                "answer".to_string()
            } else {
                w.segment_type
            };
            let start = offset;
            let end = start + content.len();
            offset = end;
            Segment {
                segment_type,
                content,
                start,
                end,
                n,
                title,
                items,
                item_src,
                item_rel,
            }
        })
        .collect();

    Ok(segments)
}

fn format_numbered_block(title: Option<&str>, items: &[String]) -> String {
    let mut lines = Vec::new();
    if let Some(title) = title.map(str::trim).filter(|t| !t.is_empty()) {
        lines.push(title.to_string());
    }
    for (i, item) in items.iter().enumerate() {
        lines.push(format!("{}. {}", i + 1, item.trim()));
    }
    lines.join("\n")
}

fn heading_title(line: &str) -> Option<String> {
    let t = line.trim();
    if let Some(rest) = t.strip_prefix('#') {
        let title = rest.trim_start_matches('#').trim();
        return (!title.is_empty()).then(|| title.to_string());
    }
    let inner = t.strip_prefix("**")?.strip_suffix("**")?.trim();
    let title = inner.trim_end_matches(':').trim();
    (!title.is_empty() && !title.contains("**")).then(|| title.to_string())
}

fn bullet_item(line: &str) -> Option<String> {
    let t = line.trim();
    let rest = t
        .strip_prefix("- ")
        .or_else(|| t.strip_prefix("* "))
        .or_else(|| t.strip_prefix("+ "));
    if let Some(rest) = rest {
        let item = rest.trim();
        return (!item.is_empty()).then(|| item.to_string());
    }
    let bytes = t.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == 0 || i >= bytes.len() {
        return None;
    }
    let sep = bytes[i];
    if sep != b'.' && sep != b')' {
        return None;
    }
    let item = t[i + 1..].trim();
    (!item.is_empty()).then(|| item.to_string())
}

/// Turn markdown headings and lists into numbered blocks.
///
/// Returns empty unless there are at least two titled sections, so a normal
/// paragraph is left alone.
pub fn parse_markdown_sections(text: &str) -> Vec<Segment> {
    struct Draft {
        title: Option<String>,
        items: Vec<String>,
        prose: Vec<String>,
    }

    let mut drafts: Vec<Draft> = Vec::new();
    let mut current = Draft {
        title: None,
        items: Vec::new(),
        prose: Vec::new(),
    };
    let mut seen_heading = false;

    let flush = |drafts: &mut Vec<Draft>, current: &mut Draft| {
        if current.title.is_some() || !current.items.is_empty() || !current.prose.is_empty() {
            drafts.push(std::mem::replace(
                current,
                Draft {
                    title: None,
                    items: Vec::new(),
                    prose: Vec::new(),
                },
            ));
        }
    };

    for line in text.lines() {
        if let Some(title) = heading_title(line) {
            flush(&mut drafts, &mut current);
            current.title = Some(title);
            seen_heading = true;
            continue;
        }
        if let Some(item) = bullet_item(line) {
            current.items.push(item);
            continue;
        }
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if !current.items.is_empty() {
            flush(&mut drafts, &mut current);
        }
        current.prose.push(t.to_string());
    }
    flush(&mut drafts, &mut current);

    let titled = drafts.iter().filter(|d| d.title.is_some()).count();
    if !seen_heading || titled < 2 {
        return Vec::new();
    }

    let mut offset = 0usize;
    let mut n = 1u32;
    let mut segments = Vec::new();
    let mut seen_titled = false;
    for draft in drafts {
        let before_first_heading = !seen_titled && draft.title.is_none();
        if draft.title.is_some() {
            seen_titled = true;
        }
        if before_first_heading && draft.items.is_empty() {
            continue;
        }
        let caveat = draft.title.is_none() && draft.items.is_empty() && seen_titled;
        let items = if draft.items.is_empty() && !caveat {
            draft.prose.clone()
        } else {
            draft.items.clone()
        };
        let segment_type = if caveat { "caveat" } else { "answer" };
        let content = if caveat {
            draft.prose.join(" ")
        } else if items.is_empty() {
            draft.title.clone().unwrap_or_default()
        } else {
            format_numbered_block(draft.title.as_deref(), &items)
        };
        if content.trim().is_empty() {
            continue;
        }
        let start = offset;
        let end = start + content.len();
        offset = end;
        segments.push(Segment {
            segment_type: segment_type.to_string(),
            content,
            start,
            end,
            n: Some(n),
            title: draft.title,
            items,

            item_src: Vec::new(),
            item_rel: Vec::new(),
        });
        n += 1;
    }
    segments
}

/// Concatenate segment bodies for the Interaction node's `content` field after structured
/// JSON is stripped (API, TUI, chat export).
pub fn segment_display_text(segments: &[Segment]) -> String {
    segments
        .iter()
        .map(|s| s.content.as_str())
        .collect::<Vec<_>>()
        .join("\n\n")
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

/// Create Content nodes for each segment and link them to the parent Interaction node
/// via Contains edges (with `order` in edge metadata). Also adds Contains edges for
/// detected nesting pairs (parent segment → child segment).
///
/// Returns the NodeId of each created segment node, in order.
pub async fn persist_segments(
    store: &Arc<GraphStore>,
    parent_id: &NodeId,
    segments: &[Segment],
    nesting: &[(usize, usize)],
) -> Result<Vec<NodeId>, AgentError> {
    let store = Arc::clone(store);
    let parent_id = parent_id.clone();
    let segments = segments.to_vec();
    let nesting = nesting.to_vec();

    tokio::task::spawn_blocking(move || {
        let mut node_ids = Vec::with_capacity(segments.len());

        for (i, segment) in segments.iter().enumerate() {
            let language = if segment.segment_type == "code" {
                Some("unknown".to_string())
            } else {
                None
            };

            let mut node = GraphNode::new(NodeType::Content(ContentData {
                content_type: segment.segment_type.clone(),
                path: None,
                body: segment.content.clone(),
                language,
            }));

            let mut metadata = serde_json::json!({
                "segment_start": segment.start,
                "segment_end": segment.end,
            });
            if let Some(n) = segment.n {
                metadata["segment_n"] = serde_json::json!(n);
            }
            if let Some(title) = &segment.title {
                metadata["segment_title"] = serde_json::json!(title);
            }
            if !segment.items.is_empty() {
                metadata["segment_items"] = serde_json::json!(segment.items);
            }
            if segment.item_src.iter().any(|row| !row.is_empty()) {
                metadata["segment_item_src"] = serde_json::json!(segment.item_src);
            }
            if segment.item_rel.iter().any(|row| !row.is_empty()) {
                metadata["segment_item_rel"] = serde_json::json!(segment.item_rel);
            }
            node.metadata = metadata;

            let node_id = store.add_node(node)?;

            let edge = GraphEdge::new(EdgeType::Contains, parent_id.clone(), node_id.clone())
                .with_metadata(serde_json::json!({ "order": i }));
            store.add_edge(edge)?;

            node_ids.push(node_id);
        }

        for (parent_idx, child_idx) in &nesting {
            let edge = GraphEdge::new(
                EdgeType::Contains,
                node_ids[*parent_idx].clone(),
                node_ids[*child_idx].clone(),
            );
            store.add_edge(edge)?;
        }

        Ok(node_ids)
    })
    .await
    .map_err(|e| AgentError::Join(e.to_string()))?
}

/// Build the system prompt suffix that instructs the LLM to produce structured
/// segment JSON output.
///
/// The suffix is appended to the agent's system prompt when
/// `SegmentConfig.structured_output` is enabled. It tells the model to emit a
/// `{"segments": [...]}` JSON envelope instead of plain text.
pub fn build_segment_prompt(labels: &[String]) -> String {
    let labels_str = labels
        .iter()
        .map(|l| format!(r#""{l}""#))
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        r#"

## Response Format

Structure your entire response as a JSON object with a "segments" array. Each segment is one numbered block:
- "n": block number, starting at 1
- "type": one of {labels_str}
- "title": section title, or null
- "items": the lines of that block. A line is a string, or an object {{"text","src","rel"}}
- "src": optional list of sources for that line. Use `file:<path>`, `tool:<id>`, or `web:<url>`. A tool result in the conversation starts with [n:<id>]. Cite that result as tool:<id>, using the id inside the brackets, not the tool name.
- "rel": optional list of relations for that line. A relation is `<type>:#<block>`. Types: supports, contradicts, depends-on, answers, picks, supersedes, qualifies, limits. A caveat that narrows a claim uses qualifies. Use contradicts only when two claims cannot both be true.

Example:
```json
{{"segments": [{{"n": 1, "type": "answer", "title": "Produce", "items": [{{"text": "Bananas", "src": ["file:list.txt"], "rel": []}}]}}, {{"n": 2, "type": "caveat", "title": null, "items": ["Adjust this for household size."]}}]}}
```

Rules:
- Every part of your response must be in a segment.
- Use the most specific type that fits.
- A list is several blocks, one per heading, not one block with the whole list inside.
- Number the blocks with "n". Number the lines by putting each line in "items".
- Output ONLY the JSON object — no text outside it.
- Do NOT encode tool invocations as segments. When you need to call a tool (read, write, edit, bash, etc.), use the native tool-calling API — never put tool calls inside a segment.
- Only use these exact segment types: {labels_str}. Do not invent new types.
"#
    )
}

/// Short JSON reminder kept on the system message sent to the model.
///
/// The full rules stay on the system prompt. This header is identical on every
/// call so the cached prefix does not change when a later human message arrives.
/// It is not stored on the graph.
pub fn segment_json_request(labels: &[String]) -> String {
    let labels_str = labels
        .iter()
        .map(|l| format!(r#""{l}""#))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "\n\nReply with only this JSON shape: {{\"segments\":[{{\"n\":1,\"type\":\"answer\",\"title\":\"Section\",\"items\":[{{\"text\":\"line one\",\"src\":[\"tool:<node_id>\"],\"rel\":[]}}]}}]}}. \
         n is the block number. items are the lines. A line may be a string, or an object with text, src, and rel. \
         src uses file:<path>, tool:<id>, or web:<url>. A tool result starts with [n:<id>]; cite tool:<id> from that tag. \
         A caveat that narrows a claim uses rel qualifies. Use contradicts only when two claims cannot both be true. \
         title may be null. type is one of: {labels_str}. Split a list into one block per heading."
    )
}

/// Split a reply on `<<<type>>>` markers. `type` must be one of `labels`.
/// Fewer than two markers returns an empty list so a single blob stays unsplit.
pub fn parse_marked_segments(text: &str, labels: &[String]) -> Vec<Segment> {
    let allowed: std::collections::HashSet<String> = labels
        .iter()
        .map(|label| label.to_ascii_lowercase())
        .collect();
    let mut markers: Vec<(usize, usize, String)> = Vec::new();
    let mut search_from = 0;
    while let Some(rel) = text[search_from..].find("<<<") {
        let open = search_from + rel;
        let after = open + 3;
        let Some(rel_close) = text[after..].find(">>>") else {
            break;
        };
        let close = after + rel_close;
        let label = text[after..close].trim().to_ascii_lowercase();
        search_from = close + 3;
        if label.is_empty()
            || label.contains(char::is_whitespace)
            || !label.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            || !allowed.contains(&label)
        {
            continue;
        }
        markers.push((open, search_from, label));
    }
    if markers.len() < 2 {
        return Vec::new();
    }
    let mut offset = 0usize;
    let mut segments = Vec::new();
    for (i, (_, content_at, label)) in markers.iter().enumerate() {
        let end = markers
            .get(i + 1)
            .map(|(next_open, _, _)| *next_open)
            .unwrap_or(text.len());
        let content = text[*content_at..end].trim().to_string();
        if content.is_empty() {
            continue;
        }
        let start = offset;
        offset += content.len();
        segments.push(Segment {
            segment_type: label.clone(),
            content,
            start,
            end: offset,
            n: None,
            title: None,
            items: Vec::new(),

            item_src: Vec::new(),
            item_rel: Vec::new(),
        });
    }
    if segments.len() < 2 {
        Vec::new()
    } else {
        segments
    }
}

/// Append [`segment_json_request`] to the system message.
///
/// The text is the same on every call, so it stays in the cached prefix.
/// Appending it to the latest human message rewrote that message on the next
/// call and invalidated the cache from the task text onward.
/// No-op when there is no system message.
pub fn append_segment_json_request(messages: &mut [graphirm_llm::LlmMessage], labels: &[String]) {
    if labels.is_empty() {
        return;
    }
    let Some(msg) = messages
        .iter_mut()
        .find(|m| m.role == graphirm_llm::Role::System)
    else {
        return;
    };
    let suffix = segment_json_request(labels);
    if let Some(graphirm_llm::ContentPart::Text { text }) = msg
        .content
        .iter_mut()
        .rev()
        .find(|part| matches!(part, graphirm_llm::ContentPart::Text { .. }))
    {
        text.push_str(&suffix);
    } else {
        msg.content.push(graphirm_llm::ContentPart::text(suffix));
    }
}

/// First line of `content` looks like a built-in tool name plus a path-like argument
/// (models often put `read foo.rs` in a `code` or `plan` segment when tools are gated or omitted).
fn segment_content_mimics_tool_invocation(content: &str) -> bool {
    let line = content.trim().lines().next().unwrap_or("").trim();
    if line.is_empty() || line.len() > 240 {
        return false;
    }
    let mut parts = line.split_whitespace();
    let Some(cmd) = parts.next() else {
        return false;
    };
    let cmd_l = cmd.to_lowercase();
    if !matches!(cmd_l.as_str(), "read" | "write" | "edit") {
        return false;
    }
    let Some(arg) = parts.next() else {
        return false;
    };
    // Require a path- or filename-like token to avoid matching "read the docs".
    looks_like_tool_path_token(arg)
}

fn looks_like_tool_path_token(s: &str) -> bool {
    if s.len() < 2 {
        return false;
    }
    if s.contains('/') || s.contains('\\') {
        return true;
    }
    s.contains('.') && !s.starts_with('.')
}

/// True when any parsed segment looks like a tool invocation encoded as text instead of
/// using the provider's native tool calls (models sometimes emit `tool_call`-style types).
pub fn has_tool_call_leak(segments: &[Segment]) -> bool {
    const TOOL_LEAK_TYPES: &[&str] = &[
        "tool_call",
        "tool_calls",
        "tool_use",
        "function_call",
        "tool",
    ];
    segments.iter().any(|s| {
        let st = s.segment_type.to_lowercase();
        if TOOL_LEAK_TYPES.contains(&st.as_str()) {
            return true;
        }
        if matches!(st.as_str(), "code" | "plan" | "answer")
            && segment_content_mimics_tool_invocation(&s.content)
        {
            return true;
        }
        false
    })
}

/// Run GLiNER2 over the raw response text with segment labels and return
/// `Segment`s with character offsets from the model.
///
/// Only available with the `local-extraction` feature flag.
/// Real integration testing requires a downloaded model; set `GLINER2_MODEL_DIR`
/// and use `OnnxExtractor::new()` to construct the extractor.
#[cfg(feature = "local-extraction")]
pub async fn segment_extract_gliner2(
    extractor: &super::local_extraction::OnnxExtractor,
    text: &str,
    labels: &[String],
    min_confidence: f64,
    label_descriptions: Option<&HashMap<String, String>>,
    label_min_confidence: Option<&HashMap<String, f64>>,
) -> Result<Vec<Segment>, AgentError> {
    let raw = extractor
        .extract_raw(
            text,
            labels,
            min_confidence,
            label_descriptions,
            label_min_confidence,
        )
        .await?;
    let segments = raw
        .into_iter()
        .map(|e| Segment {
            segment_type: e.entity_type,
            content: e.text,
            start: e.start,
            end: e.end,
            n: None,
            title: None,
            items: Vec::new(),

            item_src: Vec::new(),
            item_rel: Vec::new(),
        })
        .collect();
    Ok(segments)
}

/// Convenience wrapper: retrieve the process-wide cached `OnnxExtractor` for
/// `model_dir` and run GLiNER2 segment extraction over `text`.
///
/// Returns `None` if the extractor cannot be initialised or returns no spans.
/// All errors are logged at `warn` level so callers can treat this as non-fatal.
/// Only available with the `local-extraction` feature flag.
#[cfg(feature = "local-extraction")]
pub async fn try_gliner2_fallback(
    model_dir: &str,
    text: &str,
    labels: &[String],
    min_confidence: f64,
    label_descriptions: Option<&HashMap<String, String>>,
    label_min_confidence: Option<&HashMap<String, f64>>,
) -> Option<Vec<Segment>> {
    let extractor = match super::local_extraction::get_or_init_onnx_extractor(model_dir).await {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(error = %e, "GLiNER2 OnnxExtractor init failed");
            return None;
        }
    };

    match segment_extract_gliner2(
        extractor.as_ref(),
        text,
        labels,
        min_confidence,
        label_descriptions,
        label_min_confidence,
    )
    .await
    {
        Ok(segs) if !segs.is_empty() => {
            tracing::info!(count = segs.len(), "GLiNER2 segment fallback succeeded");
            Some(segs)
        }
        Ok(_) => {
            tracing::debug!("GLiNER2 returned no segments");
            None
        }
        Err(e) => {
            tracing::warn!(error = %e, "GLiNER2 segment extraction failed");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_display_text_joins_segments() {
        let json = r#"{"segments": [
            {"type": "reasoning", "content": "a"},
            {"type": "code", "content": "b"}
        ]}"#;
        let segments = parse_structured_segments(json).unwrap();
        assert_eq!(segment_display_text(&segments), "a\n\nb");
    }

    #[test]
    fn test_parse_item_src_and_rel() {
        let json = r#"{"segments": [{"n": 1, "type": "answer", "title": "Fix", "items": [
            {"text": "The test failed.", "src": ["tool:11111111-1111-1111-1111-111111111111"], "rel": ["supports:#fix"]},
            "plain line"
        ]}]}"#;
        let segments = parse_structured_segments(json).unwrap();
        assert_eq!(segments[0].items.len(), 2);
        assert_eq!(
            segments[0].item_src[0],
            vec!["tool:11111111-1111-1111-1111-111111111111".to_string()]
        );
        assert_eq!(segments[0].item_rel[0], vec!["supports:#fix".to_string()]);
        assert!(segments[0].item_src[1].is_empty());
    }

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
                n: None,
                title: None,
                items: Vec::new(),

                item_src: Vec::new(),
                item_rel: Vec::new(),
            },
            Segment {
                segment_type: "code".into(),
                content: "fn f(){}".into(),
                start: 30,
                end: 60,
                n: None,
                title: None,
                items: Vec::new(),

                item_src: Vec::new(),
                item_rel: Vec::new(),
            },
            Segment {
                segment_type: "answer".into(),
                content: "done".into(),
                start: 110,
                end: 130,
                n: None,
                title: None,
                items: Vec::new(),

                item_src: Vec::new(),
                item_rel: Vec::new(),
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
                n: None,
                title: None,
                items: Vec::new(),

                item_src: Vec::new(),
                item_rel: Vec::new(),
            },
            Segment {
                segment_type: "code".into(),
                content: "b".into(),
                start: 55,
                end: 100,
                n: None,
                title: None,
                items: Vec::new(),

                item_src: Vec::new(),
                item_rel: Vec::new(),
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
                n: None,
                title: None,
                items: Vec::new(),

                item_src: Vec::new(),
                item_rel: Vec::new(),
            },
            Segment {
                segment_type: "code".into(),
                content: "x".into(),
                start: 0,
                end: 10,
                n: None,
                title: None,
                items: Vec::new(),

                item_src: Vec::new(),
                item_rel: Vec::new(),
            },
        ];
        let pairs = detect_nesting(&segments);
        assert!(pairs.is_empty()); // identical span = not nested
    }

    #[tokio::test]
    async fn test_persist_segments_creates_nodes_and_edges() {
        use graphirm_graph::{EdgeType, GraphNode, GraphStore, InteractionData, NodeType};
        use std::sync::Arc;

        let store = Arc::new(GraphStore::open(":memory:").unwrap());
        let parent = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".into(),
            content: "raw text".into(),
            token_count: Some(10),
        }));
        let parent_id = store.add_node(parent).unwrap();

        let segments = vec![
            Segment {
                segment_type: "reasoning".into(),
                content: "because X".into(),
                start: 0,
                end: 9,
                n: None,
                title: None,
                items: Vec::new(),

                item_src: Vec::new(),
                item_rel: Vec::new(),
            },
            Segment {
                segment_type: "code".into(),
                content: "fn f(){}".into(),
                start: 10,
                end: 18,
                n: None,
                title: None,
                items: Vec::new(),

                item_src: Vec::new(),
                item_rel: Vec::new(),
            },
        ];

        let node_ids = persist_segments(&store, &parent_id, &segments, &[])
            .await
            .unwrap();
        assert_eq!(node_ids.len(), 2);

        for (i, nid) in node_ids.iter().enumerate() {
            let node = store.get_node(nid).unwrap();
            match node.node_type {
                NodeType::Content(data) => {
                    assert_eq!(data.content_type, segments[i].segment_type);
                    assert_eq!(data.body, segments[i].content);
                }
                _ => panic!("Expected Content node at index {i}"),
            }
        }

        let edges = store.edges_for_node(&parent_id).unwrap();
        let contains: Vec<_> = edges
            .iter()
            .filter(|e| e.edge_type == EdgeType::Contains && e.source == parent_id)
            .collect();
        assert_eq!(contains.len(), 2);
    }

    #[test]
    fn numbered_json_becomes_one_block_per_section() {
        let json = r#"{"segments":[
            {"n":1,"type":"answer","title":"Produce","items":["Bananas","Apples"]},
            {"n":2,"type":"caveat","title":null,"items":["Adjust this for household size."]}
        ]}"#;
        let segments = parse_structured_segments(json).unwrap();
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].n, Some(1));
        assert_eq!(segments[0].title.as_deref(), Some("Produce"));
        assert_eq!(segments[0].items, vec!["Bananas", "Apples"]);
        assert!(segments[0].content.contains("1. Bananas"));
        assert_eq!(segments[1].segment_type, "caveat");
        assert_eq!(segments[1].n, Some(2));
    }

    #[test]
    fn markdown_headings_become_numbered_blocks() {
        let text = "\
Here's a starter list.

**Produce**
- Bananas
- Apples

**Dairy**
- Milk

Adjust this for household size.";
        let segments = parse_markdown_sections(text);
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].n, Some(1));
        assert_eq!(segments[0].title.as_deref(), Some("Produce"));
        assert_eq!(segments[0].items, vec!["Bananas", "Apples"]);
        assert_eq!(segments[1].title.as_deref(), Some("Dairy"));
        assert_eq!(segments[2].segment_type, "caveat");
        assert!(segments[2].content.contains("household size"));
        assert!(parse_markdown_sections("just a sentence").is_empty());
    }

    #[test]
    fn marked_segments_split_on_answer_markers() {
        let labels = vec!["answer".into(), "plan".into()];
        let text = "\
<<<answer>>>
**Produce**
- Apples

<<<answer>>>
**Dairy**
- Milk";
        let segments = parse_marked_segments(text, &labels);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].segment_type, "answer");
        assert!(segments[0].content.contains("Produce"));
        assert!(segments[1].content.contains("Dairy"));
        assert!(parse_marked_segments("just a list", &labels).is_empty());
        assert!(parse_marked_segments("<<<answer>>>\nonly one", &labels).is_empty());
    }

    #[test]
    fn json_request_stays_on_the_system_message() {
        let labels = vec!["answer".into(), "plan".into()];
        let mut messages = vec![
            graphirm_llm::LlmMessage::system("rules"),
            graphirm_llm::LlmMessage::human("earlier"),
            graphirm_llm::LlmMessage::assistant("ok"),
            graphirm_llm::LlmMessage::human("list of groceries"),
        ];
        append_segment_json_request(&mut messages, &labels);
        let graphirm_llm::ContentPart::Text { text } = &messages[0].content[0] else {
            panic!("expected text");
        };
        assert!(text.starts_with("rules"));
        assert!(text.contains(r#""answer""#));
        assert!(text.contains(r#""plan""#));
        assert!(text.contains("\"items\""));
        assert!(text.contains("one block per heading"));
        let graphirm_llm::ContentPart::Text { text: task } = &messages[1].content[0] else {
            panic!("expected text");
        };
        assert_eq!(task, "earlier");
        let graphirm_llm::ContentPart::Text { text: latest } = &messages[3].content[0] else {
            panic!("expected text");
        };
        assert_eq!(latest, "list of groceries");
    }

    #[test]
    fn test_build_segment_prompt_contains_labels() {
        let labels = vec!["code".into(), "reasoning".into(), "answer".into()];
        let prompt = build_segment_prompt(&labels);
        assert!(prompt.contains(r#""code""#));
        assert!(prompt.contains(r#""reasoning""#));
        assert!(prompt.contains(r#""answer""#));
        assert!(prompt.contains("segments"));
        assert!(prompt.contains("type"));
        assert!(prompt.contains("items"));
        assert!(prompt.contains("\"n\""));
    }

    #[test]
    fn test_build_segment_prompt_includes_caveat_label() {
        let labels = vec!["reasoning".into(), "caveat".into()];
        let prompt = build_segment_prompt(&labels);
        assert!(prompt.contains(r#""caveat""#));
        assert!(prompt.contains("qualifies"));
        assert!(prompt.contains("limits"));
        assert!(prompt.contains("cannot both be true"));
    }

    #[test]
    fn test_segment_prompt_forbids_tool_calls() {
        let labels = vec![
            "observation".into(),
            "reasoning".into(),
            "code".into(),
            "plan".into(),
            "answer".into(),
        ];
        let prompt = build_segment_prompt(&labels);
        assert!(
            prompt.contains("Do NOT"),
            "prompt should contain explicit prohibition"
        );
        assert!(
            prompt.to_lowercase().contains("tool"),
            "prompt should mention tool calls"
        );
    }

    #[test]
    fn test_has_tool_call_leak_detects_tool_call_segment() {
        let segments = vec![
            Segment {
                segment_type: "observation".into(),
                content: "thinking".into(),
                start: 0,
                end: 8,
                n: None,
                title: None,
                items: Vec::new(),

                item_src: Vec::new(),
                item_rel: Vec::new(),
            },
            Segment {
                segment_type: "tool_call".into(),
                content: "read foo.rs".into(),
                start: 8,
                end: 19,
                n: None,
                title: None,
                items: Vec::new(),

                item_src: Vec::new(),
                item_rel: Vec::new(),
            },
        ];
        assert!(has_tool_call_leak(&segments));
    }

    #[test]
    fn test_has_tool_call_leak_clean_segments() {
        let segments = vec![
            Segment {
                segment_type: "reasoning".into(),
                content: "thinking".into(),
                start: 0,
                end: 8,
                n: None,
                title: None,
                items: Vec::new(),

                item_src: Vec::new(),
                item_rel: Vec::new(),
            },
            Segment {
                segment_type: "answer".into(),
                content: "here you go".into(),
                start: 8,
                end: 19,
                n: None,
                title: None,
                items: Vec::new(),

                item_src: Vec::new(),
                item_rel: Vec::new(),
            },
        ];
        assert!(!has_tool_call_leak(&segments));
    }

    #[test]
    fn test_has_tool_call_leak_detects_read_in_code_segment() {
        let segments = vec![Segment {
            segment_type: "code".into(),
            content: "read architecture_decisions.md".into(),
            start: 0,
            end: 30,
            n: None,
            title: None,
            items: Vec::new(),

            item_src: Vec::new(),
            item_rel: Vec::new(),
        }];
        assert!(has_tool_call_leak(&segments));
    }

    #[test]
    fn test_has_tool_call_leak_ignores_normal_code_segment() {
        let segments = vec![Segment {
            segment_type: "code".into(),
            content: "fn main() {\n  println!(\"hi\");\n}".into(),
            start: 0,
            end: 40,
            n: None,
            title: None,
            items: Vec::new(),

            item_src: Vec::new(),
            item_rel: Vec::new(),
        }];
        assert!(!has_tool_call_leak(&segments));
    }

    /// Verify `segment_extract_gliner2` exists and the module compiles with the feature enabled.
    /// Real integration testing requires a downloaded GLiNER2 model (`GLINER2_MODEL_DIR`).
    #[cfg(feature = "local-extraction")]
    #[test]
    fn test_segment_extract_gliner2_compiles() {
        // Compilation with `--features local-extraction` is the primary check.
        // We cannot instantiate OnnxExtractor without a real model directory,
        // so this test body is intentionally empty.
    }

    #[tokio::test]
    async fn test_persist_segments_with_nesting() {
        use graphirm_graph::{EdgeType, GraphNode, GraphStore, InteractionData, NodeType};
        use std::sync::Arc;

        let store = Arc::new(GraphStore::open(":memory:").unwrap());
        let parent = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".into(),
            content: "raw".into(),
            token_count: Some(5),
        }));
        let parent_id = store.add_node(parent).unwrap();

        let segments = vec![
            Segment {
                segment_type: "plan".into(),
                content: "plan with code inside".into(),
                start: 0,
                end: 100,
                n: None,
                title: None,
                items: Vec::new(),

                item_src: Vec::new(),
                item_rel: Vec::new(),
            },
            Segment {
                segment_type: "code".into(),
                content: "fn f(){}".into(),
                start: 30,
                end: 60,
                n: None,
                title: None,
                items: Vec::new(),

                item_src: Vec::new(),
                item_rel: Vec::new(),
            },
        ];
        let nesting = vec![(0usize, 1usize)];

        let node_ids = persist_segments(&store, &parent_id, &segments, &nesting)
            .await
            .unwrap();
        assert_eq!(node_ids.len(), 2);

        let edges = store.edges_for_node(&node_ids[0]).unwrap();
        let nested: Vec<_> = edges
            .iter()
            .filter(|e| {
                e.edge_type == EdgeType::Contains
                    && e.source == node_ids[0]
                    && e.target == node_ids[1]
            })
            .collect();
        assert_eq!(nested.len(), 1);
    }
}
