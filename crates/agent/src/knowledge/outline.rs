//! Post-hoc outline extraction: markdown headings → `outline_item` Content nodes.

use std::sync::Arc;

use graphirm_graph::{ContentData, EdgeType, GraphEdge, GraphNode, GraphStore, NodeId, NodeType};
use serde_json::json;

use crate::config::OutlineConfig;
use crate::error::AgentError;

/// `ContentData.content_type` for outline rows.
pub const OUTLINE_CONTENT_TYPE: &str = "outline_item";

/// One section parsed from markdown (`#` … `######`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedOutlineItem {
    /// Heading level 1–6.
    pub level: usize,
    pub title: String,
    pub body: String,
    /// Index into the same vec for the parent outline item, if any.
    pub parent_index: Option<usize>,
}

/// Map heading level to a stable kind label (catalog-aligned defaults).
pub fn infer_outline_kind(level: usize, catalog: &[String]) -> String {
    let default = match level {
        1 => "vision",
        2 => "epic",
        3 => "phase",
        _ => "misc",
    };
    if catalog.is_empty() {
        return default.to_string();
    }
    if catalog.iter().any(|k| k == default) {
        return default.to_string();
    }
    catalog
        .get(level.saturating_sub(1))
        .cloned()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "misc".to_string())
}

/// Parse markdown into a flat list of outline items with parent links (stack by heading depth).
pub fn parse_markdown_outline(text: &str) -> Vec<ParsedOutlineItem> {
    let lines: Vec<&str> = text.lines().collect();
    let mut headings: Vec<(usize, usize, String)> = Vec::new(); // (line_idx, level, title)

    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        if !t.starts_with('#') {
            continue;
        }
        let mut rest = t;
        let mut level = 0usize;
        while rest.starts_with('#') && level < 6 {
            level += 1;
            rest = &rest[1..];
        }
        if level == 0 {
            continue;
        }
        let title = rest.trim().to_string();
        if title.is_empty() {
            continue;
        }
        headings.push((i, level, title));
    }

    if headings.is_empty() {
        return Vec::new();
    }

    let mut items = Vec::new();
    let mut stack: Vec<(usize, usize)> = Vec::new(); // (level, item_index)

    for (hi, (line_idx, level, title)) in headings.iter().cloned().enumerate() {
        while let Some(&(sl, _)) = stack.last() {
            if sl >= level {
                stack.pop();
            } else {
                break;
            }
        }
        let parent_index = stack.last().map(|(_, idx)| *idx);

        let body_start = line_idx + 1;
        let body_end = headings
            .get(hi + 1)
            .map(|(li, _, _)| *li)
            .unwrap_or(lines.len());

        let body = lines
            .get(body_start..body_end)
            .map(|sl| sl.join("\n").trim().to_string())
            .unwrap_or_default();

        let idx = items.len();
        items.push(ParsedOutlineItem {
            level,
            title,
            body,
            parent_index,
        });
        stack.push((level, idx));
    }

    items
}

/// Persist outline items as child Content nodes under the assistant Interaction.
pub async fn persist_outline_items(
    store: &Arc<GraphStore>,
    parent_id: &NodeId,
    items: &[ParsedOutlineItem],
    config: &OutlineConfig,
) -> Result<Vec<NodeId>, AgentError> {
    let store = Arc::clone(store);
    let parent_id = parent_id.clone();
    let items = items.to_vec();
    let catalog = config.kinds.clone();

    tokio::task::spawn_blocking(move || {
        let mut node_ids = Vec::with_capacity(items.len());

        for (i, item) in items.iter().enumerate() {
            let kind = infer_outline_kind(item.level, &catalog);
            let outline_item_id = NodeId::new();

            let body_display = if item.body.is_empty() {
                item.title.clone()
            } else {
                format!("{}\n\n{}", item.title, item.body)
            };

            let mut node = GraphNode::new(NodeType::Content(ContentData {
                content_type: OUTLINE_CONTENT_TYPE.to_string(),
                path: None,
                body: body_display,
                language: None,
            }));

            let mut meta = serde_json::Map::new();
            meta.insert("outline_kind".to_string(), serde_json::Value::String(kind));
            meta.insert(
                "outline_title".to_string(),
                serde_json::Value::String(item.title.clone()),
            );
            meta.insert("outline_item_id".to_string(), json!(outline_item_id.0));
            meta.insert("outline_source".to_string(), json!("markdown"));
            meta.insert("outline_order".to_string(), json!(i as u32));
            meta.insert("user_edited".to_string(), json!(false));
            meta.insert("user_authored".to_string(), json!(false));
            meta.insert("hidden".to_string(), json!(false));
            if let Some(p) = item.parent_index {
                meta.insert("parent_outline_index".to_string(), json!(p));
            }
            node.metadata = serde_json::Value::Object(meta);

            let node_id = store.add_node(node)?;

            let edge = GraphEdge::new(EdgeType::Contains, parent_id.clone(), node_id.clone())
                .with_metadata(json!({ "order": i, "outline": true }));
            store.add_edge(edge)?;

            node_ids.push(node_id);
        }

        Ok(node_ids)
    })
    .await
    .map_err(|e| AgentError::Join(e.to_string()))?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_two_level_headings() {
        let md = r#"# Vision

Top.

## Epic A

Body a.

## Epic B

Body b.
"#;
        let items = parse_markdown_outline(md);
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].level, 1);
        assert_eq!(items[0].title, "Vision");
        assert_eq!(items[0].parent_index, None);
        assert_eq!(items[1].title, "Epic A");
        assert_eq!(items[1].parent_index, Some(0));
        assert_eq!(items[2].parent_index, Some(0));
    }

    #[test]
    fn infer_kind_respects_catalog() {
        let catalog = vec!["vision".into(), "epic".into(), "phase".into()];
        assert_eq!(infer_outline_kind(1, &catalog), "vision");
        assert_eq!(infer_outline_kind(2, &catalog), "epic");
    }

    #[test]
    fn empty_text_no_items() {
        assert!(parse_markdown_outline("no headings here").is_empty());
    }
}
