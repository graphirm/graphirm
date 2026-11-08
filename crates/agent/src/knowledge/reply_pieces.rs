//! Write an assistant reply's pieces as `reply_part` / `reply_line` Content nodes.
//!
//! Shared by Pi replies (`pi_delegate::graph`) and Graphirm's own loop
//! (`[agent.reply_pieces]`). The cutting lives in `pi_delegate::pieces`.

use graphirm_graph::GraphStore;
use graphirm_graph::edges::{EdgeType, GraphEdge};
use graphirm_graph::nodes::{ContentData, GraphNode, NodeId, NodeType};
use serde_json::json;

use crate::error::AgentError;
use crate::pi_delegate::pieces::{
    HTML_INDEX_SHAPE_VERSION, PIECE_BASELINE_VERSION, PIECE_PARSER_VERSION, Piece, PieceItem,
    cut_html_index, structure_segment,
};

/// How a reply's pieces were cut. Stored as `pieces_source` on the reply node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PiecesSource {
    /// The reply was a clean HTML index page; kinds are the model's own.
    Html,
    /// The reply was not an HTML index; blocks and kinds come from the markdown
    /// parser and its baseline labeler.
    Markdown,
    /// The reply was over the parser's character cap; one `statement` piece.
    OverCap,
}

impl PiecesSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Html => "html",
            Self::Markdown => "markdown",
            Self::OverCap => "over_cap",
        }
    }
}

/// A clean HTML index becomes `reply_part` and `reply_line` Content nodes.
/// A failed cut leaves the reply as it was recorded. Used for Pi replies.
pub fn attach_html_pieces(
    graph: &GraphStore,
    parent: &NodeId,
    text: &str,
) -> Result<(), AgentError> {
    let Ok(pieces) = cut_html_index(text) else {
        return Ok(());
    };
    write_pieces(graph, parent, text, &pieces, None)
}

/// Cut any reply with text: the HTML index when the page is one, otherwise the
/// markdown parser (`narration` for text written next to tool calls). Writes
/// the pieces and stamps `pieces_source` on the reply node, so every reply
/// records how it was cut.
pub fn record_reply_pieces(
    graph: &GraphStore,
    parent: &NodeId,
    text: &str,
    narration: bool,
) -> Result<PiecesSource, AgentError> {
    let (pieces, source) = match cut_html_index(text) {
        Ok(pieces) if !pieces.is_empty() => (pieces, PiecesSource::Html),
        _ => {
            let segment = structure_segment(text, narration);
            let source = if segment.parsed {
                PiecesSource::Markdown
            } else {
                PiecesSource::OverCap
            };
            (segment.pieces, source)
        }
    };
    write_pieces(graph, parent, text, &pieces, Some(source))?;
    let mut node = graph.get_node(parent)?;
    node.metadata["pieces_source"] = json!(source.as_str());
    node.metadata["pieces_count"] = json!(pieces.len());
    graph.update_node(parent, node)?;
    Ok(source)
}

/// `source` is `None` for Pi replies, which keep their original part metadata.
fn write_pieces(
    graph: &GraphStore,
    parent: &NodeId,
    text: &str,
    pieces: &[Piece],
    source: Option<PiecesSource>,
) -> Result<(), AgentError> {
    for piece in pieces {
        let part_id = graph.add_node(part_node(text, piece, source))?;
        graph.add_edge(GraphEdge::new(
            EdgeType::Contains,
            parent.clone(),
            part_id.clone(),
        ))?;
        // A markdown paragraph has no items; its text becomes the one line, so
        // every part can be read back from its lines.
        let whole;
        let items: &[PieceItem] = if piece.items.is_empty() && source.is_some() {
            whole = [PieceItem {
                position: 1,
                text: text
                    .get(piece.start..piece.end)
                    .unwrap_or("")
                    .trim()
                    .to_string(),
                src: None,
                start: piece.start,
                end: piece.end,
            }];
            &whole
        } else {
            &piece.items
        };
        for item in items {
            let line_id = graph.add_node(line_node(item))?;
            graph.add_edge(GraphEdge::new(EdgeType::Contains, part_id.clone(), line_id))?;
        }
    }
    Ok(())
}

fn part_node(text: &str, piece: &Piece, source: Option<PiecesSource>) -> GraphNode {
    let slice = text.get(piece.start..piece.end).unwrap_or("");
    let mut node = GraphNode::new(NodeType::Content(ContentData {
        content_type: "reply_part".to_string(),
        path: None,
        body: piece.heading.clone().unwrap_or_default(),
        language: None,
    }));
    node.metadata = json!({
        "order": piece.order,
        "id": element_id(slice).unwrap_or_default(),
        "kind": piece.kind.as_label(),
        "heading": piece.heading,
        "rel": piece.rel,
        "shape_version": HTML_INDEX_SHAPE_VERSION,
        "start": piece.start,
        "end": piece.end,
    });
    if let Some(source) = source {
        node.metadata["pieces_source"] = json!(source.as_str());
        if source != PiecesSource::Html {
            node.metadata["parser_version"] = json!(PIECE_PARSER_VERSION);
            node.metadata["baseline_version"] = json!(PIECE_BASELINE_VERSION);
        }
    }
    node
}

fn line_node(item: &PieceItem) -> GraphNode {
    let mut node = GraphNode::new(NodeType::Content(ContentData {
        content_type: "reply_line".to_string(),
        path: None,
        body: item.text.clone(),
        language: None,
    }));
    node.metadata = json!({
        "position": item.position,
        "src": item.src,
        "start": item.start,
        "end": item.end,
    });
    node
}

fn element_id(slice: &str) -> Option<String> {
    for (mark, quote) in [("id=\"", '"'), ("id='", '\'')] {
        let Some(at) = slice.find(mark) else {
            continue;
        };
        let rest = &slice[at + mark.len()..];
        let Some(end) = rest.find(quote) else {
            continue;
        };
        return Some(rest[..end].to_string());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphirm_graph::Direction;
    use graphirm_graph::nodes::InteractionData;

    const HTML: &str = r##"<nav id="index"><ul><li><a href="#a">1.0.0 [result] Ran the tests</a></li></ul></nav>
<section id="a" class="result"><h2>1.0.0 Ran the tests</h2><p data-n="1.0.1" data-src="cmd">1.0.1 3 of 120 failed.</p></section>"##;

    fn reply(graph: &GraphStore, text: &str) -> NodeId {
        graph
            .add_node(GraphNode::new(NodeType::Interaction(InteractionData {
                role: "assistant".to_string(),
                content: text.to_string(),
                token_count: None,
            })))
            .unwrap()
    }

    fn children(graph: &GraphStore, id: &NodeId) -> Vec<GraphNode> {
        graph
            .neighbors(id, Some(EdgeType::Contains), Direction::Outgoing)
            .unwrap()
    }

    fn body(node: &GraphNode) -> &str {
        match &node.node_type {
            NodeType::Content(data) => &data.body,
            _ => panic!("expected Content"),
        }
    }

    #[test]
    fn html_reply_keeps_the_models_kinds() {
        let graph = GraphStore::open_memory().unwrap();
        let id = reply(&graph, HTML);
        assert_eq!(
            record_reply_pieces(&graph, &id, HTML, false).unwrap(),
            PiecesSource::Html
        );
        let parts = children(&graph, &id);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].metadata["kind"], "result");
        assert_eq!(parts[0].metadata["pieces_source"], "html");
        let lines = children(&graph, &parts[0].id);
        assert_eq!(body(&lines[0]), "3 of 120 failed.");
        assert_eq!(lines[0].metadata["src"], "cmd");
        let node = graph.get_node(&id).unwrap();
        assert_eq!(node.metadata["pieces_source"], "html");
        assert_eq!(node.metadata["pieces_count"], 1);
    }

    #[test]
    fn plain_reply_is_cut_by_the_markdown_parser() {
        let graph = GraphStore::open_memory().unwrap();
        let text = "Let me reproduce the issue first.\n\nThen I will fix it.";
        let id = reply(&graph, text);
        assert_eq!(
            record_reply_pieces(&graph, &id, text, true).unwrap(),
            PiecesSource::Markdown
        );
        let parts = children(&graph, &id);
        assert!(!parts.is_empty());
        assert!(
            parts
                .iter()
                .all(|p| p.metadata["pieces_source"] == "markdown")
        );
        // Every part can be read back from its lines.
        let first = &children(&graph, &parts[0].id)[0];
        assert_eq!(body(first), "Let me reproduce the issue first.");
    }

    #[test]
    fn over_cap_reply_is_one_statement() {
        let graph = GraphStore::open_memory().unwrap();
        let text = "x".repeat(crate::pi_delegate::pieces::MAX_SEGMENT_CHARS + 1);
        let id = reply(&graph, &text);
        assert_eq!(
            record_reply_pieces(&graph, &id, &text, false).unwrap(),
            PiecesSource::OverCap
        );
        let parts = children(&graph, &id);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].metadata["kind"], "statement");
    }

    #[test]
    fn pi_html_pieces_keep_their_metadata_and_skip_plain_text() {
        let graph = GraphStore::open_memory().unwrap();
        let id = reply(&graph, HTML);
        attach_html_pieces(&graph, &id, HTML).unwrap();
        let parts = children(&graph, &id);
        assert!(parts[0].metadata.get("pieces_source").is_none());
        let plain = reply(&graph, "plain");
        attach_html_pieces(&graph, &plain, "plain").unwrap();
        assert!(children(&graph, &plain).is_empty());
    }
}
