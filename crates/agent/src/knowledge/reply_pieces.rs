//! Write an assistant reply's pieces as `reply_part` / `reply_line` Content nodes.
//!
//! Shared by Pi replies (`pi_delegate::graph`) and Graphirm's own loop
//! (`[agent.reply_pieces]`). The cutting lives in `pi_delegate::pieces`.

use graphirm_graph::edges::{EdgeType, GraphEdge};
use graphirm_graph::nodes::{ContentData, GraphNode, NodeId, NodeType};
use graphirm_graph::{Direction, GraphStore};
use serde_json::json;

use crate::error::AgentError;
use crate::pi_delegate::pieces::{
    HTML_INDEX_SHAPE_VERSION, PIECE_BASELINE_VERSION, PIECE_PARSER_VERSION, Piece, PieceItem,
    cut_html_index, structure_segment,
};

/// A minimal page in the shape `cut_html_index` accepts. Without it the model
/// wrote sections with no kind class and numbered lines `N.0.0` (smoke run
/// 2026-11-08), and every page fell back to the markdown parser.
const PAGE_EXAMPLE: &str = r##"
Each <section> has the id its index link points to and class set to its kind. Section N has an <h2> with its number and heading, and its lines are numbered N.0.1, N.0.2, and so on.
Copy this shape and replace the content with your own:
<nav id="index">
<ul>
<li><a href="#cause">1.0.0 [statement] Why the test fails</a></li>
<li><a href="#fix">2.0.0 [code] The fix in stats.py</a></li>
<li><a href="#ran">3.0.0 [result] Test run after the fix</a></li>
</ul>
</nav>
<section id="cause" class="statement">
<h2>1.0.0 Why the test fails</h2>
<p data-n="1.0.1" data-src="file:stats.py">1.0.1 median returns the upper middle value for an even-length list.</p>
</section>
<section id="fix" class="code">
<h2>2.0.0 The fix in stats.py</h2>
<pre data-n="2.0.1"><code>return (xs[n // 2 - 1] + xs[n // 2]) / 2</code></pre>
</section>
<section id="ran" class="result">
<h2>3.0.0 Test run after the fix</h2>
<p data-n="3.0.1" data-src="tool:bash">3.0.1 Ran pytest; 2 of 2 tests passed.</p>
</section>
To cite a part of an earlier reply, set data-src on the citing line to part:<part id> or reply:<reply id>#<part number>, with the ids exactly as graph_query mode "parts" prints them. Never guess an id.
"##;

/// The reply format Graphirm's own loop asks for: Pi's HTML index contract
/// plus an example page.
pub fn loop_contract() -> String {
    format!(
        "{}{PAGE_EXAMPLE}",
        crate::pi_delegate::tool::HTML_PIECE_INDEX_CONTRACT
    )
}

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
    write_pieces(graph, parent, text, &pieces, None).map(|_| ())
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
    let citations = match write_pieces(graph, parent, text, &pieces, Some(source)) {
        Ok(citations) => citations,
        Err(e) => {
            // Some parts may be written; say so on the reply instead of leaving it unstamped.
            let mut node = graph.get_node(parent)?;
            node.metadata["pieces_source"] = json!("failed");
            graph.update_node(parent, node)?;
            return Err(e);
        }
    };
    let mut node = graph.get_node(parent)?;
    node.metadata["pieces_source"] = json!(source.as_str());
    node.metadata["pieces_count"] = json!(pieces.len());
    node.metadata["citations_resolved"] = json!(citations.resolved);
    node.metadata["citations_unresolved"] = json!(citations.unresolved);
    graph.update_node(parent, node)?;
    Ok(source)
}

/// How many of a reply's citations found their part (ADR-0002).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct CitationCounts {
    resolved: usize,
    unresolved: usize,
}

/// What a line's `data-src` says about an earlier reply part.
enum Citation {
    /// Not a citation: no `data-src`, or one like `file:…` / `tool:…`.
    None,
    Resolved(NodeId),
    Unresolved,
}

fn is_reply_part(node: &GraphNode) -> bool {
    matches!(&node.node_type, NodeType::Content(c) if c.content_type == "reply_part")
}

/// `part:<part id>` or `reply:<reply id>#<N>`, checked against the graph: the
/// target must exist and be a `reply_part`.
fn resolve_citation(graph: &GraphStore, src: Option<&str>) -> Result<Citation, AgentError> {
    let Some(src) = src.map(str::trim) else {
        return Ok(Citation::None);
    };
    let found = if let Some(id) = src.strip_prefix("part:") {
        graph
            .get_node(&NodeId(id.trim().to_string()))
            .ok()
            .filter(is_reply_part)
            .map(|node| node.id)
    } else if let Some(rest) = src.strip_prefix("reply:") {
        let number = rest
            .rsplit_once('#')
            .and_then(|(id, n)| Some((id.trim(), n.trim().parse::<usize>().ok()?)));
        match number {
            Some((reply_id, n)) if n >= 1 => {
                let reply_id = NodeId(reply_id.to_string());
                if graph.get_node(&reply_id).is_err() {
                    None
                } else {
                    let mut parts: Vec<GraphNode> = graph
                        .neighbors(&reply_id, Some(EdgeType::Contains), Direction::Outgoing)?
                        .into_iter()
                        .filter(is_reply_part)
                        .collect();
                    parts.sort_by_key(|p| p.metadata["order"].as_u64().unwrap_or(0));
                    parts.into_iter().nth(n - 1).map(|node| node.id)
                }
            }
            _ => None,
        }
    } else {
        return Ok(Citation::None);
    };
    Ok(found.map_or(Citation::Unresolved, Citation::Resolved))
}

/// `source` is `None` for Pi replies, which keep their original part metadata
/// and are not checked for citations (Pi cannot look a part id up).
fn write_pieces(
    graph: &GraphStore,
    parent: &NodeId,
    text: &str,
    pieces: &[Piece],
    source: Option<PiecesSource>,
) -> Result<CitationCounts, AgentError> {
    let mut citations = CitationCounts::default();
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
            let citation = match source {
                Some(_) => resolve_citation(graph, item.src.as_deref())?,
                None => Citation::None,
            };
            let mut line = line_node(item);
            if matches!(citation, Citation::Unresolved) {
                line.metadata["citation_unresolved"] = json!(true);
            }
            let line_id = graph.add_node(line)?;
            graph.add_edge(GraphEdge::new(
                EdgeType::Contains,
                part_id.clone(),
                line_id.clone(),
            ))?;
            match citation {
                Citation::Resolved(cited) => {
                    graph.add_edge(
                        GraphEdge::new(EdgeType::DerivedFrom, line_id, cited)
                            .with_metadata(json!({ "citation": true, "src": item.src })),
                    )?;
                    citations.resolved += 1;
                }
                Citation::Unresolved => {
                    tracing::warn!(
                        reply = %parent,
                        src = item.src.as_deref().unwrap_or(""),
                        "Reply line cites a part that does not exist; not linked"
                    );
                    citations.unresolved += 1;
                }
                Citation::None => {}
            }
        }
    }
    Ok(citations)
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
    fn the_example_page_in_the_contract_cuts_cleanly() {
        let contract = loop_contract();
        // The contract's prose also names `<nav id="index">`; the page is the last one.
        let start = contract.rfind("<nav id=\"index\">").expect("example page");
        let pieces = cut_html_index(&contract[start..]).expect("example cuts");
        let kinds: Vec<&str> = pieces.iter().map(|p| p.kind.as_label()).collect();
        assert_eq!(kinds, vec!["statement", "code", "result"]);
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

    /// A one-line statement page whose line has the given `data-src`.
    fn citing_page(src: &str) -> String {
        format!(
            r##"<nav id="index"><ul><li><a href="#b">1.0.0 [statement] What failed</a></li></ul></nav>
<section id="b" class="statement"><h2>1.0.0 What failed</h2><p data-n="1.0.1" data-src="{src}">1.0.1 The same 3 tests.</p></section>"##
        )
    }

    /// Records `citing_page(src)` as a new reply; returns the reply and its one line.
    fn cite(graph: &GraphStore, src: &str) -> (GraphNode, GraphNode) {
        let page = citing_page(src);
        let id = reply(graph, &page);
        assert_eq!(
            record_reply_pieces(graph, &id, &page, false).unwrap(),
            PiecesSource::Html
        );
        let part = children(graph, &id).remove(0);
        let line = children(graph, &part.id).remove(0);
        (graph.get_node(&id).unwrap(), line)
    }

    fn citation_edges(graph: &GraphStore, line: &NodeId) -> Vec<GraphEdge> {
        graph
            .edges_for_node(line)
            .unwrap()
            .into_iter()
            .filter(|e| e.edge_type == EdgeType::DerivedFrom)
            .collect()
    }

    #[test]
    fn a_citation_links_the_line_to_the_cited_part() {
        let graph = GraphStore::open_memory().unwrap();
        let earlier = reply(&graph, HTML);
        record_reply_pieces(&graph, &earlier, HTML, false).unwrap();
        let cited = children(&graph, &earlier).remove(0);

        for src in [format!("part:{}", cited.id), format!("reply:{earlier}#1")] {
            let (node, line) = cite(&graph, &src);
            let edges = citation_edges(&graph, &line.id);
            assert_eq!(edges.len(), 1, "{src}");
            assert_eq!(edges[0].source, line.id);
            assert_eq!(edges[0].target, cited.id);
            assert_eq!(edges[0].metadata["citation"], true);
            assert_eq!(edges[0].metadata["src"], src.as_str());
            assert!(line.metadata.get("citation_unresolved").is_none());
            assert_eq!(node.metadata["citations_resolved"], 1);
            assert_eq!(node.metadata["citations_unresolved"], 0);
        }
    }

    #[test]
    fn an_unknown_citation_target_is_reported_not_linked() {
        let graph = GraphStore::open_memory().unwrap();
        let earlier = reply(&graph, HTML);
        record_reply_pieces(&graph, &earlier, HTML, false).unwrap();

        // No such node, a node that is not a part, a part number the reply lacks, no number.
        for src in [
            "part:no-such-node".to_string(),
            format!("part:{earlier}"),
            format!("reply:{earlier}#2"),
            format!("reply:{earlier}"),
        ] {
            let (node, line) = cite(&graph, &src);
            assert!(citation_edges(&graph, &line.id).is_empty(), "{src}");
            assert_eq!(line.metadata["citation_unresolved"], true, "{src}");
            assert_eq!(node.metadata["citations_resolved"], 0);
            assert_eq!(node.metadata["citations_unresolved"], 1);
        }
    }

    #[test]
    fn other_sources_are_not_citations() {
        let graph = GraphStore::open_memory().unwrap();
        let (node, line) = cite(&graph, "file:stats.py");
        assert!(citation_edges(&graph, &line.id).is_empty());
        assert!(line.metadata.get("citation_unresolved").is_none());
        assert_eq!(node.metadata["citations_resolved"], 0);
        assert_eq!(node.metadata["citations_unresolved"], 0);
    }

    #[test]
    fn pi_replies_are_not_checked_for_citations() {
        let graph = GraphStore::open_memory().unwrap();
        let page = citing_page("part:no-such-node");
        let id = reply(&graph, &page);
        attach_html_pieces(&graph, &id, &page).unwrap();
        let part = children(&graph, &id).remove(0);
        let line = children(&graph, &part.id).remove(0);
        assert!(line.metadata.get("citation_unresolved").is_none());
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
