//! Session export — renders a session's graph nodes as a Markdown document.

use chrono::{DateTime, Utc};
use graphirm_graph::{GraphNode, NodeType};

pub fn render_session_markdown(
    session_name: &str,
    model: &str,
    created_at: DateTime<Utc>,
    nodes: &[GraphNode],
) -> String {
    let mut out = String::with_capacity(4096);

    // Header
    out.push_str(&format!("# Session: {session_name}\n\n"));
    out.push_str(&format!("**Model:** {model}  \n"));
    out.push_str(&format!(
        "**Created:** {}  \n",
        created_at.format("%Y-%m-%d %H:%M UTC")
    ));
    out.push_str(&format!(
        "**Exported:** {}  \n\n",
        Utc::now().format("%Y-%m-%d %H:%M UTC")
    ));
    out.push_str("---\n\n");

    // Conversation — user and assistant only, sorted by created_at
    let mut interactions: Vec<&GraphNode> = nodes
        .iter()
        .filter(|n| {
            if let NodeType::Interaction(d) = &n.node_type {
                d.role != "tool" && d.role != "system"
            } else {
                false
            }
        })
        .collect();
    interactions.sort_by_key(|n| n.created_at);

    out.push_str("## Conversation\n\n");

    if interactions.is_empty() {
        out.push_str("*No messages.*\n\n");
    } else {
        for node in &interactions {
            if let NodeType::Interaction(d) = &node.node_type {
                let label = match d.role.as_str() {
                    "user" => "**User**",
                    "assistant" => "**Assistant**",
                    other => other,
                };
                out.push_str(&format!("{label}\n\n"));
                out.push_str(&d.content);
                out.push_str("\n\n---\n\n");
            }
        }
    }

    // Knowledge table
    let mut knowledge: Vec<&GraphNode> = nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeType::Knowledge(_)))
        .collect();
    knowledge.sort_by_key(|n| n.created_at);

    if !knowledge.is_empty() {
        out.push_str("## Extracted Knowledge\n\n");
        out.push_str("| Entity | Type | Summary | Confidence |\n");
        out.push_str("|--------|------|---------|------------|\n");
        for node in &knowledge {
            if let NodeType::Knowledge(k) = &node.node_type {
                let entity = k.entity.replace('|', "\\|");
                let entity_type = k.entity_type.replace('|', "\\|");
                let summary = k.summary.replace('|', "\\|");
                out.push_str(&format!(
                    "| {} | {} | {} | {:.0}% |\n",
                    entity,
                    entity_type,
                    summary,
                    k.confidence * 100.0,
                ));
            }
        }
        out.push('\n');
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use graphirm_graph::{InteractionData, KnowledgeData};

    fn interaction(role: &str, content: &str) -> GraphNode {
        GraphNode::new(NodeType::Interaction(InteractionData {
            role: role.to_string(),
            content: content.to_string(),
            token_count: None,
        }))
    }

    fn knowledge(entity: &str, entity_type: &str, summary: &str) -> GraphNode {
        GraphNode::new(NodeType::Knowledge(KnowledgeData {
            entity: entity.to_string(),
            entity_type: entity_type.to_string(),
            summary: summary.to_string(),
            confidence: 0.9,
        }))
    }

    #[test]
    fn renders_metadata_header() {
        let nodes = vec![interaction("user", "Hello")];
        let md = render_session_markdown("test-session", "claude-sonnet", Utc::now(), &nodes);
        assert!(md.contains("# Session: test-session"));
        assert!(md.contains("claude-sonnet"));
    }

    #[test]
    fn renders_user_and_assistant_turns() {
        let nodes = vec![
            interaction("user", "What is Rust?"),
            interaction("assistant", "Rust is a systems language."),
        ];
        let md = render_session_markdown("s", "m", Utc::now(), &nodes);
        assert!(md.contains("**User**"));
        assert!(md.contains("What is Rust?"));
        assert!(md.contains("**Assistant**"));
        assert!(md.contains("Rust is a systems language."));
    }

    #[test]
    fn skips_tool_interactions() {
        let nodes = vec![
            interaction("user", "Run tests"),
            interaction("tool", r#"{"tool":"bash","result":"ok"}"#),
            interaction("assistant", "Tests pass."),
        ];
        let md = render_session_markdown("s", "m", Utc::now(), &nodes);
        assert!(!md.contains("**Tool**"));
        assert!(md.contains("Tests pass."));
    }

    #[test]
    fn renders_knowledge_section() {
        let nodes = vec![knowledge("Rust", "language", "A systems programming language.")];
        let md = render_session_markdown("s", "m", Utc::now(), &nodes);
        assert!(md.contains("## Extracted Knowledge"));
        assert!(md.contains("Rust"));
        assert!(md.contains("language"));
        assert!(md.contains("A systems programming language."));
    }

    #[test]
    fn empty_session_produces_valid_markdown() {
        let md = render_session_markdown("empty", "m", Utc::now(), &[]);
        assert!(md.contains("# Session: empty"));
        assert!(md.contains("*No messages.*"));
    }
}
