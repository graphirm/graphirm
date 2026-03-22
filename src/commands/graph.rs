use std::path::Path;

use crate::GraphAction;
use crate::error::GraphirmError;

pub fn run(action: GraphAction, db_path: &Path) -> Result<(), GraphirmError> {
    let graph = super::open_store(db_path)?;

    match action {
        GraphAction::Stats => {
            let nodes = graph.node_count_db()?;
            let edges = graph.edge_count_db()?;
            let by_type = graph.node_counts_by_type()?;

            println!("Graph: {}", db_path.display());
            println!("  Nodes : {nodes}");
            println!("  Edges : {edges}");
            if !by_type.is_empty() {
                println!("  By type:");
                for (t, c) in by_type {
                    println!("    {t:<15} {c}");
                }
            }
        }
        GraphAction::List { limit, r#type } => {
            let nodes = graph.list_recent_nodes(limit)?;
            let nodes: Vec<_> = if let Some(ref filter) = r#type {
                nodes
                    .into_iter()
                    .filter(|n| n.node_type.type_name() == filter.as_str())
                    .collect()
            } else {
                nodes
            };

            if nodes.is_empty() {
                println!("No nodes found.");
                return Ok(());
            }

            println!("{:<38}  {:<12}  LABEL", "ID", "TYPE");
            println!("{}", "-".repeat(90));
            for node in nodes {
                let label = node_display_label(&node);
                println!(
                    "{:<38}  {:<12}  {}",
                    &node.id.to_string()[..36.min(node.id.to_string().len())],
                    node.node_type.type_name(),
                    label
                );
            }
        }
    }
    Ok(())
}

pub fn node_display_label(node: &graphirm_graph::nodes::GraphNode) -> String {
    use graphirm_graph::nodes::NodeType;
    if let Some(label) = node.label() {
        return label.to_string();
    }
    match &node.node_type {
        NodeType::Interaction(d) => {
            let preview: String = d.content.chars().take(60).collect();
            let ellipsis = if d.content.len() > 60 { "…" } else { "" };
            format!("[{}] {}{}", d.role, preview, ellipsis)
        }
        NodeType::Agent(d) => format!("[agent] {} ({})", d.name, d.status),
        NodeType::Content(d) => {
            let name = d.path.as_deref().unwrap_or(&d.content_type);
            format!("[content] {}", name)
        }
        NodeType::Task(d) => format!("[task] {} — {}", d.title, d.status),
        NodeType::Knowledge(d) => format!("[{}] {}", d.entity_type, d.entity),
    }
}

#[cfg(test)]
mod tests {
    use super::node_display_label;
    use graphirm_graph::nodes::{GraphNode, InteractionData, NodeType};

    #[test]
    fn node_display_label_prefers_metadata_label() {
        let mut node = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".to_string(),
            content: "Fallback preview".to_string(),
            token_count: None,
        }));
        node.set_label("interaction_1_2_1");

        assert_eq!(node_display_label(&node), "interaction_1_2_1");
    }
}
