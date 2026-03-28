use std::path::Path;

use crate::KnowledgeAction;
use crate::error::GraphirmError;

pub fn run(action: KnowledgeAction, db_path: &Path) -> Result<(), GraphirmError> {
    let store = super::open_store(db_path)?;

    match action {
        KnowledgeAction::List { limit, json } => {
            let nodes = store.list_pinned_knowledge(limit)?;
            if nodes.is_empty() {
                println!("No pinned knowledge nodes.");
            } else if json {
                let arr: Vec<serde_json::Value> = nodes
                    .iter()
                    .filter_map(|node| {
                        if let graphirm_graph::NodeType::Knowledge(ref kd) = node.node_type {
                            let mut obj = serde_json::Map::new();
                            obj.insert("id".to_string(), serde_json::json!(node.id.0));
                            obj.insert("entity".to_string(), serde_json::json!(kd.entity));
                            obj.insert(
                                "entity_type".to_string(),
                                serde_json::json!(kd.entity_type),
                            );
                            obj.insert("summary".to_string(), serde_json::json!(kd.summary));
                            obj.insert("confidence".to_string(), serde_json::json!(kd.confidence));
                            Some(serde_json::Value::Object(obj))
                        } else {
                            None
                        }
                    })
                    .collect();
                println!("{}", serde_json::to_string_pretty(&arr)?);
            } else {
                for node in &nodes {
                    if let graphirm_graph::NodeType::Knowledge(ref kd) = node.node_type {
                        println!(
                            "{}  [{}] {}: {}",
                            node.id.0, kd.entity_type, kd.entity, kd.summary
                        );
                    }
                }
                println!("\n{} pinned node(s)", nodes.len());
            }
        }
        KnowledgeAction::Pin {
            entity,
            summary,
            entity_type,
        } => {
            let mut metadata = serde_json::Map::new();
            metadata.insert("pinned".to_string(), serde_json::json!(true));
            let node = graphirm_graph::GraphNode::new(graphirm_graph::NodeType::Knowledge(
                graphirm_graph::KnowledgeData {
                    entity: entity.clone(),
                    entity_type,
                    summary,
                    confidence: 1.0,
                },
            ));
            let node = graphirm_graph::GraphNode {
                metadata: serde_json::Value::Object(metadata),
                ..node
            };
            let id = store.add_node(node)?;
            println!("Pinned: {} ({})", entity, id.0);
        }
        KnowledgeAction::Unpin { id } => {
            let node_id = graphirm_graph::NodeId(id);
            let mut node = store.get_node(&node_id)?;
            if let Some(obj) = node.metadata.as_object_mut() {
                obj.remove("pinned");
            }
            store.update_node(&node_id, node)?;
            println!("Unpinned: {}", node_id.0);
        }
    }
    Ok(())
}
