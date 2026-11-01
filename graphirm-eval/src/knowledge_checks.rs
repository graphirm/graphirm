//! Quality checks on a session's Knowledge nodes: repeated names, empty summaries,
//! and whether each node's `derived_from` message actually mentions it.
//! Pure functions over the JSON the server returns, so they are unit-tested here
//! and called from `TestHarness::check_verifier`.

use std::collections::{HashMap, HashSet};

use serde_json::Value;

use crate::client::GraphResponse;

/// Lowercase letters and digits only, so `/api/health` and `API Health` match.
pub fn name_key(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

pub fn entity(node: &Value) -> &str {
    node["node_type"]["entity"].as_str().unwrap_or("")
}

fn summary(node: &Value) -> &str {
    node["node_type"]["summary"].as_str().unwrap_or("")
}

/// `GET /api/graph/{session}/knowledge` lists a node once per message it derives from,
/// so a node linked to two messages appears twice. Keep the first copy of each id.
pub fn distinct(nodes: &[Value]) -> Vec<&Value> {
    let mut seen = HashSet::new();
    nodes
        .iter()
        .filter(|node| seen.insert(node["id"].as_str().unwrap_or("").to_string()))
        .collect()
}

/// Names shared by more than one distinct node, sorted.
pub fn duplicate_names(nodes: &[&Value]) -> Vec<String> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for node in nodes {
        *counts.entry(name_key(entity(node))).or_default() += 1;
    }
    let mut names: Vec<String> = counts
        .into_iter()
        .filter(|(key, count)| !key.is_empty() && *count > 1)
        .map(|(key, _)| key)
        .collect();
    names.sort();
    names
}

/// Share of nodes with a non-empty summary; `None` when there are no nodes.
pub fn summary_rate(nodes: &[&Value]) -> Option<f64> {
    if nodes.is_empty() {
        return None;
    }
    let with = nodes
        .iter()
        .filter(|node| !summary(node).trim().is_empty())
        .count();
    Some(with as f64 / nodes.len() as f64)
}

/// A node is grounded when at least one of its `derived_from` targets is an
/// Interaction whose content contains the entity name (case-insensitive).
/// `subgraph` is the depth-1 subgraph rooted at the knowledge node.
pub fn is_grounded(knowledge_id: &str, name: &str, subgraph: &GraphResponse) -> bool {
    let needle = name.to_lowercase();
    if needle.trim().is_empty() {
        return false;
    }
    let targets: HashSet<&str> = subgraph
        .edges
        .iter()
        .filter(|edge| {
            edge["edge_type"].as_str() == Some("derived_from")
                && edge["source"].as_str() == Some(knowledge_id)
        })
        .filter_map(|edge| edge["target"].as_str())
        .collect();
    subgraph.nodes.iter().any(|node| {
        node["id"].as_str().is_some_and(|id| targets.contains(id))
            && node["node_type"]["content"]
                .as_str()
                .is_some_and(|content| content.to_lowercase().contains(&needle))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn knowledge(id: &str, entity: &str, summary: &str) -> Value {
        json!({ "id": id, "node_type": { "type": "Knowledge", "entity": entity, "entity_type": "function", "summary": summary, "confidence": 0.8 } })
    }

    #[test]
    fn distinct_drops_repeated_ids_and_keeps_order() {
        let nodes = vec![
            knowledge("a", "Pin", ""),
            knowledge("b", "Unpin", ""),
            knowledge("a", "Pin", ""),
        ];
        let ids: Vec<&str> = distinct(&nodes)
            .iter()
            .map(|n| n["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["a", "b"]);
    }

    #[test]
    fn duplicate_names_uses_the_normalised_name() {
        let nodes = [
            knowledge("a", "Pin", ""),
            knowledge("b", "pin", ""),
            knowledge("c", "/api/health", ""),
            knowledge("d", "API health", ""),
            knowledge("e", "List", ""),
        ];
        let refs: Vec<&Value> = nodes.iter().collect();
        assert_eq!(duplicate_names(&refs), ["apihealth", "pin"]);
    }

    #[test]
    fn summary_rate_counts_non_blank_summaries() {
        let nodes = [
            knowledge("a", "Pin", "creates a pinned node"),
            knowledge("b", "Unpin", "  "),
            knowledge("c", "List", ""),
        ];
        let refs: Vec<&Value> = nodes.iter().collect();
        assert!((summary_rate(&refs).unwrap() - 1.0 / 3.0).abs() < 1e-9);
        assert_eq!(summary_rate(&[]), None);
    }

    #[test]
    fn grounded_needs_a_derived_from_target_that_mentions_the_name() {
        let subgraph = GraphResponse {
            nodes: vec![
                knowledge("k", "Pin", ""),
                json!({ "id": "m1", "node_type": { "type": "Interaction", "role": "assistant", "content": "Added `/api/health` session_count" } }),
                json!({ "id": "m2", "node_type": { "type": "Interaction", "role": "user", "content": "The `Pin` subcommand pins a node" } }),
                json!({ "id": "k2", "node_type": { "type": "Knowledge", "entity": "Pin" } }),
            ],
            edges: vec![
                json!({ "edge_type": "derived_from", "source": "k", "target": "m1" }),
                json!({ "edge_type": "relates_to", "source": "k", "target": "m2" }),
            ],
        };
        assert!(!is_grounded("k", "Pin", &subgraph));
        let mut linked = subgraph;
        linked
            .edges
            .push(json!({ "edge_type": "derived_from", "source": "k", "target": "m2" }));
        assert!(is_grounded("k", "pin", &linked));
        assert!(!is_grounded("k", "", &linked));
    }
}
