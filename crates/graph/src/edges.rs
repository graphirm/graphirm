use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::nodes::NodeId;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EdgeId(pub String);

impl EdgeId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

impl Default for EdgeId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for EdgeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for EdgeId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeType {
    RespondsTo,
    SpawnedBy,
    DelegatesTo,
    DependsOn,
    Produces,
    Reads,
    Modifies,
    Summarizes,
    Contains,
    FollowsUp,
    Steers,
    RelatesTo,
    /// Links a knowledge node back to the interaction it was derived from.
    DerivedFrom,
}

impl EdgeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            EdgeType::RespondsTo => "responds_to",
            EdgeType::SpawnedBy => "spawned_by",
            EdgeType::DelegatesTo => "delegates_to",
            EdgeType::DependsOn => "depends_on",
            EdgeType::Produces => "produces",
            EdgeType::Reads => "reads",
            EdgeType::Modifies => "modifies",
            EdgeType::Summarizes => "summarizes",
            EdgeType::Contains => "contains",
            EdgeType::FollowsUp => "follows_up",
            EdgeType::Steers => "steers",
            EdgeType::RelatesTo => "relates_to",
            EdgeType::DerivedFrom => "derived_from",
        }
    }

    pub fn all() -> &'static [EdgeType] {
        &[
            EdgeType::RespondsTo,
            EdgeType::SpawnedBy,
            EdgeType::DelegatesTo,
            EdgeType::DependsOn,
            EdgeType::Produces,
            EdgeType::Reads,
            EdgeType::Modifies,
            EdgeType::Summarizes,
            EdgeType::Contains,
            EdgeType::FollowsUp,
            EdgeType::Steers,
            EdgeType::RelatesTo,
            EdgeType::DerivedFrom,
        ]
    }
}

impl fmt::Display for EdgeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub id: EdgeId,
    pub edge_type: EdgeType,
    pub source: NodeId,
    pub target: NodeId,
    pub weight: f64,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
}

impl GraphEdge {
    pub fn new(edge_type: EdgeType, source: NodeId, target: NodeId) -> Self {
        Self {
            id: EdgeId::new(),
            edge_type,
            source,
            target,
            weight: 1.0,
            metadata: serde_json::Value::Object(serde_json::Map::new()),
            created_at: Utc::now(),
        }
    }

    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_id_display() {
        let id = EdgeId("edge-456".to_string());
        assert_eq!(id.to_string(), "edge-456");
    }

    #[test]
    fn edge_id_serde_roundtrip() {
        let id = EdgeId("edge-rt".to_string());
        let json = serde_json::to_string(&id).unwrap();
        let back: EdgeId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, back);
    }

    #[test]
    fn edge_type_display() {
        assert_eq!(EdgeType::RespondsTo.to_string(), "responds_to");
        assert_eq!(EdgeType::DelegatesTo.to_string(), "delegates_to");
        assert_eq!(EdgeType::FollowsUp.to_string(), "follows_up");
    }

    #[test]
    fn edge_type_serde_roundtrip() {
        for et in EdgeType::all() {
            let json = serde_json::to_string(et).unwrap();
            let back: EdgeType = serde_json::from_str(&json).unwrap();
            assert_eq!(*et, back);
        }
    }

    #[test]
    fn edge_type_all_has_thirteen_variants() {
        assert_eq!(EdgeType::all().len(), 13);
    }

    #[test]
    fn graph_edge_new_defaults() {
        let src = NodeId::from("src");
        let tgt = NodeId::from("tgt");
        let edge = GraphEdge::new(EdgeType::RespondsTo, src.clone(), tgt.clone());
        assert_eq!(edge.source, src);
        assert_eq!(edge.target, tgt);
        assert_eq!(edge.edge_type, EdgeType::RespondsTo);
        assert!((edge.weight - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn graph_edge_with_weight() {
        let edge = GraphEdge::new(EdgeType::DependsOn, NodeId::from("a"), NodeId::from("b"))
            .with_weight(0.75);
        assert!((edge.weight - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn graph_edge_serde_roundtrip() {
        let edge = GraphEdge::new(
            EdgeType::Produces,
            NodeId::from("agent-1"),
            NodeId::from("content-1"),
        )
        .with_weight(0.9)
        .with_metadata(serde_json::json!({"tool": "bash"}));

        let json = serde_json::to_string(&edge).unwrap();
        let back: GraphEdge = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, edge.id);
        assert_eq!(back.edge_type, EdgeType::Produces);
        assert_eq!(back.source.0, "agent-1");
        assert_eq!(back.target.0, "content-1");
        assert!((back.weight - 0.9).abs() < f64::EPSILON);
        assert_eq!(back.metadata["tool"], "bash");
    }
}
