use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EdgeId(pub String);

impl EdgeId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
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
}
