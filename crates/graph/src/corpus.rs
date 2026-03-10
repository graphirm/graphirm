//! Corpus export for structured-LLM-response discovery.
//!
//! Produces JSONL of assistant turns (session_id, turn_index, role, text) from a GraphStore
//! for downstream GLiNER2 label exploration.

use serde::{Deserialize, Serialize};

use crate::error::GraphError;
use crate::nodes::NodeType;
use crate::store::GraphStore;

/// A single turn in the corpus: one row per assistant (or optionally all) interaction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusTurn {
    pub session_id: String,
    /// 0-based index of this turn within the session's interaction sequence.
    pub turn_index: u32,
    pub role: String,
    pub text: String,
}

/// Export assistant turns from the graph to JSONL.
///
/// Opens the graph at `db_path`, lists all sessions via Agent nodes, and for each session
/// writes one JSON object per assistant turn (session_id, turn_index, role, text).
/// Writes to `out` (e.g. a file or stdout).
pub fn export_corpus_to_jsonl(
    store: &GraphStore,
    out: &mut impl std::io::Write,
    assistant_only: bool,
) -> Result<u64, GraphError> {
    let agent_nodes = store.get_agent_nodes()?;
    let mut count: u64 = 0;

    for (node, _) in agent_nodes {
        let session_id = node.id.0.clone();
        let interactions = store.get_session_interactions(&session_id)?;

        for (turn_index, interaction) in interactions.iter().enumerate() {
            let NodeType::Interaction(ref data) = interaction.node_type else {
                continue;
            };
            if assistant_only && data.role != "assistant" {
                continue;
            }
            let turn = CorpusTurn {
                session_id: session_id.clone(),
                turn_index: turn_index as u32,
                role: data.role.clone(),
                text: data.content.clone(),
            };
            let line = serde_json::to_string(&turn).map_err(GraphError::Serde)?;
            writeln!(out, "{line}").map_err(|e| {
                GraphError::NodeNotFound(format!("write error: {e}"))
            })?;
            count += 1;
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nodes::{AgentData, GraphNode, InteractionData, NodeType};

    #[test]
    fn corpus_turn_roundtrip() {
        let turn = CorpusTurn {
            session_id: "sess-1".to_string(),
            turn_index: 2,
            role: "assistant".to_string(),
            text: "Here is the fix.\n\n```rust\nfn main() {}\n```".to_string(),
        };
        let json = serde_json::to_string(&turn).unwrap();
        let back: CorpusTurn = serde_json::from_str(&json).unwrap();
        assert_eq!(turn, back);
    }

    #[test]
    fn export_corpus_writes_assistant_turns_only() {
        let store = GraphStore::open_memory().unwrap();

        let mut agent = GraphNode::new(NodeType::Agent(AgentData {
            name: "test".to_string(),
            model: "gpt-4".to_string(),
            system_prompt: None,
            status: "active".to_string(),
        }));
        agent.metadata["session_id"] = serde_json::json!(agent.id.0);
        let agent_id = store.add_node(agent).unwrap();

        let mut u = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "Hello".to_string(),
            token_count: None,
        }));
        u.metadata["session_id"] = serde_json::json!(agent_id.0);
        store.add_node(u).unwrap();

        let mut a = GraphNode::new(NodeType::Interaction(InteractionData {
            role: "assistant".to_string(),
            content: "Hi there!".to_string(),
            token_count: None,
        }));
        a.metadata["session_id"] = serde_json::json!(agent_id.0);
        store.add_node(a).unwrap();

        let mut out = Vec::new();
        let count = export_corpus_to_jsonl(&store, &mut out, true).unwrap();
        assert_eq!(count, 1);
        let line = std::str::from_utf8(&out).unwrap().trim();
        let parsed: CorpusTurn = serde_json::from_str(line).unwrap();
        assert_eq!(parsed.role, "assistant");
        assert_eq!(parsed.text, "Hi there!");
        assert_eq!(parsed.turn_index, 1); // second interaction in session
    }
}
