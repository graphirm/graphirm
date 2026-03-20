use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;

use graphirm_graph::GraphStore;
use graphirm_tools::impact::{ImpactBrief, ImpactProvider, KnowledgeNote, compute_risk};

/// GraphImpactProvider implements ImpactProvider by querying ripgrep for dependents
/// and the graph for related Knowledge nodes.
pub struct GraphImpactProvider {
    graph: Arc<GraphStore>,
    workspace: PathBuf,
}

impl GraphImpactProvider {
    /// Create a new GraphImpactProvider.
    pub fn new(graph: Arc<GraphStore>, workspace: PathBuf) -> Self {
        Self { graph, workspace }
    }

    /// Count how many files depend on the target path using ripgrep.
    ///
    /// Returns `Some(count)` if rg is available, `None` if rg is not found or fails.
    /// Filters with `--glob !.git --glob !target --glob !node_modules` to keep searches fast.
    pub async fn count_dependents(&self, path: &PathBuf) -> Option<usize> {
        // Get filename for ripgrep search
        let file_stem = path.file_stem()?.to_string_lossy().to_string();

        let output = tokio::process::Command::new("rg")
            .args(&[
                "--files-with-matches",
                "--no-messages",
                "--glob", "!.git",
                "--glob", "!target",
                "--glob", "!node_modules",
                &file_stem,
            ])
            .current_dir(&self.workspace)
            .output()
            .await
            .ok()?;

        if output.status.success() {
            let count = output.stdout.iter().filter(|&&b| b == b'\n').count();
            Some(count)
        } else {
            None
        }
    }

    /// Find Knowledge notes mentioning the target path from OTHER sessions.
    ///
    /// Matches by file stem (case-insensitive) in entity or summary.
    /// Skips Knowledge nodes from the current session.
    /// Returns at most 5 notes to keep briefs concise.
    pub fn find_knowledge_notes(&self, path: &PathBuf, session_id: &str) -> Vec<KnowledgeNote> {
        let file_stem = match path.file_stem() {
            Some(stem) => stem.to_string_lossy().to_string().to_lowercase(),
            None => return Vec::new(),
        };

        // Query for Knowledge nodes matching the path stem, excluding current session
        let search_result = self.graph.search_knowledge(
            &file_stem,
            None,     // no entity_type filter
            None,     // no session_id filter — we'll filter manually
            1000,     // get many to filter
        );

        let nodes = match search_result {
            Ok(nodes) => nodes,
            Err(_) => return Vec::new(),
        };

        let mut notes = Vec::new();
        for node in nodes {
            // Skip nodes from the current session
            if let Some(node_session_id) = node.metadata.get("session_id").and_then(|v| v.as_str()) {
                if node_session_id == session_id {
                    continue;
                }
            }

            // Extract turn number from metadata
            let turn = node.metadata
                .get("turn")
                .and_then(|v| v.as_u64())
                .map(|t| t as u32)
                .unwrap_or(0);

            // Get text from the node's Knowledge variant
            let text = if let graphirm_graph::nodes::NodeType::Knowledge(kd) = &node.node_type {
                // Use entity if present, otherwise use summary
                if !kd.entity.is_empty() {
                    kd.entity.clone()
                } else {
                    kd.summary.clone()
                }
            } else {
                continue;
            };

            notes.push(KnowledgeNote { text, turn });

            if notes.len() >= 5 {
                break;
            }
        }

        notes
    }
}

#[async_trait]
impl ImpactProvider for GraphImpactProvider {
    /// Analyze the impact of modifying the given paths.
    /// Returns a Vec of ImpactBrief, skipping empty briefs (no dependents and no knowledge notes).
    async fn analyze(&self, paths: &[PathBuf]) -> Result<Vec<ImpactBrief>, String> {
        let mut briefs = Vec::new();

        // Placeholder session_id for now — in real usage this would come from context
        let session_id = "placeholder-session";

        for path in paths {
            let dependent_count = self.count_dependents(path).await;
            let knowledge_notes = self.find_knowledge_notes(path, session_id);

            // Compute risk
            let risk = compute_risk(dependent_count, !knowledge_notes.is_empty());

            let brief = ImpactBrief {
                path: path.clone(),
                dependent_count,
                knowledge_notes,
                risk,
            };

            // Skip empty briefs via threshold gate
            if !brief.is_empty() {
                briefs.push(brief);
            }
        }

        Ok(briefs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphirm_graph::nodes::NodeType;

    /// Test 1: find_knowledge_notes matches by file stem.
    #[test]
    fn find_knowledge_notes_matches_by_stem() {
        let store = Arc::new(GraphStore::open_memory().expect("Failed to open memory graph"));

        // Create a Knowledge node from "other-session" mentioning "tokens.rs"
        let knowledge_node = graphirm_graph::nodes::GraphNode {
            id: graphirm_graph::nodes::NodeId("know-1".to_string()),
            node_type: NodeType::Knowledge(graphirm_graph::nodes::KnowledgeData {
                entity: "tokens.rs".to_string(),
                entity_type: "file".to_string(),
                summary: "Authentication token handler".to_string(),
                confidence: 0.95,
            }),
            metadata: serde_json::json!({"session_id": "other-session", "turn": 1}),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        store.add_node(knowledge_node).expect("Failed to add node");

        let provider = GraphImpactProvider::new(store, PathBuf::from("."));

        // Query for src/auth/tokens.rs from "current-session"
        let notes = provider.find_knowledge_notes(&PathBuf::from("src/auth/tokens.rs"), "current-session");

        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].text, "tokens.rs");
        assert_eq!(notes[0].turn, 1);
    }

    /// Test 2: find_knowledge_notes skips the current session.
    #[test]
    fn find_knowledge_notes_skips_current_session() {
        let store = Arc::new(GraphStore::open_memory().expect("Failed to open memory graph"));

        // Create a Knowledge node with current_session id
        let knowledge_node = graphirm_graph::nodes::GraphNode {
            id: graphirm_graph::nodes::NodeId("know-2".to_string()),
            node_type: NodeType::Knowledge(graphirm_graph::nodes::KnowledgeData {
                entity: "tokens.rs".to_string(),
                entity_type: "file".to_string(),
                summary: "Token handler".to_string(),
                confidence: 0.85,
            }),
            metadata: serde_json::json!({"session_id": "current-session", "turn": 1}),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        store.add_node(knowledge_node).expect("Failed to add node");

        let provider = GraphImpactProvider::new(store, PathBuf::from("."));

        // Query from same session
        let notes = provider.find_knowledge_notes(&PathBuf::from("src/auth/tokens.rs"), "current-session");

        // Should be empty
        assert_eq!(notes.len(), 0);
    }

    /// Test 3: find_knowledge_notes returns empty when no match.
    #[test]
    fn find_knowledge_notes_returns_empty_on_no_match() {
        let store = Arc::new(GraphStore::open_memory().expect("Failed to open memory graph"));

        // Create a Knowledge node about "database"
        let knowledge_node = graphirm_graph::nodes::GraphNode {
            id: graphirm_graph::nodes::NodeId("know-3".to_string()),
            node_type: NodeType::Knowledge(graphirm_graph::nodes::KnowledgeData {
                entity: "database".to_string(),
                entity_type: "component".to_string(),
                summary: "PostgreSQL integration".to_string(),
                confidence: 0.90,
            }),
            metadata: serde_json::json!({"session_id": "other-session", "turn": 2}),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        store.add_node(knowledge_node).expect("Failed to add node");

        let provider = GraphImpactProvider::new(store, PathBuf::from("."));

        // Query for unrelated path
        let notes = provider.find_knowledge_notes(&PathBuf::from("src/auth.rs"), "current-session");

        // Should be empty
        assert_eq!(notes.len(), 0);
    }

    /// Test 4: analyze skips empty briefs (threshold gate).
    #[test]
    fn analyze_skips_empty_briefs() {
        let store = Arc::new(GraphStore::open_memory().expect("Failed to open memory graph"));
        // Use a nonexistent workspace so rg fails quickly
        let provider = GraphImpactProvider::new(store, PathBuf::from("/nonexistent/workspace"));

        let rt = tokio::runtime::Runtime::new().expect("Failed to create runtime");
        let briefs = rt.block_on(async {
            // Nonexistent path, no Knowledge nodes
            provider.analyze(&[PathBuf::from("/nonexistent/path.rs")]).await
        }).expect("analyze failed");

        // Should be empty due to threshold gate
        assert_eq!(briefs.len(), 0);
    }
}
