// Context telemetry: statistics about graph context utilization

use serde::{Deserialize, Serialize};

/// Statistics about context utilization from the graph.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ContextStats {
    /// Number of Knowledge nodes included in context
    pub knowledge_count: u32,
    /// Number of cross-session links surfaced
    pub cross_session_links_count: u32,
    /// Number of pinned convention nodes included
    pub pinned_conventions_count: u32,
    /// Percentage of context budget used by graph data (0.0-100.0)
    pub graph_token_percentage: f64,
    /// Was repo briefing injected
    pub repo_briefing_included: bool,
    /// Was context compaction triggered this turn
    pub compaction_triggered: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_stats_creation_with_all_fields() {
        let stats = ContextStats {
            knowledge_count: 5,
            cross_session_links_count: 2,
            pinned_conventions_count: 3,
            graph_token_percentage: 45.5,
            repo_briefing_included: true,
            compaction_triggered: false,
        };

        assert_eq!(stats.knowledge_count, 5);
        assert_eq!(stats.cross_session_links_count, 2);
        assert_eq!(stats.pinned_conventions_count, 3);
        assert!((stats.graph_token_percentage - 45.5).abs() < f64::EPSILON);
        assert!(stats.repo_briefing_included);
        assert!(!stats.compaction_triggered);
    }

    #[test]
    fn context_stats_serialize_to_json() {
        let stats = ContextStats {
            knowledge_count: 5,
            cross_session_links_count: 2,
            pinned_conventions_count: 3,
            graph_token_percentage: 45.5,
            repo_briefing_included: true,
            compaction_triggered: false,
        };

        let json = serde_json::to_string(&stats).unwrap();
        let parsed: ContextStats = serde_json::from_str(&json).unwrap();

        assert_eq!(stats.knowledge_count, parsed.knowledge_count);
        assert_eq!(
            stats.cross_session_links_count,
            parsed.cross_session_links_count
        );
        assert_eq!(
            stats.pinned_conventions_count,
            parsed.pinned_conventions_count
        );
        assert!(
            (stats.graph_token_percentage - parsed.graph_token_percentage).abs() < f64::EPSILON
        );
        assert_eq!(stats.repo_briefing_included, parsed.repo_briefing_included);
        assert_eq!(stats.compaction_triggered, parsed.compaction_triggered);
    }

    #[test]
    fn context_stats_default_values() {
        let stats = ContextStats::default();

        assert_eq!(stats.knowledge_count, 0);
        assert_eq!(stats.cross_session_links_count, 0);
        assert_eq!(stats.pinned_conventions_count, 0);
        assert!((stats.graph_token_percentage - 0.0).abs() < f64::EPSILON);
        assert!(!stats.repo_briefing_included);
        assert!(!stats.compaction_triggered);
    }

    #[test]
    fn context_stats_field_access() {
        let stats = ContextStats {
            knowledge_count: 10,
            cross_session_links_count: 5,
            pinned_conventions_count: 2,
            graph_token_percentage: 75.3,
            repo_briefing_included: true,
            compaction_triggered: true,
        };

        assert_eq!(stats.knowledge_count, 10);
        assert_eq!(stats.cross_session_links_count, 5);
        assert_eq!(stats.pinned_conventions_count, 2);
        assert!((stats.graph_token_percentage - 75.3).abs() < f64::EPSILON);
        assert!(stats.repo_briefing_included);
        assert!(stats.compaction_triggered);
    }
}
