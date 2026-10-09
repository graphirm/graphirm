//! Stops a `graph_query` loop inside one agent run.
//!
//! An identical call returns the result already in hand. A run of different
//! calls stops after [`GRAPH_QUERY_CALL_CAP`], which is enough for list-then-bfs
//! plus one retry.

use serde_json::Value;
use std::collections::BTreeMap;

/// Distinct `graph_query` calls allowed in one agent loop.
/// The BFS eval prompt needs two (list a node, then traverse). Four leaves a retry.
pub const GRAPH_QUERY_CALL_CAP: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphQueryDecision {
    Run,
    /// Same arguments already produced `prior`.
    Repeat {
        notice: String,
    },
    Cap {
        notice: String,
    },
}

#[derive(Debug, Default)]
pub struct GraphQueryGuard {
    seen: BTreeMap<String, String>,
    calls: usize,
}

impl GraphQueryGuard {
    pub fn prepare(&mut self, args: &Value) -> GraphQueryDecision {
        let key = canonical_args(args);
        if let Some(prior) = self.seen.get(&key) {
            let mut notice = "This graph_query already ran with the same arguments. \
                 Use that result and do not call graph_query with these arguments again."
                .to_string();
            if !prior.is_empty() {
                notice.push_str("\n\nPrevious result:\n");
                notice.push_str(prior);
            }
            return GraphQueryDecision::Repeat { notice };
        }
        if self.calls >= GRAPH_QUERY_CALL_CAP {
            return GraphQueryDecision::Cap {
                notice: format!(
                    "graph_query has been called {GRAPH_QUERY_CALL_CAP} times this turn. \
                     Stop querying and answer from the results you already have."
                ),
            };
        }
        self.calls += 1;
        self.seen.insert(key, String::new());
        GraphQueryDecision::Run
    }

    pub fn remember(&mut self, args: &Value, output: &str) {
        let key = canonical_args(args);
        self.seen.insert(key, output.to_string());
    }
}

fn canonical_args(args: &Value) -> String {
    serde_json::to_string(&sort_json(args)).unwrap_or_else(|_| args.to_string())
}

fn sort_json(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut sorted = BTreeMap::new();
            for (key, child) in map {
                sorted.insert(key.clone(), sort_json(child));
            }
            Value::Object(sorted.into_iter().collect())
        }
        Value::Array(items) => Value::Array(items.iter().map(sort_json).collect()),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_repeated_graph_query_returns_the_previous_result() {
        let mut guard = GraphQueryGuard::default();
        let args = json!({"mode": "bfs", "node_id": "abc", "depth": 2});
        assert!(matches!(guard.prepare(&args), GraphQueryDecision::Run));
        guard.remember(&args, "BFS from abc");
        match guard.prepare(&args) {
            GraphQueryDecision::Repeat { notice } => {
                assert!(notice.contains("already ran"));
                assert!(notice.contains("BFS from abc"));
            }
            other => panic!("expected repeat, got {other:?}"),
        }
    }

    #[test]
    fn argument_order_does_not_make_a_new_query() {
        let mut guard = GraphQueryGuard::default();
        let first = json!({"mode": "search", "query": "test"});
        let flipped = json!({"query": "test", "mode": "search"});
        assert!(matches!(guard.prepare(&first), GraphQueryDecision::Run));
        guard.remember(&first, "0 results");
        match guard.prepare(&flipped) {
            GraphQueryDecision::Repeat { notice } => assert!(notice.contains("0 results")),
            other => panic!("expected repeat, got {other:?}"),
        }
    }

    #[test]
    fn drifting_graph_queries_stop_after_the_cap() {
        let mut guard = GraphQueryGuard::default();
        for index in 0..GRAPH_QUERY_CALL_CAP {
            let args = json!({"mode": "list_type", "node_type": "content", "n": index});
            assert!(
                matches!(guard.prepare(&args), GraphQueryDecision::Run),
                "call {index} should run"
            );
            guard.remember(&args, "ok");
        }
        let extra = json!({"mode": "list_type", "node_type": "agent"});
        match guard.prepare(&extra) {
            GraphQueryDecision::Cap { notice } => {
                assert!(notice.contains("Stop querying"));
            }
            other => panic!("expected cap, got {other:?}"),
        }
    }
}
