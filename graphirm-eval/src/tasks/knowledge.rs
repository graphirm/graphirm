use crate::task::{EvalTask, Verifier};

pub fn tasks() -> Vec<EvalTask> {
    vec![
        EvalTask {
            id: "entity-recall".to_string(),
            name: "Knowledge node created after factual statement".to_string(),
            tags: vec!["knowledge".to_string()],
            prompts: vec![
                // Explicit "just acknowledge" phrasing prevents the agent from
                // deciding to verify the statement by reading files with tools.
                "Acknowledge this fact (no tools needed): Rust uses ownership and the \
                 borrow checker to guarantee memory safety at compile time without a \
                 garbage collector. The borrow checker is a static analysis pass in rustc."
                    .to_string(),
            ],
            verifier: Verifier::KnowledgeNodeCount { min_count: 1 },
            max_turns: 2,
            timeout_secs: 60,
            enable_segments: false,
        },
        EvalTask {
            id: "multi-entity".to_string(),
            name: "Multiple knowledge nodes from code-heavy message".to_string(),
            tags: vec!["knowledge".to_string()],
            prompts: vec![
                // Explicit "just acknowledge" phrasing prevents the agent from
                // deciding to verify the statement by reading files with tools.
                "Acknowledge these facts (no tools needed): The system uses three key \
                 patterns: 1) PageRank for relevance scoring of graph nodes, \
                 2) HNSW (Hierarchical Navigable Small World) for vector similarity search, \
                 3) BFS (Breadth-First Search) for graph traversal. \
                 Each is used in the context engine to build LLM context windows."
                    .to_string(),
            ],
            verifier: Verifier::KnowledgeNodeCount { min_count: 3 },
            max_turns: 2,
            timeout_secs: 60,
            enable_segments: false,
        },
    ]
}
