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
            segment_filter: None,
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
            segment_filter: None,
        },
        EvalTask {
            // One extracted node per name, however often the name is mentioned.
            // The text is the assistant message that produced nine nodes, three
            // of them repeats, in the March graph (docs/plans/2026-11-01-knowledge-extraction-fixes.md).
            id: "repeated-mentions".to_string(),
            name: "One knowledge node per name, linked to every message extraction read"
                .to_string(),
            tags: vec!["knowledge".to_string()],
            prompts: vec![
                "Acknowledge this summary in one sentence (no tools needed): I've added the \
                 `graphirm knowledge` CLI subcommand with three actions. The `KnowledgeAction` \
                 enum defines the three subcommands: `List` shows pinned Knowledge nodes, `Pin` \
                 creates a new pinned Knowledge node, and `Unpin` removes the pinned flag. The \
                 match arm follows the `ImportCursor` pattern: it handles `List` by calling \
                 `list_pinned_knowledge()`, handles `Pin` by creating a node with `pinned: true`, \
                 and handles `Unpin` by removing the `pinned` key."
                    .to_string(),
            ],
            verifier: Verifier::All(vec![
                Verifier::KnowledgeNodeCount { min_count: 3 },
                Verifier::KnowledgeUniqueNames,
                Verifier::KnowledgeCoversWindow { min_fraction: 1.0 },
            ]),
            max_turns: 2,
            timeout_secs: 60,
            enable_segments: false,
            segment_filter: None,
        },
        EvalTask {
            // The entity is only in the user message; the reply is one word. Its
            // derived_from links must include the user message, not just the reply.
            id: "user-only-entity".to_string(),
            name: "Entity named only by the user links to the user message".to_string(),
            tags: vec!["knowledge".to_string()],
            prompts: vec![
                "Reply with the single word OK and nothing else (no tools needed). Context for \
                 later: the `RetryBudget` struct in the agent loop caps tool retries at three \
                 per turn, and `RetryBudget::exhausted()` ends the turn with a tool error."
                    .to_string(),
            ],
            verifier: Verifier::All(vec![
                Verifier::KnowledgeNodeCount { min_count: 1 },
                Verifier::KnowledgeCoversWindow { min_fraction: 1.0 },
            ]),
            max_turns: 2,
            timeout_secs: 60,
            enable_segments: false,
            segment_filter: None,
        },
        EvalTask {
            // Same prompt as multi-entity, scored on summaries. The local GLiNER2
            // backend writes none by design, so this task is expected to fail there.
            id: "knowledge-summaries".to_string(),
            name: "Extracted knowledge nodes carry a summary".to_string(),
            tags: vec!["knowledge".to_string()],
            prompts: vec![
                "Acknowledge these facts (no tools needed): The system uses three key \
                 patterns: 1) PageRank for relevance scoring of graph nodes, \
                 2) HNSW (Hierarchical Navigable Small World) for vector similarity search, \
                 3) BFS (Breadth-First Search) for graph traversal. \
                 Each is used in the context engine to build LLM context windows."
                    .to_string(),
            ],
            verifier: Verifier::All(vec![
                Verifier::KnowledgeNodeCount { min_count: 1 },
                Verifier::KnowledgeSummaryRate { min_fraction: 0.9 },
            ]),
            max_turns: 2,
            timeout_secs: 60,
            enable_segments: false,
            segment_filter: None,
        },
    ]
}
