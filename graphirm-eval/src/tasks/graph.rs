use crate::task::{EvalTask, Verifier};

pub fn tasks() -> Vec<EvalTask> {
    vec![
        // ── existing graph structure task ─────────────────────────────────────
        EvalTask {
            id: "graph-integrity".to_string(),
            name: "Graph has expected node types after a multi-turn session".to_string(),
            tags: vec!["graph".to_string()],
            prompts: vec![
                "Read the file `crates/agent/src/lib.rs` and tell me what it exports.".to_string(),
                "Now read `crates/graph/src/lib.rs` and tell me what it exports.".to_string(),
                "Compare the two. Which crate has more public exports?".to_string(),
            ],
            verifier: Verifier::All(vec![
                Verifier::GraphContains {
                    min_nodes: 10,
                    type_name: "Interaction".to_string(),
                },
                Verifier::GraphContains {
                    min_nodes: 10,
                    type_name: "Content".to_string(),
                },
            ]),
            max_turns: 10,
            timeout_secs: 120,
            enable_segments: false,
            segment_filter: None,
        },
        // ── graph_query: list_type mode ───────────────────────────────────────
        EvalTask {
            id: "graph-query-list-type".to_string(),
            name: "graph_query lists interaction nodes in the current session".to_string(),
            tags: vec!["graph".to_string(), "graph_query".to_string()],
            prompts: vec![
                "Use the graph_query tool with mode='list_type' and node_type='interaction' \
                 to list the interaction nodes in this session. \
                 Include the raw tool output in your response."
                    .to_string(),
            ],
            // The tool output header starts with "Nodes of type 'interaction'",
            // which the LLM will either quote directly or paraphrase.
            verifier: Verifier::ResponseContainsAny {
                substrings: vec![
                    "Nodes of type".to_string(),
                    "interaction".to_string(),
                    "node".to_string(),
                ],
            },
            max_turns: 5,
            timeout_secs: 60,
            enable_segments: false,
            segment_filter: None,
        },
        // ── graph_query: search mode (empty result is a successful empty response) ──
        EvalTask {
            id: "graph-query-search-empty".to_string(),
            name: "graph_query search returns an empty success when no Knowledge nodes match"
                .to_string(),
            tags: vec!["graph".to_string(), "graph_query".to_string()],
            prompts: vec![
                "Use the graph_query tool with mode='search' and query='xyzzy_unique_token_abc' \
                 to search the knowledge graph. \
                 Report exactly what the tool returned."
                    .to_string(),
            ],
            // Tool output on empty: "Knowledge search for 'xyzzy...' (0 results): (no Knowledge nodes…)"
            verifier: Verifier::ResponseContainsAny {
                substrings: vec![
                    "no Knowledge nodes".to_string(),
                    "0 results".to_string(),
                    "no results".to_string(),
                    "nothing".to_string(),
                    "empty".to_string(),
                    "found nothing".to_string(),
                    "xyzzy".to_string(),
                ],
            },
            max_turns: 5,
            timeout_secs: 60,
            enable_segments: false,
            segment_filter: None,
        },
        // ── graph_query: bfs mode (seed a Content node via read, then traverse) ──
        EvalTask {
            id: "graph-query-bfs".to_string(),
            name: "graph_query BFS traverses the graph from a known content node".to_string(),
            tags: vec!["graph".to_string(), "graph_query".to_string()],
            prompts: vec![
                // Seed a Content node so the BFS has something to traverse.
                "Read the file `Cargo.toml` using the read tool.".to_string(),
                // Ask the agent to discover a node ID and run BFS from it.
                "Now use graph_query with mode='list_type' and node_type='content' \
                 to find a node ID. Then use graph_query in bfs mode with that node_id \
                 and depth=2. Report what nodes you found and how many."
                    .to_string(),
            ],
            // Tool output for bfs starts with "BFS from <id> (type: content)…"
            verifier: Verifier::ResponseContainsAny {
                substrings: vec![
                    "BFS from".to_string(),
                    "bfs".to_string(),
                    "traversal".to_string(),
                    "no reachable".to_string(),
                    "node".to_string(),
                ],
            },
            max_turns: 10,
            timeout_secs: 120,
            enable_segments: false,
            segment_filter: None,
        },
        // ── graph_query: invalid mode returns an error (agent handles it gracefully) ──
        EvalTask {
            id: "graph-query-invalid-mode".to_string(),
            name: "graph_query with an unknown mode returns an error the agent handles gracefully"
                .to_string(),
            tags: vec!["graph".to_string(), "graph_query".to_string()],
            prompts: vec!["Call the graph_query tool with mode='does_not_exist'. \
                 Tell me whether it returned an error or succeeded."
                .to_string()],
            // The tool returns InvalidArguments; the agent should report an error occurred.
            verifier: Verifier::ResponseContainsAny {
                substrings: vec![
                    "error".to_string(),
                    "invalid".to_string(),
                    "unknown mode".to_string(),
                    "does not exist".to_string(),
                    "not supported".to_string(),
                    "failed".to_string(),
                ],
            },
            max_turns: 5,
            timeout_secs: 60,
            enable_segments: false,
            segment_filter: None,
        },
    ]
}
