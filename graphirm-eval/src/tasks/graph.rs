use crate::task::{EvalTask, Verifier};

pub fn tasks() -> Vec<EvalTask> {
    vec![
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
                Verifier::GraphContains { min_nodes: 10, type_name: "Interaction".to_string() },
                Verifier::GraphContains { min_nodes: 10, type_name: "Content".to_string() },
            ]),
            max_turns: 10,
            timeout_secs: 120,
        },
    ]
}
