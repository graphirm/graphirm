use crate::task::{EvalTask, Verifier};

pub fn tasks() -> Vec<EvalTask> {
    vec![EvalTask {
        id: "segment-extraction".to_string(),
        name: "Structured response segments are stored as Content nodes".to_string(),
        tags: vec!["segments".to_string()],
        prompts: vec![
            "Write a short Rust function called `add` that takes two i32 arguments \
             and returns their sum. Show the code and explain what it does."
                .to_string(),
        ],
        verifier: Verifier::All(vec![
            // The graph must have Content nodes (created by file reads or segment extraction)
            Verifier::GraphContains {
                min_nodes: 3,
                type_name: "Content".to_string(),
            },
            // At least one Content node must be a "code" segment
            Verifier::GraphContainsContentType {
                content_type: "code".to_string(),
            },
        ]),
        max_turns: 5,
        timeout_secs: 90,
        enable_segments: true,
    }]
}
