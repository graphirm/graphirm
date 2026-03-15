use crate::task::{EvalTask, Verifier};

pub fn tasks() -> Vec<EvalTask> {
    vec![
        EvalTask {
            id: "grep-and-explain".to_string(),
            name: "Read store.rs and explain get_node and add_node".to_string(),
            tags: vec!["basic".to_string(), "tool-use".to_string()],
            prompts: vec![
                "Read the file `crates/graph/src/store.rs` and tell me what \
                 `get_node` and `add_node` do. Quote the function signatures."
                    .to_string(),
            ],
            verifier: Verifier::All(vec![
                Verifier::ResponseContains { substring: "get_node".to_string() },
                Verifier::ResponseContains { substring: "add_node".to_string() },
            ]),
            max_turns: 5,
            timeout_secs: 120,
            enable_segments: false,
            segment_filter: None,
        },
        EvalTask {
            id: "read-line-count".to_string(),
            name: "Count lines in workflow.rs using read tool".to_string(),
            tags: vec!["basic".to_string(), "tool-use".to_string()],
            prompts: vec![
                "Read `crates/agent/src/workflow.rs` and tell me how many lines it has."
                    .to_string(),
            ],
            // Dynamically compare against `wc -l` so the test doesn't break
            // every time the file changes.
            verifier: Verifier::ResponseContainsCommandOutput {
                command: "sh".to_string(),
                args: vec![
                    "-c".to_string(),
                    "wc -l crates/agent/src/workflow.rs | awk '{print $1}'".to_string(),
                ],
            },
            max_turns: 3,
            timeout_secs: 60,
            enable_segments: false,
            segment_filter: None,
        },
        EvalTask {
            id: "bash-line-count".to_string(),
            name: "Count lines in workflow.rs using bash".to_string(),
            tags: vec!["tool-use".to_string()],  // removed "basic" tag
            prompts: vec![
                "How many lines are in `crates/agent/src/workflow.rs`? \
                 Use bash to count them precisely."
                    .to_string(),
            ],
            verifier: Verifier::ResponseContainsCommandOutput {
                command: "sh".to_string(),
                args: vec![
                    "-c".to_string(),
                    "wc -l crates/agent/src/workflow.rs | awk '{print $1}'".to_string(),
                ],
            },
            max_turns: 3,
            timeout_secs: 90,
            enable_segments: false,
            segment_filter: None,
        },
        EvalTask {
            id: "write-fibonacci".to_string(),
            name: "Write a Fibonacci function to a file".to_string(),
            tags: vec!["coding".to_string()],
            prompts: vec![
                "Write a Rust function `fn fibonacci(n: u64) -> u64` to the file \
                 `/tmp/eval_fib.rs`. The function should return the nth Fibonacci number \
                 using recursion with memoization."
                    .to_string(),
            ],
            verifier: Verifier::All(vec![
                Verifier::FileContains {
                    path: "/tmp/eval_fib.rs".to_string(),
                    substring: "fn fibonacci".to_string(),
                },
                Verifier::FileContains {
                    path: "/tmp/eval_fib.rs".to_string(),
                    substring: "u64".to_string(),
                },
            ]),
            max_turns: 5,
            timeout_secs: 120,
            enable_segments: false,
            segment_filter: None,
        },
        EvalTask {
            id: "multi-turn-read-write".to_string(),
            name: "Write a file then read it back in next turn".to_string(),
            tags: vec!["multi-turn".to_string()],
            prompts: vec![
                "Write the string 'EVAL_MARKER_9a3f' to the file `/tmp/eval_marker.txt`.".to_string(),
                "Read the file `/tmp/eval_marker.txt` and tell me its contents.".to_string(),
            ],
            verifier: Verifier::ResponseContains {
                substring: "EVAL_MARKER_9a3f".to_string(),
            },
            max_turns: 3,
            timeout_secs: 90,
            enable_segments: false,
            segment_filter: None,
        },
    ]
}
