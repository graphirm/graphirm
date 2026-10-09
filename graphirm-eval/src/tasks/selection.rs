use crate::task::{EvalTask, Verifier};
use crate::workspace::SELECTION_TOKEN;

const READ_ORDER: &str = "Read these files in order, each exactly once: \
selection/early.txt, selection/mid_a.txt, selection/mid_b.txt, selection/late.txt. \
The token is the single word in early.txt that starts with TOKEN_EARLY_. \
Do not read early.txt again after the first read.";

pub fn tasks() -> Vec<EvalTask> {
    vec![
        EvalTask {
            id: "selection-carry-token".to_string(),
            name: "An early file token survives into a later write".to_string(),
            tags: vec!["selection".to_string()],
            prompts: vec![format!(
                "{READ_ORDER} Write selection/answer.txt now. \
                 The file must contain only that token."
            )],
            verifier: Verifier::FileContains {
                path: "selection/answer.txt".to_string(),
                substring: SELECTION_TOKEN.to_string(),
            },
            max_turns: 8,
            timeout_secs: 180,
            enable_segments: false,
            segment_filter: None,
        },
        EvalTask {
            id: "selection-edit-from-early".to_string(),
            name: "A later edit uses a token read from an earlier file".to_string(),
            tags: vec!["selection".to_string()],
            prompts: vec![format!(
                "{READ_ORDER} Edit selection/late.txt now. \
                 Replace the line LABEL=unset with LABEL= followed immediately by that token. \
                 Do not write any other file."
            )],
            verifier: Verifier::FileContains {
                path: "selection/late.txt".to_string(),
                substring: format!("LABEL={SELECTION_TOKEN}"),
            },
            max_turns: 8,
            timeout_secs: 180,
            enable_segments: false,
            segment_filter: None,
        },
    ]
}
