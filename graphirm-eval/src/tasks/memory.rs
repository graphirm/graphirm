#![allow(dead_code)]

use crate::task::{CrossSessionTask, Verifier};

pub fn cross_session_tasks() -> Vec<CrossSessionTask> {
    vec![
        CrossSessionTask {
            id: "single-fact-recall".to_string(),
            name: "Agent recalls unique fact from previous session".to_string(),
            tags: vec!["memory".to_string()],
            session1_prompt: "Remember this secret code for later: EVAL_SECRET_xk7q2".to_string(),
            session2_prompt: "What was the secret code I mentioned?".to_string(),
            verifier: Verifier::ResponseContains { substring: "xk7q2".to_string() },
            timeout_secs: 60,
        },
    ]
}
