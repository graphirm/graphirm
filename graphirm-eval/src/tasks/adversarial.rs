//! Adversarial eval tasks — failure modes, edge cases, and stress tests.
//!
//! Unlike the basic suite (happy path, clear task, one shot), these tasks probe
//! the agent's robustness under realistic difficult conditions:
//!
//! 1. `missing-file-resilience`   — agent must report error, not hallucinate content
//! 2. `precise-edit-no-collateral`— edit one function without corrupting others
//! 3. `grep-exact-count`          — precise counting against dynamic ground truth
//! 4. `cascading-pipeline`        — 3-turn write→script→run with full context retention
//! 5. `fix-broken-script`         — diagnose a broken script, fix it, then run it

use crate::task::{EvalTask, Verifier};

pub fn tasks() -> Vec<EvalTask> {
    vec![
        // ── 1. Missing file resilience ────────────────────────────────────────
        // A well-behaved agent reads the file, gets a "not found" error from
        // the tool, and reports that error to the user. A failing agent either
        // hallucinates plausible file contents or silently ignores the error.
        EvalTask {
            id: "missing-file-resilience".to_string(),
            name: "Agent must report missing file, not hallucinate".to_string(),
            tags: vec!["adversarial".to_string(), "robustness".to_string()],
            prompts: vec![
                "Read the file `/tmp/eval_nonexistent_xk9z7q.txt` and summarize \
                 its contents."
                    .to_string(),
            ],
            verifier: Verifier::All(vec![
                // Must acknowledge the file doesn't exist in some way
                Verifier::ResponseContainsAny {
                    substrings: vec![
                        "not found".to_string(),
                        "does not exist".to_string(),
                        "doesn't exist".to_string(),
                        "no such file".to_string(),
                        "cannot read".to_string(),
                        "could not read".to_string(),
                        "failed to read".to_string(),
                        "unable to read".to_string(),
                        "error".to_string(),
                    ],
                },
                // Must NOT claim to have read any content
                Verifier::ResponseNotContains {
                    substring: "contents of".to_string(),
                },
            ]),
            max_turns: 3,
            timeout_secs: 60,
            enable_segments: false,
            segment_filter: None,
        },

        // ── 2. Precise edit — no collateral damage ────────────────────────────
        // The agent must modify exactly one function in a multi-function file.
        // Common failure: agent re-writes the whole file and subtly changes or
        // drops one of the untouched functions.
        EvalTask {
            id: "precise-edit-no-collateral".to_string(),
            name: "Edit one function without corrupting the others".to_string(),
            tags: vec!["adversarial".to_string(), "tool-use".to_string()],
            prompts: vec![
                // Turn 1: write a known file
                "Write the following Python code to `/tmp/eval_functions.py` \
                 exactly as shown:\n\n\
                 ```python\n\
                 def add(a, b):\n    return a + b\n\n\
                 def multiply(a, b):\n    return a * b\n\n\
                 def subtract(a, b):\n    return a - b\n\
                 ```"
                    .to_string(),
                // Turn 2: targeted edit
                "Edit `/tmp/eval_functions.py` to change `multiply` so it returns \
                 `a * b * 2` instead of `a * b`. Do not change `add` or `subtract`."
                    .to_string(),
            ],
            verifier: Verifier::All(vec![
                // The targeted change was made
                Verifier::FileContains {
                    path: "/tmp/eval_functions.py".to_string(),
                    substring: "a * b * 2".to_string(),
                },
                // The other functions are intact
                Verifier::FileContains {
                    path: "/tmp/eval_functions.py".to_string(),
                    substring: "return a + b".to_string(),
                },
                Verifier::FileContains {
                    path: "/tmp/eval_functions.py".to_string(),
                    substring: "return a - b".to_string(),
                },
            ]),
            max_turns: 5,
            timeout_secs: 120,
            enable_segments: false,
            segment_filter: None,
        },

        // ── 3. Grep exact count ───────────────────────────────────────────────
        // Ask the agent to count how many times a specific token appears in the
        // source tree. Verifier runs the same command and compares dynamically —
        // the count changes as we add spawn_blocking calls, so no hardcoding.
        // Failure mode: agent answers from "knowledge" without running bash,
        // producing a plausible-but-wrong number. The prompt explicitly requires
        // the agent to run the command and quote its output.
        EvalTask {
            id: "grep-exact-count".to_string(),
            name: "Count spawn_blocking occurrences precisely across all Rust files".to_string(),
            tags: vec!["adversarial".to_string(), "tool-use".to_string()],
            prompts: vec![
                "Run this exact bash command and tell me the number it prints:\n\n\
                 ```bash\n\
                 grep -r --include='*.rs' -c spawn_blocking crates/ src/ \
                 | awk -F: '{sum+=$2} END {print sum}'\n\
                 ```\n\n\
                 Do not estimate — run the command and quote the output."
                    .to_string(),
            ],
            verifier: Verifier::ResponseContainsCommandOutput {
                command: "sh".to_string(),
                args: vec![
                    "-c".to_string(),
                    "grep -r --include='*.rs' -c spawn_blocking crates/ src/ \
                     | awk -F: '{sum+=$2} END {print sum}'"
                        .to_string(),
                ],
            },
            max_turns: 3,
            timeout_secs: 60,
            enable_segments: false,
            segment_filter: None,
        },

        // ── 4. Cascading pipeline ─────────────────────────────────────────────
        // Three-turn chain: write a value, write a script that reads that value,
        // run the script. Tests that the agent retains full context across turns
        // and can produce executable shell scripts that compose correctly.
        // Failure modes: script has a bug, agent forgets the value from turn 1,
        // agent confuses the file paths between turns.
        EvalTask {
            id: "cascading-pipeline".to_string(),
            name: "3-turn write→script→run pipeline".to_string(),
            tags: vec!["adversarial".to_string(), "multi-turn".to_string()],
            prompts: vec![
                "Write the text `SECRET_VALUE=turquoise42` to `/tmp/eval_secret.txt`.".to_string(),
                "Write a bash script to `/tmp/eval_reader.sh` that reads \
                 `/tmp/eval_secret.txt` and prints its contents prefixed with `FOUND:`. \
                 Make the script executable."
                    .to_string(),
                "Run `/tmp/eval_reader.sh` and tell me exactly what it prints."
                    .to_string(),
            ],
            verifier: Verifier::All(vec![
                Verifier::ResponseContains {
                    substring: "FOUND:".to_string(),
                },
                Verifier::ResponseContains {
                    substring: "turquoise42".to_string(),
                },
            ]),
            max_turns: 3,
            timeout_secs: 120,
            enable_segments: false,
            segment_filter: None,
        },

        // ── 5. Fix broken script ──────────────────────────────────────────────
        // The agent is handed a syntactically broken Python script. It must:
        //   a) read or run it to discover the error
        //   b) fix the error in the file
        //   c) run it successfully
        // Verifier runs the script from the harness side — it must exit 0.
        // Failure modes: agent doesn't actually fix the file (just describes the
        // fix), fixes one error and introduces another, or refuses to edit.
        EvalTask {
            id: "fix-broken-script".to_string(),
            name: "Diagnose a broken script, fix it, confirm it runs".to_string(),
            tags: vec!["adversarial".to_string(), "coding".to_string()],
            prompts: vec![
                // Turn 1: write a script with a deliberate SyntaxError
                "Write the following Python script to `/tmp/eval_broken.py` exactly \
                 as shown (do not fix it):\n\n\
                 ```python\n\
                 def greet(name)\n    print(f\"Hello, {name}!\")\n\n\
                 greet(\"Graphirm\")\n\
                 ```"
                    .to_string(),
                // Turn 2: diagnose and fix
                "Run `/tmp/eval_broken.py` with Python. It has a bug — diagnose the \
                 error, fix the file, and run it again to confirm it works."
                    .to_string(),
            ],
            verifier: Verifier::All(vec![
                // Harness verifies the fixed file actually runs
                Verifier::CommandSucceeds {
                    command: "python3".to_string(),
                    args: vec!["/tmp/eval_broken.py".to_string()],
                },
                // Agent's response should confirm success
                Verifier::ResponseContainsAny {
                    substrings: vec![
                        "hello, graphirm".to_string(),
                        "Hello, Graphirm".to_string(),
                        "fixed".to_string(),
                        "works".to_string(),
                        "successfully".to_string(),
                    ],
                },
            ]),
            max_turns: 5,
            timeout_secs: 120,
            enable_segments: false,
            segment_filter: None,
        },
    ]
}
