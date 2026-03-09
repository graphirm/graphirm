mod coding;
pub mod graph;
pub mod knowledge;
pub mod memory;

use crate::task::EvalTask;

/// Returns all registered eval tasks (excludes cross-session memory tasks).
pub fn all_tasks() -> Vec<EvalTask> {
    let mut tasks = vec![];
    tasks.extend(coding::tasks());
    tasks.extend(knowledge::tasks());
    tasks.extend(graph::tasks());
    tasks
}
