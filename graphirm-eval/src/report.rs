//! Results reporter — writes Markdown and JSON output files.

use std::path::Path;

use crate::task::{SuiteScore, TaskOutcome, TaskResult};

pub fn write_report(
    results: &[TaskResult],
    path: &Path,
    experiment_id: Option<&str>,
) -> anyhow::Result<()> {
    // JSON
    std::fs::create_dir_all(path.parent().unwrap_or(Path::new(".")))?;
    let json = serde_json::to_string_pretty(results)?;
    std::fs::write(path, json)?;

    // Markdown (same path, .md extension)
    let md_path = path.with_extension("md");
    let mut md = format!(
        "# graphirm-eval results\n\n**Date:** {}\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M UTC")
    );
    if let Some(id) = experiment_id {
        md.push_str(&format!("**Experiment:** `{id}`\n"));
    }
    md.push('\n');
    let score = SuiteScore::from_results(results);
    md.push_str(&format!(
        "**Score:** {}/{} ({:.0}%)\n",
        score.passed,
        score.scored(),
        score.percent()
    ));
    if score.errored > 0 {
        md.push_str(&format!(
            "**Excluded:** {} infrastructure error{}\n",
            score.errored,
            if score.errored == 1 { "" } else { "s" }
        ));
    }
    md.push('\n');
    md.push_str("| Task | Result | Turns | Time |\n|---|---|---|---|\n");
    for r in results {
        let icon = match r.outcome {
            TaskOutcome::Pass => "✅",
            TaskOutcome::Fail => "❌",
            TaskOutcome::Error => "⚠",
        };
        let detail = match r.outcome {
            TaskOutcome::Pass => "",
            TaskOutcome::Fail | TaskOutcome::Error => r.failure_reason.as_deref().unwrap_or("-"),
        };
        md.push_str(&format!(
            "| {} | {} {} | {} | {:.1}s |\n",
            r.task_id, icon, detail, r.turns_used, r.elapsed_secs
        ));
    }
    std::fs::write(md_path, md)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::TaskResult;

    #[test]
    fn report_excludes_infrastructure_errors_from_the_score() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("eval.json");
        let results = vec![
            TaskResult::pass("a", 1, 1.0),
            TaskResult::fail("b", "verifier returned false"),
            TaskResult::error("c", "rate limit exhausted"),
        ];
        write_report(&results, &path, None).unwrap();
        let md = std::fs::read_to_string(path.with_extension("md")).unwrap();
        assert!(md.contains("**Score:** 1/2 (50%)"));
        assert!(md.contains("1 infrastructure"));
        assert!(md.contains("rate limit exhausted"));
        assert!(!md.contains("❌ rate limit exhausted"));
    }
}
