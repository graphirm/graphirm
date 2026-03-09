//! Results reporter — writes Markdown and JSON output files.

use std::path::Path;

use crate::task::TaskResult;

pub fn write_report(results: &[TaskResult], path: &Path) -> anyhow::Result<()> {
    // JSON
    std::fs::create_dir_all(path.parent().unwrap_or(Path::new(".")))?;
    let json = serde_json::to_string_pretty(results)?;
    std::fs::write(path, json)?;

    // Markdown (same path, .md extension)
    let md_path = path.with_extension("md");
    let mut md = format!(
        "# graphirm-eval results\n\n**Date:** {}\n\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M UTC")
    );
    let passed = results.iter().filter(|r| r.passed).count();
    md.push_str(&format!(
        "**Score:** {}/{} ({:.0}%)\n\n",
        passed,
        results.len(),
        if results.is_empty() {
            0.0
        } else {
            passed as f64 / results.len() as f64 * 100.0
        }
    ));
    md.push_str("| Task | Result | Turns | Time |\n|---|---|---|---|\n");
    for r in results {
        let icon = if r.passed { "✅" } else { "❌" };
        let reason = r.failure_reason.as_deref().unwrap_or("-");
        md.push_str(&format!(
            "| {} | {} {} | {} | {:.1}s |\n",
            r.task_id,
            icon,
            if r.passed { "" } else { reason },
            r.turns_used,
            r.elapsed_secs
        ));
    }
    std::fs::write(md_path, md)?;
    Ok(())
}
