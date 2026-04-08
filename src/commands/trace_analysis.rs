use std::path::Path;
use std::sync::Arc;

use crate::error::GraphirmError;

pub fn run(db_path: &Path, max_sessions: usize, format: &str) -> Result<(), GraphirmError> {
    let graph = Arc::new(crate::commands::open_store(db_path)?);
    let report = graphirm_agent::trace_analysis::build_trace_report(&graph, max_sessions);

    match format {
        "json" => {
            println!(
                "{}",
                serde_json::to_string_pretty(&report).expect("JSON serialization")
            );
        }
        _ => print_markdown_report(&report),
    }
    Ok(())
}

fn print_markdown_report(report: &graphirm_agent::trace_analysis::TraceReport) {
    println!("# Trace Analysis Report\n");
    println!("Sessions analyzed: {}\n", report.sessions_analyzed);

    if !report.patterns.is_empty() {
        println!("## Patterns Detected\n");
        for pattern in &report.patterns {
            println!(
                "### {} ({:?}) — {} occurrence(s)",
                pattern.pattern, pattern.severity, pattern.occurrences
            );
            println!("{}\n", pattern.description);
        }
    }

    if !report.per_session.is_empty() {
        println!("## Per-Session Summary\n");
        for s in &report.per_session {
            let finding_count = s.findings.len();
            println!(
                "- **{}** ({}) — {} turns, {} tokens, {} finding(s)",
                s.agent_name, s.status, s.turn_count, s.token_total, finding_count
            );
        }
        println!();
    }

    if !report.suggestions.is_empty() {
        println!("## Suggestions\n");
        for s in &report.suggestions {
            println!("- {s}");
        }
    }
}
