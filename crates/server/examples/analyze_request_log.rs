//! Analyze request log JSONL files produced by usage discovery scenarios.
//!
//! Reads one or more JSONL files and produces a report showing:
//! - Endpoint frequency (which endpoints are called most)
//! - Endpoint group distribution (sessions vs graph vs events)
//! - Method distribution (GET vs POST vs DELETE)
//! - Call sequences (what follows what)
//! - Timing statistics (p50, p95, max per endpoint)
//! - Session patterns (how many calls per session)
//!
//! Usage: cargo run -p graphirm-server --example analyze_request_log -- <file1.jsonl> [file2.jsonl ...]

use std::collections::HashMap;
use std::io::BufRead;

use graphirm_server::request_log::RequestLogEntry as LogEntry;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("Usage: analyze_request_log <file1.jsonl> [file2.jsonl ...]");
        std::process::exit(1);
    }

    let mut entries: Vec<LogEntry> = Vec::new();

    for path in &args {
        let file = std::fs::File::open(path).unwrap_or_else(|e| {
            eprintln!("Failed to open {path}: {e}");
            std::process::exit(1);
        });

        for line in std::io::BufReader::new(file).lines() {
            let line = line.unwrap();
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<LogEntry>(&line) {
                Ok(entry) => entries.push(entry),
                Err(e) => eprintln!("Skipping malformed line: {e}"),
            }
        }
    }

    if entries.is_empty() {
        println!("No entries found.");
        return;
    }

    println!("═══════════════════════════════════════════════════");
    println!("  USAGE DISCOVERY REPORT");
    println!("  {} entries from {} file(s)", entries.len(), args.len());
    println!("═══════════════════════════════════════════════════\n");

    report_endpoint_frequency(&entries);
    report_group_distribution(&entries);
    report_method_distribution(&entries);
    report_call_sequences(&entries);
    report_timing_stats(&entries);
    report_session_patterns(&entries);
}

fn report_endpoint_frequency(entries: &[LogEntry]) {
    println!("── Endpoint Frequency ──────────────────────────────\n");

    let mut counts: HashMap<String, usize> = HashMap::new();
    for entry in entries {
        let key = normalize_path(&entry.method, &entry.path);
        *counts.entry(key).or_default() += 1;
    }

    let mut sorted: Vec<_> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));

    let total = entries.len() as f64;
    println!("{:<45}  {:>5}  {:>6}", "ENDPOINT", "COUNT", "  %");
    println!("{}", "-".repeat(60));
    for (endpoint, count) in &sorted {
        let pct = (*count as f64 / total) * 100.0;
        println!("{:<45}  {:>5}  {:>5.1}%", endpoint, count, pct);
    }
    println!();
}

fn report_group_distribution(entries: &[LogEntry]) {
    println!("── Endpoint Group Distribution ─────────────────────\n");

    let mut counts: HashMap<&str, usize> = HashMap::new();
    for entry in entries {
        *counts.entry(entry.endpoint_group.as_str()).or_default() += 1;
    }

    let mut sorted: Vec<_> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));

    let total = entries.len() as f64;
    for (group, count) in &sorted {
        let pct = (*count as f64 / total) * 100.0;
        let bar_len = (pct / 2.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("  {:<12}  {:>5}  ({:>5.1}%)  {}", group, count, pct, bar);
    }
    println!();
}

fn report_method_distribution(entries: &[LogEntry]) {
    println!("── Method Distribution ─────────────────────────────\n");

    let mut counts: HashMap<&str, usize> = HashMap::new();
    for entry in entries {
        *counts.entry(entry.method.as_str()).or_default() += 1;
    }

    let mut sorted: Vec<_> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));

    for (method, count) in &sorted {
        let pct = (*count as f64 / entries.len() as f64) * 100.0;
        println!("  {:<8}  {:>5}  ({:>5.1}%)", method, count, pct);
    }
    println!();
}

fn report_call_sequences(entries: &[LogEntry]) {
    println!("── Call Sequences (bigrams) ─────────────────────────\n");

    let mut bigrams: HashMap<String, usize> = HashMap::new();
    for window in entries.windows(2) {
        let a = normalize_path(&window[0].method, &window[0].path);
        let b = normalize_path(&window[1].method, &window[1].path);
        let key = format!("{a}  →  {b}");
        *bigrams.entry(key).or_default() += 1;
    }

    let mut sorted: Vec<_> = bigrams.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));

    println!("{:<80}  {:>5}", "SEQUENCE", "COUNT");
    println!("{}", "-".repeat(87));
    for (seq, count) in sorted.iter().take(15) {
        println!("{:<80}  {:>5}", seq, count);
    }
    println!();
}

fn report_timing_stats(entries: &[LogEntry]) {
    println!("── Timing (ms) ─────────────────────────────────────\n");

    let mut by_endpoint: HashMap<String, Vec<f64>> = HashMap::new();
    for entry in entries {
        let key = normalize_path(&entry.method, &entry.path);
        by_endpoint.entry(key).or_default().push(entry.duration_ms);
    }

    println!(
        "{:<45}  {:>6}  {:>6}  {:>6}  {:>6}",
        "ENDPOINT", "P50", "P95", "MAX", "N"
    );
    println!("{}", "-".repeat(75));

    let mut sorted: Vec<_> = by_endpoint.into_iter().collect();
    sorted.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    for (endpoint, mut times) in sorted {
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = times.len();
        let p50 = times[n / 2];
        let p95 = times[(n * 95 / 100).min(n - 1)];
        let max = times[n - 1];
        println!(
            "{:<45}  {:>6.1}  {:>6.1}  {:>6.1}  {:>6}",
            endpoint, p50, p95, max, n
        );
    }
    println!();
}

fn report_session_patterns(entries: &[LogEntry]) {
    println!("── Session Patterns ────────────────────────────────\n");

    let mut by_session: HashMap<String, Vec<String>> = HashMap::new();
    for entry in entries {
        if let Some(ref sid) = entry.session_id {
            let key = normalize_path(&entry.method, &entry.path);
            by_session.entry(sid.clone()).or_default().push(key);
        }
    }

    if by_session.is_empty() {
        println!("  No session-scoped requests found.\n");
        return;
    }

    println!("  Sessions observed: {}", by_session.len());
    println!("  Avg calls/session: {:.1}", {
        let total: usize = by_session.values().map(|v| v.len()).sum();
        total as f64 / by_session.len() as f64
    });

    // Show call sequence for each session
    println!();
    for (sid, calls) in &by_session {
        println!("  Session {}:", &sid[..8.min(sid.len())]);
        for (i, call) in calls.iter().enumerate() {
            let prefix = if i == calls.len() - 1 {
                "└─"
            } else {
                "├─"
            };
            println!("    {prefix} {call}");
        }
        println!();
    }
}

/// Normalize a path by replacing UUIDs/IDs with `:id` placeholders.
fn normalize_path(method: &str, path: &str) -> String {
    let segments: Vec<&str> = path.split('/').collect();
    let normalized: Vec<&str> = segments
        .iter()
        .enumerate()
        .map(|(i, seg)| {
            if i >= 3
                && (seg.len() > 8 || seg.contains('-'))
                && ![
                    "prompt",
                    "messages",
                    "abort",
                    "children",
                    "tasks",
                    "knowledge",
                    "node",
                    "subgraph",
                ]
                .contains(seg)
            {
                ":id"
            } else {
                seg
            }
        })
        .collect();

    format!("{} {}", method, normalized.join("/"))
}
