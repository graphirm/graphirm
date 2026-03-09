mod client;
mod harness;
mod report;
mod task;
mod tasks;

use clap::Parser;

#[derive(Parser)]
#[command(name = "graphirm-eval", about = "Graphirm evaluation harness")]
struct Cli {
    /// Path to the graphirm binary
    #[arg(long, default_value = "target/release/graphirm")]
    binary: std::path::PathBuf,

    /// Only run tasks whose tags include this value
    #[arg(long)]
    filter: Option<String>,

    /// Write results JSON to this path
    #[arg(long, default_value = "results/latest.json")]
    report: std::path::PathBuf,

    /// Skip memory tasks (requires EMBEDDING_BACKEND)
    #[arg(long)]
    skip_memory: bool,

    /// Just list available tasks without running them
    #[arg(long)]
    list: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    let mut tasks = tasks::all_tasks();

    if let Some(tag) = &cli.filter {
        tasks.retain(|t| t.tags.iter().any(|tg| tg == tag));
    }
    if cli.skip_memory {
        tasks.retain(|t| !t.tags.contains(&"memory".to_string()));
    }

    if cli.list {
        println!("Available tasks ({}):", tasks.len());
        for t in &tasks {
            println!("  [{:20}] {} {:?}", t.id, t.name, t.tags);
        }
        return Ok(());
    }

    let harness = harness::TestHarness::start(cli.binary).await?;

    println!("Running {} tasks...\n", tasks.len());

    // Delay between tasks (ms). Overridable via EVAL_INTER_TASK_DELAY_MS.
    let inter_task_delay_ms: u64 = std::env::var("EVAL_INTER_TASK_DELAY_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);

    let mut results = vec![];
    for (i, task) in tasks.iter().enumerate() {
        if i > 0 && inter_task_delay_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(inter_task_delay_ms)).await;
        }
        print!("  [{}] {} ... ", task.id, task.name);
        let result = harness.run_task(task).await;
        let icon = if result.passed { "✅" } else { "❌" };
        println!("{icon} ({:.1}s, {} turns)", result.elapsed_secs, result.turns_used);
        if let Some(ref reason) = result.failure_reason {
            println!("      ↳ {reason}");
        }
        results.push(result);
    }

    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();
    if total > 0 {
        println!("\n{}/{} tasks passed ({:.0}%)", passed, total, passed as f64 / total as f64 * 100.0);
    }

    report::write_report(&results, &cli.report)?;
    println!("Report written to {}", cli.report.display());

    Ok(())
}
