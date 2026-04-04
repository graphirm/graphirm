//! CLI: run GLiNER2 span extraction with the same labels as `SegmentConfig` defaults.

use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;

use clap::Parser;
use graphirm_agent::knowledge::local_extraction::get_or_init_onnx_extractor;
use graphirm_agent::knowledge::segments::segment_extract_gliner2;

#[derive(Parser, Debug)]
#[command(name = "gliner2-segment-probe")]
#[command(about = "Print GLiNER2 segment spans as JSON (graphirm-agent segment fallback path)")]
struct Cli {
    /// Directory with gliner2 ONNX files (gliner2_config.json, *.onnx).
    #[arg(long, env = "GLINER2_MODEL_DIR")]
    model_dir: String,

    /// Input text file. If omitted, read stdin.
    #[arg(short, long)]
    file: Option<PathBuf>,

    /// Minimum span confidence (0.0–1.0).
    #[arg(long, default_value_t = 0.5)]
    min_confidence: f64,

    /// Comma-separated labels (default: observation,reasoning,code,plan,answer).
    #[arg(long, value_delimiter = ',')]
    labels: Option<Vec<String>>,
}

fn default_labels() -> Vec<String> {
    vec![
        "observation".into(),
        "reasoning".into(),
        "code".into(),
        "plan".into(),
        "answer".into(),
    ]
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let text = if let Some(path) = &cli.file {
        fs::read_to_string(path).map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?
    } else {
        let mut buf = String::new();
        io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| anyhow::anyhow!("read stdin: {e}"))?;
        buf
    };

    if text.trim().is_empty() {
        anyhow::bail!("input text is empty");
    }

    let labels = cli.labels.unwrap_or_else(default_labels);
    if labels.is_empty() {
        anyhow::bail!("at least one --labels entry is required");
    }

    let extractor = get_or_init_onnx_extractor(&cli.model_dir).await?;
    let segments = segment_extract_gliner2(extractor.as_ref(), &text, &labels, cli.min_confidence)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let value = serde_json::to_value(&segments)?;
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}
