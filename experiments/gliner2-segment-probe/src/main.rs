//! CLI: run GLiNER2 span extraction with the same labels as `SegmentConfig` defaults.

use std::collections::HashMap;
use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use graphirm_agent::knowledge::local_extraction::get_or_init_onnx_extractor;
use graphirm_agent::knowledge::segments::segment_extract_gliner2;
use serde::Serialize;
use serde_json::json;

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

    /// Repeatable `label=description` (appended after `entities` as `[DESCRIPTION] label: ...`).
    #[arg(long = "description", value_name = "LABEL=TEXT")]
    description: Vec<String>,

    /// TOML file: top-level `label = "description"` strings (same keys as `--labels`).
    #[arg(long)]
    descriptions_file: Option<PathBuf>,

    /// Repeatable `label=0.35` (per-label min confidence).
    #[arg(long, value_name = "LABEL=SCORE")]
    label_threshold: Vec<String>,

    /// TOML file: top-level `label = 0.35` floats.
    #[arg(long)]
    thresholds_file: Option<PathBuf>,

    /// Write before/after sweep JSON here (runs each threshold with and without descriptions).
    #[arg(long)]
    sweep_json_out: Option<PathBuf>,

    /// Comma-separated thresholds for `--sweep-json-out` (default: 0.5,0.4,0.35,0.25,0.2).
    #[arg(long, default_value = "0.5,0.4,0.35,0.25,0.2")]
    sweep_thresholds: String,

    /// When using `--sweep-json-out`, skip printing segment JSON to stdout.
    #[arg(long, default_value_t = false)]
    sweep_quiet: bool,
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

fn parse_kv_str(s: &str) -> anyhow::Result<(String, String)> {
    let (k, v) = s
        .split_once('=')
        .ok_or_else(|| anyhow::anyhow!("expected KEY=VALUE, got {:?}", s))?;
    if k.is_empty() {
        anyhow::bail!("empty key before '='");
    }
    Ok((k.to_string(), v.to_string()))
}

fn parse_kv_f64(s: &str) -> anyhow::Result<(String, f64)> {
    let (k, v) = s
        .split_once('=')
        .ok_or_else(|| anyhow::anyhow!("expected KEY=VALUE, got {:?}", s))?;
    if k.is_empty() {
        anyhow::bail!("empty key before '='");
    }
    let f: f64 = v
        .parse()
        .with_context(|| format!("parse threshold value {:?}", v))?;
    Ok((k.to_string(), f))
}

fn load_string_map_from_toml(path: &PathBuf) -> anyhow::Result<HashMap<String, String>> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("parse TOML {}", path.display()))
}

fn load_f64_map_from_toml(path: &PathBuf) -> anyhow::Result<HashMap<String, f64>> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("parse TOML {}", path.display()))
}

fn merge_descriptions(
    file: Option<&PathBuf>,
    cli_pairs: &[String],
) -> anyhow::Result<Option<HashMap<String, String>>> {
    let mut m: HashMap<String, String> = HashMap::new();
    if let Some(path) = file {
        m.extend(load_string_map_from_toml(path)?);
    }
    for s in cli_pairs {
        let (k, v) = parse_kv_str(s)?;
        m.insert(k, v);
    }
    if m.is_empty() { Ok(None) } else { Ok(Some(m)) }
}

fn merge_thresholds(
    file: Option<&PathBuf>,
    cli_pairs: &[String],
) -> anyhow::Result<Option<HashMap<String, f64>>> {
    let mut m: HashMap<String, f64> = HashMap::new();
    if let Some(path) = file {
        m.extend(load_f64_map_from_toml(path)?);
    }
    for s in cli_pairs {
        let (k, v) = parse_kv_f64(s)?;
        m.insert(k, v);
    }
    if m.is_empty() { Ok(None) } else { Ok(Some(m)) }
}

#[derive(Serialize)]
struct SweepRun {
    min_confidence: f64,
    segment_count: usize,
    segments: serde_json::Value,
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

    let labels = cli.labels.clone().unwrap_or_else(default_labels);
    if labels.is_empty() {
        anyhow::bail!("at least one --labels entry is required");
    }

    let label_descriptions = merge_descriptions(cli.descriptions_file.as_ref(), &cli.description)?;
    let label_min_confidence =
        merge_thresholds(cli.thresholds_file.as_ref(), &cli.label_threshold)?;

    let extractor = get_or_init_onnx_extractor(&cli.model_dir).await?;

    if let Some(out_path) = &cli.sweep_json_out {
        let thresholds: Vec<f64> = cli
            .sweep_thresholds
            .split(',')
            .map(|s| s.trim().parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!("invalid --sweep-thresholds: {e}"))?;

        let mut baseline_runs = Vec::new();
        for &min_c in &thresholds {
            let segs =
                segment_extract_gliner2(extractor.as_ref(), &text, &labels, min_c, None, None)
                    .await?;
            baseline_runs.push(SweepRun {
                min_confidence: min_c,
                segment_count: segs.len(),
                segments: serde_json::to_value(&segs)?,
            });
        }

        let with_desc_runs = if let Some(ref desc) = label_descriptions {
            let mut runs = Vec::new();
            for &min_c in &thresholds {
                let segs = segment_extract_gliner2(
                    extractor.as_ref(),
                    &text,
                    &labels,
                    min_c,
                    Some(desc),
                    label_min_confidence.as_ref(),
                )
                .await?;
                runs.push(SweepRun {
                    min_confidence: min_c,
                    segment_count: segs.len(),
                    segments: serde_json::to_value(&segs)?,
                });
            }
            Some(runs)
        } else {
            None
        };

        let fixture = cli
            .file
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| "(stdin)".to_string());

        let with_descriptions = match (label_descriptions.as_ref(), &with_desc_runs) {
            (Some(d), Some(runs)) => json!({
                "label_descriptions": d,
                "label_min_confidence": label_min_confidence,
                "runs": runs,
            }),
            _ => serde_json::Value::Null,
        };

        let doc = json!({
            "generated_at": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            "host": "local (gliner2-segment-probe --sweep-json-out)",
            "fixture": fixture,
            "model_dir": cli.model_dir,
            "labels": labels,
            "tool": "gliner2-segment-probe sweep (segment_extract_gliner2)",
            "sweep_thresholds": thresholds,
            "baseline": {
                "label_descriptions": serde_json::Value::Null,
                "label_min_confidence": serde_json::Value::Null,
                "runs": baseline_runs,
            },
            "with_descriptions": with_descriptions,
            "notes": [
                "Offsets in each run's segments are byte indices into the UTF-8 input.",
                "baseline uses no label_descriptions; with_descriptions is null when no --description/--descriptions-file.",
                "Reproduce: GLINER2_MODEL_DIR=... cargo run -p gliner2-segment-probe --release -- --file <fixture> --sweep-json-out <path> [--descriptions-file <toml>]",
            ],
        });

        fs::write(out_path, serde_json::to_string_pretty(&doc)?)
            .with_context(|| format!("write {}", out_path.display()))?;
        if !cli.sweep_quiet {
            println!("{}", serde_json::to_string_pretty(&doc)?);
        }
        return Ok(());
    }

    let segments = segment_extract_gliner2(
        extractor.as_ref(),
        &text,
        &labels,
        cli.min_confidence,
        label_descriptions.as_ref(),
        label_min_confidence.as_ref(),
    )
    .await
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    let value = serde_json::to_value(&segments)?;
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}
