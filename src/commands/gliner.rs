use std::io::Write;
use std::path::PathBuf;

use crate::error::GraphirmError;

pub async fn run_label_explore(
    corpus_path: PathBuf,
    labels_str: String,
    min_confidence: f64,
    out: Option<PathBuf>,
) -> Result<(), GraphirmError> {
    use std::io::BufReader;

    let model_dir = std::env::var("GLINER2_MODEL_DIR").map_err(|_| {
        GraphirmError::Config(
            "GLINER2_MODEL_DIR not set. Run `graphirm model download` and set the env var.".into(),
        )
    })?;
    let model_dir = std::path::Path::new(&model_dir);

    let file = std::fs::File::open(&corpus_path).map_err(|e| {
        GraphirmError::Config(format!("open corpus {}: {}", corpus_path.display(), e))
    })?;
    let turns = graphirm_agent::knowledge::label_explore::read_corpus_jsonl(BufReader::new(file))?;
    let total = turns.len();
    if total == 0 {
        eprintln!("Corpus is empty.");
        return Ok(());
    }

    let labels: Vec<String> = labels_str
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if labels.is_empty() {
        return Err(GraphirmError::Config(
            "At least one --labels value required".into(),
        ));
    }

    let extractor = graphirm_agent::knowledge::local_extraction::OnnxExtractor::new(model_dir)
        .map_err(|e| GraphirmError::Config(format!("load GLiNER2 model: {}", e)))?;

    eprintln!(
        "Running GLiNER2 on {} turns with {} labels...",
        total,
        labels.len()
    );
    let report = graphirm_agent::knowledge::label_explore::run_label_exploration(
        &extractor,
        &turns,
        &labels,
        min_confidence,
    )
    .await?;

    let json =
        serde_json::to_string_pretty(&report).map_err(|e| GraphirmError::Config(e.to_string()))?;
    if let Some(path) = out {
        std::fs::write(&path, json).map_err(GraphirmError::Io)?;
        eprintln!("Wrote report to {}", path.display());
    } else {
        println!("{}", json);
    }
    eprintln!(
        "Coverage: {:.1}% ({} / {} chars in {} turns)",
        report.corpus_stats.coverage_pct,
        report.corpus_stats.covered_chars,
        report.corpus_stats.total_chars,
        report.corpus_stats.turns_with_any_label
    );
    Ok(())
}

pub fn run_schema_suggest(
    report_path: PathBuf,
    out: Option<PathBuf>,
) -> Result<(), GraphirmError> {
    let json = std::fs::read_to_string(&report_path).map_err(|e| {
        GraphirmError::Config(format!("read report {}: {}", report_path.display(), e))
    })?;
    let report: graphirm_agent::knowledge::label_explore::LabelExplorationReport =
        serde_json::from_str(&json)
            .map_err(|e| GraphirmError::Config(format!("parse report JSON: {}", e)))?;
    let rec = graphirm_agent::knowledge::schema_suggest::analyse_report(&report);
    let out_json =
        serde_json::to_string_pretty(&rec).map_err(|e| GraphirmError::Config(e.to_string()))?;
    if let Some(path) = out {
        std::fs::write(&path, out_json).map_err(GraphirmError::Io)?;
        eprintln!("Wrote schema recommendation to {}", path.display());
    } else {
        println!("{}", out_json);
    }
    eprintln!(
        "Recommended segment types ({}): {}",
        rec.recommended_segment_types.len(),
        rec.recommended_segment_types.join(", ")
    );
    Ok(())
}

pub async fn run_predict_spans(
    corpus_path: PathBuf,
    labels_str: String,
    min_confidence: f64,
    out: Option<PathBuf>,
    batch_size: Option<u64>,
) -> Result<(), GraphirmError> {
    use std::io::BufReader;

    let model_dir = std::env::var("GLINER2_MODEL_DIR").map_err(|_| {
        GraphirmError::Config(
            "GLINER2_MODEL_DIR not set. Run `graphirm model download` and set the env var.".into(),
        )
    })?;
    let model_dir = std::path::Path::new(&model_dir);

    let labels: Vec<String> = labels_str
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if labels.is_empty() {
        return Err(GraphirmError::Config(
            "At least one --labels value required".into(),
        ));
    }

    let extractor = graphirm_agent::knowledge::local_extraction::OnnxExtractor::new(model_dir)
        .map_err(|e| GraphirmError::Config(format!("load GLiNER2 model: {}", e)))?;

    let batch_size_usize = batch_size.map(|n| n as usize);

    let mut writer: Box<dyn std::io::Write> = if let Some(path) = &out {
        Box::new(std::fs::File::create(path).map_err(GraphirmError::Io)?)
    } else {
        Box::new(std::io::stdout())
    };

    let mut total_written = 0usize;

    if let Some(batch_size_n) = batch_size_usize {
        let file = std::fs::File::open(&corpus_path).map_err(|e| {
            GraphirmError::Config(format!("open corpus {}: {}", corpus_path.display(), e))
        })?;
        let mut reader = BufReader::new(file);
        loop {
            let turns = graphirm_agent::knowledge::label_explore::read_corpus_jsonl_batch(
                &mut reader,
                batch_size_n,
            )?;
            if turns.is_empty() {
                break;
            }
            let rows = graphirm_agent::knowledge::predict_spans::run_predict_spans(
                &extractor,
                &turns,
                &labels,
                min_confidence,
            )
            .await?;
            for row in &rows {
                let line =
                    serde_json::to_string(row).map_err(|e| GraphirmError::Config(e.to_string()))?;
                writeln!(writer, "{line}").map_err(GraphirmError::Io)?;
            }
            total_written += rows.len();
            if out.is_some() {
                eprintln!(
                    "  processed batch: {} turns (total {} so far)",
                    rows.len(),
                    total_written
                );
            }
        }
    } else {
        let file = std::fs::File::open(&corpus_path).map_err(|e| {
            GraphirmError::Config(format!("open corpus {}: {}", corpus_path.display(), e))
        })?;
        let turns =
            graphirm_agent::knowledge::label_explore::read_corpus_jsonl(BufReader::new(file))?;
        if turns.is_empty() {
            eprintln!("Corpus is empty.");
            return Ok(());
        }
        let rows = graphirm_agent::knowledge::predict_spans::run_predict_spans(
            &extractor,
            &turns,
            &labels,
            min_confidence,
        )
        .await?;
        for row in &rows {
            let line =
                serde_json::to_string(row).map_err(|e| GraphirmError::Config(e.to_string()))?;
            writeln!(writer, "{line}").map_err(GraphirmError::Io)?;
        }
        total_written = rows.len();
    }

    if out.is_some() && total_written > 0 {
        eprintln!("Wrote {} turn spans to output.", total_written);
    }
    Ok(())
}

pub fn run_validate_agreement(
    human_path: PathBuf,
    gliner_path: PathBuf,
    threshold: f64,
    out: Option<PathBuf>,
) -> Result<(), GraphirmError> {
    use std::io::BufReader;

    let human_file = std::fs::File::open(&human_path).map_err(|e| {
        GraphirmError::Config(format!(
            "open human annotations {}: {}",
            human_path.display(),
            e
        ))
    })?;
    let human = graphirm_agent::knowledge::validate_agreement::read_annotations_jsonl(
        BufReader::new(human_file),
    )?;

    let gliner_file = std::fs::File::open(&gliner_path).map_err(|e| {
        GraphirmError::Config(format!(
            "open gliner spans {}: {}",
            gliner_path.display(),
            e
        ))
    })?;
    let gliner =
        graphirm_agent::knowledge::predict_spans::read_spans_jsonl(BufReader::new(gliner_file))?;

    const OVERLAP_RATIO_MIN: f64 = 0.5;
    let report = graphirm_agent::knowledge::validate_agreement::validate_agreement(
        &human,
        &gliner,
        threshold,
        OVERLAP_RATIO_MIN,
    );

    let out_json =
        serde_json::to_string_pretty(&report).map_err(|e| GraphirmError::Config(e.to_string()))?;
    if let Some(path) = out {
        std::fs::write(&path, out_json).map_err(GraphirmError::Io)?;
        eprintln!("Wrote agreement report to {}", path.display());
    } else {
        println!("{}", out_json);
    }
    eprintln!(
        "Agreement: {:.1}% ({} / {} segments) — {}",
        report.agreement_pct,
        report.matched_segments,
        report.total_human_segments,
        if report.pass { "PASS" } else { "FAIL" }
    );
    Ok(())
}
