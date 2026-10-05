//! `graphirm label-pieces` — type a kind for each piece of a final Pi reply.

use std::fs::OpenOptions;
use std::io::{BufReader, Write};
use std::path::PathBuf;

use graphirm_agent::pi_delegate::{label_line, load_final_segments, segments_still_to_label};

use crate::error::GraphirmError;

pub fn run(path: PathBuf, out: PathBuf, show_baseline: bool) -> Result<(), GraphirmError> {
    let segments = load_final_segments(&path).map_err(|error| {
        GraphirmError::Config(format!("cannot read {}: {error}", path.display()))
    })?;
    if segments.is_empty() {
        return Err(GraphirmError::Config(format!(
            "no final replies in {}",
            path.display()
        )));
    }
    let existing = std::fs::read_to_string(&out).unwrap_or_default();
    let pending = segments_still_to_label(&segments, &existing);
    if pending.is_empty() {
        println!(
            "all {} final replies are already in {}",
            segments.len(),
            out.display()
        );
        return Ok(());
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out)
        .map_err(|error| {
            GraphirmError::Config(format!("cannot open {}: {error}", out.display()))
        })?;
    println!(
        "{} final replies, {} still to label. Labels append to {}. Type quit to stop.",
        segments.len(),
        pending.len(),
        out.display()
    );
    if show_baseline {
        println!("baseline kinds are visible. The first pass should omit --show-baseline.");
    }
    let stdin = std::io::stdin();
    let mut input = BufReader::new(stdin.lock());
    let mut output = std::io::stdout().lock();
    let labeled = graphirm_agent::pi_delegate::run_label_session(
        &pending,
        show_baseline,
        &mut input,
        &mut output,
    )
    .map_err(|error| GraphirmError::Config(format!("label session: {error}")))?;
    for reply in &labeled {
        writeln!(file, "{}", label_line(reply)).map_err(|error| {
            GraphirmError::Config(format!("cannot write {}: {error}", out.display()))
        })?;
    }
    println!("labeled {} replies into {}", labeled.len(), out.display());
    Ok(())
}
