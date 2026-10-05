//! `graphirm label-pieces` — type a kind for each piece of a final Pi reply.

use std::fs::OpenOptions;
use std::io::{BufReader, Write};
use std::path::PathBuf;

use graphirm_agent::pi_delegate::{LabeledReply, load_final_replies};
use serde_json::json;

use crate::error::GraphirmError;

pub fn run(path: PathBuf, out: PathBuf, show_baseline: bool) -> Result<(), GraphirmError> {
    let replies = load_final_replies(&path).map_err(|error| {
        GraphirmError::Config(format!("cannot read {}: {error}", path.display()))
    })?;
    if replies.is_empty() {
        return Err(GraphirmError::Config(format!(
            "no final replies in {}",
            path.display()
        )));
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out)
        .map_err(|error| {
            GraphirmError::Config(format!("cannot open {}: {error}", out.display()))
        })?;
    println!(
        "{} final replies. Labels append to {}. Type quit to stop.",
        replies.len(),
        out.display()
    );
    if show_baseline {
        println!("baseline kinds are visible. The first pass should omit --show-baseline.");
    }
    let stdin = std::io::stdin();
    let mut input = BufReader::new(stdin.lock());
    let mut output = std::io::stdout().lock();
    let labeled = graphirm_agent::pi_delegate::run_label_session(
        &replies,
        show_baseline,
        &mut input,
        &mut output,
    )
    .map_err(|error| GraphirmError::Config(format!("label session: {error}")))?;
    for reply in &labeled {
        writeln!(file, "{}", label_row(&path, reply)).map_err(|error| {
            GraphirmError::Config(format!("cannot write {}: {error}", out.display()))
        })?;
    }
    println!("labeled {} replies into {}", labeled.len(), out.display());
    Ok(())
}

fn label_row(source: &std::path::Path, reply: &LabeledReply) -> String {
    json!({
        "source": source.display().to_string(),
        "reply_index": reply.reply_index,
        "pieces": reply.pieces.iter().map(|piece| json!({
            "order": piece.order,
            "start": piece.start,
            "end": piece.end,
            "kind": piece.kind.as_label(),
        })).collect::<Vec<_>>(),
    })
    .to_string()
}
