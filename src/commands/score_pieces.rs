//! `graphirm score-pieces` — compare hand labels with the current parser.

use std::path::PathBuf;

use graphirm_agent::pi_delegate::{format_score, score_label_file};

use crate::error::GraphirmError;

pub fn run(labels: PathBuf) -> Result<(), GraphirmError> {
    let report = score_label_file(&labels).map_err(|error| {
        GraphirmError::Config(format!("cannot read {}: {error}", labels.display()))
    })?;
    print!("{}", format_score(&report));
    Ok(())
}
