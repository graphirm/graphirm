use std::path::{Path, PathBuf};

use crate::error::GraphirmError;

pub fn run(
    db_path: &Path,
    out: Option<PathBuf>,
    limit: Option<u64>,
    all_roles: bool,
) -> Result<(), GraphirmError> {
    let graph = super::open_store(db_path)?;
    let assistant_only = !all_roles;
    let count = if let Some(path) = out {
        let mut f = std::fs::File::create(path)?;
        graphirm_graph::export_corpus_to_jsonl(&graph, &mut f, assistant_only, limit)?
    } else {
        let mut stdout = std::io::stdout();
        graphirm_graph::export_corpus_to_jsonl(&graph, &mut stdout, assistant_only, limit)?
    };
    if all_roles {
        eprintln!("Exported {} turns (user + assistant).", count);
    } else {
        eprintln!("Exported {} assistant turns.", count);
    }
    Ok(())
}
