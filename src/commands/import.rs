use std::path::{Path, PathBuf};

use crate::error::GraphirmError;

pub fn run(path: PathBuf, dry_run: bool, db_path: &Path) -> Result<(), GraphirmError> {
    let store = super::open_store(db_path)?;

    let files: Vec<PathBuf> = if path.is_dir() {
        std::fs::read_dir(&path)?
            .filter_map(|e: std::io::Result<std::fs::DirEntry>| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "txt"))
            .collect()
    } else {
        vec![path]
    };

    if files.is_empty() {
        println!("No .txt files found.");
        return Ok(());
    }

    for file in &files {
        let text = std::fs::read_to_string(file)?;
        let source_name = file
            .file_name()
            .and_then(|n: &std::ffi::OsStr| n.to_str())
            .unwrap_or("unknown");
        let transcript = graphirm_agent::import::cursor::parse_transcript(source_name, &text);
        if dry_run {
            println!(
                "[dry-run] {} — {} turns",
                source_name,
                transcript.turns.len()
            );
            continue;
        }
        let result = graphirm_agent::import::cursor::write_transcript(&store, &transcript)?;
        if result.skipped {
            println!("Already imported, skipping: {source_name}");
        } else {
            println!("Imported {} turns from {source_name}", result.turns_written);
        }
    }
    Ok(())
}
