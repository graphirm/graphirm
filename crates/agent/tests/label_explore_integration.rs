//! Integration tests for label exploration (corpus reading and, with model, full pipeline).

#![cfg(feature = "local-extraction")]

use std::io::BufReader;

#[test]
fn read_synthetic_corpus_fixture() {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture = manifest.join("tests/fixtures/corpus_synthetic.jsonl");
    let file = std::fs::File::open(&fixture).expect("open fixture");
    let turns = graphirm_agent::knowledge::label_explore::read_corpus_jsonl(BufReader::new(file))
        .expect("parse corpus");
    assert!(
        turns.len() >= 10,
        "synthetic fixture should have at least 10 turns, got {}",
        turns.len()
    );
    assert!(turns.iter().all(|t| t.role == "assistant"));
    assert!(turns.iter().any(|t| t.text.contains("```")));
}
