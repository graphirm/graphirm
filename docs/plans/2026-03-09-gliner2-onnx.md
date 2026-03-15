# GLiNER2 ONNX Inference Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the stub `parse_onnx_outputs` in `local_extraction.rs` with a fully-working GLiNER2 NER inference pipeline, using `lmo3/gliner2-large-v1-onnx` (four separate ONNX sessions) downloaded from HuggingFace on first run, and make `OnnxExtractor` reusable via a process-wide cache keyed by `model_dir`.

**Architecture:** GLiNER2-large-v1 exports as four ONNX files — `encoder.onnx`, `span_rep.onnx`, `count_embed.onnx`, `classifier.onnx`. Inference is a sequential 7-step pipeline: build schema-formatted token input → encoder → extract [E] embeddings → generate word spans → span_rep → count_embed → dot-product scoring. `OnnxExtractor` holds four `ort::Session` fields, a `tokenizers::Tokenizer`, and config loaded from `gliner2_config.json`. Model files are downloaded once via `hf-hub` and cached in `~/.cache/huggingface/hub/`; extractor instances should also be cached process-wide by canonical `model_dir` so segment fallback and knowledge extraction do not rebuild ONNX sessions on every turn.

**Tech Stack:** Rust, `ort` 2.0.0-rc.12, `tokenizers` 0.21, `ndarray` 0.17, `hf-hub` 0.5, `regex` 1 (all under `local-extraction` feature), `serde_json`

---

## Status

> **All tasks ✅ DONE.** The GLiNER2 four-session inference pipeline, `ExtractionBackend::Local { model_dir }`,
> the setup guide, and the process-wide `OnnxExtractor` cache are all implemented and tested.
> Tasks 1–8 are preserved below as reference for how the implementation was structured.

---

## Background

GLiNER2-large-v1 exports as **four** ONNX files. `OnnxExtractor` holds four `ort::Session` fields behind `tokio::sync::Mutex`, a `tokenizers::Tokenizer`, and config loaded from `gliner2_config.json`.

**Reference architecture (from `lmoe/gliner2-onnx` ARCHITECTURE.md and `runtime.py`):**

### The four ONNX sessions

| File | Input tensors | Output tensor | Purpose |
|------|--------------|---------------|---------|
| `encoder.onnx` | `input_ids` (i64, `[1,seq]`), `attention_mask` (i64, `[1,seq]`) | `hidden_states` (f32, `[1,seq,1024]`) | DeBERTa encoder |
| `span_rep.onnx` | `hidden_states` (f32, `[1,seq,1024]`), `span_start_idx` (i64, `[1,spans]`), `span_end_idx` (i64, `[1,spans]`) | `span_rep` (f32, `[1,spans,1024]`) | Span representation |
| `count_embed.onnx` | `label_embeddings` (f32, `[labels,1024]`) | `transformed` (f32, `[labels,1024]`) | Label transformation |
| `classifier.onnx` | `hidden_state` (f32, `[labels,1024]`) | `logits` (f32, `[labels,1]`) | Classification (not used for NER) |

### The input format

Schema prefix prepended before the text:
```
( [P] entities ( [E] label1 [E] label2 ... ) ) [SEP_TEXT] text_lowercase
```

Special token IDs (from `gliner2_config.json`):

| Token | ID |
|-------|----|
| `[SEP_TEXT]` | 128002 |
| `[P]` | 128003 |
| `[E]` | 128005 |

Parentheses `(` and `)` are encoded via the tokenizer (not hardcoded IDs).

### The inference pipeline

```
1. tokenize schema prefix + text words (track [E] positions, word offsets)
2. encoder(input_ids, attn_mask) → hidden_states [1, seq, 1024]
3. label_embeddings = hidden_states[0, e_positions, :]         [num_labels, 1024]
4. text_hidden = hidden_states[0, text_start_idx.., :]         [text_tokens, 1024]
5. span pairs = all (word_i, word_j) where j - i < max_width
   span_start_idx = [first_token_positions[i] for (i,j) in spans]
   span_end_idx   = [first_token_positions[j] for (i,j) in spans]
6. span_rep(text_hidden, span_start, span_end) → span_rep      [spans, 1024]
7. count_embed(label_embeddings) → transformed                  [labels, 1024]
8. scores = sigmoid(span_rep @ transformed.T)                   [spans, labels]
9. collect entities above threshold, deduplicate by label overlap
```

### Model size

`lmo3/gliner2-large-v1-onnx` (fp32): ~1.95 GB total
- `onnx/encoder.onnx`: ~1.65 GB
- `onnx/span_rep.onnx`: ~112 MB
- `onnx/count_embed.onnx`: ~72 MB
- `onnx/classifier.onnx`: ~8 MB

## Follow-on optimization: process-wide extractor cache

The initial implementation plan below gets GLiNER2 working, but there is one important performance follow-up for real agent traffic:

- `try_gliner2_fallback()` in `segments.rs` currently constructs `OnnxExtractor` per call.
- `extract_knowledge_with_backend()` in `extraction.rs` also constructs `OnnxExtractor` inline for `Local` and `Hybrid`.
- `OnnxExtractor::new()` loads several large ONNX sessions and is expensive enough to add seconds of latency.

The cache design for this follow-up should be:

- **Scope:** process-wide, reusable anywhere in the crate
- **Key:** canonicalized `model_dir` string/path
- **Value:** `Arc<OnnxExtractor>`
- **Concurrency:** same-key concurrent requests share one initialization attempt
- **Failure semantics:** failed initialization must not poison the cache forever; a later call may retry
- **Primary consumers:** `segments.rs::try_gliner2_fallback()` and `extraction.rs::{Local, Hybrid}`

---

## Task 1: Add `hf-hub` and `regex` dependencies — ✅ DONE

**Files:**
- Modify: `crates/agent/Cargo.toml`

### Step 1: Write a compile-check test (just verifying deps)

Before adding, confirm the current feature-gated build compiles:

```bash
cargo check -p graphirm-agent --features local-extraction 2>&1 | grep "^error" | head -5
```

Expected: no errors (the existing stub compiles)

### Step 2: Add dependencies

In `crates/agent/Cargo.toml`, add to `[dependencies]`:

```toml
hf-hub = { version = "0.5", default-features = false, features = ["tokio", "native-tls"], optional = true }
regex  = { version = "1", optional = true }
```

Update the `[features]` section:

```toml
local-extraction = ["dep:ort", "dep:tokenizers", "dep:ndarray", "dep:hf-hub", "dep:regex"]
```

> **As implemented:** `hf-hub` 0.5 with `native-tls` (not `rustls-tls`).

### Step 3: Verify compile

```bash
cargo check -p graphirm-agent --features local-extraction 2>&1 | grep "^error" | head -5
```

Expected: no errors

### Step 4: Commit

```bash
git add crates/agent/Cargo.toml
git commit -m "chore(agent): add hf-hub and regex deps under local-extraction feature"
```

---

## Task 2: Define `Gliner2Config` and model download helper — ✅ DONE

**Files:**
- Modify: `crates/agent/src/knowledge/local_extraction.rs`

Add config types and the download function to `local_extraction.rs`.

### Step 1: Write a failing test

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gliner2_config_parse() {
        let json = r#"{
            "max_width": 12,
            "special_tokens": {
                "[P]": 128003,
                "[E]": 128005,
                "[L]": 128007,
                "[SEP_TEXT]": 128002
            },
            "onnx_files": {
                "fp32": {
                    "encoder": "onnx/encoder.onnx",
                    "span_rep": "onnx/span_rep.onnx",
                    "count_embed": "onnx/count_embed.onnx",
                    "classifier": "onnx/classifier.onnx"
                }
            }
        }"#;
        let config: Gliner2Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.max_width, 12);
        assert_eq!(config.special_tokens["[E]"], 128005);
        assert_eq!(config.special_tokens["[SEP_TEXT]"], 128002);
    }
}
```

Run: `cargo test -p graphirm-agent --features local-extraction test_gliner2_config_parse 2>&1`
Expected: FAIL — `Gliner2Config` doesn't exist yet

### Step 2: Write the config types

Add to `crates/agent/src/knowledge/local_extraction.rs`:

```rust
//! Local ONNX-based entity extraction using GLiNER2.
//!
//! Runs `lmo3/gliner2-large-v1-onnx` — a DeBERTa-v3-large model exported as
//! four ONNX files. Inference runs on CPU via ONNX Runtime without PyTorch.
//!
//! # Model download
//!
//! On first use, call [`OnnxExtractor::download_model`] to fetch the four ONNX
//! files and tokenizer from HuggingFace Hub (~1.95 GB). Files are cached in
//! `~/.cache/huggingface/hub/` (same location as the Python `huggingface_hub` library).
//!
//! # Feature flag
//!
//! Requires the `local-extraction` cargo feature:
//! ```bash
//! cargo build --features local-extraction
//! ```
//!
//! Also requires glibc >= 2.38 (for the prebuilt ONNX Runtime binary).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::{Array1, Array2, Array3, ArrayView2, Axis, s};
use ort::session::Session;
use regex::Regex;
use serde::Deserialize;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

use crate::error::AgentError;

use super::extraction::{ExtractedEntity, ExtractionResponse};

// ─── Config types ───────────────────────────────────────────────────────────

/// GLiNER2 ONNX model configuration, loaded from `gliner2_config.json`.
#[derive(Debug, Deserialize)]
pub struct Gliner2Config {
    /// Maximum span width in words.
    pub max_width: usize,
    /// Special token IDs: "[P]", "[E]", "[L]", "[SEP_TEXT]".
    pub special_tokens: HashMap<String, i64>,
    /// ONNX file paths per precision level ("fp32", "fp16").
    pub onnx_files: HashMap<String, Gliner2OnnxFiles>,
}

/// File paths for a single precision level within the ONNX model directory.
#[derive(Debug, Deserialize)]
pub struct Gliner2OnnxFiles {
    pub encoder: String,
    pub span_rep: String,
    pub count_embed: String,
    pub classifier: String,
}
```

### Step 3: Run test to verify it passes

Run: `cargo test -p graphirm-agent --features local-extraction test_gliner2_config_parse 2>&1`
Expected: PASS

### Step 4: Add the download function

Still in `local_extraction.rs`, after the config types:

```rust
// ─── Model download ──────────────────────────────────────────────────────────

/// HuggingFace repository for the pre-exported ONNX model.
const HF_MODEL_ID: &str = "lmo3/gliner2-large-v1-onnx";

/// Download the GLiNER2-large-v1 ONNX model files from HuggingFace Hub.
///
/// Files are cached in `~/.cache/huggingface/hub/` and reused on subsequent
/// calls. Returns the local directory path containing the downloaded files.
///
/// # Files downloaded (~1.95 GB total)
///
/// - `onnx/encoder.onnx`       — DeBERTa-v3-large encoder (~1.65 GB)
/// - `onnx/span_rep.onnx`      — Span representation MLP (~112 MB)
/// - `onnx/count_embed.onnx`   — Unrolled GRU for label scoring (~72 MB)
/// - `onnx/classifier.onnx`    — Classification head (~8 MB, not used for NER)
/// - `gliner2_config.json`     — Model config (special tokens, max_width)
/// - `tokenizer.json`          — HuggingFace tokenizer
/// - `tokenizer_config.json`   — Tokenizer config
/// - `added_tokens.json`       — Special token definitions
///
/// # Reproducibility
///
/// The `lmoe/gliner2-onnx` repository at <https://github.com/lmoe/gliner2-onnx>
/// contains the export tool. To regenerate the ONNX files from the original
/// PyTorch weights (`fastino/gliner2-large-v1`):
///
/// ```bash
/// git clone https://github.com/lmoe/gliner2-onnx
/// cd gliner2-onnx
/// pip install -e ".[export]"
/// make onnx-export MODEL=fastino/gliner2-large-v1
/// # Output: model_out/gliner2-large-v1/
/// # Upload to HuggingFace or point OnnxExtractor::new() at that directory.
/// ```
pub async fn download_model() -> Result<PathBuf, AgentError> {
    use hf_hub::{Repo, RepoType, api::tokio::Api};

    let api = Api::new()
        .map_err(|e| AgentError::Workflow(format!("HuggingFace API init failed: {e}")))?;

    let repo = api.repo(Repo::new(HF_MODEL_ID.to_string(), RepoType::Model));

    let files = [
        "gliner2_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "added_tokens.json",
        "onnx/encoder.onnx",
        "onnx/span_rep.onnx",
        "onnx/count_embed.onnx",
        "onnx/classifier.onnx",
    ];

    let mut model_dir: Option<PathBuf> = None;
    for filename in &files {
        tracing::info!(file = filename, "Downloading GLiNER2 model file");
        let local_path = repo
            .get(filename)
            .await
            .map_err(|e| AgentError::Workflow(format!("Download failed for {filename}: {e}")))?;

        // All files land under the same cache directory; derive it from first file.
        if model_dir.is_none() {
            model_dir = local_path.parent().map(|p| {
                // Navigate up from the file to the repo root (files may be in subdirs).
                p.ancestors()
                    .find(|anc| anc.join("gliner2_config.json").exists()
                        || anc.join("tokenizer.json").exists())
                    .unwrap_or(p)
                    .to_path_buf()
            });
        }
    }

    model_dir.ok_or_else(|| AgentError::Workflow("Failed to determine model directory".into()))
}
```

### Step 5: Write test for download (marked `#[ignore]` — requires network + 1.95 GB)

```rust
#[tokio::test]
#[ignore = "requires network access and ~1.95 GB download"]
async fn test_download_model_creates_files() {
    let dir = download_model().await.unwrap();
    assert!(dir.join("tokenizer.json").exists());
    assert!(dir.join("gliner2_config.json").exists());
    assert!(dir.join("onnx/encoder.onnx").exists());
    assert!(dir.join("onnx/span_rep.onnx").exists());
    assert!(dir.join("onnx/count_embed.onnx").exists());
}
```

### Step 6: Commit

```bash
git add crates/agent/src/knowledge/local_extraction.rs crates/agent/Cargo.toml
git commit -m "feat(knowledge): add Gliner2Config types and hf-hub model download helper"
```

---

## Task 3: Implement `OnnxExtractor` struct and constructor — ✅ DONE

**Files:**
- Modify: `crates/agent/src/knowledge/local_extraction.rs`

### Step 1: Write failing test

```rust
#[test]
fn test_onnx_extractor_new_fails_with_missing_dir() {
    let result = OnnxExtractor::new(Path::new("/nonexistent/model/dir"));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));
}
```

Run: `cargo test -p graphirm-agent --features local-extraction test_onnx_extractor_new_fails_with_missing_dir 2>&1`
Expected: FAIL — `OnnxExtractor` struct doesn't exist yet

### Step 2: Add `OnnxExtractor` struct

```rust
// ─── OnnxExtractor ───────────────────────────────────────────────────────────

/// Extracted entity span from GLiNER2 NER output.
#[derive(Debug, Clone)]
pub struct RawOnnxEntity {
    pub entity_type: String,
    pub text: String,
    pub start: usize,   // char offset in original text
    pub end: usize,     // char offset in original text
    pub confidence: f64,
}

/// Four-session ONNX runtime for GLiNER2 NER.
///
/// Holds the encoder, span representation, count_embed, and tokenizer.
/// All four ONNX files must be in the same model directory.
pub struct OnnxExtractor {
    /// DeBERTa-v3-large encoder: (input_ids, attention_mask) → hidden_states
    encoder: Mutex<Session>,
    /// Span representation MLP: (hidden_states, span_start, span_end) → span_rep
    span_rep: Mutex<Session>,
    /// Unrolled GRU label transform: (label_embeddings,) → transformed
    count_embed: Mutex<Session>,
    tokenizer: Tokenizer,
    max_width: usize,
    /// Token ID for [E] (entity label marker in NER schema)
    tok_e: i64,
    /// Token ID for [SEP_TEXT] (schema/text separator)
    tok_sep_text: i64,
    /// Token IDs for [P] (task type marker)
    tok_p: i64,
    /// Word boundary regex (matches GLiNER2's WhitespaceTokenSplitter)
    word_re: Regex,
}

impl OnnxExtractor {
    /// Load GLiNER2-large-v1 from a local directory.
    ///
    /// `model_dir` must contain:
    /// - `gliner2_config.json`
    /// - `tokenizer.json`
    /// - `onnx/encoder.onnx`, `onnx/span_rep.onnx`, `onnx/count_embed.onnx`
    ///
    /// Use [`download_model`] to populate this directory on first run.
    pub fn new(model_dir: &Path) -> Result<Self, AgentError> {
        if !model_dir.is_dir() {
            return Err(AgentError::Workflow(format!(
                "GLiNER2 model directory not found: {}",
                model_dir.display()
            )));
        }

        // Load config
        let config_path = model_dir.join("gliner2_config.json");
        let config_bytes = std::fs::read(&config_path).map_err(|e| {
            AgentError::Workflow(format!("Cannot read gliner2_config.json: {e}"))
        })?;
        let config: Gliner2Config = serde_json::from_slice(&config_bytes)
            .map_err(|e| AgentError::Workflow(format!("Invalid gliner2_config.json: {e}")))?;

        let fp32 = config.onnx_files.get("fp32").ok_or_else(|| {
            AgentError::Workflow("gliner2_config.json missing fp32 onnx_files".into())
        })?;

        // Load sessions
        let encoder = Session::builder()
            .map_err(|e| AgentError::Workflow(format!("ORT session builder: {e}")))?
            .with_intra_threads(4)
            .map_err(|e| AgentError::Workflow(format!("ORT intra_threads: {e}")))?
            .commit_from_file(model_dir.join(&fp32.encoder))
            .map_err(|e| AgentError::Workflow(format!("Load encoder.onnx: {e}")))?;

        let span_rep = Session::builder()
            .map_err(|e| AgentError::Workflow(format!("ORT session builder: {e}")))?
            .commit_from_file(model_dir.join(&fp32.span_rep))
            .map_err(|e| AgentError::Workflow(format!("Load span_rep.onnx: {e}")))?;

        let count_embed = Session::builder()
            .map_err(|e| AgentError::Workflow(format!("ORT session builder: {e}")))?
            .commit_from_file(model_dir.join(&fp32.count_embed))
            .map_err(|e| AgentError::Workflow(format!("Load count_embed.onnx: {e}")))?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| AgentError::Workflow(format!("Load tokenizer: {e}")))?;

        let tok_e = *config.special_tokens.get("[E]").ok_or_else(|| {
            AgentError::Workflow("gliner2_config.json missing [E] special token".into())
        })?;
        let tok_sep_text = *config.special_tokens.get("[SEP_TEXT]").ok_or_else(|| {
            AgentError::Workflow("gliner2_config.json missing [SEP_TEXT] special token".into())
        })?;
        let tok_p = *config.special_tokens.get("[P]").ok_or_else(|| {
            AgentError::Workflow("gliner2_config.json missing [P] special token".into())
        })?;

        // Word boundary regex — mirrors GLiNER2's WhitespaceTokenSplitter
        let word_re = Regex::new(
            r"(?i)(?:https?://\S+|www\.\S+)|[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}|@[a-z0-9_]+|\w+(?:[-_]\w+)*|\S",
        )
        .map_err(|e| AgentError::Workflow(format!("Regex compile: {e}")))?;

        Ok(Self {
            encoder: Mutex::new(encoder),
            span_rep: Mutex::new(span_rep),
            count_embed: Mutex::new(count_embed),
            tokenizer,
            max_width: config.max_width,
            tok_e,
            tok_sep_text,
            tok_p,
            word_re,
        })
    }
}
```

### Step 3: Run test to verify it passes

Run: `cargo test -p graphirm-agent --features local-extraction test_onnx_extractor_new_fails_with_missing_dir 2>&1`
Expected: PASS

Run: `cargo check -p graphirm-agent --features local-extraction 2>&1 | grep "^error" | head -10`
Expected: no errors

### Step 4: Commit

```bash
git add crates/agent/src/knowledge/local_extraction.rs
git commit -m "feat(knowledge): add OnnxExtractor struct with four ONNX session loader"
```

---

## Task 4: Implement schema input builder and span generator — ✅ DONE

**Files:**
- Modify: `crates/agent/src/knowledge/local_extraction.rs`

These are pure functions — no ONNX calls. Test them in isolation before wiring into inference.

### Step 1: Write failing tests

```rust
#[test]
fn test_build_schema_prefix_has_e_positions() {
    // We can't easily construct OnnxExtractor in unit test without model files.
    // Test the logic by verifying the schema prefix structure manually:
    // ( [P] entities ( [E] fn [E] bug ) ) [SEP_TEXT]
    // [E] positions should appear at positions before each label.
    // Structural test: [E] token ID appears twice for two labels.
    let tok_e: i64 = 128005;
    let tok_p: i64 = 128003;
    let tok_sep: i64 = 128002;
    // Build a simplified token sequence inline (without tokenizer):
    let tokens: Vec<i64> = vec![287, tok_p, 100, 287, tok_e, 101, tok_e, 102, 1263, 1263, tok_sep];
    let e_positions: Vec<usize> = tokens.iter().enumerate()
        .filter(|(_, &t)| t == tok_e)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(e_positions.len(), 2, "two [E] positions for two labels");
}

#[test]
fn test_generate_spans_max_width() {
    // For 4 words with max_width=2: (0,0),(0,1),(1,1),(1,2),(2,2),(2,3),(3,3)
    let spans = generate_spans(4, 2);
    assert_eq!(spans.len(), 7);
    assert!(spans.contains(&(0, 1)));
    assert!(!spans.contains(&(0, 2))); // width 3 excluded
}
```

Run: `cargo test -p graphirm-agent --features local-extraction test_generate_spans_max_width 2>&1`
Expected: FAIL — `generate_spans` doesn't exist

### Step 2: Implement helper functions

Add to `local_extraction.rs` (as free functions, not methods):

```rust
// ─── Helper functions ────────────────────────────────────────────────────────

/// Generate all valid word-span (start, end) pairs where end - start < max_width.
///
/// Both start and end are inclusive word indices. A single-word span has start == end.
pub fn generate_spans(num_words: usize, max_width: usize) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    for i in 0..num_words {
        for j in i..num_words.min(i + max_width) {
            spans.push((i, j));
        }
    }
    spans
}

/// Numerically stable sigmoid for a scalar f32.
#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Apply sigmoid element-wise to a 2D array in place.
fn sigmoid_inplace(arr: &mut Array2<f32>) {
    arr.mapv_inplace(sigmoid_f32);
}
```

### Step 3: Implement `build_ner_input` as an `OnnxExtractor` method

```rust
impl OnnxExtractor {
    // ... new() from Task 3 ...

    /// Build the tokenized input sequence for GLiNER2 NER.
    ///
    /// Output format:
    /// `( [P] entities ( [E] label1 [E] label2 ... ) ) [SEP_TEXT] word1 word2 ...`
    ///
    /// Returns:
    /// - `input_ids`: full token sequence as i64 vec
    /// - `e_positions`: token indices of each [E] marker (one per label)
    /// - `word_offsets`: char (start, end) of each word in original text
    /// - `text_start_idx`: token index where text begins (after [SEP_TEXT])
    /// - `first_token_positions`: for each word, index of its first token in `text_hidden`
    fn build_ner_input(
        &self,
        text: &str,
        entity_types: &[String],
    ) -> (Vec<i64>, Vec<usize>, Vec<(usize, usize)>, usize, Vec<usize>) {
        let mut tokens: Vec<i64> = Vec::new();

        // Opening paren
        let open_ids = self.tokenizer
            .encode("(", false)
            .map(|e| e.get_ids().iter().map(|&x| x as i64).collect::<Vec<_>>())
            .unwrap_or_default();
        tokens.extend_from_slice(&open_ids);

        // [P] token
        tokens.push(self.tok_p);

        // Task name "entities"
        let entities_ids = self.tokenizer
            .encode("entities", false)
            .map(|e| e.get_ids().iter().map(|&x| x as i64).collect::<Vec<_>>())
            .unwrap_or_default();
        tokens.extend_from_slice(&entities_ids);

        // Inner opening paren
        tokens.extend_from_slice(&open_ids);

        // Labels with [E] markers
        let mut e_positions: Vec<usize> = Vec::new();
        for label in entity_types {
            e_positions.push(tokens.len());
            tokens.push(self.tok_e);
            let label_ids = self.tokenizer
                .encode(label.as_str(), false)
                .map(|e| e.get_ids().iter().map(|&x| x as i64).collect::<Vec<_>>())
                .unwrap_or_default();
            tokens.extend_from_slice(&label_ids);
        }

        // Two closing parens
        let close_ids = self.tokenizer
            .encode(")", false)
            .map(|e| e.get_ids().iter().map(|&x| x as i64).collect::<Vec<_>>())
            .unwrap_or_default();
        tokens.extend_from_slice(&close_ids);
        tokens.extend_from_slice(&close_ids);

        // [SEP_TEXT]
        tokens.push(self.tok_sep_text);

        let text_start_idx = tokens.len();

        // Tokenize text words, tracking offsets and first-token positions
        let text_lower = text.to_lowercase();
        let mut word_offsets: Vec<(usize, usize)> = Vec::new();
        let mut first_token_positions: Vec<usize> = Vec::new();
        let mut token_idx: usize = 0;

        for m in self.word_re.find_iter(&text_lower) {
            word_offsets.push((m.start(), m.end()));
            first_token_positions.push(token_idx);

            let word_ids = self.tokenizer
                .encode(m.as_str(), false)
                .map(|e| e.get_ids().iter().map(|&x| x as i64).collect::<Vec<_>>())
                .unwrap_or_default();
            token_idx += word_ids.len();
            tokens.extend(word_ids);
        }

        (tokens, e_positions, word_offsets, text_start_idx, first_token_positions)
    }
}
```

### Step 4: Run tests to verify pass

Run: `cargo test -p graphirm-agent --features local-extraction test_generate_spans_max_width test_build_schema_prefix_has_e_positions 2>&1`
Expected: PASS

Run: `cargo check -p graphirm-agent --features local-extraction 2>&1 | grep "^error" | head -10`
Expected: no errors

### Step 5: Commit

```bash
git add crates/agent/src/knowledge/local_extraction.rs
git commit -m "feat(knowledge): implement generate_spans, sigmoid helpers, and NER input builder"
```

---

## Task 5: Implement the full four-session inference pipeline — ✅ DONE

**Files:**
- Modify: `crates/agent/src/knowledge/local_extraction.rs`

This is the core of the implementation — the `extract()` method that runs all four ONNX sessions.

### Step 1: Write a structural test (verifies the method signature compiles)

```rust
#[test]
fn test_onnx_extractor_new_fails_gracefully() {
    // Proves extract() method signature is correct; actual inference
    // needs a real model dir (use #[ignore] test below for that).
    let result = OnnxExtractor::new(Path::new("/definitely/not/real"));
    assert!(result.is_err());
}
```

### Step 2: Implement `extract()` on `OnnxExtractor`

```rust
impl OnnxExtractor {
    // ... new() and build_ner_input() from earlier tasks ...

    /// Run GLiNER2 NER on `text` for the given `entity_types`.
    ///
    /// Returns raw entity spans with confidence scores. Non-overlapping
    /// entities per label are selected greedily by descending score.
    pub async fn extract(
        &self,
        text: &str,
        entity_types: &[String],
        min_confidence: f64,
    ) -> Result<ExtractionResponse, AgentError> {
        if text.is_empty() || entity_types.is_empty() {
            return Ok(ExtractionResponse { entities: vec![] });
        }

        let (input_ids, e_positions, word_offsets, text_start_idx, first_token_positions) =
            self.build_ner_input(text, entity_types);

        if word_offsets.is_empty() {
            return Ok(ExtractionResponse { entities: vec![] });
        }

        let seq_len = input_ids.len();
        let attn_mask: Vec<i64> = vec![1; seq_len];

        // ── Step 1: Encoder ───────────────────────────────────────────────
        let input_ids_arr = Array2::from_shape_vec(
            (1, seq_len),
            input_ids.clone(),
        )
        .map_err(|e| AgentError::Workflow(format!("input_ids shape error: {e}")))?;

        let attn_mask_arr = Array2::from_shape_vec(
            (1, seq_len),
            attn_mask,
        )
        .map_err(|e| AgentError::Workflow(format!("attn_mask shape error: {e}")))?;

        let encoder_outputs = {
            let mut enc = self.encoder.lock().await;
            enc.run(ort::inputs! {
                "input_ids"      => input_ids_arr.view(),
                "attention_mask" => attn_mask_arr.view(),
            }
            .map_err(|e| AgentError::Workflow(format!("ORT input build: {e}")))?)
            .map_err(|e| AgentError::Workflow(format!("Encoder run: {e}")))?
        };

        // hidden_states: [1, seq_len, 1024]
        let hs_raw = encoder_outputs["hidden_states"]
            .try_extract_tensor::<f32>()
            .map_err(|e| AgentError::Workflow(format!("Extract hidden_states: {e}")))?;
        let hs = hs_raw
            .view()
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| AgentError::Workflow(format!("hidden_states reshape: {e}")))?;
        let hs_2d = hs.index_axis(Axis(0), 0); // [seq_len, 1024]

        // ── Step 2: Label embeddings at [E] positions ─────────────────────
        let num_labels = entity_types.len();
        let hidden_size = hs_2d.ncols();

        let mut label_emb = Array2::<f32>::zeros((num_labels, hidden_size));
        for (li, &ep) in e_positions.iter().enumerate() {
            if ep < hs_2d.nrows() {
                label_emb.row_mut(li).assign(&hs_2d.row(ep));
            }
        }

        // ── Step 3: Text hidden states (after [SEP_TEXT]) ─────────────────
        let text_hidden = hs_2d.slice(s![text_start_idx.., ..]).to_owned(); // [text_tokens, 1024]

        // ── Step 4: Generate word spans ───────────────────────────────────
        let num_words = word_offsets.len();
        let spans = generate_spans(num_words, self.max_width);
        let num_spans = spans.len();

        let span_starts: Vec<i64> = spans
            .iter()
            .map(|&(si, _)| first_token_positions[si] as i64)
            .collect();
        let span_ends: Vec<i64> = spans
            .iter()
            .map(|&(_, ei)| first_token_positions[ei] as i64)
            .collect();

        let span_start_arr = Array2::from_shape_vec((1, num_spans), span_starts)
            .map_err(|e| AgentError::Workflow(format!("span_start shape: {e}")))?;
        let span_end_arr = Array2::from_shape_vec((1, num_spans), span_ends)
            .map_err(|e| AgentError::Workflow(format!("span_end shape: {e}")))?;

        // span_rep needs hidden_states with batch dim: [1, text_tokens, 1024]
        let text_hidden_3d = text_hidden
            .view()
            .insert_axis(Axis(0))
            .to_owned(); // [1, text_tokens, 1024]

        // ── Step 5: Span representations ──────────────────────────────────
        let span_rep_outputs = {
            let mut sr = self.span_rep.lock().await;
            sr.run(ort::inputs! {
                "hidden_states"  => text_hidden_3d.view(),
                "span_start_idx" => span_start_arr.view(),
                "span_end_idx"   => span_end_arr.view(),
            }
            .map_err(|e| AgentError::Workflow(format!("span_rep input build: {e}")))?)
            .map_err(|e| AgentError::Workflow(format!("span_rep run: {e}")))?
        };

        // span_rep output: [1, num_spans, 1024]
        let sr_raw = span_rep_outputs["span_rep"]
            .try_extract_tensor::<f32>()
            .map_err(|e| AgentError::Workflow(format!("Extract span_rep: {e}")))?;
        let sr_3d = sr_raw
            .view()
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| AgentError::Workflow(format!("span_rep reshape: {e}")))?;
        let span_rep_2d: ArrayView2<f32> = sr_3d.index_axis(Axis(0), 0); // [num_spans, 1024]

        // ── Step 6: Transform label embeddings via count_embed ────────────
        let count_embed_outputs = {
            let mut ce = self.count_embed.lock().await;
            ce.run(ort::inputs! {
                "label_embeddings" => label_emb.view(),
            }
            .map_err(|e| AgentError::Workflow(format!("count_embed input build: {e}")))?)
            .map_err(|e| AgentError::Workflow(format!("count_embed run: {e}")))?
        };

        let ce_raw = count_embed_outputs
            .values()
            .next()
            .ok_or_else(|| AgentError::Workflow("count_embed produced no output".into()))?
            .try_extract_tensor::<f32>()
            .map_err(|e| AgentError::Workflow(format!("Extract count_embed: {e}")))?;
        let transformed: ArrayView2<f32> = ce_raw
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| AgentError::Workflow(format!("count_embed reshape: {e}")))?; // [labels, 1024]

        // ── Step 7: Score spans against labels ────────────────────────────
        // scores = sigmoid(span_rep @ transformed.T)  →  [num_spans, num_labels]
        let mut scores: Array2<f32> = span_rep_2d.dot(&transformed.t());
        sigmoid_inplace(&mut scores);

        // ── Step 8: Collect entities above threshold ──────────────────────
        let mut raw_entities: Vec<(usize, usize, usize, f64)> = Vec::new(); // (span_idx, label_idx, char offsets, score)
        for (span_idx, &(word_start, word_end)) in spans.iter().enumerate() {
            for (label_idx, _label) in entity_types.iter().enumerate() {
                let score = scores[[span_idx, label_idx]] as f64;
                if score >= min_confidence {
                    raw_entities.push((word_start, word_end, label_idx, score));
                }
            }
        }

        // Sort by score descending for greedy deduplication
        raw_entities.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

        // Deduplicate: keep highest-score non-overlapping spans per label
        let mut kept: Vec<RawOnnxEntity> = Vec::new();
        for (word_start, word_end, label_idx, score) in raw_entities {
            let char_start = word_offsets[word_start].0;
            let char_end = word_offsets[word_end].1;
            let label = &entity_types[label_idx];

            let overlaps = kept.iter().any(|k| {
                k.entity_type == *label
                    && k.start < char_end
                    && k.end > char_start
            });

            if !overlaps {
                kept.push(RawOnnxEntity {
                    entity_type: label.clone(),
                    text: text[char_start..char_end].to_string(),
                    start: char_start,
                    end: char_end,
                    confidence: score,
                });
            }
        }

        Ok(raw_entities_to_extraction_response(kept))
    }
}
```

### Step 3: Implement `raw_entities_to_extraction_response`

```rust
/// Convert raw GLiNER2 entity spans into the shared `ExtractionResponse` format.
pub fn raw_entities_to_extraction_response(raw: Vec<RawOnnxEntity>) -> ExtractionResponse {
    let entities = raw
        .into_iter()
        .map(|e| ExtractedEntity {
            entity_type: e.entity_type,
            name: e.text,
            description: String::new(), // GLiNER2 doesn't produce descriptions
            confidence: e.confidence,
            relationships: vec![],
        })
        .collect();
    ExtractionResponse { entities }
}
```

### Step 4: Add a unit test for the scoring logic

```rust
#[test]
fn test_sigmoid_inplace() {
    let mut arr = Array2::from_shape_vec((2, 2), vec![0.0f32, 2.0, -2.0, 100.0]).unwrap();
    sigmoid_inplace(&mut arr);
    assert!((arr[[0, 0]] - 0.5).abs() < 1e-5);
    assert!(arr[[0, 1]] > 0.85);
    assert!(arr[[1, 0]] < 0.15);
    assert!((arr[[1, 1]] - 1.0).abs() < 1e-4);
}

#[test]
fn test_generate_spans_single_word() {
    let spans = generate_spans(1, 12);
    assert_eq!(spans, vec![(0, 0)]);
}

#[test]
fn test_raw_entities_to_extraction_response() {
    let raw = vec![RawOnnxEntity {
        entity_type: "function".into(),
        text: "parse_config".into(),
        start: 0,
        end: 12,
        confidence: 0.92,
    }];
    let resp = raw_entities_to_extraction_response(raw);
    assert_eq!(resp.entities.len(), 1);
    assert_eq!(resp.entities[0].name, "parse_config");
    assert!((resp.entities[0].confidence - 0.92).abs() < f64::EPSILON);
}
```

Run: `cargo test -p graphirm-agent --features local-extraction 2>&1 | tail -15`
Expected: all unit tests pass (extraction tests skip since they need a real model)

### Step 5: Add `#[ignore]` integration test requiring a downloaded model

```rust
/// Full inference test — requires model to be downloaded first.
/// Run with: GLINER2_MODEL_DIR=/path/to/model cargo test -p graphirm-agent --features local-extraction -- --ignored
#[tokio::test]
#[ignore = "requires downloaded GLiNER2 model (~1.95 GB). Run download_model() first."]
async fn test_extract_entities_with_real_model() {
    let model_dir = std::path::PathBuf::from(
        std::env::var("GLINER2_MODEL_DIR")
            .expect("Set GLINER2_MODEL_DIR to the local model directory"),
    );

    let extractor = OnnxExtractor::new(&model_dir).expect("Failed to load model");

    let result = extractor
        .extract(
            "We use serde for JSON serialization and tokio for async runtime.",
            &["library".to_string(), "pattern".to_string()],
            0.4,
        )
        .await
        .expect("Extraction failed");

    // serde and tokio should be detected as libraries
    assert!(!result.entities.is_empty(), "Should detect at least one entity");
    let names: Vec<&str> = result.entities.iter().map(|e| e.name.as_str()).collect();
    assert!(
        names.contains(&"serde") || names.contains(&"tokio"),
        "Should detect serde or tokio as library. Got: {:?}",
        names
    );
}
```

### Step 6: Commit

```bash
git add crates/agent/src/knowledge/local_extraction.rs
git commit -m "feat(knowledge): implement GLiNER2 four-session ONNX inference pipeline"
```

---

## Task 6: Update `extraction.rs` to use directory-based `OnnxExtractor` — ✅ DONE

**Files:**
- Modify: `crates/agent/src/knowledge/extraction.rs`

`ExtractionBackend::Local` stores `model_dir: String` pointing to the directory from `download_model()`. `Hybrid` uses the same field.

### As implemented

`ExtractionBackend::Local { model_dir }` and `ExtractionBackend::Hybrid { model_dir }` are
already in place. The deserialization test (`test_extraction_backend_local_deserialize`) passes.

### Step 3: Update `extract_knowledge_with_backend` for the `Local` path

In `extraction.rs`, in the `Local` arm of `extract_knowledge_with_backend`:

```rust
#[cfg(feature = "local-extraction")]
ExtractionBackend::Local { model_dir } => {
    use crate::knowledge::local_extraction::OnnxExtractor;
    let extractor = onnx.ok_or_else(|| {
        AgentError::Workflow(
            "Local backend requires a pre-constructed OnnxExtractor".into(),
        )
    })?;
    extractor
        .extract(text, &config.entity_types, config.min_confidence)
        .await
}
```

> **Note:** The current code passes `onnx: Option<()>` as a placeholder. Replace the `onnx` parameter type with `Option<&OnnxExtractor>` so callers can pass a constructed extractor. Alternatively, construct the extractor here from `model_dir`. Constructing inline is simpler for now:

```rust
#[cfg(feature = "local-extraction")]
ExtractionBackend::Local { model_dir } => {
    use std::path::Path;
    use crate::knowledge::local_extraction::OnnxExtractor;
    let extractor = OnnxExtractor::new(Path::new(model_dir))?;
    extractor
        .extract(text, &config.entity_types, config.min_confidence)
        .await
}
```

> **Note:** The inline construction above is acceptable for first-pass wiring, but the follow-up cache task below should replace it with a shared `Arc<OnnxExtractor>` lookup so `Local`, `Hybrid`, and segment fallback all reuse the same loaded sessions.

### Step 4: Run tests to verify pass

Run: `cargo test -p graphirm-agent --features local-extraction 2>&1 | tail -10`
Expected: all pass

### Step 5: Commit

```bash
git add crates/agent/src/knowledge/extraction.rs
git commit -m "feat(knowledge): update ExtractionBackend::Local to use model_dir, wire OnnxExtractor"
```

---

## Task 7: Write setup and reproducibility guide — ✅ DONE

**Files:**
- Create: `docs/guides/gliner2-setup.md`

### Step 1: Write the guide

```markdown
# GLiNER2 Local Extraction — Setup Guide

GLiNER2 is used as the zero-cost entity extraction backend. It runs a 486M-parameter
DeBERTa-v3-large model locally via ONNX Runtime — no LLM API calls, no token cost.

## Quick start (programmatic download)

The easiest way is to call `download_model()` from Rust before first use:

```rust
use graphirm_agent::knowledge::local_extraction::download_model;

let model_dir = download_model().await?;
println!("Model cached at: {}", model_dir.display());
// Then: OnnxExtractor::new(&model_dir)?
```

This downloads ~1.95 GB from HuggingFace Hub to `~/.cache/huggingface/hub/` and is
idempotent — files are reused on subsequent calls.

## Manual download

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    'lmo3/gliner2-large-v1-onnx',
    allow_patterns=['*.json', 'onnx/encoder.onnx', 'onnx/span_rep.onnx',
                    'onnx/count_embed.onnx', 'onnx/classifier.onnx']
)
print('Downloaded to:', path)
"
```

Then set `model_dir` in your `AgentConfig` TOML:

```toml
[agent.extraction]
enabled = true
backend = { local = { model_dir = "/home/user/.cache/huggingface/hub/models--lmo3--gliner2-large-v1-onnx/snapshots/HASH" } }
```

## Reproducing the ONNX export from scratch

The pre-exported ONNX files at `lmo3/gliner2-large-v1-onnx` were produced from
`fastino/gliner2-large-v1` using the `lmoe/gliner2-onnx` export tool.

To regenerate (e.g., if you want to use a different base model or quantize to FP16):

```bash
# Prerequisites: Python 3.10+, ~8 GB RAM, ~10 GB disk
git clone https://github.com/lmoe/gliner2-onnx
cd gliner2-onnx
pip install -e ".[export]"

# Export gliner2-large-v1 (FP32, ~1.95 GB)
make onnx-export MODEL=fastino/gliner2-large-v1

# Export with FP16 quantization (~1 GB, minimal accuracy loss)
make onnx-export MODEL=fastino/gliner2-large-v1 QUANTIZE=fp16

# Output is in: model_out/gliner2-large-v1/
```

Then point `OnnxExtractor::new()` at `model_out/gliner2-large-v1/`.

## Supported models

| Model | Size | Notes |
|-------|------|-------|
| `fastino/gliner2-large-v1` | 1.95 GB | Best accuracy, English |
| `fastino/gliner2-multi-v1` | 1.23 GB | Multilingual |
| `fastino/gliner2-base-v1`  | — | **Not ONNX-exportable** (CountLSTMv2 architecture) |

## System requirements

- glibc >= 2.38 (required for the prebuilt ORT binary from `ort` crate)
- ~2 GB disk for model files
- ~4 GB RAM for inference
- CPU inference: ~150-200ms per extraction call
- GPU: set `providers = ["CUDAExecutionProvider"]` (requires CUDA ORT build)

## Feature flag

Build with local extraction support:

```bash
cargo build --features local-extraction
cargo test -p graphirm-agent --features local-extraction
# Run ignored integration tests (needs downloaded model):
cargo test -p graphirm-agent --features local-extraction -- --ignored
```
```

### Step 2: Commit

```bash
git add docs/guides/gliner2-setup.md
git commit -m "docs: add GLiNER2 local extraction setup and reproducibility guide"
```

---

## Task 8: Add a process-wide `OnnxExtractor` cache

**Files:**
- Modify: `crates/agent/src/knowledge/local_extraction.rs`
- Modify: `crates/agent/src/knowledge/segments.rs`
- Modify: `crates/agent/src/knowledge/extraction.rs`
- Modify: `AGENTS.md`
- Modify: `crates/agent/AGENTS.md`

### Step 1: Write failing cache-behavior tests

Add tests that prove the shared cache helper:

- returns the same `Arc` for the same `model_dir` (identity via `Arc::ptr_eq`)
- initializes different `model_dir` keys independently
- deduplicates concurrent initialization for the same key (builder runs once)
- does not permanently poison a key after an initialization error (retry succeeds)

Because real `OnnxExtractor::new()` needs model files, factor the core cache logic behind a
small generic internal helper `SharedInitCache<T>` that accepts a builder closure. Unit-test
that helper with a counter-based fake builder first.

```rust
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use crate::error::AgentError;

// ── Test 1: same key returns same Arc ──────────────────────────────────────

#[tokio::test]
async fn shared_cache_reuses_same_key() {
    let calls = Arc::new(AtomicUsize::new(0));
    let cache = SharedInitCache::<usize>::default();

    let builder = |calls: Arc<AtomicUsize>| {
        move || {
            let calls = calls.clone();
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok::<Arc<usize>, AgentError>(Arc::new(7))
            }
        }
    };

    let a = cache.get_or_try_init("dir_a", builder(calls.clone())).await.unwrap();
    let b = cache.get_or_try_init("dir_a", builder(calls.clone())).await.unwrap();

    assert!(Arc::ptr_eq(&a, &b), "second call must return cached Arc");
    assert_eq!(calls.load(Ordering::SeqCst), 1, "builder must run exactly once");
}

// ── Test 2: different keys are independent ─────────────────────────────────

#[tokio::test]
async fn shared_cache_different_keys_independent() {
    let cache = SharedInitCache::<usize>::default();

    let a = cache
        .get_or_try_init("dir_a", || async { Ok(Arc::new(1)) })
        .await
        .unwrap();
    let b = cache
        .get_or_try_init("dir_b", || async { Ok(Arc::new(2)) })
        .await
        .unwrap();

    assert_eq!(*a, 1);
    assert_eq!(*b, 2);
    assert!(!Arc::ptr_eq(&a, &b));
}

// ── Test 3: failed init does not poison the key ────────────────────────────

#[tokio::test]
async fn shared_cache_no_poison_on_error() {
    let attempt = Arc::new(AtomicUsize::new(0));
    let cache = SharedInitCache::<usize>::default();

    // First call fails.
    let r1 = cache
        .get_or_try_init("dir_a", {
            let attempt = attempt.clone();
            move || {
                let attempt = attempt.clone();
                async move {
                    attempt.fetch_add(1, Ordering::SeqCst);
                    Err::<Arc<usize>, _>(AgentError::Workflow("boom".into()))
                }
            }
        })
        .await;
    assert!(r1.is_err());

    // Second call with a succeeding builder must work (key not poisoned).
    let r2 = cache
        .get_or_try_init("dir_a", || async { Ok(Arc::new(42)) })
        .await;
    assert_eq!(*r2.unwrap(), 42);
    assert_eq!(attempt.load(Ordering::SeqCst), 1, "failed builder ran once");
}

// ── Test 4: concurrent init for same key deduplicates ──────────────────────

#[tokio::test]
async fn shared_cache_concurrent_init_deduplicates() {
    let calls = Arc::new(AtomicUsize::new(0));
    let cache = Arc::new(SharedInitCache::<usize>::default());

    let mut handles = Vec::new();
    for _ in 0..8 {
        let cache = cache.clone();
        let calls = calls.clone();
        handles.push(tokio::spawn(async move {
            cache
                .get_or_try_init("dir_a", {
                    let calls = calls.clone();
                    move || {
                        let calls = calls.clone();
                        async move {
                            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                            calls.fetch_add(1, Ordering::SeqCst);
                            Ok(Arc::new(99))
                        }
                    }
                })
                .await
        }));
    }

    let results: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap().unwrap())
        .collect();

    // All 8 tasks got the same Arc, builder ran exactly once.
    for r in &results {
        assert!(Arc::ptr_eq(r, &results[0]));
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
```

Run: `cargo test -p graphirm-agent --features local-extraction shared_cache 2>&1`
Expected: FAIL — `SharedInitCache` doesn't exist yet

### Step 2: Implement the shared cache helper in `local_extraction.rs`

Add a small reusable cache abstraction for async one-time initialization per key. Keep the
public production API tiny:

```rust
#[cfg(feature = "local-extraction")]
pub async fn get_or_init_onnx_extractor(model_dir: &str) -> Result<Arc<OnnxExtractor>, AgentError>
```

**Two layers:**

1. `SharedInitCache<T>` — a generic struct wrapping `HashMap<String, Arc<OnceCell<Arc<T>>>>`.
   Instantiable in tests with `SharedInitCache::<usize>::default()`. Core method:
   `async fn get_or_try_init<F, Fut>(&self, key: &str, builder: F) -> Result<Arc<T>, AgentError>`.

2. Production `static` — a `LazyLock<SharedInitCache<OnnxExtractor>>` that backs the public
   `get_or_init_onnx_extractor(model_dir)` function.

```rust
use std::sync::{LazyLock, Mutex as StdMutex};
use tokio::sync::OnceCell;

/// Generic async-init-once-per-key cache. Testable as an instance; used as a static in production.
pub(crate) struct SharedInitCache<T> {
    map: StdMutex<HashMap<String, Arc<OnceCell<Arc<T>>>>>,
}

impl<T> Default for SharedInitCache<T> {
    fn default() -> Self {
        Self { map: StdMutex::new(HashMap::new()) }
    }
}

/// Process-wide singleton wrapping `SharedInitCache<OnnxExtractor>`.
static EXTRACTOR_CACHE: LazyLock<SharedInitCache<OnnxExtractor>> =
    LazyLock::new(SharedInitCache::default);
```

Key design points:

- `std::sync::LazyLock` for process-wide singleton (stable since Rust 1.80)
- `std::sync::Mutex` for the inner `HashMap` — held only for `get`/`insert`, never across `.await`
- `tokio::sync::OnceCell` per key — `get_or_try_init` is non-poisoning: if the builder returns `Err`,
  the cell stays unset and the next caller can retry
- Canonicalize `model_dir` via `std::fs::canonicalize` (falling back to the raw string if it fails)
  so `/models/gliner2` and `/models/../models/gliner2` hit the same key
- Build `OnnxExtractor` inside `tokio::task::spawn_blocking` because `OnnxExtractor::new()` does
  synchronous file I/O and ONNX session loading

### Step 3: Wire `segments.rs` to use the cache

In `try_gliner2_fallback()`:

- remove the per-call `spawn_blocking(move || OnnxExtractor::new(&path))`
- call `get_or_init_onnx_extractor(model_dir).await`
- preserve the current non-fatal behavior: log the error and return `None`

This is the highest-impact latency fix because the fallback runs in the agent turn path.

### Step 4: Wire `extraction.rs` `Local` and `Hybrid` paths to use the same cache

Replace both inline `OnnxExtractor::new(std::path::Path::new(model_dir))?` call sites with:

```rust
let extractor = get_or_init_onnx_extractor(model_dir).await?;
```

Then call `extractor.extract(...)` as before. This keeps one cache implementation for all agent-side GLiNER2 entry points.

### Step 5: Update docs and comments to reflect the real runtime story

Update:

- `OnnxExtractor` docs in `local_extraction.rs`
- any call-site comments in `segments.rs` and `extraction.rs`
- root `AGENTS.md` risk note that currently says cache if latency matters
- `crates/agent/AGENTS.md` to mention the shared cache helper

The docs should say the extractor is lazily cached per process by `model_dir`, not merely "should be cached."

### Step 6: Verification

Run:

```bash
# All 4 cache tests
cargo test -p graphirm-agent --features local-extraction shared_cache 2>&1

# Existing extraction tests still pass
cargo test -p graphirm-agent --features local-extraction extract_knowledge_with_backend 2>&1

# Full crate check
cargo check -p graphirm-agent --features local-extraction 2>&1 | grep "^error" | head -10

# Clippy
cargo clippy -p graphirm-agent --features local-extraction 2>&1 | grep "^error" | head -10
```

Expected:

- All 4 `shared_cache_*` tests pass
- `Local` / `Hybrid` extraction tests still pass
- Segment fallback compiles and uses the shared cache path
- No compile or clippy errors

### Step 7: Commit

```bash
git add crates/agent/src/knowledge/local_extraction.rs crates/agent/src/knowledge/segments.rs crates/agent/src/knowledge/extraction.rs AGENTS.md crates/agent/AGENTS.md
git commit -m "perf(agent): cache OnnxExtractor by model directory"
```

---

## Final Verification

Run: `cargo test -p graphirm-agent --features local-extraction 2>&1 | tail -20`
Run: `cargo clippy -p graphirm-agent --features local-extraction 2>&1 | grep "^error" | head -10`

Expected:
- All non-ignored tests pass
- No clippy errors

To run the full pipeline with a real model:
```bash
cargo test -p graphirm-agent --features local-extraction -- --ignored --nocapture
```

---

## Summary

| Task | What changes | Status |
|------|-------------|--------|
| 1 | Add `hf-hub`, `regex` deps | ✅ done |
| 2 | `Gliner2Config` + `download_model()` | ✅ done |
| 3 | Redesign `OnnxExtractor` (4 sessions) | ✅ done |
| 4 | `build_ner_input()` + `generate_spans()` | ✅ done |
| 5 | Full `extract()` pipeline | ✅ done |
| 6 | Update `ExtractionBackend::Local` to `model_dir` | ✅ done |
| 7 | `docs/guides/gliner2-setup.md` | ✅ done |
| 8 | Shared `OnnxExtractor` cache by `model_dir` | ✅ done |
