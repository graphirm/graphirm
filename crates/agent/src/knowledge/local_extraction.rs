//! Local ONNX-based entity extraction using GLiNER2.
//!
//! Runs `lmo3/gliner2-large-v1-onnx` — a DeBERTa-v3-large model exported as
//! four ONNX files. Inference runs on CPU via ONNX Runtime without PyTorch.
//!
//! # Model download
//!
//! On first use, call [`download_model`] to fetch the four ONNX files and
//! tokenizer from HuggingFace Hub (~1.95 GB). Files are cached in
//! `~/.cache/huggingface/hub/` (same location as the Python `huggingface_hub`
//! library).
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
use std::path::PathBuf;

use ndarray::Array2;
use serde::Deserialize;

use crate::error::AgentError;

// ─── Config types ─────────────────────────────────────────────────────────────

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

// ─── OnnxExtractor ────────────────────────────────────────────────────────────

/// Extracted entity span from GLiNER2 NER output.
#[derive(Debug, Clone)]
pub struct RawOnnxEntity {
    pub entity_type: String,
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub confidence: f64,
}

/// Four-session ONNX runtime for GLiNER2 NER.
///
/// Holds the encoder, span representation, count_embed sessions and the
/// tokenizer. All four ONNX files must be present in the same model directory.
///
/// Construct once at startup and reuse — loading four large ONNX sessions is
/// expensive (several seconds). Pass as `Arc<OnnxExtractor>` across tasks.
// Fields are used by extract() added in Task 5; suppress until then.
#[allow(dead_code)]
pub struct OnnxExtractor {
    /// DeBERTa-v3-large encoder: (input_ids, attention_mask) → hidden_states
    encoder: tokio::sync::Mutex<ort::session::Session>,
    /// Span representation MLP: (hidden_states, span_start, span_end) → span_rep
    span_rep: tokio::sync::Mutex<ort::session::Session>,
    /// Unrolled GRU label transform: (label_embeddings,) → transformed
    count_embed: tokio::sync::Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
    max_width: usize,
    /// Token ID for [E] (entity label marker in NER schema)
    tok_e: i64,
    /// Token ID for [SEP_TEXT] (schema/text separator)
    tok_sep_text: i64,
    /// Token ID for [P] (task type marker)
    tok_p: i64,
    /// Word boundary regex (matches GLiNER2's WhitespaceTokenSplitter)
    word_re: regex::Regex,
}

impl std::fmt::Debug for OnnxExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxExtractor")
            .field("max_width", &self.max_width)
            .finish_non_exhaustive()
    }
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
    pub fn new(model_dir: &std::path::Path) -> Result<Self, AgentError> {
        if !model_dir.is_dir() {
            return Err(AgentError::Workflow(format!(
                "GLiNER2 model directory not found: {}",
                model_dir.display()
            )));
        }

        let config_path = model_dir.join("gliner2_config.json");
        let config_bytes = std::fs::read(&config_path).map_err(|e| {
            AgentError::Workflow(format!("Cannot read gliner2_config.json: {e}"))
        })?;
        let config: Gliner2Config = serde_json::from_slice(&config_bytes)
            .map_err(|e| AgentError::Workflow(format!("Invalid gliner2_config.json: {e}")))?;

        let fp32 = config.onnx_files.get("fp32").ok_or_else(|| {
            AgentError::Workflow("gliner2_config.json missing fp32 onnx_files".into())
        })?;

        let encoder = ort::session::Session::builder()
            .map_err(|e| AgentError::Workflow(format!("ORT session builder: {e}")))?
            .with_intra_threads(4)
            .map_err(|e| AgentError::Workflow(format!("ORT intra_threads: {e}")))?
            .commit_from_file(model_dir.join(&fp32.encoder))
            .map_err(|e| AgentError::Workflow(format!("Load encoder.onnx: {e}")))?;

        let span_rep = ort::session::Session::builder()
            .map_err(|e| AgentError::Workflow(format!("ORT session builder: {e}")))?
            .commit_from_file(model_dir.join(&fp32.span_rep))
            .map_err(|e| AgentError::Workflow(format!("Load span_rep.onnx: {e}")))?;

        let count_embed = ort::session::Session::builder()
            .map_err(|e| AgentError::Workflow(format!("ORT session builder: {e}")))?
            .commit_from_file(model_dir.join(&fp32.count_embed))
            .map_err(|e| AgentError::Workflow(format!("Load count_embed.onnx: {e}")))?;

        let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
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

        // Mirrors GLiNER2's WhitespaceTokenSplitter — matches URLs, emails,
        // @handles, hyphenated words, and single non-whitespace characters.
        let word_re = regex::Regex::new(
            r"(?i)(?:https?://\S+|www\.\S+)|[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}|@[a-z0-9_]+|\w+(?:[-_]\w+)*|\S",
        )
        .map_err(|e| AgentError::Workflow(format!("Regex compile: {e}")))?;

        Ok(Self {
            encoder: tokio::sync::Mutex::new(encoder),
            span_rep: tokio::sync::Mutex::new(span_rep),
            count_embed: tokio::sync::Mutex::new(count_embed),
            tokenizer,
            max_width: config.max_width,
            tok_e,
            tok_sep_text,
            tok_p,
            word_re,
        })
    }

    /// Stub — `extract()` implementation added in Task 5.
    ///
    /// Returns an error until the inference pipeline is wired up. This stub
    /// exists only to satisfy the call site in `extraction.rs`; it will be
    /// replaced wholesale in Task 5.
    pub async fn extract(
        &self,
        _text: &str,
        _entity_types: &[String],
        _min_confidence: f64,
    ) -> Result<super::extraction::ExtractionResponse, AgentError> {
        Err(AgentError::Workflow(
            "OnnxExtractor::extract not yet implemented — coming in Task 5".into(),
        ))
    }

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
    #[allow(dead_code)] // called in Task 5 extract()
    fn build_ner_input(
        &self,
        text: &str,
        entity_types: &[String],
    ) -> (Vec<i64>, Vec<usize>, Vec<(usize, usize)>, usize, Vec<usize>) {
        let mut tokens: Vec<i64> = Vec::new();

        // Tokenize a literal string to i64 ids via the HF tokenizer.
        let encode_str = |s: &str| -> Vec<i64> {
            self.tokenizer
                .encode(s, false)
                .map(|e| e.get_ids().iter().map(|&x| x as i64).collect::<Vec<_>>())
                .unwrap_or_default()
        };

        let open_ids = encode_str("(");
        let close_ids = encode_str(")");

        // ( [P] entities (
        tokens.extend_from_slice(&open_ids);
        tokens.push(self.tok_p);
        tokens.extend_from_slice(&encode_str("entities"));
        tokens.extend_from_slice(&open_ids);

        // [E] label1 [E] label2 ...
        let mut e_positions: Vec<usize> = Vec::new();
        for label in entity_types {
            e_positions.push(tokens.len());
            tokens.push(self.tok_e);
            tokens.extend_from_slice(&encode_str(label.as_str()));
        }

        // ) ) [SEP_TEXT]
        tokens.extend_from_slice(&close_ids);
        tokens.extend_from_slice(&close_ids);
        tokens.push(self.tok_sep_text);

        let text_start_idx = tokens.len();

        // Tokenize text words, tracking char offsets and first-token positions.
        let text_lower = text.to_lowercase();
        let mut word_offsets: Vec<(usize, usize)> = Vec::new();
        let mut first_token_positions: Vec<usize> = Vec::new();
        let mut token_idx: usize = 0;

        for m in self.word_re.find_iter(&text_lower) {
            word_offsets.push((m.start(), m.end()));
            first_token_positions.push(token_idx);

            let word_ids = encode_str(m.as_str());
            token_idx += word_ids.len();
            tokens.extend(word_ids);
        }

        (tokens, e_positions, word_offsets, text_start_idx, first_token_positions)
    }
}

// ─── Helper functions ─────────────────────────────────────────────────────────

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
#[allow(dead_code)] // used in Task 5 extract()
#[inline]
pub(crate) fn sigmoid_f32(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Apply sigmoid element-wise to a 2D ndarray in place.
#[allow(dead_code)] // used in Task 5 extract()
pub(crate) fn sigmoid_inplace(arr: &mut Array2<f32>) {
    arr.mapv_inplace(sigmoid_f32);
}

// ─── Model download ───────────────────────────────────────────────────────────

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
    use hf_hub::{api::tokio::Api, Repo, RepoType};

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
                    .find(|anc| {
                        anc.join("gliner2_config.json").exists()
                            || anc.join("tokenizer.json").exists()
                    })
                    .unwrap_or(p)
                    .to_path_buf()
            });
        }
    }

    model_dir.ok_or_else(|| AgentError::Workflow("Failed to determine model directory".into()))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_schema_prefix_has_e_positions() {
        let tok_e: i64 = 128005;
        let tokens: Vec<i64> = vec![287, 128003, 100, 287, tok_e, 101, tok_e, 102, 1263, 1263, 128002];
        let e_positions: Vec<usize> = tokens.iter().enumerate()
            .filter(|(_, t)| **t == tok_e)
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

    #[test]
    fn test_onnx_extractor_new_fails_with_missing_dir() {
        let result = OnnxExtractor::new(std::path::Path::new("/nonexistent/model/dir"));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

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
}
