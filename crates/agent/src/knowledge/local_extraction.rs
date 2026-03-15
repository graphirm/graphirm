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
//! # Extractor caching
//!
//! Call [`get_or_init_onnx_extractor`] instead of [`OnnxExtractor::new`] directly.
//! It maintains a process-wide cache keyed by canonical `model_dir` so that
//! the expensive ONNX session loading (~seconds) happens at most once per unique
//! directory, regardless of how many concurrent callers or agent turns request it.
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
/// Prefer [`get_or_init_onnx_extractor`] over constructing directly: it returns
/// a process-wide cached `Arc<OnnxExtractor>` so the expensive session loading
/// (~seconds) runs at most once per unique `model_dir`.
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

    /// Run GLiNER2 NER on `text` for the given `entity_types`.
    ///
    /// Returns raw entity spans with character offsets and confidence. Use this
    /// when you need offsets for overlap or coverage analysis. For the standard
    /// knowledge-extraction response format, use [`extract`](Self::extract).
    ///
    /// # Performance
    ///
    /// ~150–200ms per call on CPU. Construct `OnnxExtractor` once at startup
    /// and reuse it via `Arc<OnnxExtractor>`.
    pub async fn extract_raw(
        &self,
        text: &str,
        entity_types: &[String],
        min_confidence: f64,
    ) -> Result<Vec<RawOnnxEntity>, AgentError> {
        if text.is_empty() || entity_types.is_empty() {
            return Ok(vec![]);
        }

        let (input_ids, e_positions, word_offsets, text_start_idx, first_token_positions) =
            self.build_ner_input(text, entity_types);

        if word_offsets.is_empty() {
            return Ok(vec![]);
        }

        let seq_len = input_ids.len();
        let attn_mask: Vec<i64> = vec![1; seq_len];

        // ── Step 1: Encoder ───────────────────────────────────────────────────
        let input_ids_arr = ndarray::Array2::from_shape_vec((1, seq_len), input_ids.clone())
            .map_err(|e| AgentError::Workflow(format!("input_ids shape error: {e}")))?;
        let attn_mask_arr = ndarray::Array2::from_shape_vec((1, seq_len), attn_mask)
            .map_err(|e| AgentError::Workflow(format!("attn_mask shape error: {e}")))?;

        // hidden_states: [1, seq_len, 1024] → extract to owned [seq_len, 1024]
        // before releasing the encoder lock.
        let hs_2d: ndarray::Array2<f32> = {
            let mut enc = self.encoder.lock().await;
            let encoder_outputs = enc
                .run(ort::inputs! {
                    "input_ids"      => ort::value::TensorRef::from_array_view(input_ids_arr.view())
                        .map_err(|e| AgentError::Workflow(format!("input_ids TensorRef: {e}")))?,
                    "attention_mask" => ort::value::TensorRef::from_array_view(attn_mask_arr.view())
                        .map_err(|e| AgentError::Workflow(format!("attn_mask TensorRef: {e}")))?,
                })
                .map_err(|e| AgentError::Workflow(format!("Encoder run: {e}")))?;
            encoder_outputs["hidden_states"]
                .try_extract_array::<f32>()
                .map_err(|e| AgentError::Workflow(format!("Extract hidden_states: {e}")))?
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|e| AgentError::Workflow(format!("hidden_states reshape: {e}")))?
                .index_axis(ndarray::Axis(0), 0)
                .to_owned()
        }; // lock released here

        // ── Step 2: Label embeddings at [E] positions ─────────────────────────
        let num_labels = entity_types.len();
        let hidden_size = hs_2d.ncols();
        let mut label_emb = ndarray::Array2::<f32>::zeros((num_labels, hidden_size));
        for (li, &ep) in e_positions.iter().enumerate() {
            if ep < hs_2d.nrows() {
                label_emb.row_mut(li).assign(&hs_2d.row(ep));
            }
        }

        // ── Step 3: Text hidden states (after [SEP_TEXT]) ────────────────────
        let text_hidden = hs_2d
            .slice(ndarray::s![text_start_idx.., ..])
            .to_owned(); // [text_tokens, 1024]

        // ── Step 4: Generate word spans ──────────────────────────────────────
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

        let span_start_arr = ndarray::Array2::from_shape_vec((1, num_spans), span_starts)
            .map_err(|e| AgentError::Workflow(format!("span_start shape: {e}")))?;
        let span_end_arr = ndarray::Array2::from_shape_vec((1, num_spans), span_ends)
            .map_err(|e| AgentError::Workflow(format!("span_end shape: {e}")))?;

        // span_rep needs hidden_states with batch dim: [1, text_tokens, 1024]
        let text_hidden_3d = text_hidden
            .view()
            .insert_axis(ndarray::Axis(0))
            .to_owned(); // [1, text_tokens, 1024]

        // ── Step 5: Span representations ─────────────────────────────────────
        // span_rep: [1, num_spans, 1024] → extract to owned [num_spans, 1024]
        let span_rep_2d: ndarray::Array2<f32> = {
            let mut sr = self.span_rep.lock().await;
            let span_rep_outputs = sr
                .run(ort::inputs! {
                    "hidden_states"  => ort::value::TensorRef::from_array_view(text_hidden_3d.view())
                        .map_err(|e| AgentError::Workflow(format!("text_hidden TensorRef: {e}")))?,
                    "span_start_idx" => ort::value::TensorRef::from_array_view(span_start_arr.view())
                        .map_err(|e| AgentError::Workflow(format!("span_start TensorRef: {e}")))?,
                    "span_end_idx"   => ort::value::TensorRef::from_array_view(span_end_arr.view())
                        .map_err(|e| AgentError::Workflow(format!("span_end TensorRef: {e}")))?,
                })
                .map_err(|e| AgentError::Workflow(format!("span_rep run: {e}")))?;
            span_rep_outputs["span_representations"]
                .try_extract_array::<f32>()
                .map_err(|e| AgentError::Workflow(format!("Extract span_representations: {e}")))?
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|e| AgentError::Workflow(format!("span_rep reshape: {e}")))?
                .index_axis(ndarray::Axis(0), 0)
                .to_owned()
        }; // lock released here

        // ── Step 6: Transform label embeddings via count_embed ────────────────
        // transformed: [num_labels, 1024]
        let transformed: ndarray::Array2<f32> = {
            let mut ce = self.count_embed.lock().await;
            let count_embed_outputs = ce
                .run(ort::inputs! {
                    "label_embeddings" => ort::value::TensorRef::from_array_view(label_emb.view())
                        .map_err(|e| AgentError::Workflow(format!("label_emb TensorRef: {e}")))?,
                })
                .map_err(|e| AgentError::Workflow(format!("count_embed run: {e}")))?;
            // count_embed.onnx has a single unnamed output; access positionally.
            // The output name varies between export versions, so positional is
            // more robust than keying by name.
            count_embed_outputs
                .values()
                .next()
                .ok_or_else(|| AgentError::Workflow("count_embed produced no output".into()))?
                .try_extract_array::<f32>()
                .map_err(|e| AgentError::Workflow(format!("Extract count_embed: {e}")))?
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| AgentError::Workflow(format!("count_embed reshape: {e}")))?
                .to_owned()
        }; // lock released here

        // ── Step 7: Score spans against labels ───────────────────────────────
        // scores = sigmoid(span_rep @ transformed.T) → [num_spans, num_labels]
        let mut scores: ndarray::Array2<f32> = span_rep_2d.dot(&transformed.t());
        sigmoid_inplace(&mut scores);

        // ── Step 8: Collect entities above threshold ──────────────────────────
        let mut raw_entities: Vec<(usize, usize, usize, f64)> = Vec::new();
        for (span_idx, &(word_start, word_end)) in spans.iter().enumerate() {
            for label_idx in 0..entity_types.len() {
                let score = scores[[span_idx, label_idx]] as f64;
                if score >= min_confidence {
                    raw_entities.push((word_start, word_end, label_idx, score));
                }
            }
        }

        // Sort descending for greedy deduplication
        raw_entities.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

        // Deduplicate: keep highest-score non-overlapping spans per label
        let mut kept: Vec<RawOnnxEntity> = Vec::new();
        for (word_start, word_end, label_idx, score) in raw_entities {
            let char_start = word_offsets[word_start].0;
            let char_end = word_offsets[word_end].1;
            let label = &entity_types[label_idx];

            let overlaps = kept.iter().any(|k| {
                k.entity_type == *label && k.start < char_end && k.end > char_start
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

        Ok(kept)
    }

    /// Run NER and return the standard extraction response (entities with name, type, confidence).
    ///
    /// Use [`extract_raw`](Self::extract_raw) when you need character offsets (e.g. for overlap or coverage).
    pub async fn extract(
        &self,
        text: &str,
        entity_types: &[String],
        min_confidence: f64,
    ) -> Result<super::extraction::ExtractionResponse, AgentError> {
        let raw = self.extract_raw(text, entity_types, min_confidence).await?;
        Ok(raw_entities_to_extraction_response(raw))
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
        //
        // Run the word regex against the *original* text to preserve byte
        // offsets for slicing back later; pass the lowercased word to the
        // tokenizer so casing matches GLiNER2's training-time preprocessing.
        let mut word_offsets: Vec<(usize, usize)> = Vec::new();
        let mut first_token_positions: Vec<usize> = Vec::new();
        let mut token_idx: usize = 0;

        for m in self.word_re.find_iter(text) {
            word_offsets.push((m.start(), m.end()));
            first_token_positions.push(token_idx);

            let word_lower = m.as_str().to_lowercase();
            let word_ids = encode_str(&word_lower);
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
pub(crate) fn sigmoid_inplace(arr: &mut Array2<f32>) {
    arr.mapv_inplace(sigmoid_f32);
}

// ─── Conversion helpers ───────────────────────────────────────────────────────

/// Convert raw GLiNER2 entity spans into the shared `ExtractionResponse` format.
pub fn raw_entities_to_extraction_response(raw: Vec<RawOnnxEntity>) -> super::extraction::ExtractionResponse {
    use super::extraction::ExtractedEntity;
    let entities = raw
        .into_iter()
        .map(|e| ExtractedEntity {
            entity_type: e.entity_type,
            name: e.text,
            description: String::new(),
            confidence: e.confidence,
            relationships: vec![],
        })
        .collect();
    super::extraction::ExtractionResponse { entities }
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
        // ONNX graph files (model structure)
        "onnx/encoder.onnx",
        "onnx/span_rep.onnx",
        "onnx/count_embed.onnx",
        "onnx/classifier.onnx",
        // ONNX external data files (model weights — split due to >2 GB size)
        "onnx/encoder.onnx.data",
        "onnx/span_rep.onnx.data",
        "onnx/count_embed.onnx.data",
        "onnx/classifier.onnx.data",
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

// ─── Process-wide OnnxExtractor cache ─────────────────────────────────────────

/// Generic async-init-once-per-key cache.
///
/// For the same key, only one builder future runs at a time; other concurrent
/// waiters block until it completes and then reuse the cached `Arc<T>`.
///
/// Non-poisoning: if the builder returns an error the cell stays unset, so the
/// next caller can retry with a fresh builder.
///
/// The `std::sync::Mutex` on the inner map is held only for the brief
/// `get`/`insert` operation — never across an `.await` point.
pub(crate) struct SharedInitCache<T> {
    map: std::sync::Mutex<HashMap<String, std::sync::Arc<tokio::sync::OnceCell<std::sync::Arc<T>>>>>,
}

impl<T> Default for SharedInitCache<T> {
    fn default() -> Self {
        Self {
            map: std::sync::Mutex::new(HashMap::new()),
        }
    }
}

impl<T: Send + Sync + 'static> SharedInitCache<T> {
    /// Return the cached value for `key`, initialising it via `builder` on first access.
    pub(crate) async fn get_or_try_init<F, Fut>(
        &self,
        key: &str,
        builder: F,
    ) -> Result<std::sync::Arc<T>, AgentError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<std::sync::Arc<T>, AgentError>>,
    {
        let cell: std::sync::Arc<tokio::sync::OnceCell<std::sync::Arc<T>>> = {
            let mut map = self
                .map
                .lock()
                .expect("SharedInitCache mutex poisoned");
            map.entry(key.to_string())
                .or_insert_with(|| std::sync::Arc::new(tokio::sync::OnceCell::new()))
                .clone()
        };
        cell.get_or_try_init(builder).await.cloned()
    }
}

/// Process-wide singleton: one `OnnxExtractor` per canonical `model_dir`.
static EXTRACTOR_CACHE: std::sync::LazyLock<SharedInitCache<OnnxExtractor>> =
    std::sync::LazyLock::new(SharedInitCache::default);

/// Return the process-wide cached `OnnxExtractor` for `model_dir`, loading it on
/// first access.
///
/// The path is normalised via `std::fs::canonicalize` so that `/a/gliner2` and
/// `/a/../a/gliner2` share a single cache entry. On error (e.g. non-existent
/// directory), the cache entry stays unset so the next caller can retry.
///
/// `OnnxExtractor::new()` runs inside `tokio::task::spawn_blocking` because it
/// performs synchronous file I/O and loads four large ONNX sessions.
pub async fn get_or_init_onnx_extractor(
    model_dir: &str,
) -> Result<std::sync::Arc<OnnxExtractor>, AgentError> {
    let canonical_key = std::fs::canonicalize(model_dir)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| model_dir.to_owned());
    let dir_owned = model_dir.to_owned();
    EXTRACTOR_CACHE
        .get_or_try_init(&canonical_key, move || async move {
            let path = std::path::PathBuf::from(dir_owned);
            tokio::task::spawn_blocking(move || {
                OnnxExtractor::new(&path).map(std::sync::Arc::new)
            })
            .await
            .map_err(|e| AgentError::Join(e.to_string()))?
        })
        .await
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
        assert!(dir.join("onnx/encoder.onnx.data").exists());
        assert!(dir.join("onnx/span_rep.onnx").exists());
        assert!(dir.join("onnx/count_embed.onnx").exists());
    }

    #[test]
    fn test_sigmoid_inplace() {
        let mut arr = ndarray::Array2::from_shape_vec((2, 2), vec![0.0f32, 2.0, -2.0, 100.0]).unwrap();
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

    // ── SharedInitCache behaviour ─────────────────────────────────────────────

    #[tokio::test]
    async fn shared_cache_reuses_same_key() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = std::sync::Arc::new(AtomicUsize::new(0));
        let cache = SharedInitCache::<usize>::default();

        let make_builder = |c: std::sync::Arc<AtomicUsize>| {
            move || {
                let c = c.clone();
                async move {
                    c.fetch_add(1, Ordering::SeqCst);
                    Ok::<std::sync::Arc<usize>, AgentError>(std::sync::Arc::new(7))
                }
            }
        };

        let a = cache.get_or_try_init("dir_a", make_builder(calls.clone())).await.unwrap();
        let b = cache.get_or_try_init("dir_a", make_builder(calls.clone())).await.unwrap();

        assert!(std::sync::Arc::ptr_eq(&a, &b), "second call must return cached Arc");
        assert_eq!(calls.load(Ordering::SeqCst), 1, "builder must run exactly once");
    }

    #[tokio::test]
    async fn shared_cache_different_keys_independent() {
        let cache = SharedInitCache::<usize>::default();

        let a = cache
            .get_or_try_init("dir_a", || async { Ok(std::sync::Arc::new(1usize)) })
            .await
            .unwrap();
        let b = cache
            .get_or_try_init("dir_b", || async { Ok(std::sync::Arc::new(2usize)) })
            .await
            .unwrap();

        assert_eq!(*a, 1);
        assert_eq!(*b, 2);
        assert!(!std::sync::Arc::ptr_eq(&a, &b));
    }

    #[tokio::test]
    async fn shared_cache_no_poison_on_error() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let attempt = std::sync::Arc::new(AtomicUsize::new(0));
        let cache = SharedInitCache::<usize>::default();

        // First call fails.
        let r1 = cache
            .get_or_try_init("dir_a", {
                let attempt = attempt.clone();
                move || {
                    let attempt = attempt.clone();
                    async move {
                        attempt.fetch_add(1, Ordering::SeqCst);
                        Err::<std::sync::Arc<usize>, _>(AgentError::Workflow("boom".into()))
                    }
                }
            })
            .await;
        assert!(r1.is_err());

        // Second call with a succeeding builder must work (key not poisoned).
        let r2 = cache
            .get_or_try_init("dir_a", || async { Ok(std::sync::Arc::new(42usize)) })
            .await;
        assert_eq!(*r2.unwrap(), 42);
        assert_eq!(attempt.load(Ordering::SeqCst), 1, "failed builder ran once");
    }

    #[tokio::test]
    async fn shared_cache_concurrent_init_deduplicates() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = std::sync::Arc::new(AtomicUsize::new(0));
        let cache = std::sync::Arc::new(SharedInitCache::<usize>::default());

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
                                Ok(std::sync::Arc::new(99usize))
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

        // All 8 tasks got the same Arc; builder ran exactly once.
        for r in &results {
            assert!(std::sync::Arc::ptr_eq(r, &results[0]));
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    /// Full inference test — requires model to be downloaded first.
    /// Run with: cargo test -p graphirm-agent --features local-extraction -- --ignored --nocapture
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

        assert!(!result.entities.is_empty(), "Should detect at least one entity");
        let names: Vec<&str> = result.entities.iter().map(|e| e.name.as_str()).collect();
        assert!(
            names.contains(&"serde") || names.contains(&"tokio"),
            "Should detect serde or tokio as library. Got: {:?}",
            names
        );
    }
}
