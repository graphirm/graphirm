//! Local ONNX-based entity and relation extraction using GLiNER2.
//!
//! Provides a fast, zero-cost extraction path that runs a 205M parameter
//! model on CPU via ONNX Runtime. Used for per-turn entity/relation extraction
//! while the LLM backend handles higher-order synthesis.
//!
//! Requires the `local-extraction` feature. The prebuilt ONNX Runtime binary
//! requires glibc >= 2.38. Enable with: `cargo build --features local-extraction`

use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

use crate::error::AgentError;

use super::extraction::{ExtractedEntity, ExtractionResponse};

/// Raw entity output from the ONNX model before conversion to ExtractionResponse.
#[derive(Debug, Clone)]
pub struct RawOnnxEntity {
    pub entity_type: String,
    pub text: String,
    pub confidence: f64,
}

/// Wraps a GLiNER2 ONNX model session and tokenizer for local entity extraction.
///
/// The `Session` is held behind a `Mutex` because `Session::run` requires `&mut self`
/// and `extract` is called from async contexts.
pub struct OnnxExtractor {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl OnnxExtractor {
    /// Load a GLiNER2 ONNX model and its tokenizer from disk.
    pub fn new(model_path: &Path, tokenizer_path: &Path) -> Result<Self, AgentError> {
        let session = Session::builder()
            .map_err(|e| AgentError::Workflow(format!("Failed to create ONNX session builder: {e}")))?
            .with_intra_threads(4)
            .map_err(|e| AgentError::Workflow(format!("Failed to configure intra threads: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| AgentError::Workflow(format!("Failed to load ONNX model: {e}")))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| AgentError::Workflow(format!("Failed to load tokenizer: {e}")))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
        })
    }

    /// Run entity extraction on the given text using the configured entity types.
    ///
    /// Returns raw entities converted to an `ExtractionResponse`. The actual ONNX
    /// inference tensor construction and output parsing is model-specific to
    /// GLiNER2's architecture:
    /// - Input: tokenized text + entity type prompts
    /// - Output: span predictions with entity type labels and confidence scores
    pub async fn extract(
        &self,
        text: &str,
        entity_types: &[String],
        min_confidence: f64,
    ) -> Result<ExtractionResponse, AgentError> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| AgentError::Workflow(format!("Tokenization failed: {e}")))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let seq_len = input_ids.len();

        let type_prompt = build_entity_type_prompt(entity_types);
        let type_encoding = self
            .tokenizer
            .encode(type_prompt.as_str(), true)
            .map_err(|e| AgentError::Workflow(format!("Entity type tokenization failed: {e}")))?;

        let _type_ids: Vec<i64> = type_encoding
            .get_ids()
            .iter()
            .map(|&id| id as i64)
            .collect();

        let input_array = ndarray::Array2::from_shape_vec((1, seq_len), input_ids)
            .map_err(|e| AgentError::Workflow(format!("Input tensor shape error: {e}")))?;
        let mask_array = ndarray::Array2::from_shape_vec((1, seq_len), attention_mask)
            .map_err(|e| AgentError::Workflow(format!("Mask tensor shape error: {e}")))?;

        let input_tensor = Tensor::from_array(input_array)
            .map_err(|e| AgentError::Workflow(format!("Input tensor creation failed: {e}")))?;
        let mask_tensor = Tensor::from_array(mask_array)
            .map_err(|e| AgentError::Workflow(format!("Mask tensor creation failed: {e}")))?;

        let mut session = self.session.lock().await;
        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_tensor,
                "attention_mask" => mask_tensor,
            ])
            .map_err(|e| AgentError::Workflow(format!("ONNX inference failed: {e}")))?;

        // TODO: Implement GLiNER2-specific output parsing once the ONNX export format is
        // validated. The model outputs span logits that need to be decoded against the
        // tokenizer's offset mapping and matched to entity type labels.
        let raw_entities: Vec<RawOnnxEntity> = parse_onnx_outputs(&outputs, &encoding, entity_types)
            .into_iter()
            .filter(|e| e.confidence >= min_confidence)
            .collect();

        Ok(raw_entities_to_extraction_response(raw_entities))
    }
}

/// Build the entity type prompt string that GLiNER2 uses to condition extraction.
pub fn build_entity_type_prompt(entity_types: &[String]) -> String {
    entity_types
        .iter()
        .map(|t| format!("<entity>{}</entity>", t))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Convert raw ONNX entity outputs into the shared ExtractionResponse format.
pub fn raw_entities_to_extraction_response(raw_entities: Vec<RawOnnxEntity>) -> ExtractionResponse {
    let entities = raw_entities
        .into_iter()
        .map(|raw| ExtractedEntity {
            entity_type: raw.entity_type,
            name: raw.text,
            description: String::new(),
            confidence: raw.confidence,
            relationships: vec![],
        })
        .collect();

    ExtractionResponse { entities }
}

/// Parse ONNX session outputs into `RawOnnxEntity` values.
///
/// This is a placeholder. The real implementation must decode GLiNER2's output tensor
/// format (span start/end logits, entity type classification logits) against the
/// tokenizer's offset mapping to recover text spans and labels.
fn parse_onnx_outputs(
    _outputs: &ort::session::SessionOutputs,
    _encoding: &tokenizers::Encoding,
    _entity_types: &[String],
) -> Vec<RawOnnxEntity> {
    // TODO: Implement once the ONNX export graph structure is known.
    vec![]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_extractor_new_fails_with_missing_model() {
        let result = OnnxExtractor::new(
            Path::new("/nonexistent/model.onnx"),
            Path::new("/nonexistent/tokenizer.json"),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_build_entity_type_prompt() {
        let types = vec![
            "function".to_string(),
            "pattern".to_string(),
            "decision".to_string(),
        ];
        let prompt = build_entity_type_prompt(&types);
        assert!(prompt.contains("function"));
        assert!(prompt.contains("pattern"));
        assert!(prompt.contains("decision"));
    }

    #[test]
    fn test_parse_onnx_output_to_extraction_response() {
        let raw_entities = vec![
            RawOnnxEntity {
                entity_type: "function".to_string(),
                text: "parse_config".to_string(),
                confidence: 0.92,
            },
            RawOnnxEntity {
                entity_type: "library".to_string(),
                text: "serde".to_string(),
                confidence: 0.88,
            },
        ];

        let response = raw_entities_to_extraction_response(raw_entities);
        assert_eq!(response.entities.len(), 2);
        assert_eq!(response.entities[0].name, "parse_config");
        assert_eq!(response.entities[0].entity_type, "function");
        assert!((response.entities[0].confidence - 0.92).abs() < f64::EPSILON);
        assert_eq!(response.entities[1].name, "serde");
    }
}
