//! Local ONNX-based entity and relation extraction using GLiNER2.
//!
//! Provides a fast, zero-cost extraction path that runs a 205M parameter
//! model on CPU via ONNX Runtime. Used for per-turn entity/relation extraction
//! while the LLM backend handles higher-order synthesis.

