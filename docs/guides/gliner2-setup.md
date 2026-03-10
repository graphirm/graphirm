# GLiNER2 Local Extraction — Setup Guide

GLiNER2 is used as the zero-cost entity extraction backend. It runs a 486M-parameter
DeBERTa-v3-large model locally via ONNX Runtime — no LLM API calls, no token cost.

## Quick start (CLI — recommended)

Build with the `local-extraction` feature, then run the built-in download command:

```bash
# One-time build
cargo build --release --features local-extraction

# Download ~1.95 GB from HuggingFace Hub (idempotent — safe to run again)
./target/release/graphirm model download
```

The command prints the cached model directory and the exact export command:

```
Model directory: /home/user/.cache/huggingface/hub/models--lmo3--gliner2-large-v1-onnx/snapshots/6adb78ae8098685d239dda324cc124d948962c21

To use the local ONNX extraction backend, set:
  export GLINER2_MODEL_DIR="..."

Then restart `graphirm serve`. Extraction will run at ~150-200ms per call
instead of 25-35s via the LLM API.
```

**Auto-detection:** If `GLINER2_MODEL_DIR` is unset, `graphirm serve` will still
auto-detect the model from the standard HuggingFace cache path and use the Local
backend automatically when the snapshot directory is found.

## Quick start (programmatic download)

The easiest way is to call `download_model()` from Rust before first use:

```rust
use graphirm_agent::knowledge::local_extraction::download_model;

let model_dir = download_model().await?;
println!("Model cached at: {}", model_dir.display());
// Then: OnnxExtractor::new(&model_dir)?
```

This downloads ~3.7 GB from HuggingFace Hub to `~/.cache/huggingface/hub/` and is
idempotent — files are reused on subsequent calls. Allow ~20 minutes on a typical
home connection.

## Manual download

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    'lmo3/gliner2-large-v1-onnx',
    allow_patterns=[
        '*.json',
        'onnx/encoder.onnx',    'onnx/encoder.onnx.data',
        'onnx/span_rep.onnx',   'onnx/span_rep.onnx.data',
        'onnx/count_embed.onnx','onnx/count_embed.onnx.data',
        'onnx/classifier.onnx', 'onnx/classifier.onnx.data',
    ]
)
print('Downloaded to:', path)
"
```

> **Note:** Each `.onnx` file has a paired `.onnx.data` file containing the external
> weight shards. Both must be present — the `.onnx` file is just the graph structure.

Then set `model_dir` in your `AgentConfig` TOML:

```toml
[agent.extraction]
enabled = true
backend = { local = { model_dir = "/home/user/.cache/huggingface/hub/models--lmo3--gliner2-large-v1-onnx/snapshots/HASH" } }
```

## Running the integration tests

```bash
# Download the model via the Rust test harness (takes ~20 min):
cargo test -p graphirm-agent --features local-extraction \
    -- --ignored --nocapture test_download_model_creates_files

# Run end-to-end NER inference (requires downloaded model):
GLINER2_MODEL_DIR=~/.cache/huggingface/hub/models--lmo3--gliner2-large-v1-onnx/snapshots/6adb78ae8098685d239dda324cc124d948962c21 \
    cargo test -p graphirm-agent --features local-extraction \
    -- --ignored --nocapture test_extract_entities_with_real_model
```

Both tests pass as of 2026-03-09 on the verified snapshot
`6adb78ae8098685d239dda324cc124d948962c21`.

## ONNX model tensor names (verified)

Inspected with `onnxruntime` on the downloaded model:

| Session | Inputs | Output |
|---------|--------|--------|
| `encoder.onnx` | `input_ids [batch, seq_len]`, `attention_mask [batch, seq_len]` | `hidden_states [batch, seq_len, 1024]` |
| `span_rep.onnx` | `hidden_states`, `span_start_idx [batch, num_spans]`, `span_end_idx` | `span_representations [batch, num_spans, 1024]` |
| `count_embed.onnx` | `label_embeddings [num_labels, 1024]` | `transformed_embeddings [num_labels, 1024]` |
| `classifier.onnx` | `hidden_state [batch, 1024]` | `logit [batch, 1]` |

> The `classifier.onnx` session is loaded but not used in the main NER inference
> path — scoring uses a direct dot-product (`span_representations @ transformed_embeddings.T`)
> followed by sigmoid activation.

## Reproducing the ONNX export from scratch

The pre-exported ONNX files at `lmo3/gliner2-large-v1-onnx` were produced from
`fastino/gliner2-large-v1` using the `lmoe/gliner2-onnx` export tool.

To regenerate (e.g., if you want to use a different base model or quantize to FP16):

```bash
# Prerequisites: Python 3.10+, ~8 GB RAM, ~10 GB disk
git clone https://github.com/lmoe/gliner2-onnx
cd gliner2-onnx
pip install -e ".[export]"

# Export gliner2-large-v1 (FP32, ~3.7 GB with external data files)
make onnx-export MODEL=fastino/gliner2-large-v1

# Export with FP16 quantization (~1.9 GB, minimal accuracy loss)
make onnx-export MODEL=fastino/gliner2-large-v1 QUANTIZE=fp16

# Output is in: model_out/gliner2-large-v1/
```

Then point `OnnxExtractor::new()` at `model_out/gliner2-large-v1/`.

## Supported models

| Model | Size | Notes |
|-------|------|-------|
| `fastino/gliner2-large-v1` | ~3.7 GB (with .data shards) | Best accuracy, English |
| `fastino/gliner2-multi-v1` | ~2.5 GB | Multilingual |
| `fastino/gliner2-base-v1`  | — | **Not ONNX-exportable** (CountLSTMv2 architecture) |

## System requirements

- glibc >= 2.35 (the crate includes a compatibility shim for glibc 2.35–2.37)
- ~4 GB disk for model files (ONNX graphs + external weight shards)
- ~4 GB RAM for inference
- CPU inference: ~150–200 ms per extraction call
- GPU: set `providers = ["CUDAExecutionProvider"]` (requires CUDA ORT build)

**Note (Ubuntu 22.04 / glibc 2.35):** The `graphirm-agent` crate includes a thin
glibc compatibility shim in `src/lib.rs` that provides `__isoc23_strtoll` and
friends. This allows ort's prebuilt binary (compiled against glibc 2.38) to load on
systems with glibc 2.35+.

## Feature flag

Build with local extraction support:

```bash
cargo build --features local-extraction
cargo test -p graphirm-agent --features local-extraction
# Run ignored integration tests (needs downloaded model):
cargo test -p graphirm-agent --features local-extraction -- --ignored --nocapture
# Pass model directory via env var:
GLINER2_MODEL_DIR=/path/to/snapshot cargo test -p graphirm-agent --features local-extraction -- --ignored --nocapture
```
