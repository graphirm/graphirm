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

**Note (Ubuntu 22.04 / glibc 2.35):** The `graphirm-agent` crate includes a thin
glibc compatibility shim in `src/lib.rs` that provides `__isoc23_strtoll` and
friends. This allows ort's prebuilt binary to load on systems with glibc < 2.38.

## Feature flag

Build with local extraction support:

```bash
cargo build --features local-extraction
cargo test -p graphirm-agent --features local-extraction
# Run ignored integration tests (needs downloaded model):
cargo test -p graphirm-agent --features local-extraction -- --ignored --nocapture
# Pass model directory via env var:
GLINER2_MODEL_DIR=/path/to/model cargo test -p graphirm-agent --features local-extraction -- --ignored --nocapture
```
