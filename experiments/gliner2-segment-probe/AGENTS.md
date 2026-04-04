# gliner2-segment-probe

Runs **GLiNER2 ONNX** over arbitrary text using the **same code path** as segment fallback in `graphirm_agent::knowledge::segments::segment_extract_gliner2` (labels + `min_confidence` match `SegmentConfig` defaults unless overridden). Optional `label_descriptions` / `label_min_confidence` are **TOML-only** for `graphirm serve`; the probe passes `None` for both (see `docs/guides/gliner2-setup.md` — Segment fallback).

## Prerequisites

- Downloaded model directory (see repo root `docs/guides/gliner2-setup.md` or `graphirm model download`).
- Env **`GLINER2_MODEL_DIR`** pointing at the snapshot directory that contains `gliner2_config.json` and ONNX files.

## Run

From repo root:

```bash
export GLINER2_MODEL_DIR=/path/to/gliner2-large-v1-onnx/snapshots/<hash>
cargo run -p gliner2-segment-probe -- \
  --file experiments/gliner2-segment-probe/fixtures/sample_planning.txt
```

Stdin (no `--file`):

```bash
cargo run -p gliner2-segment-probe -- < experiments/gliner2-segment-probe/fixtures/sample_planning.txt
```

Custom labels (comma-separated):

```bash
cargo run -p gliner2-segment-probe -- \
  --labels plan,answer,reasoning \
  --min-confidence 0.45 \
  --file fixtures/sample_planning.txt
```

## Output

JSON array of objects: `segment_type`, `content`, `start`, `end` (byte offsets into the **original** input string). Order follows GLiNER dedupe logic (score-based), not necessarily top-to-bottom reading order.

A full multi-threshold sweep for `fixtures/sample_planning.txt` (local run, pinned model path in `meta`) is saved at **`results/sample_planning-gliner2-sweep.json`**.

## How to test

```bash
cargo build -p gliner2-segment-probe
```

Full run requires the ONNX model on disk; CI does not execute inference here.
