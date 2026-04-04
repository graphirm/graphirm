# gliner2-segment-probe

Runs **GLiNER2 ONNX** over arbitrary text using the **same code path** as segment fallback in `graphirm_agent::knowledge::segments::segment_extract_gliner2` (labels + `min_confidence`, optional `label_descriptions` / `label_min_confidence` — same as `[agent.segments]` in TOML). See `docs/guides/gliner2-setup.md` (Segment fallback).

## Prerequisites

- Downloaded model directory (see repo root `docs/guides/gliner2-setup.md` or `graphirm model download`).
- Env **`GLINER2_MODEL_DIR`** pointing at the snapshot directory that contains `gliner2_config.json` and ONNX files.

## Run

From repo root:

```bash
export GLINER2_MODEL_DIR=/path/to/gliner2-large-v1-onnx/snapshots/<hash>
cargo run -p gliner2-segment-probe --release -- \
  --file experiments/gliner2-segment-probe/fixtures/sample_planning.txt
```

Stdin (no `--file`):

```bash
cargo run -p gliner2-segment-probe --release -- \
  < experiments/gliner2-segment-probe/fixtures/sample_planning.txt
```

### Optional: label descriptions (same as server TOML)

Repeatable `KEY=VALUE` or a TOML file of `label = "description"` strings:

```bash
cargo run -p gliner2-segment-probe --release -- \
  --file fixtures/hirekey_planning_long.txt \
  --descriptions-file fixtures/label_descriptions_hirekey.toml \
  --min-confidence 0.5
```

Per-label confidence overrides: `--label-threshold plan=0.35` or `--thresholds-file` (TOML with float values).

### Multi-threshold sweep JSON (baseline vs with descriptions)

Writes **before/after** in one file: `baseline.runs` (no descriptions) and `with_descriptions.runs` (when `--descriptions-file` or `--description` is set).

```bash
cargo run -p gliner2-segment-probe --release -- \
  --file fixtures/hirekey_planning_long.txt \
  --descriptions-file fixtures/label_descriptions_hirekey.toml \
  --sweep-json-out results/hirekey_planning_long-gliner2-sweep.json \
  --sweep-quiet
```

`--sweep-thresholds` defaults to `0.5,0.4,0.35,0.25,0.2`. Use `--sweep-quiet` to only write the file (no stdout dump).

## Output

Single run: JSON array of `segment_type`, `content`, `start`, `end` (byte offsets into the **original** input). Order follows GLiNER dedupe logic (score-based), not necessarily top-to-bottom reading order.

## Captured results (local GLiNER runs)

| Artifact | Fixture |
|----------|---------|
| `results/sample_planning-gliner2-sweep.json` | Short `sample_planning.txt`, **baseline only** (historical, pre–description CLI) |
| `results/sample_planning-gliner2-sweep-v2.json` | Same short fixture, **baseline + with_descriptions** (`fixtures/label_descriptions_hirekey.toml`) |
| `results/hirekey_planning_long-gliner2-sweep.json` | Long HireKey planning paste, **baseline + with_descriptions** |

## How to test

```bash
cargo build -p gliner2-segment-probe
```

Full run requires the ONNX model on disk; CI does not execute inference here.
