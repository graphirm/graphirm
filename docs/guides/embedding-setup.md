# Embedding Providers — Setup Guide

Cross-session memory lets Graphirm recall facts from past sessions. It works by embedding
`Knowledge` nodes after each turn and injecting the most relevant ones into new sessions via
HNSW vector search. This guide covers setup for each supported backend.

## Quick start

Set one environment variable and start the server:

```bash
export MISTRAL_API_KEY=<your-key>
export EMBEDDING_BACKEND=mistral/codestral-embed
graphirm serve
```

Omitting `EMBEDDING_BACKEND` disables cross-session memory silently — the agent still works
normally, it just won't remember anything across sessions.

---

## Supported backends

### `mistral/codestral-embed` — recommended

Code-optimised Mistral model, 1536-dimensional vectors. Best discrimination score in
benchmarks (0.305 vs 0.169 for `mistral-embed`).

**Requirements:** `MISTRAL_API_KEY`

```bash
export EMBEDDING_BACKEND=mistral/codestral-embed
```

**Cost:** $0.10 / 1M tokens. A typical turn with one Knowledge node costs ~$0.000002.

---

### `mistral/mistral-embed` — fallback API option

General-purpose Mistral model, 1024-dimensional vectors. Lower discrimination than
`codestral-embed` — not recommended unless `codestral-embed` is unavailable.

```bash
export EMBEDDING_BACKEND=mistral/mistral-embed
```

---

### `fastembed/nomic-embed-text-v1` — free, local

Local ONNX inference via [`fastembed-rs`](https://github.com/Anush008/fastembed-rs).
768-dimensional vectors, ~270 MB model download on first use. No API key, no per-call cost.

**Requirements:**
- glibc ≥ 2.38 (Ubuntu 24.04+; the ONNX Runtime pre-built binary uses C23 symbols)
- Build with `--features local-embed`

```bash
# Build
cargo build --features local-embed

# Run
export EMBEDDING_BACKEND=fastembed/nomic-embed-text-v1
graphirm serve
```

Model files are downloaded from HuggingFace Hub to `~/.cache/huggingface/hub/` on first
call and reused on subsequent starts (~270 MB, allow ~2 minutes on a home connection).

**Other local models:**

| Spec | Dim | Size | Notes |
|---|---|---|---|
| `fastembed/nomic-embed-text-v1` | 768 | ~270 MB | Best quality, recommended |
| `fastembed/bge-small-en-v1.5` | 384 | ~125 MB | Fastest, smallest |
| `fastembed/bge-base-en-v1.5` | 768 | ~435 MB | — |
| `fastembed/bge-large-en-v1.5` | 1024 | ~1.3 GB | Highest quality local |

---

## Benchmark results (2026-03-09)

Measured on a 20-text software engineering corpus (10 related pairs + 1 unrelated pair).
Discrimination = mean related-pair cosine similarity − unrelated-pair similarity.
Higher is better; ≥ 0.3 is production-grade.

| Provider | Dim | Avg latency | Discrimination | Verdict |
|---|---|---|---|---|
| `mistral/codestral-embed` | 1536 | 417 ms | **0.305** | ✅ Good |
| `mistral/mistral-embed` | 1024 | 373 ms | 0.169 | ⚠ Poor |
| `fastembed/nomic-embed-text-v1` | 768 | ~5–30 ms† | not measured† | Expected good |

† fastembed requires glibc ≥ 2.38; benchmark host (Ubuntu 22.04) had glibc 2.35.
Run `cargo run --bin embed_bench --features local-embed` on Ubuntu 24.04 to get results.

**Decision rule:**
- If fastembed discrimination ≥ 0.255 → prefer fastembed (free, faster after warmup)
- Otherwise → use `mistral/codestral-embed`

---

## Running the benchmark yourself

```bash
source .env   # loads MISTRAL_API_KEY

# API providers only (works on any glibc)
cargo run --bin embed_bench

# Include fastembed (Ubuntu 24.04+ only)
cargo run --bin embed_bench --features local-embed
```

Results are printed to stdout. Update the header comment in `src/bin/embed_bench.rs` with
your findings.

---

## Architecture

```
EMBEDDING_BACKEND env var
        │
        ▼
create_embedding_provider(spec, key)   [crates/llm/src/factory.rs]
        │
        ├── "mistral/*"   → MistralEmbeddingProvider  (REST API)
        └── "fastembed/*" → FastEmbedProvider          (local ONNX, spawn_blocking)
        │
        ▼
MemoryRetriever::from_store(graph, provider, dim)
        │
        ├── post-turn: embed new Knowledge nodes → HNSW index
        └── pre-loop:  top-5 retrieval → appended to system prompt
```

---

## Known limitations

- **In-memory HNSW only** — the vector index is rebuilt from scratch each server start.
  Embeddings computed in one run are lost on restart. Persistent memory (serialising the
  index to the graph store) is Phase 9 work.
- **Dimension must match** — `EMBEDDING_BACKEND` and the actual model output must agree on
  dimension. Mismatches cause a panic at the first HNSW insert. If you switch backends,
  restart with a fresh `--db` file.
- **fastembed on Ubuntu 22.04** — the pre-built ONNX Runtime binary calls `__isoc23_strtoull`
  (a glibc 2.38 C23 symbol). The code compiles but the binary links fail at runtime. Either
  upgrade to Ubuntu 24.04 or use an API backend.
