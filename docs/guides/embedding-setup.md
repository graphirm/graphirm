# Embedding Providers — Setup Guide

Cross-session memory lets Graphirm recall facts from past sessions. It works by embedding
`Knowledge` nodes after each turn and injecting the most relevant ones into new sessions via
HNSW vector search. This guide covers setup for each supported backend.

## Quick start

No API key needed. Build with the local-embed feature and set one env var:

```bash
cargo build --release --features local-embed
export EMBEDDING_BACKEND=fastembed/bge-small-en-v1.5
graphirm serve
```

The model (~130 MB) downloads from HuggingFace on first use and is cached at
`~/.cache/huggingface/hub/`. Subsequent starts are instant.

Omitting `EMBEDDING_BACKEND` disables cross-session memory silently — the agent still works
normally, it just won't remember anything across sessions.

---

## Supported backends

### `fastembed/bge-small-en-v1.5` — recommended ✅

Local ONNX inference via [`fastembed-rs`](https://github.com/Anush008/fastembed-rs).
384-dimensional vectors, ~130 MB model. Best discrimination-to-cost ratio in benchmarks
(0.334, free, 12ms/call).

**Requirements:**
- glibc ≥ 2.38 (Ubuntu 24.04+; the ONNX Runtime pre-built binary uses C23 symbols)
- Build with `--features local-embed`

```bash
cargo build --release --features local-embed
export EMBEDDING_BACKEND=fastembed/bge-small-en-v1.5
graphirm serve
```

---

### Other local fastembed models

All require `--features local-embed` and glibc ≥ 2.38.

| Spec | Dim | Latency | Discrimination | Size |
|---|---|---|---|---|
| `fastembed/bge-small-en-v1.5` | 384 | 12ms | **0.334** ✅ | ~130 MB |
| `fastembed/bge-base-en-v1.5` | 768 | 21ms | **0.346** ✅ | ~435 MB |
| `fastembed/bge-large-en-v1.5` | 1024 | 58ms | **0.372** ✅ | ~1.3 GB |
| `fastembed/nomic-embed-text-v1` | 768 | 25ms | 0.224 ⚠ | ~270 MB |

---

### `mistral/codestral-embed` — API fallback

Code-optimised Mistral model, 1536-dimensional vectors. Use when you can't build with
`--features local-embed` (e.g. glibc < 2.38 host).

**Requirements:** `MISTRAL_API_KEY`

```bash
export MISTRAL_API_KEY=<your-key>
export EMBEDDING_BACKEND=mistral/codestral-embed
graphirm serve
```

**Cost:** $0.10 / 1M tokens (~$0.000002 per Knowledge node).

---

### `mistral/mistral-embed`

General-purpose Mistral model, 1024-dim. Discrimination 0.169 — well below threshold.
Not recommended.

```bash
export EMBEDDING_BACKEND=mistral/mistral-embed
```

---

## Benchmark results

Measured on a 20-text software engineering corpus (9 related pairs + 1 unrelated pair).
Discrimination = mean related-pair cosine similarity − unrelated-pair similarity.
Higher is better; ≥ 0.3 is production-grade.

### 2026-03-10 (Hetzner spoke, Ubuntu 24.04, glibc 2.39) — full local benchmark

| Provider | Dim | Latency | Discrimination | Verdict |
|---|---|---|---|---|
| `fastembed/bge-large-en-v1.5` | 1024 | 58ms | **0.372** | ✅ Best quality |
| `fastembed/bge-base-en-v1.5` | 768 | 21ms | **0.346** | ✅ Good |
| `fastembed/bge-small-en-v1.5` | 384 | 12ms | **0.334** | ✅ **Recommended** |
| `mistral/codestral-embed` | 1536 | 417ms | 0.305 | ✅ Good (API) |
| `fastembed/nomic-embed-text-v1` | 768 | 25ms | 0.224 | ⚠ Below threshold |
| `mistral/mistral-embed` | 1024 | 373ms | 0.169 | ⚠ Poor |

**Decision:** `bge-small-en-v1.5` wins. All three BGE models beat `codestral-embed` on discrimination while being free, offline, and 20×+ faster. `bge-small` is chosen over `bge-base`/`bge-large` because the discrimination gap is marginal (0.012) while storage and model size are half.

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

- **HNSW is restored from SQLite on startup** — embeddings are persisted to the `embeddings`
  table in the graph DB. On the next server start, `hydrate_from_graph` loads them back into
  the in-memory HNSW index. Memory is durable as long as the `.db` file is preserved.
- **Dimension mismatch on backend change** — if you switch `EMBEDDING_BACKEND` between runs,
  old embeddings will be skipped (dimension mismatch warning logged). Cross-session memory
  will be empty until new embeddings are generated. Clear the DB or re-embed if you need
  to migrate old embeddings.
- **fastembed on Ubuntu 22.04** — the pre-built ONNX Runtime binary calls `__isoc23_strtoull`
  (a glibc 2.38 C23 symbol). The code compiles but the binary links fail at runtime. Either
  upgrade to Ubuntu 24.04 or use an API backend.
