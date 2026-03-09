//! Embedding provider benchmark.
//!
//! Runs Mistral (mistral-embed + codestral-embed) and fastembed (nomic-embed-text-v1)
//! on a 20-text software-engineering corpus and prints a comparison table.
//!
//! Usage:
//!   MISTRAL_API_KEY=... cargo run --bin embed_bench --features local-embed
//!
//! BENCHMARK RESULTS — RUN 1 (2026-03-09, Ubuntu 22.04 glibc 2.35, local machine):
//! fastembed/nomic-embed-text-v1 skipped — requires glibc >= 2.38 (ort prebuilt binary).
//!
//! ── mistral/mistral-embed ($0.10/1M tok) ──
//!   Dimension:         1024
//!   Avg latency:       373ms per call
//!   Related-sim avg:   0.8337
//!   Unrelated-sim:     0.6650
//!   Discrimination:    0.1686   ← POOR (< 0.3)
//!   Cost (20 texts):   $0.000041
//!   Per-pair sims: ["0.849","0.827","0.809","0.844","0.837","0.826","0.869","0.798","0.843"]
//!
//! ── mistral/codestral-embed ($0.10/1M tok) ──
//!   Dimension:         1536     ← NOTE: plan assumed 1024; actual is 1536
//!   Avg latency:       417ms per call
//!   Related-sim avg:   0.6851
//!   Unrelated-sim:     0.3806
//!   Discrimination:    0.3046   ← GOOD (0.3–0.5 range)
//!   Cost (20 texts):   $0.000041
//!   Per-pair sims: ["0.722","0.715","0.557","0.743","0.617","0.736","0.730","0.638","0.707"]
//!
//! BENCHMARK RESULTS — RUN 2 (2026-03-09, Ubuntu 24.04 glibc 2.39, Hetzner spoke):
//! All three providers ran including fastembed.
//!
//! ── mistral/mistral-embed ($0.10/1M tok) ──
//!   Dimension:         1024
//!   Avg latency:       170ms per call
//!   Related-sim avg:   0.8336
//!   Unrelated-sim:     0.6650
//!   Discrimination:    0.1686   ← POOR (< 0.3)
//!   Cost (20 texts):   $0.000041
//!   Per-pair sims: ["0.849","0.827","0.809","0.844","0.837","0.826","0.868","0.798","0.843"]
//!
//! ── mistral/codestral-embed ($0.10/1M tok) ──
//!   Dimension:         1536
//!   Avg latency:       256ms per call
//!   Related-sim avg:   0.6851
//!   Unrelated-sim:     0.3806
//!   Discrimination:    0.3046   ← GOOD (0.3–0.5 range)
//!   Cost (20 texts):   $0.000041
//!   Per-pair sims: ["0.722","0.715","0.557","0.743","0.617","0.736","0.730","0.638","0.707"]
//!
//! ── fastembed/nomic-embed-text-v1 (free) ──
//!   Dimension:         768
//!   Avg latency:       29ms per call   ← 10x faster than codestral-embed
//!   Related-sim avg:   0.5565
//!   Unrelated-sim:     0.3327
//!   Discrimination:    0.2238   ← BELOW THRESHOLD (need >= 0.255 to beat codestral-embed)
//!   Cost (20 texts):   $0.000000
//!   Per-pair sims: ["0.501","0.599","0.554","0.566","0.561","0.453","0.663","0.517","0.594"]
//!
//! DECISION: codestral-embed remains the primary backend.
//! fastembed is 10x faster and free but discrimination (0.224) is below the 0.255 cutoff.
//! codestral-embed (0.305) cleanly separates related from unrelated; fastembed does not.
//! EMBEDDING_BACKEND="mistral/codestral-embed" is the recommended default.

use std::time::Instant;

use graphirm_llm::{EmbeddingProvider, MistralEmbedModel, MistralEmbeddingProvider};

// 20 software-engineering texts: 10 related pairs (indices 0-1, 2-3, ..., 18-19)
const CORPUS: &[&str] = &[
    // Pair 1: Rust memory safety
    "Rust's ownership model guarantees memory safety at compile time without a garbage collector.",
    "The borrow checker enforces that references do not outlive the data they point to.",
    // Pair 2: async/await
    "Tokio is the most widely used async runtime for Rust, powering production servers.",
    "async/await syntax in Rust transforms futures into state machines compiled to efficient code.",
    // Pair 3: graph databases
    "A knowledge graph stores entities as nodes and relationships as typed edges.",
    "PageRank assigns importance scores to graph nodes by traversing incoming edge weights.",
    // Pair 4: LLM context windows
    "Large language models process input as a fixed-size context window of tokens.",
    "Token limits require truncation or summarization strategies for long conversations.",
    // Pair 5: HNSW vector search
    "HNSW builds a hierarchical proximity graph for approximate nearest-neighbor search.",
    "Vector similarity search retrieves semantically related documents using cosine distance.",
    // Pair 6: CI/CD pipelines
    "GitHub Actions runs automated tests on every pull request to catch regressions early.",
    "Continuous integration ensures every commit is built and tested before merging.",
    // Pair 7: database indexing
    "B-tree indexes allow O(log n) lookup for range queries in relational databases.",
    "SQLite stores its index structure as a B-tree on disk for efficient sorted access.",
    // Pair 8: code generation
    "Transformer models fine-tuned on code learn the statistical patterns of programming languages.",
    "Codestral is Mistral's code-completion model optimised for fill-in-the-middle tasks.",
    // Pair 9: error handling
    "Rust's Result type forces callers to explicitly handle error paths at compile time.",
    "The ? operator propagates errors up the call stack without hidden control flow.",
    // Pair 10: UNRELATED pair (should have LOW similarity)
    "The Eiffel Tower was built in 1889 and stands 330 metres tall in Paris.",
    "Ownership in Rust means each value has exactly one owner at any point in time.",
];

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

async fn bench_provider(name: &str, provider: &dyn EmbeddingProvider, cost_per_1m_tokens: f64) {
    println!("\n── {name} ──");

    let mut latencies = Vec::new();
    let mut embeddings = Vec::new();
    let mut total_chars = 0usize;

    for text in CORPUS {
        let t0 = Instant::now();
        match provider.embed(text).await {
            Ok(emb) => {
                latencies.push(t0.elapsed().as_millis());
                total_chars += text.len();
                embeddings.push(emb);
            }
            Err(e) => {
                println!("  ERROR embedding text: {e}");
                return;
            }
        }
    }

    let avg_latency = latencies.iter().sum::<u128>() / latencies.len() as u128;
    let dim = embeddings[0].len();

    // Related pairs: indices 0-1, 2-3, ..., 16-17 (9 related pairs, skip last unrelated)
    let related_sims: Vec<f32> = (0..9)
        .map(|i| cosine(&embeddings[i * 2], &embeddings[i * 2 + 1]))
        .collect();
    let avg_related = related_sims.iter().sum::<f32>() / related_sims.len() as f32;

    // Unrelated pair: indices 18-19
    let unrelated_sim = cosine(&embeddings[18], &embeddings[19]);

    // Rough cost estimate: assume 1 token ≈ 4 chars
    let total_tokens = total_chars / 4;
    let cost_usd = (total_tokens as f64 / 1_000_000.0) * cost_per_1m_tokens;

    println!("  Dimension:         {dim}");
    println!("  Avg latency:       {avg_latency}ms per call");
    println!("  Related-sim avg:   {avg_related:.4} (higher = better, max 1.0)");
    println!("  Unrelated-sim:     {unrelated_sim:.4} (lower = better)");
    println!(
        "  Discrimination:    {:.4}",
        avg_related - unrelated_sim
    );
    println!("  Cost (20 texts):   ${cost_usd:.6}");
    println!(
        "  Per-pair sims:     {:?}",
        related_sims
            .iter()
            .map(|s| format!("{s:.3}"))
            .collect::<Vec<_>>()
    );
}

#[tokio::main]
async fn main() {
    let mistral_key = std::env::var("MISTRAL_API_KEY").expect("MISTRAL_API_KEY not set");

    println!("=== Embedding Provider Benchmark ===");
    println!(
        "Corpus: {} texts, 9 related pairs + 1 unrelated pair\n",
        CORPUS.len()
    );

    // Mistral mistral-embed
    let mistral_embed =
        MistralEmbeddingProvider::new(&mistral_key, MistralEmbedModel::MistralEmbed);
    bench_provider(
        "mistral/mistral-embed ($0.10/1M tok)",
        &mistral_embed,
        0.10,
    )
    .await;

    // Mistral codestral-embed
    let codestral_embed =
        MistralEmbeddingProvider::new(&mistral_key, MistralEmbedModel::CodestralEmbed);
    bench_provider(
        "mistral/codestral-embed ($0.10/1M tok)",
        &codestral_embed,
        0.10,
    )
    .await;

    // fastembed nomic-embed-text-v1 (only available with --features local-embed on glibc >= 2.38)
    #[cfg(feature = "local-embed")]
    {
        use graphirm_llm::FastEmbedProvider;
        match FastEmbedProvider::new("nomic-embed-text-v1") {
            Ok(fe) => bench_provider("fastembed/nomic-embed-text-v1 (free)", &fe, 0.0).await,
            Err(e) => println!("\n── fastembed/nomic-embed-text-v1 ──\n  SKIP: {e}"),
        }
    }

    #[cfg(not(feature = "local-embed"))]
    println!("\n── fastembed/nomic-embed-text-v1 ──\n  SKIP: build with --features local-embed to enable");

    println!("\n=== Summary ===");
    println!("Discrimination = related_sim_avg - unrelated_sim (higher = better)");
    println!("Cost is for this 20-text benchmark run only.");
}
