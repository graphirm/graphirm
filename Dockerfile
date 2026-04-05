# Stage 1: Build the React web-app
FROM node:22-bookworm-slim AS web-builder

WORKDIR /app/web-app
COPY web-app/package.json web-app/package-lock.json ./
RUN npm ci
COPY web-app/ ./
RUN npm run build

# Stage 2: Install cargo-chef into a base Rust image (reused by stages 3 and 4)
FROM rust:1.88-bookworm AS chef
RUN cargo install cargo-chef --locked
WORKDIR /app

# Stage 3: Compute the dependency recipe from workspace manifests
FROM chef AS planner
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY src/ src/
COPY graphirm-eval/ graphirm-eval/
COPY experiments/gliner2-segment-probe/ experiments/gliner2-segment-probe/
RUN cargo chef prepare --recipe-path recipe.json

# Stage 4: Cook (cache) dependencies, then compile the binary
FROM chef AS rust-builder
COPY --from=planner /app/recipe.json recipe.json
# This layer is cached as long as Cargo.toml / Cargo.lock don't change
RUN cargo chef cook --release -p graphirm --recipe-path recipe.json
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY src/ src/
COPY graphirm-eval/ graphirm-eval/
COPY experiments/gliner2-segment-probe/ experiments/gliner2-segment-probe/
RUN cargo build --release -p graphirm

# Stage 5: Slim runtime image
FROM debian:bookworm-slim AS runtime

# curl is required by the HEALTHCHECK instruction below
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=rust-builder /app/target/release/graphirm /usr/local/bin/graphirm
COPY --from=web-builder /app/web-app/dist /app/web-app/dist
COPY config /app/config

WORKDIR /app

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:3000/api/health || exit 1

CMD ["graphirm", "--db", "/data/graph.db", "serve", "--host", "0.0.0.0", "--port", "3000"]
