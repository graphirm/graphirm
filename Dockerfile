# Stage 1: Build the React web-app
FROM node:22-bookworm-slim AS web-builder

WORKDIR /app/web-app
COPY web-app/package.json web-app/package-lock.json ./
RUN npm ci
COPY web-app/ ./
RUN npm run build

# Stage 2: Build the Rust binary
FROM rust:1.88-bookworm AS rust-builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY src/ src/
COPY graphirm-eval/ graphirm-eval/
RUN cargo build --release -p graphirm

# Stage 3: Runtime image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=rust-builder /app/target/release/graphirm /usr/local/bin/graphirm
COPY --from=web-builder /app/web-app/dist /app/web-app/dist
COPY config /app/config

WORKDIR /app

EXPOSE 3000

CMD ["graphirm", "--db", "/data/graph.db", "serve", "--host", "0.0.0.0", "--port", "3000"]
