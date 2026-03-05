# YGN-SAGE Enterprise MCP Gateway - SOTA 2026
# Optimized for Google Cloud Run (Serverless, Scale-to-Zero)

# Stage 1: Build Rust core
FROM rust:slim AS builder

RUN apt-get update && apt-get install -y \
    clang \
    libclang-dev \
    pkg-config \
    libssl-dev \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Cargo.toml Cargo.lock /app/
COPY sage-core/ /app/sage-core/

# Use a virtual environment for maturin to avoid PEP 668 issues
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN pip install --no-cache-dir maturin \
    && cd /app/sage-core \
    && maturin build --release --out /app/wheels

# Stage 2: Final Python Runtime
FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the built wheels from the builder stage
COPY --from=builder /app/wheels /app/wheels

# Install Python dependencies and the built Rust core
COPY sage-python/pyproject.toml /app/sage-python/
RUN pip install --no-cache-dir mcp-use z3-solver \
    && pip install --no-cache-dir /app/wheels/*.whl

# Copy application code
COPY sage-python/src/ /app/sage-python/src/
COPY sage-discover/ /app/sage-discover/
COPY docs/ /app/docs/
COPY ui/ /app/ui/

# Expose the MCP Gateway port
EXPOSE 8080

# Environment variables for execution
ENV PYTHONPATH="/app/sage-python/src"
ENV HOST="0.0.0.0"
ENV PORT="8080"

# Entrypoint for the MCP Server
CMD ["python", "sage-discover/mcp_gateway.py"]
