# Skills Lab — Docker Image
# Build:  docker build -t skills-lab .
# Run:    docker compose up

FROM python:3.12-slim

LABEL maintainer="Skills Lab"
LABEL description="MCP Server for AI coding agents with semantic search"

WORKDIR /app

# Install system deps needed by onnxruntime (no DLL issues in Docker)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (ONNX + core)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir onnxruntime>=1.16 tokenizers>=0.13 huggingface_hub>=0.16

# Copy project
COPY . .

# Pre-download embedding model during build (cached in Docker layer)
RUN skills-lab download-model 2>/dev/null || echo "Model download skipped (will download on first run)"

# Workspace volume
VOLUME ["/app/workspace"]

# MCP server (stdio)
EXPOSE 7788

CMD ["python", "-m", "cli", "run-dashboard"]
