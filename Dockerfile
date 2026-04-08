# ─── Warehouse Order Fulfillment RL Environment ────────────
# Build : docker build -t warehouse-env .
# Run   : docker run -p 8000:8000 warehouse-env
# Test  : docker run warehouse-env python test.py
# Eval  : docker run warehouse-env python evaluate.py
# ────────────────────────────────────────────────────────────

FROM python:3.11-slim

LABEL maintainer="Rakesh"
LABEL description="Warehouse Order Fulfillment RL Environment (OpenEnv MCP Server)"
LABEL tags="openenv,reinforcement-learning,warehouse"

WORKDIR /app

# Install git (needed for openenv-core from github)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy package files
COPY warehouse_env/ ./warehouse_env/
COPY pyproject.toml train.py test.py evaluate.py inference.py README.md openenv.yaml ./

# Install the package from the root directory
RUN pip install --no-cache-dir .

# Expose MCP server port (8000 standard, 7860 for HuggingFace Spaces)
EXPOSE 8000 7860


# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default: start the MCP server
CMD ["python", "-m", "warehouse_env.server.app"]
