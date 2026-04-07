# ─── Warehouse Order Fulfillment RL Environment ────────────
# Build : docker build -t warehouse-env .
# Run   : docker run -p 8000:8000 warehouse-env
# ────────────────────────────────────────────────────────────

FROM python:3.11-slim

LABEL maintainer="Rakesh"
LABEL description="Warehouse Order Fulfillment RL Environment (MCP Server)"

WORKDIR /app

# Install git (needed for openenv-core from github)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY warehouse_env/ ./warehouse_env/
COPY train.py test.py README.md ./

# Install the package and its dependencies
RUN pip install --no-cache-dir ./warehouse_env

# Expose MCP server port
EXPOSE 8000

# Default: start the MCP server
# Use --host 0.0.0.0 to allow external connections
CMD ["python", "-m", "warehouse_env.server.app"]
