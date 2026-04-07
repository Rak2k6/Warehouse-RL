# 🏭 Warehouse Order Fulfillment — RL Environment (OpenEnv)

> An OpenAI Gym-compatible reinforcement learning environment with **MCP Server** support. 
> Designed for the **Meta PyTorch OpenEnv Hackathon**.

The agent learns to optimally assign warehouse orders to workers, minimizing total fulfillment time.

---

## 🏗️ Project Structure (OpenEnv/MCP Style)

```
├── warehouse_env/          # Dedicated Python package
│   ├── envs/               # Core Gymnasium Logic
│   ├── server/             # MCP Server (FastAPI + FastMCP)
│   ├── client.py           # MCP Client Library
│   ├── pyproject.toml      # Package Metadata & Deps
│   └── openenv.yaml        # OpenEnv Config
├── train.py                # RL Training Script (Q-Learning)
├── test.py                 # Comprehensive Test Suite
├── Dockerfile              # Containerization (MCP Server)
└── README.md
```

---

## 🚀 Getting Started

### 1. Installation

```bash
# Clone the repository
git clone <repo-url>
cd Traffic\ RL

# Install the package locally
pip install ./warehouse_env
```

### 2. Running the MCP Server

The environment can be served as an MCP (Model Context Protocol) server, making it accessible to AI agents and external tools.

```bash
# Start the server
python -m warehouse_env.server.app
```
*Server runs at http://localhost:8000*

### 3. Training the RL Agent

Train a tabular Q-learning agent against a random baseline:

```bash
python train.py --episodes 1000
```

### 4. Testing

Run the full test suite (Gym API, Determinism, Edge Cases, Demo):

```bash
python test.py
```

To test the **MCP Client-Server** interaction (ensure server is running in another terminal):

```bash
python test.py --mcp
```

---

## 🧠 Environment Design

| Component | Details |
|-----------|---------|
| **State** | Pending order count, worker busy-times, worker utilization, queue processing times, queue wait times. |
| **Actions** | Discrete: assign order index `0..N` from queue, or `wait_step`. |
| **MCP Tools** | `warehouse_env.assign_order`, `warehouse_env.wait_step` |
| **Reward** | Progress-based with fulfillment speed bonuses and idle penalties. |

---

## 🐳 Docker Support

Build and run the MCP server in a container:

```bash
docker build -t warehouse-env .
docker run -p 8000:8000 warehouse-env
```

---

## 🤗 Hugging Face Spaces Compatibility

This structure is perfectly aligned with Hugging Face Spaces. The `Dockerfile` handles the installation of the `warehouse_env` package and starts the FastAPI server, which exposes the OpenEnv interface.

---

## 📄 License
MIT
