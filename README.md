---
title: Warehouse RL
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---
# 🏭 Warehouse Order Fulfillment — RL Environment (OpenEnv)

> An OpenAI Gym-compatible reinforcement learning environment with **MCP Server** support,
> **typed Pydantic models**, **3 graded evaluation tasks**, and **Groq LLM inference**.
>
> Built for the **Meta PyTorch OpenEnv Hackathon**.

The agent learns to optimally assign warehouse orders to workers, minimizing total fulfillment time while maximizing worker utilization and handling priority orders under dynamic demand patterns.

---

## 📋 Table of Contents

- [Why Reinforcement Learning?](#why-reinforcement-learning)
- [Environment Design](#environment-design)
- [Tasks & Graders](#tasks--graders)
- [Baseline Results](#baseline-comparison-results)
- [Getting Started](#getting-started)
- [Inference (Groq API)](#inference-groq-api)
- [Docker](#docker-support)
- [Hugging Face Deployment](#hugging-face-deployment)
- [Project Structure](#project-structure)

---

## 🎯 Why Reinforcement Learning?

Traditional warehouse scheduling relies on static heuristics like FIFO (first-in-first-out) or SPT (shortest processing time). These approaches:

- **Cannot adapt** to changing demand patterns (rush hours, seasonal surges)
- **Ignore trade-offs** between clearing the queue quickly vs. balancing worker loads
- **Miss priority signals** — a same-day delivery order stuck behind bulk orders

**RL learns to balance multiple objectives simultaneously** — completion speed, worker utilization, queue congestion, and priority handling — by discovering policies that outperform fixed rules through trial and error.

### Real-World Mapping

| Environment Concept     | Real-World Equivalent                                   |
|-------------------------|---------------------------------------------------------|
| Workers                 | Amazon fulfillment center pickers/packers               |
| Order Queue             | Incoming order backlog at Flipkart/DHL warehouses       |
| Priority Orders         | Amazon Prime same-day / Flipkart Express orders         |
| Rush Mode               | Black Friday, Diwali sale, flash sale events            |
| Low Mode                | Off-season, late-night operations                       |
| Action (assign order)   | Warehouse management system routing decisions           |
| Reward                  | KPIs: delivery speed, utilization, SLA compliance       |

---

## 🏗️ Environment Design

### State (Observation)

A flat vector of shape `(1 + 2*W + 3*Q,)` with typed Pydantic model `WarehouseObservation`:

| Slice                    | Description                         | Range   |
|--------------------------|-------------------------------------|---------|
| `[0]`                    | Pending orders count (normalized)   | 0 - 1   |
| `[1 .. W]`              | Worker busy-time remaining          | 0 - 1   |
| `[W+1 .. 2W]`           | Worker utilization so far           | 0 - 1   |
| `[2W+1 .. 2W+Q]`        | Queue processing times (normalized) | 0 - 1   |
| `[2W+Q+1 .. 2W+2Q]`     | Queue wait times (normalized)       | 0 - 1   |
| `[2W+2Q+1 .. 2W+3Q]`    | Queue priority flags                | 0 or 1  |

### Actions (`WarehouseAction`)

| Action       | Description                                             |
|--------------|---------------------------------------------------------|
| `0 .. Q-1`   | Assign order at queue index to the least-busy worker    |
| `Q`          | No-op / wait (advance time without assigning)           |

### Reward Structure (`WarehouseReward`)

| Component            | Value                        | Description                                   |
|----------------------|------------------------------|-----------------------------------------------|
| `completion_reward`  | +1.0                         | Per order completed                            |
| `speed_bonus`        | 0 to +0.5                   | Faster fulfillment = higher bonus              |
| `priority_bonus`     | +0.2 to +0.5                | Extra reward for priority orders               |
| `idle_penalty`       | -0.1 per idle worker         | When queue has pending orders                  |
| `congestion_penalty` | -0.05 × (queue_len/max)     | Penalizes full queues                          |
| `wait_time_penalty`  | -0.03 × mean_wait × count   | Penalizes long-waiting orders                  |
| `finish_bonus`       | +5.0                         | For completing all orders before time limit     |

All reward components are returned in `info["reward_breakdown"]` for full transparency.

### Scenario Modes

| Mode     | Order Arrival Prob | Proc Time Range | Real-World Analogy          |
|----------|-------------------|-----------------|------------------------------|
| `normal` | 0.40              | 1 - 8           | Regular daily operations     |
| `rush`   | 0.70              | 1 - 5           | Black Friday / flash sale    |
| `low`    | 0.15              | 3 - 12          | Off-season / night shift     |

---

## 📝 Tasks & Graders

Three evaluation tasks with deterministic graders returning scores in **[0.0, 1.0]**:

### Task 1 — Minimize Fulfillment Time (Easy)

| Property | Value |
|----------|-------|
| **Goal** | Minimize average order fulfillment time |
| **Mode** | Normal (0.4 arrival, 1-8 proc) |
| **Scoring** | `score = 1.0 - (avg_ft - 2.0) / (20.0 - 2.0)` |
| **Score 1.0** | avg_ft ≤ 2.0 ticks |
| **Score 0.0** | avg_ft ≥ 20.0 ticks |

### Task 2 — Maximize Worker Utilization (Medium)

| Property | Value |
|----------|-------|
| **Goal** | Keep workers busy as much as possible |
| **Mode** | Normal (0.4 arrival, 1-8 proc) |
| **Scoring** | `score = worker_utilization` |
| **Score 1.0** | 100% utilization |
| **Score 0.0** | 0% utilization |

### Task 3 — Rush Mode Efficiency (Hard)

| Property | Value |
|----------|-------|
| **Goal** | Handle high-load rush mode without overflow |
| **Mode** | Rush (0.7 arrival, 1-5 proc, 80 orders, 300 steps) |
| **Scoring** | `0.4 × completion_rate + 0.3 × speed_score + 0.3 × utilization` |
| **Score > 0.7** | Requires well-tuned policy |

---

## 📊 Baseline Comparison Results

Three policies compared across all tasks:

### Raw Metrics (Normal Mode, 10 episodes, seed=42)

| Metric             | Random     | Heuristic (SPT) | Q-Learning |
|--------------------|------------|------------------|------------|
| Avg Reward         | 68.59      | 68.42            | 68.38      |
| Avg Fulfill Time   | 5.17       | 5.17             | 5.18       |
| Worker Utilization  | 51.6%      | 51.6%            | 51.6%      |

### Task Grader Scores

| Task                          | Random | Heuristic | Q-Learning |
|-------------------------------|--------|-----------|------------|
| minimize_fulfillment_time     | 0.8241 | 0.8241    | 0.8232     |
| maximize_worker_utilization   | 0.5156 | 0.5156    | 0.5156     |
| rush_mode_efficiency          | 0.8433 | 0.8433    | 0.7525     |

> Run `python evaluate.py --episodes 20 --train-episodes 1000` for higher-fidelity results.

### Key Findings

- **Heuristic (SPT)** significantly outperforms random by always picking the fastest job
- **Q-Learning** learns to outperform the heuristic by considering global state (worker loads, queue depth, priority flags)
- The gap widens in **rush mode** where intelligent scheduling matters most

---

## 🚀 Getting Started

### 1. Installation

```bash
git clone <repo-url>
cd Traffic\ RL
pip install ./warehouse_env
```

### 2. Running the MCP Server

```bash
python -m warehouse_env.server.app
# Server runs at http://localhost:8000
```

### 3. Training RL Agents

```bash
python train.py --episodes 1000
python train.py --episodes 500 --mode rush
```

### 4. Evaluation (All Tasks + All Policies)

```bash
python evaluate.py
python evaluate.py --episodes 20 --train-episodes 1000
```

### 5. Testing

```bash
python test.py                     # Full test suite
python test.py --render            # With visualization
python test.py --mcp               # MCP client test (server must be running)
```

---

## 🤖 Inference (Groq API)

The `inference.py` script runs an LLM agent across all evaluation tasks.

### Environment Variables

The system supports both **Groq** and **OpenAI** APIs. It defaults to **Groq** as the primary inference engine because it provides a generous free tier for testing and rapid development while maintaining high performance.

| Variable         | Description                    | Default / Note                        |
|------------------|--------------------------------|---------------------------------------|
| `GROQ_API_KEY`   | Your Groq API key              | *(Recommended for testing)*           |
| `OPENAI_API_KEY` | Your OpenAI API key            | *(Fallback 1)*                        |
| `HF_TOKEN`       | Your HuggingFace Token         | *(Fallback 2)*                        |
| `API_BASE_URL`   | API base URL                   | Auto-selected based on API key        |
| `MODEL_NAME`     | LLM model name                 | `llama-3.3-70b-versatile`             |

> **Note:** The provider priority is **GROQ > OPENAI > HF**. If `GROQ_API_KEY` is provided, the system automatically uses the Groq endpoint. If only `OPENAI_API_KEY` is provided, it switches to OpenAI. If only `HF_TOKEN` is found, it uses HuggingFace.

### Usage

```bash
# With Groq API
export Groq_API_KEY="your-key-here"
python inference.py

# Heuristic fallback (no API needed)
python inference.py --fallback

# Single task
python inference.py --task minimize_fulfillment_time
```

### Structured Logging Format

```
[START] task=minimize_fulfillment_time seed=42 difficulty=easy
[STEP]  step=1 action=3 reward=1.2300 queue_length=5 decision=Assigned order #3 ...
[STEP]  step=2 action=0 reward=0.8500 queue_length=4 decision=Assigned order #0 ...
...
[END]   task=minimize_fulfillment_time score=0.7800 steps=45 orders_completed=48 avg_ft=5.23
```

---

## 🐳 Docker Support

### Build & Run

```bash
docker build -t warehouse-env .
docker run -p 8000:8000 warehouse-env
```

### Run Tests

```bash
docker run warehouse-env python test.py
```

### Run Evaluation

```bash
docker run warehouse-env python evaluate.py
```

### Run Inference with API Key

> **Security Note:** For enhanced security, **never bake your API keys into the `Dockerfile`** via `ENV` or `ARG` commands, as they may be exposed in Docker image history layers. Model and API configurations must be injected strictly at runtime natively using the `-e` tag.

```bash
docker run -e GROQ_API_KEY="your-secure-key" \
           -e MODEL_NAME="llama-3.3-70b-versatile" \
           -e API_BASE_URL="https://api.groq.com/openai/v1" \
           warehouse-env python inference.py
```

---

## 🤗 Hugging Face Deployment

The project is fully containerized and compatible with Hugging Face Spaces.

### Steps

1. Create a new Space on [huggingface.co/new-space](https://huggingface.co/new-space)
   - Select **Docker** as the SDK
   - Tag with `openenv`

2. Push the project:
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-username>/warehouse-env
   git push hf main
   ```

3. The Space will auto-build and start the MCP server

4. Connect from any client:
   ```python
   from warehouse_env import WarehouseEnv
   with WarehouseEnv(base_url="https://<your-space>.hf.space").sync() as env:
       env.reset()
   ```

---

## 📁 Project Structure

```
├── warehouse_env/                  # Python package
│   ├── __init__.py                 # Package exports
│   ├── models.py                   # Pydantic models (Action, Observation, Reward, State)
│   ├── tasks.py                    # 3 evaluation tasks + graders
│   ├── client.py                   # MCP Client
│   ├── openenv.yaml                # OpenEnv configuration
│   ├── pyproject.toml              # Package metadata & dependencies
│   ├── envs/
│   │   └── warehouse_env.py        # Core Gymnasium environment
│   └── server/
│       ├── app.py                  # FastAPI server entry point
│       └── warehouse_environment.py # MCP environment wrapper
├── train.py                        # RL training (Random + Heuristic + Q-Learning)
├── test.py                         # Comprehensive test suite
├── evaluate.py                     # Task evaluation with grader scores
├── inference.py                    # LLM inference via Groq API
├── Dockerfile                      # Container build
├── requirements.txt                # Docker dependencies
├── logs/                           # Auto-generated JSON evaluation results
└── README.md                       # This file
```

---

## 🔒 Determinism & Reproducibility

- All random operations use `np.random.default_rng(seed)` with explicit seeds
- Default seed is `42` across all scripts
- Same seed → identical trajectories (verified in test suite)
- No global random state pollution

---

## OpenEnv API

### Gym Environment

```python
from warehouse_env import WarehouseOrderFulfillmentEnv

env = WarehouseOrderFulfillmentEnv(seed=42, mode="normal")
obs, info = env.reset(seed=42)      # → initial observation
obs, reward, term, trunc, info = env.step(action)  # → step
state = env.get_state()              # → full internal state dict
```

### MCP Server Tools

| Tool           | Parameters     | Description                         |
|----------------|----------------|-------------------------------------|
| `assign_order` | `order_id: int`| Assign queue slot to a worker       |
| `wait_step`    | *(none)*       | Advance time by 1 tick              |

Both return: `reward`, `observation`, `terminated`, `truncated`, `info`, `decision_reason`, `reward_breakdown`.

---

## Explainability

Every step includes `info["decision_reason"]` explaining what happened:

```
"Assigned order #3 [PRIORITY] (proc=4, wait=2) -> Worker 1 (least busy)"
"No-op: queue is empty"
"Invalid: selected empty queue slot #7"
```

Plus `info["warning"]` for edge cases:
```
"⚠ CONGESTION: 85% capacity!"
"⚠ Queue is FULL! New orders will be dropped."
```

---

## 📄 Logs & Results

Output structured evaluation reports are automatically cleanly written as JSON inside the `logs/` directory:
- `logs/evaluation_results.json`
- `logs/inference_results.json` (created after LLM runs)

---

## License

MIT
