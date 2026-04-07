# Warehouse Order Fulfillment - RL Environment (OpenEnv)

> An OpenAI Gym-compatible reinforcement learning environment with **MCP Server** support.
> Designed for the **Meta PyTorch OpenEnv Hackathon**.

The agent learns to optimally assign warehouse orders to workers, minimizing total fulfillment time while maximizing worker utilization and handling priority orders.

---

## Why Reinforcement Learning?

Traditional warehouse scheduling relies on static heuristics like FIFO (first-in-first-out) or SPT (shortest processing time). These approaches:

- **Cannot adapt** to changing demand patterns (rush hours, seasonal surges)
- **Ignore trade-offs** between clearing the queue quickly vs. balancing worker loads
- **Miss priority signals** - a same-day delivery order stuck behind bulk orders

**RL learns to balance multiple objectives simultaneously** - completion speed, worker utilization, queue congestion, and priority handling - by discovering policies that outperform fixed rules through trial and error.

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

## Project Structure (OpenEnv/MCP Style)

```
+-- warehouse_env/              # Dedicated Python package
|   +-- envs/                   # Core Gymnasium Logic
|   |   +-- warehouse_env.py    # Environment: state, actions, rewards
|   +-- server/                 # MCP Server (FastAPI + FastMCP)
|   |   +-- app.py              # Server entry point
|   |   +-- warehouse_environment.py  # MCP tool definitions
|   +-- client.py               # MCP Client Library
|   +-- pyproject.toml          # Package Metadata & Deps
|   +-- openenv.yaml            # OpenEnv Config
+-- train.py                    # RL Training (Random + Heuristic + Q-Learning)
+-- test.py                     # Comprehensive Test Suite
+-- Dockerfile                  # Containerization (MCP Server)
+-- README.md
```

---

## Environment Design

### State (Observation)

A flat vector of shape `(1 + 2*W + 3*Q,)`:

| Slice                    | Description                         | Range   |
|--------------------------|-------------------------------------|---------|
| `[0]`                    | Pending orders count (normalized)   | 0 - 1   |
| `[1 .. W]`               | Worker busy-time remaining          | 0 - 1   |
| `[W+1 .. 2W]`            | Worker utilization so far           | 0 - 1   |
| `[2W+1 .. 2W+Q]`         | Queue processing times (normalized) | 0 - 1   |
| `[2W+Q+1 .. 2W+2Q]`      | Queue wait times (normalized)       | 0 - 1   |
| `[2W+2Q+1 .. 2W+3Q]`     | Queue priority flags                | 0 or 1  |

### Actions

| Action       | Description                                             |
|--------------|---------------------------------------------------------|
| `0 .. Q-1`   | Assign order at queue index to the least-busy worker    |
| `Q`          | No-op / wait (advance time without assigning)           |

### Reward (Multi-Objective)

| Component            | Value                        | Description                                   |
|----------------------|------------------------------|-----------------------------------------------|
| `completion_reward`  | +1.0                         | Per order completed                            |
| `speed_bonus`        | 0 to +0.5                   | Faster fulfillment = higher bonus              |
| `priority_bonus`     | +0.2 to +0.5                | Extra reward for priority orders               |
| `idle_penalty`       | -0.1 per idle worker         | When queue has pending orders                  |
| `congestion_penalty` | -0.05 * (queue_len/max)     | Penalizes full queues                           |
| `wait_time_penalty`  | -0.03 * mean_wait * count   | Penalizes long-waiting orders                  |
| `finish_bonus`       | +5.0                         | For completing all orders before time limit     |

All reward components are returned in `info["reward_breakdown"]` for full transparency.

---

## Scenario Modes

Simulate different real-world demand patterns:

| Mode     | Order Arrival Prob | Proc Time Range | Real-World Analogy          |
|----------|-------------------|-----------------|-----------------------------|
| `normal` | 0.40              | 1 - 8           | Regular daily operations     |
| `rush`   | 0.70              | 1 - 5           | Black Friday / flash sale    |
| `low`    | 0.15              | 3 - 12          | Off-season / night shift     |

```bash
python train.py --mode rush --episodes 500
python test.py --mode low
```

---

## Priority Orders

~20% of orders are flagged as **priority** (configurable via `priority_prob`).

- Priority orders appear as `1.0` in the observation vector
- Completing them quickly yields a **+0.5 bonus**; late completion still gets **+0.2**
- Shown with a star marker in the console visualization

This models real-world expedited shipping (Amazon Prime, same-day delivery).

---

## Getting Started

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

Train and compare three policies (Random vs Heuristic vs Q-Learning):

```bash
python train.py --episodes 1000
python train.py --episodes 500 --mode rush
```

### 4. Testing

```bash
# Full test suite (API, determinism, priority, modes, edge cases, demo)
python test.py

# With visualization
python test.py --render

# Test MCP client-server interaction (server must be running)
python test.py --mcp
```

---

## Evaluation Results

Three policies are compared across key metrics:

| Metric             | Random     | Heuristic (SPT) | Q-Learning |
|--------------------|------------|------------------|------------|
| Avg Reward         | ~20-30     | ~35-45           | ~40-55     |
| Avg Fulfill Time   | ~8-10      | ~5-7             | ~4-6       |
| Orders Completed   | ~35-40     | ~45-48           | ~45-50     |
| Worker Utilization  | ~40-50%    | ~55-65%          | ~60-70%    |

> Results vary by scenario mode and episode count. Run `python train.py` for exact numbers.

### Key Findings

- **Heuristic (SPT)** significantly outperforms random by always picking the fastest job
- **Q-Learning** learns to outperform the heuristic by considering global state (worker loads, queue depth, priority flags)
- The gap widens in **rush mode** where intelligent scheduling matters most

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
"Warning: High congestion detected! Queue at 85% capacity (17/20)"
"Warning: Queue is FULL! New orders will be dropped."
```

---

## OpenEnv MCP Support

The environment exposes two MCP tools:

| Tool           | Parameters     | Description                         |
|----------------|----------------|-------------------------------------|
| `assign_order` | `order_id: int`| Assign queue slot to a worker       |
| `wait_step`    | *(none)*       | Advance time by 1 tick              |

Both return: `reward`, `observation`, `terminated`, `truncated`, `info`, `decision_reason`, `reward_breakdown`.

---

## Docker Support

```bash
docker build -t warehouse-env .
docker run -p 8000:8000 warehouse-env
```

Compatible with Hugging Face Spaces deployment.

---

## License

MIT
