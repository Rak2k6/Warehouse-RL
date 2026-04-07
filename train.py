"""
Training Script — Warehouse Order Fulfillment Environment
==========================================================
Implements two baselines for comparison:
  1. Random Policy  — selects a random valid action each step.
  2. Q-Learning      — tabularized Q-learning with ε-greedy exploration.

The Q-table uses a discretized state key derived from the observation
vector (pending-order count bucket, busiest-worker bucket, shortest-
order bucket) to keep the table tractable.

Usage:
    python train.py                     # default: 1000 episodes
    python train.py --episodes 2000     # custom episode count
    python train.py --render            # show per-step rendering
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from warehouse_env.envs.warehouse_env import WarehouseOrderFulfillmentEnv


# ── Helpers ──────────────────────────────────────────────────────────

def discretize_state(obs: np.ndarray, num_workers: int, max_queue: int) -> tuple:
    """
    Convert continuous observation into a compact discrete key:
      - pending_bucket   (0..4)  → how full the queue is
      - busy_bucket      (0..4)  → how busy the busiest worker is
      - shortest_order   (0..4)  → shortest pending order proc time
    """
    pending_norm = obs[0]
    pending_bucket = int(np.clip(pending_norm * 5, 0, 4))

    worker_offset = 1
    worker_busy = obs[worker_offset : worker_offset + num_workers]
    busy_bucket = int(np.clip(np.max(worker_busy) * 5, 0, 4))

    queue_offset = 1 + 2 * num_workers
    queue_times = obs[queue_offset : queue_offset + max_queue]
    nonzero = queue_times[queue_times > 0]
    if len(nonzero) > 0:
        shortest = int(np.clip(np.min(nonzero) * 5, 0, 4))
    else:
        shortest = 0

    return (pending_bucket, busy_bucket, shortest)


def pick_valid_action(
    queue_proc_time: np.ndarray,
    max_queue: int,
    rng: np.random.Generator,
) -> int:
    """Return a random valid queue index, or no-op if queue is empty."""
    valid = np.where(queue_proc_time > 0)[0]
    if len(valid) == 0:
        return max_queue  # no-op
    return int(rng.choice(valid))


# ── Policies ─────────────────────────────────────────────────────────

def run_random_policy(
    env: WarehouseOrderFulfillmentEnv,
    episodes: int,
    rng: np.random.Generator,
    render: bool = False,
) -> list[dict]:
    """Run a purely random policy and collect metrics."""
    logs: list[dict] = []

    for ep in range(episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 2**31)))
        total_reward = 0.0
        done = False

        while not done:
            action = pick_valid_action(env.queue_proc_time, env.max_queue, rng)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        logs.append({
            "episode": ep,
            "policy": "random",
            "total_reward": round(total_reward, 2),
            "orders_completed": info["orders_completed"],
            "avg_fulfillment_time": round(info["avg_fulfillment_time"], 2),
            "worker_utilization": round(info["worker_utilization"], 4),
            "steps": info["step"],
        })

        if (ep + 1) % max(1, episodes // 10) == 0:
            print(
                f"  [Random] Episode {ep+1:>5}/{episodes}  "
                f"reward={total_reward:>7.2f}  "
                f"completed={info['orders_completed']:>3}  "
                f"avg_ft={info['avg_fulfillment_time']:.2f}"
            )

    return logs


def run_q_learning(
    env: WarehouseOrderFulfillmentEnv,
    episodes: int,
    rng: np.random.Generator,
    render: bool = False,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
) -> tuple[list[dict], dict]:
    """
    Train a tabular Q-learning agent.
    Returns episode logs and the Q-table.
    """
    Q: dict[tuple, np.ndarray] = defaultdict(
        lambda: np.zeros(env.action_space.n, dtype=np.float64)
    )
    logs: list[dict] = []
    epsilon = epsilon_start

    for ep in range(episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 2**31)))
        state = discretize_state(obs, env.num_workers, env.max_queue)
        total_reward = 0.0
        done = False

        while not done:
            # ε-greedy action selection (only from valid actions)
            valid = np.where(env.queue_proc_time > 0)[0]
            valid_actions = list(valid) + [env.max_queue]

            if rng.random() < epsilon:
                action = int(rng.choice(valid_actions))
            else:
                q_vals = Q[state]
                # Mask invalid actions to -inf
                masked = np.full_like(q_vals, -np.inf)
                for a in valid_actions:
                    masked[a] = q_vals[a]
                action = int(np.argmax(masked))

            obs_next, reward, terminated, truncated, info = env.step(action)
            next_state = discretize_state(obs_next, env.num_workers, env.max_queue)
            total_reward += reward
            done = terminated or truncated

            # Q-update
            best_next = np.max(Q[next_state]) if not done else 0.0
            Q[state][action] += alpha * (
                reward + gamma * best_next - Q[state][action]
            )

            state = next_state

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        logs.append({
            "episode": ep,
            "policy": "q_learning",
            "total_reward": round(total_reward, 2),
            "orders_completed": info["orders_completed"],
            "avg_fulfillment_time": round(info["avg_fulfillment_time"], 2),
            "worker_utilization": round(info["worker_utilization"], 4),
            "steps": info["step"],
            "epsilon": round(epsilon, 4),
        })

        if (ep + 1) % max(1, episodes // 10) == 0:
            recent = logs[-max(1, episodes // 10) :]
            avg_reward = np.mean([l["total_reward"] for l in recent])
            avg_completed = np.mean([l["orders_completed"] for l in recent])
            print(
                f"  [Q-Learn] Episode {ep+1:>5}/{episodes}  "
                f"ε={epsilon:.3f}  "
                f"avg_reward={avg_reward:>7.2f}  "
                f"avg_completed={avg_completed:>5.1f}"
            )

    return logs, dict(Q)


# ── Evaluation helpers ───────────────────────────────────────────────

def evaluate_q_policy(
    env: WarehouseOrderFulfillmentEnv,
    Q: dict,
    episodes: int = 50,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Evaluate the learned Q-policy greedily (no exploration)."""
    if rng is None:
        rng = np.random.default_rng(42)

    logs: list[dict] = []
    for ep in range(episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 2**31)))
        state = discretize_state(obs, env.num_workers, env.max_queue)
        total_reward = 0.0
        done = False

        while not done:
            valid = np.where(env.queue_proc_time > 0)[0]
            valid_actions = list(valid) + [env.max_queue]

            q_vals = Q.get(state, np.zeros(env.action_space.n))
            masked = np.full_like(q_vals, -np.inf)
            for a in valid_actions:
                masked[a] = q_vals[a]
            action = int(np.argmax(masked))

            obs, reward, terminated, truncated, info = env.step(action)
            state = discretize_state(obs, env.num_workers, env.max_queue)
            total_reward += reward
            done = terminated or truncated

        logs.append({
            "episode": ep,
            "policy": "q_greedy",
            "total_reward": round(total_reward, 2),
            "orders_completed": info["orders_completed"],
            "avg_fulfillment_time": round(info["avg_fulfillment_time"], 2),
            "worker_utilization": round(info["worker_utilization"], 4),
        })
    return logs


def print_summary(label: str, logs: list[dict]):
    rewards = [l["total_reward"] for l in logs]
    completed = [l["orders_completed"] for l in logs]
    ft = [l["avg_fulfillment_time"] for l in logs]
    util = [l["worker_utilization"] for l in logs]
    print("-" * 60)
    print(f"  {label}")
    print("-" * 60)
    print(f"  Episodes         : {len(logs)}")
    print(f"  Avg Reward       : {np.mean(rewards):>8.2f} ± {np.std(rewards):.2f}")
    print(f"  Avg Completed    : {np.mean(completed):>8.1f} ± {np.std(completed):.1f}")
    print(f"  Avg Fulfill Time : {np.mean(ft):>8.2f} ± {np.std(ft):.2f}")
    print(f"  Avg Utilization  : {np.mean(util)*100:>7.1f}%")
    print(f"{'─'*60}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train RL agents on the Warehouse Order Fulfillment Environment"
    )
    parser.add_argument("--episodes", type=int, default=1000, help="Training episodes")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--render", action="store_true", help="Render during training")
    parser.add_argument("--save-logs", type=str, default="logs", help="Directory to save logs")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    render_mode = "human" if args.render else None
    env = WarehouseOrderFulfillmentEnv(
        num_workers=4,
        max_queue=20,
        max_orders=50,
        max_steps=200,
        render_mode=render_mode,
        seed=args.seed,
    )

    print("+----------------------------------------------------------+")
    print("     Warehouse Order Fulfillment - RL Training            ")
    print("+----------------------------------------------------------+")
    print(f"  Workers: {env.num_workers}  |  Max Queue: {env.max_queue}  |  "
          f"Max Orders: {env.max_orders}")
    print(f"  Episodes: {args.episodes}  |  Seed: {args.seed}")
    print()

    # ---------- 1. Random baseline ----------
    print("▶ Training Random Policy Baseline …")
    t0 = time.time()
    random_logs = run_random_policy(env, args.episodes, rng, render=args.render)
    print(f"  Done in {time.time()-t0:.1f}s")
    print_summary("Random Policy (Training)", random_logs)

    # ---------- 2. Q-Learning ----------
    print("\n▶ Training Q-Learning Agent …")
    t0 = time.time()
    q_logs, Q_table = run_q_learning(
        env, args.episodes, rng, render=args.render
    )
    print(f"  Done in {time.time()-t0:.1f}s  |  Q-table states: {len(Q_table)}")
    print_summary("Q-Learning (Training)", q_logs)

    # ---------- 3. Evaluation ----------
    print("\n▶ Evaluating Learned Q-Policy (greedy) …")
    eval_logs = evaluate_q_policy(env, Q_table, args.eval_episodes, rng)
    print_summary("Q-Learning (Greedy Eval)", eval_logs)

    # ---------- 4. Comparison ----------
    random_last = random_logs[-args.eval_episodes:]
    print("\n+----------------------------------------------------------+")
    print("                     COMPARISON                           ")
    print("+----------------------------------------------------------+")

    r_rew = np.mean([l["total_reward"] for l in random_last])
    q_rew = np.mean([l["total_reward"] for l in eval_logs])
    r_ft = np.mean([l["avg_fulfillment_time"] for l in random_last])
    q_ft = np.mean([l["avg_fulfillment_time"] for l in eval_logs])
    r_comp = np.mean([l["orders_completed"] for l in random_last])
    q_comp = np.mean([l["orders_completed"] for l in eval_logs])

    print(f"  {'Metric':<25} {'Random':>10} {'Q-Learn':>10} {'Δ':>10}")
    print("  " + "-" * 55)
    print(f"  {'Avg Reward':<25} {r_rew:>10.2f} {q_rew:>10.2f} {q_rew-r_rew:>+10.2f}")
    print(f"  {'Avg Fulfill Time':<25} {r_ft:>10.2f} {q_ft:>10.2f} {q_ft-r_ft:>+10.2f}")
    print(f"  {'Avg Completed':<25} {r_comp:>10.1f} {q_comp:>10.1f} {q_comp-r_comp:>+10.1f}")
    print("+----------------------------------------------------------+")

    # ---------- 5. Save logs ----------
    log_dir = Path(args.save_logs)
    log_dir.mkdir(exist_ok=True)
    with open(log_dir / "random_logs.json", "w") as f:
        json.dump(random_logs, f, indent=2)
    with open(log_dir / "q_learning_logs.json", "w") as f:
        json.dump(q_logs, f, indent=2)
    with open(log_dir / "eval_logs.json", "w") as f:
        json.dump(eval_logs, f, indent=2)
    print(f"\n  📁 Logs saved to {log_dir.resolve()}/")

    env.close()
    print("\n✅ Training complete.")


if __name__ == "__main__":
    main()
