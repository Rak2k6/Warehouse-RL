"""
Training Script - Warehouse Order Fulfillment Environment
==========================================================
Implements three policies for comparison:
  1. Random Policy    - selects a random valid action each step.
  2. Heuristic (SPT)  - shortest processing time first, priority tiebreak.
  3. Q-Learning       - tabularized Q-learning with e-greedy exploration.

The Q-table uses a discretized state key derived from the observation
vector (pending-order count bucket, busiest-worker bucket, shortest-
order bucket) to keep the table tractable.

Usage:
    python train.py                          # default: 1000 episodes, normal mode
    python train.py --episodes 2000          # custom episode count
    python train.py --mode rush              # rush scenario
    python train.py --render                 # show per-step rendering
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


# -- Helpers ---------------------------------------------------------------

def discretize_state(obs: np.ndarray, num_workers: int, max_queue: int) -> tuple:
    """
    Convert continuous observation into a compact discrete key:
      - pending_bucket   (0..4)  -> how full the queue is
      - busy_bucket      (0..4)  -> how busy the busiest worker is
      - shortest_order   (0..4)  -> shortest pending order proc time
      - has_priority     (0..1)  -> any priority order in queue
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

    # Priority awareness
    prio_offset = 1 + 2 * num_workers + 2 * max_queue
    prio_flags = obs[prio_offset : prio_offset + max_queue]
    has_priority = int(np.any(prio_flags > 0.5))

    return (pending_bucket, busy_bucket, shortest, has_priority)


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


def pick_heuristic_action(
    queue_proc_time: np.ndarray,
    queue_priority: np.ndarray,
    max_queue: int,
) -> int:
    """
    Heuristic: Shortest Processing Time (SPT) first, with priority tiebreak.
    Priority orders are preferred when processing times are equal.
    """
    valid = np.where(queue_proc_time > 0)[0]
    if len(valid) == 0:
        return max_queue  # no-op

    # Sort by: priority DESC (handle priority first), then proc_time ASC
    proc_times = queue_proc_time[valid]
    priorities = queue_priority[valid]

    # Create a sorting key: primary = -priority, secondary = proc_time
    sort_keys = np.column_stack((-priorities, proc_times))
    best_idx = valid[np.lexsort((sort_keys[:, 1], sort_keys[:, 0]))[0]]
    return int(best_idx)


def print_episode_summary(ep: int, info: dict, policy_name: str):
    """Print a clean one-line episode summary."""
    summary = info.get("episode_summary", {})
    if summary:
        print(
            f"  [{policy_name}] Ep {ep+1:>5}  "
            f"reward={summary.get('total_reward', 0):>7.2f}  "
            f"completed={summary.get('orders_completed', 0):>3}/"
            f"{summary.get('orders_generated', 0)}  "
            f"avg_ft={summary.get('avg_fulfillment_time', 0):.2f}  "
            f"util={summary.get('worker_utilization', 0)*100:.1f}%"
        )


# -- Policies --------------------------------------------------------------

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
            print_episode_summary(ep, info, "Random")

    return logs


def run_heuristic_policy(
    env: WarehouseOrderFulfillmentEnv,
    episodes: int,
    rng: np.random.Generator,
    render: bool = False,
) -> list[dict]:
    """Run the SPT heuristic policy and collect metrics."""
    logs: list[dict] = []

    for ep in range(episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 2**31)))
        total_reward = 0.0
        done = False

        while not done:
            action = pick_heuristic_action(
                env.queue_proc_time, env.queue_priority, env.max_queue
            )
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        logs.append({
            "episode": ep,
            "policy": "heuristic_spt",
            "total_reward": round(total_reward, 2),
            "orders_completed": info["orders_completed"],
            "avg_fulfillment_time": round(info["avg_fulfillment_time"], 2),
            "worker_utilization": round(info["worker_utilization"], 4),
            "steps": info["step"],
        })

        if (ep + 1) % max(1, episodes // 10) == 0:
            print_episode_summary(ep, info, "SPT")

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
            # e-greedy action selection (only from valid actions)
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
                f"e={epsilon:.3f}  "
                f"avg_reward={avg_reward:>7.2f}  "
                f"avg_completed={avg_completed:>5.1f}"
            )

    return logs, dict(Q)


# -- Evaluation helpers ----------------------------------------------------

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
    print("\u2500" * 60)
    print(f"  {label}")
    print("\u2500" * 60)
    print(f"  Episodes         : {len(logs)}")
    print(f"  Avg Reward       : {np.mean(rewards):>8.2f} +/- {np.std(rewards):.2f}")
    print(f"  Avg Completed    : {np.mean(completed):>8.1f} +/- {np.std(completed):.1f}")
    print(f"  Avg Fulfill Time : {np.mean(ft):>8.2f} +/- {np.std(ft):.2f}")
    print(f"  Avg Utilization  : {np.mean(util)*100:>7.1f}%")
    print("\u2500" * 60)


def print_comparison_table(
    random_logs: list[dict],
    heuristic_logs: list[dict],
    q_logs: list[dict],
):
    """Print a side-by-side comparison of all three policies."""
    def avg(logs, key):
        return np.mean([l[key] for l in logs])

    r_rew = avg(random_logs, "total_reward")
    h_rew = avg(heuristic_logs, "total_reward")
    q_rew = avg(q_logs, "total_reward")

    r_ft = avg(random_logs, "avg_fulfillment_time")
    h_ft = avg(heuristic_logs, "avg_fulfillment_time")
    q_ft = avg(q_logs, "avg_fulfillment_time")

    r_comp = avg(random_logs, "orders_completed")
    h_comp = avg(heuristic_logs, "orders_completed")
    q_comp = avg(q_logs, "orders_completed")

    r_util = avg(random_logs, "worker_utilization")
    h_util = avg(heuristic_logs, "worker_utilization")
    q_util = avg(q_logs, "worker_utilization")

    print()
    print("\u2554" + "\u2550" * 68 + "\u2557")
    print("\u2551" + "  POLICY COMPARISON".center(68) + "\u2551")
    print("\u2560" + "\u2550" * 68 + "\u2563")
    print(f"\u2551  {'Metric':<25} {'Random':>10} {'Heuristic':>10} {'Q-Learn':>10} {'Best':>8}  \u2551")
    print("\u2551  " + "\u2500" * 64 + "  \u2551")

    # Reward (higher is better)
    best_rew = "Q" if q_rew >= h_rew and q_rew >= r_rew else ("H" if h_rew >= r_rew else "R")
    print(f"\u2551  {'Avg Reward':<25} {r_rew:>10.2f} {h_rew:>10.2f} {q_rew:>10.2f} {best_rew:>8}  \u2551")

    # Fulfillment time (lower is better)
    best_ft = "Q" if q_ft <= h_ft and q_ft <= r_ft else ("H" if h_ft <= r_ft else "R")
    print(f"\u2551  {'Avg Fulfill Time':<25} {r_ft:>10.2f} {h_ft:>10.2f} {q_ft:>10.2f} {best_ft:>8}  \u2551")

    # Completed (higher is better)
    best_comp = "Q" if q_comp >= h_comp and q_comp >= r_comp else ("H" if h_comp >= r_comp else "R")
    print(f"\u2551  {'Avg Completed':<25} {r_comp:>10.1f} {h_comp:>10.1f} {q_comp:>10.1f} {best_comp:>8}  \u2551")

    # Utilization (higher is better)
    best_util = "Q" if q_util >= h_util and q_util >= r_util else ("H" if h_util >= r_util else "R")
    print(f"\u2551  {'Avg Utilization':<25} {r_util*100:>9.1f}% {h_util*100:>9.1f}% {q_util*100:>9.1f}% {best_util:>8}  \u2551")

    print("\u255a" + "\u2550" * 68 + "\u255d")
    print("  Legend: R=Random, H=Heuristic(SPT), Q=Q-Learning")
    print()


# -- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train RL agents on the Warehouse Order Fulfillment Environment"
    )
    parser.add_argument("--episodes", type=int, default=1000, help="Training episodes")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--render", action="store_true", help="Render during training")
    parser.add_argument(
        "--mode", type=str, default="normal",
        choices=["normal", "rush", "low"],
        help="Scenario mode: normal, rush (high traffic), low (low traffic)"
    )
    parser.add_argument("--save-logs", type=str, default="logs", help="Directory to save logs")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    render_mode = "human" if args.render else None
    env = WarehouseOrderFulfillmentEnv(
        num_workers=4,
        max_queue=20,
        max_orders=50,
        max_steps=200,
        mode=args.mode,
        render_mode=render_mode,
        seed=args.seed,
    )

    print("\u2554" + "\u2550" * 58 + "\u2557")
    print("\u2551" + "  Warehouse Order Fulfillment - RL Training".center(58) + "\u2551")
    print("\u255a" + "\u2550" * 58 + "\u255d")
    print(f"  Workers: {env.num_workers}  |  Max Queue: {env.max_queue}  |  "
          f"Max Orders: {env.max_orders}")
    print(f"  Mode: {args.mode.upper()}  |  Episodes: {args.episodes}  |  Seed: {args.seed}")
    print(f"  Order arrival prob: {env.new_order_prob}  |  "
          f"Proc time range: {env.order_time_range}")
    print()

    # ---------- 1. Random baseline ----------
    print("\u25b6 Training Random Policy Baseline ...")
    t0 = time.time()
    random_logs = run_random_policy(env, args.episodes, rng, render=args.render)
    print(f"  Done in {time.time()-t0:.1f}s")
    print_summary("Random Policy (Training)", random_logs)

    # ---------- 2. Heuristic (SPT) baseline ----------
    print("\n\u25b6 Running Heuristic (SPT + Priority) Policy ...")
    t0 = time.time()
    heuristic_logs = run_heuristic_policy(env, args.episodes, rng, render=args.render)
    print(f"  Done in {time.time()-t0:.1f}s")
    print_summary("Heuristic SPT (Training)", heuristic_logs)

    # ---------- 3. Q-Learning ----------
    print("\n\u25b6 Training Q-Learning Agent ...")
    t0 = time.time()
    q_logs, Q_table = run_q_learning(
        env, args.episodes, rng, render=args.render
    )
    print(f"  Done in {time.time()-t0:.1f}s  |  Q-table states: {len(Q_table)}")
    print_summary("Q-Learning (Training)", q_logs)

    # ---------- 4. Evaluation ----------
    print("\n\u25b6 Evaluating Learned Q-Policy (greedy) ...")
    eval_logs = evaluate_q_policy(env, Q_table, args.eval_episodes, rng)
    print_summary("Q-Learning (Greedy Eval)", eval_logs)

    # ---------- 5. Heuristic eval for fair comparison ----------
    heuristic_eval = heuristic_logs[-args.eval_episodes:]
    random_eval = random_logs[-args.eval_episodes:]

    # ---------- 6. Comparison table ----------
    print_comparison_table(random_eval, heuristic_eval, eval_logs)

    # ---------- 7. Save logs ----------
    log_dir = Path(args.save_logs)
    log_dir.mkdir(exist_ok=True)
    with open(log_dir / "random_logs.json", "w") as f:
        json.dump(random_logs, f, indent=2)
    with open(log_dir / "heuristic_logs.json", "w") as f:
        json.dump(heuristic_logs, f, indent=2)
    with open(log_dir / "q_learning_logs.json", "w") as f:
        json.dump(q_logs, f, indent=2)
    with open(log_dir / "eval_logs.json", "w") as f:
        json.dump(eval_logs, f, indent=2)
    print(f"  \U0001f4c1 Logs saved to {log_dir.resolve()}/")

    env.close()
    print("\n\u2705 Training complete.")


if __name__ == "__main__":
    main()
