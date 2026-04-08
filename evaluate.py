"""
Evaluation Script — Warehouse Order Fulfillment Environment
=============================================================
Runs all 3 policies (Random, Heuristic, Q-Learning) against all 3 tasks
and outputs a comparison table with grader scores.

Usage:
  python evaluate.py
  python evaluate.py --episodes 20
  python evaluate.py --train-episodes 500
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from warehouse_env.envs.warehouse_env import WarehouseOrderFulfillmentEnv
from warehouse_env.models import EpisodeResult
from warehouse_env.tasks import TASKS, Task, run_task_grader
from warehouse_env.utils import ensure_utf8_stdout, icon


# ── Helpers from train.py ────────────────────────────────────────────

def discretize_state(obs: np.ndarray, num_workers: int, max_queue: int) -> tuple:
    pending_norm = obs[0]
    pending_bucket = int(np.clip(pending_norm * 5, 0, 4))
    worker_offset = 1
    worker_busy = obs[worker_offset : worker_offset + num_workers]
    busy_bucket = int(np.clip(np.max(worker_busy) * 5, 0, 4))
    queue_offset = 1 + 2 * num_workers
    queue_times = obs[queue_offset : queue_offset + max_queue]
    nonzero = queue_times[queue_times > 0]
    shortest = int(np.clip(np.min(nonzero) * 5, 0, 4)) if len(nonzero) > 0 else 0
    prio_offset = 1 + 2 * num_workers + 2 * max_queue
    prio_flags = obs[prio_offset : prio_offset + max_queue]
    has_priority = int(np.any(prio_flags > 0.5))
    return (pending_bucket, busy_bucket, shortest, has_priority)


def pick_random_action(env, rng):
    valid = np.where(env.queue_proc_time > 0)[0]
    if len(valid) == 0:
        return env.max_queue
    return int(rng.choice(valid))


def pick_heuristic_action(env):
    valid = np.where(env.queue_proc_time > 0)[0]
    if len(valid) == 0:
        return env.max_queue
    priorities = env.queue_priority[valid]
    proc_times = env.queue_proc_time[valid]
    sort_keys = np.column_stack((-priorities, proc_times))
    best_idx = valid[np.lexsort((sort_keys[:, 1], sort_keys[:, 0]))[0]]
    return int(best_idx)


# ── Q-Learning Training ─────────────────────────────────────────────

def train_q_learning(
    env_config: dict,
    episodes: int = 500,
    seed: int = 42,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
) -> dict:
    """Train a Q-table and return it."""
    rng = np.random.default_rng(seed)
    env = WarehouseOrderFulfillmentEnv(seed=seed, **env_config)
    Q: dict[tuple, np.ndarray] = defaultdict(
        lambda: np.zeros(env.action_space.n, dtype=np.float64)
    )
    epsilon = epsilon_start

    for ep in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        state = discretize_state(obs, env.num_workers, env.max_queue)
        done = False

        while not done:
            valid = np.where(env.queue_proc_time > 0)[0]
            valid_actions = list(valid) + [env.max_queue]

            if rng.random() < epsilon:
                action = int(rng.choice(valid_actions))
            else:
                q_vals = Q[state]
                masked = np.full_like(q_vals, -np.inf)
                for a in valid_actions:
                    masked[a] = q_vals[a]
                action = int(np.argmax(masked))

            obs_next, reward, terminated, truncated, info = env.step(action)
            next_state = discretize_state(obs_next, env.num_workers, env.max_queue)
            done = terminated or truncated

            best_next = np.max(Q[next_state]) if not done else 0.0
            Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])
            state = next_state

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    env.close()
    return dict(Q)


# ── Evaluate a policy on a task ──────────────────────────────────────

def evaluate_policy(
    policy_name: str,
    task: Task,
    num_episodes: int,
    Q_table: dict | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Run a policy on a task for num_episodes and return results."""
    if rng is None:
        rng = np.random.default_rng(task.seed)

    scores = []
    rewards = []
    ft_values = []
    util_values = []
    completed_values = []

    for ep in range(num_episodes):
        seed = task.seed + ep
        env = WarehouseOrderFulfillmentEnv(seed=seed, **task.env_config)
        obs, _ = env.reset(seed=seed)
        total_reward = 0.0
        done = False

        # Each policy gets its own action RNG so random differs from heuristic
        action_rng = np.random.default_rng(seed + 1000 * (hash(policy_name) % 100))

        if policy_name == "q_learning" and Q_table is not None:
            state_key = discretize_state(obs, env.num_workers, env.max_queue)

        while not done:
            if policy_name == "random":
                action = pick_random_action(env, action_rng)
            elif policy_name == "heuristic":
                action = pick_heuristic_action(env)
            elif policy_name == "q_learning" and Q_table is not None:
                valid = np.where(env.queue_proc_time > 0)[0]
                valid_actions = list(valid) + [env.max_queue]
                q_vals = Q_table.get(state_key, np.zeros(env.action_space.n))
                masked = np.full_like(q_vals, -np.inf)
                for a in valid_actions:
                    masked[a] = q_vals[a]
                action = int(np.argmax(masked))
            else:
                action = pick_heuristic_action(env)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            if policy_name == "q_learning" and Q_table is not None:
                state_key = discretize_state(obs, env.num_workers, env.max_queue)

        summary = info.get("episode_summary", {})
        result = EpisodeResult(
            orders_completed=summary.get("orders_completed", env.orders_completed),
            orders_generated=summary.get("orders_generated", env.total_orders_generated),
            priority_orders_completed=summary.get("priority_completed", env.priority_orders_completed),
            avg_fulfillment_time=summary.get("avg_fulfillment_time", 0.0),
            worker_utilization=summary.get("worker_utilization", 0.0),
            total_reward=summary.get("total_reward", total_reward),
            steps=env.current_step,
            mode=task.env_config.get("mode", "normal"),
            terminated=terminated,
            truncated=truncated,
            queue_overflow_count=getattr(env, "_queue_overflow_count", 0),
        )

        score = run_task_grader(task, result)
        scores.append(score)
        rewards.append(total_reward)
        ft_values.append(result.avg_fulfillment_time)
        util_values.append(result.worker_utilization)
        completed_values.append(result.orders_completed)
        env.close()

    return {
        "policy": policy_name,
        "task": task.name,
        "avg_score": round(float(np.mean(scores)), 4),
        "std_score": round(float(np.std(scores)), 4),
        "avg_reward": round(float(np.mean(rewards)), 2),
        "avg_fulfillment_time": round(float(np.mean(ft_values)), 2),
        "avg_utilization": round(float(np.mean(util_values)), 4),
        "avg_completed": round(float(np.mean(completed_values)), 1),
        "episodes": num_episodes,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ensure_utf8_stdout()

    parser = argparse.ArgumentParser(
        description="Evaluate policies on all tasks with grader scores"
    )
    parser.add_argument("--episodes", type=int, default=10, help="Eval episodes per task")
    parser.add_argument("--train-episodes", type=int, default=500, help="Q-learning training episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="logs/evaluation_results.json")
    args = parser.parse_args()

    print("=" * 62)
    print("  Warehouse RL -- Full Evaluation".center(62))
    print("=" * 62)
    print()

    policies = ["random", "heuristic", "q_learning"]
    all_results: list[dict] = []

    # Pre-train Q-tables for each unique env config
    q_tables: dict[str, dict] = {}
    configs_seen: set[str] = set()

    for task in TASKS:
        config_key = json.dumps(task.env_config, sort_keys=True)
        if config_key not in configs_seen:
            configs_seen.add(config_key)
            print(f"  {icon('clock')} Training Q-Learning for config: {task.env_config.get('mode', 'normal')} mode ...")
            t0 = time.time()
            q_tables[config_key] = train_q_learning(
                task.env_config,
                episodes=args.train_episodes,
                seed=args.seed,
            )
            print(f"     Done in {time.time()-t0:.1f}s  |  States: {len(q_tables[config_key])}")

    # Evaluate all policies on all tasks
    print()
    for task in TASKS:
        print(f"─── Task: {task.name} ({task.difficulty}) ───")
        config_key = json.dumps(task.env_config, sort_keys=True)

        for policy in policies:
            rng = np.random.default_rng(args.seed)
            Q = q_tables.get(config_key)
            result = evaluate_policy(
                policy, task, args.episodes,
                Q_table=Q, rng=rng,
            )
            all_results.append(result)
            print(
                f"  {policy:<12} | score={result['avg_score']:.4f} "
                f"| reward={result['avg_reward']:>7.2f} "
                f"| ft={result['avg_fulfillment_time']:.2f} "
                f"| util={result['avg_utilization']*100:.1f}%"
            )
        print()

    # ── Summary table ──
    print()
    print("=" * 80)
    print("  EVALUATION COMPARISON TABLE".center(80))
    print("=" * 80)
    print(f"  {'Task':<30} {'Policy':<12} {'Score':>8} {'Reward':>8} {'Avg FT':>8} {'Util %':>8}")
    print("  " + "-" * 74)

    for r in all_results:
        print(
            f"  {r['task']:<30} {r['policy']:<12} "
            f"{r['avg_score']:>8.4f} {r['avg_reward']:>8.2f} "
            f"{r['avg_fulfillment_time']:>8.2f} {r['avg_utilization']*100:>7.1f}%"
        )

    print("=" * 80)

    # Best policy per task
    print()
    for task in TASKS:
        task_results = [r for r in all_results if r["task"] == task.name]
        best = max(task_results, key=lambda r: r["avg_score"])
        print(f"  {icon('trophy')} {task.name}: Best = {best['policy']} (score={best['avg_score']:.4f})")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  {icon('folder')} Results saved to {args.output}")
    print(f"\n{icon('check')} Evaluation complete.")


if __name__ == "__main__":
    main()
