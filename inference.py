"""
Inference Script — Warehouse Order Fulfillment Environment
============================================================
Runs an LLM-based agent (via OpenAI-compatible API / Groq) across all
defined tasks and outputs structured, reproducible scores.

Environment variables:
  API_BASE_URL   — API base URL (default: https://api.groq.com/openai/v1)
  MODEL_NAME     — Model to use (default: llama-3.3-70b-versatile)
  GROQ_API_KEY   — Your Groq API key
  OPENAI_API_KEY — Or your OpenAI API key (fallback)

Usage:
  python inference.py
  python inference.py --fallback   # Use heuristic fallback (no API needed)

Logging format:
  [START] task=<name> seed=<seed> difficulty=<difficulty>
  [STEP]  step=<n> action=<a> reward=<r> queue_length=<q> decision=<reason>
  [END]   task=<name> score=<s> steps=<n> orders_completed=<c> avg_ft=<t>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Optional

import numpy as np

from warehouse_env.envs.warehouse_env import WarehouseOrderFulfillmentEnv
from warehouse_env.models import EpisodeResult
from warehouse_env.tasks import TASKS, Task, run_task_grader
from warehouse_env.utils import ensure_utf8_stdout, icon
from warehouse_env.llm_client import (
    get_llm_client,
    llm_chat,
    parse_action,
    random_valid_action,
)


# ── LLM Agent (OpenAI-compatible) ───────────────────────────────────

class LLMAgent:
    """Agent that uses an OpenAI-compatible API to select actions."""

    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name
        self.api_disabled = False  # Stateful fallback if API fails

    def select_action(
        self,
        env: WarehouseOrderFulfillmentEnv,
        obs: np.ndarray,
        step: int,
    ) -> int:
        """Ask the LLM to select an action given current state."""
        # ---- Efficiency & Fallback Checks ----
        valid_actions = [i for i in range(env.max_queue) if env.queue_proc_time[i] > 0]
        valid_actions.append(env.max_queue)  # no-op

        # 1. Permanent fallback if API already failed once this session
        if self.api_disabled:
            return self._heuristic_fallback(env)

        # 2. Skip LLM if no decision is needed (only 1 valid action)
        if len(valid_actions) <= 1:
            return self._heuristic_fallback(env)

        # Build a concise state description
        queue_items = []
        for i in range(env.max_queue):
            if env.queue_proc_time[i] > 0:
                prio = " [PRIORITY]" if env.queue_priority[i] > 0.5 else ""
                queue_items.append(
                    f"  slot {i}: proc_time={env.queue_proc_time[i]:.0f}, "
                    f"wait={env.queue_wait_time[i]:.0f}{prio}"
                )

        workers = []
        for i in range(env.num_workers):
            status = f"busy({env.worker_busy[i]:.0f}t)" if env.worker_busy[i] > 0 else "idle"
            util = env.worker_work_time[i] / max(env.current_step, 1)
            workers.append(f"  W{i}: {status}, utilization={util:.1%}")

        queue_str = "\n".join(queue_items) if queue_items else "  (empty)"
        worker_str = "\n".join(workers)

        prompt = f"""You are an AI agent controlling a warehouse order fulfillment system.

CURRENT STATE (step {step}/{env.max_steps}):
- Orders completed: {env.orders_completed}/{env.total_orders_generated}
- Queue ({len(queue_items)}/{env.max_queue} slots used):
{queue_str}
- Workers:
{worker_str}

VALID ACTIONS: {valid_actions}
- Actions 0-{env.max_queue-1}: Assign order at that queue slot to the least-busy worker
- Action {env.max_queue}: Wait/no-op (skip this turn)

GOAL: Minimize fulfillment time and maximize worker utilization.
STRATEGY: Prefer priority orders first, then shortest processing time.
If all workers are very busy and queue has short orders, it may be better to wait.

Respond with ONLY a single integer representing your chosen action. Nothing else."""

        # Use centralized LLM call with try/except built in
        text = llm_chat(
            self.client, self.model_name, prompt,
            temperature=0.0, max_tokens=10,
        )
        action = parse_action(text, valid_actions)
        if action is not None:
            return action

        # If LLM failed (text is None), enter permanent fallback mode for this run
        if text is None:
            print(f"    [INFO] Permanently switching to heuristic fallback for this task run.")
            self.api_disabled = True

        return self._heuristic_fallback(env)

    def _heuristic_fallback(self, env: WarehouseOrderFulfillmentEnv) -> int:
        """Fastest-first heuristic when LLM fails."""
        valid = np.where(env.queue_proc_time > 0)[0]
        if len(valid) == 0:
            return env.max_queue
        # Priority first, then shortest processing time
        priorities = env.queue_priority[valid]
        proc_times = env.queue_proc_time[valid]
        sort_keys = np.column_stack((-priorities, proc_times))
        best_idx = valid[np.lexsort((sort_keys[:, 1], sort_keys[:, 0]))[0]]
        return int(best_idx)


class HeuristicAgent:
    """Deterministic heuristic agent (SPT + priority). No API needed."""

    def select_action(
        self,
        env: WarehouseOrderFulfillmentEnv,
        obs: np.ndarray,
        step: int,
    ) -> int:
        valid = np.where(env.queue_proc_time > 0)[0]
        if len(valid) == 0:
            return env.max_queue
        priorities = env.queue_priority[valid]
        proc_times = env.queue_proc_time[valid]
        sort_keys = np.column_stack((-priorities, proc_times))
        best_idx = valid[np.lexsort((sort_keys[:, 1], sort_keys[:, 0]))[0]]
        return int(best_idx)


# ── Run a single task ────────────────────────────────────────────────

def run_task(
    task: Task,
    agent,
    verbose: bool = True,
) -> tuple[float, list[dict]]:
    """Run an agent on a task across multiple episodes and return the avg score."""
    scores: list[float] = []
    all_logs: list[dict] = []

    for ep in range(task.num_episodes):
        seed = task.seed + ep
        env = WarehouseOrderFulfillmentEnv(seed=seed, **task.env_config)
        obs, info = env.reset(seed=seed)

        if verbose and ep == 0:
            print(f"[START] task={task.name} seed={seed} difficulty={task.difficulty}")

        total_reward = 0.0
        done = False
        step = 0

        while not done:
            action = agent.select_action(env, obs, step)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1

            if verbose and ep == 0 and step <= 10:
                print(
                    f"[STEP] step={step} action={action} "
                    f"reward={reward:.4f} queue_length={info['queue_length']} "
                    f"decision={info.get('decision_reason', '')}"
                )

        # Build episode result for grading
        summary = info.get("episode_summary", {})
        result = EpisodeResult(
            orders_completed=summary.get("orders_completed", env.orders_completed),
            orders_generated=summary.get("orders_generated", env.total_orders_generated),
            priority_orders_completed=summary.get("priority_completed", env.priority_orders_completed),
            avg_fulfillment_time=summary.get("avg_fulfillment_time", 0.0),
            worker_utilization=summary.get("worker_utilization", 0.0),
            total_reward=summary.get("total_reward", total_reward),
            steps=step,
            mode=task.env_config.get("mode", "normal"),
            terminated=terminated,
            truncated=truncated,
            queue_overflow_count=getattr(env, "_queue_overflow_count", 0),
        )

        score = run_task_grader(task, result)
        scores.append(score)

        all_logs.append({
            "episode": ep,
            "seed": seed,
            "score": score,
            "orders_completed": result.orders_completed,
            "orders_generated": result.orders_generated,
            "avg_fulfillment_time": result.avg_fulfillment_time,
            "worker_utilization": result.worker_utilization,
            "total_reward": round(total_reward, 2),
            "steps": step,
        })

        env.close()

    avg_score = float(np.mean(scores))

    if verbose:
        if all_logs:
            best = max(all_logs, key=lambda x: x["score"])
            print(
                f"[END] task={task.name} score={avg_score:.4f} "
                f"steps={best['steps']} orders_completed={best['orders_completed']} "
                f"avg_ft={best['avg_fulfillment_time']:.2f}"
            )
        else:
            print(f"[END] task={task.name} score=0.0000 steps=0 orders_completed=0 avg_ft=0.00")

    return avg_score, all_logs


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ensure_utf8_stdout()

    parser = argparse.ArgumentParser(
        description="Run inference on the Warehouse RL environment"
    )
    parser.add_argument(
        "--fallback", action="store_true",
        help="Use heuristic agent instead of LLM API"
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="Run a specific task (by name). Default: all tasks."
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print step-by-step logs"
    )
    parser.add_argument(
        "--output", type=str, default="logs/inference_results.json",
        help="Path to save results JSON"
    )
    args = parser.parse_args()

    # ── Determine agent via centralized LLM client ──
    client, model_name, api_key = get_llm_client()

    if args.fallback or client is None:
        if client is None and not args.fallback:
            print("[INFO] No API key set (GROQ_API_KEY / OPENAI_API_KEY). "
                  "Falling back to heuristic agent.")
        agent = HeuristicAgent()
        agent_name = "heuristic"
    else:
        print(f"[INFO] Using LLM API: model={model_name}")
        agent = LLMAgent(client=client, model_name=model_name)
        agent_name = f"llm/{model_name}"

    # ── Select tasks ──
    if args.task:
        tasks = [t for t in TASKS if t.name == args.task]
        if not tasks:
            print(f"[ERROR] Task '{args.task}' not found. Available: {[t.name for t in TASKS]}")
            sys.exit(1)
    else:
        tasks = TASKS

    if not tasks:
        print("[INFO] No tasks to run.")
        return

    # ── Run ──
    print()
    print("=" * 60)
    print(f"  Warehouse RL {icon('dash')} Inference ({agent_name})")
    print("=" * 60)
    print()

    all_results: dict[str, Any] = {
        "agent": agent_name,
        "tasks": {},
    }

    for task in tasks:
        print(f"\n{icon('dash') * 50}")
        print(f"  Task: {task.name} ({task.difficulty})")
        print(f"  {task.description}")
        print(f"{icon('dash') * 50}")

        t0 = time.time()
        avg_score, logs = run_task(task, agent, verbose=args.verbose)
        elapsed = time.time() - t0

        all_results["tasks"][task.name] = {
            "difficulty": task.difficulty,
            "avg_score": round(avg_score, 4),
            "episodes": len(logs),
            "elapsed_s": round(elapsed, 1),
            "logs": logs,
        }

        print(f"  {icon('arrow')} Avg Score: {avg_score:.4f}  ({elapsed:.1f}s)")

    # ── Summary table ──
    print()
    print("=" * 60)
    print("  INFERENCE RESULTS".center(60))
    print("=" * 60)
    print(f"  {'Task':<35} {'Difficulty':<10} {'Score':>8}")
    print("  " + "-" * 54)
    for task in tasks:
        tr = all_results["tasks"][task.name]
        print(f"  {task.name:<35} {tr['difficulty']:<10} {tr['avg_score']:>8.4f}")
    print("=" * 60)

    # ── Save results ──
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  {icon('folder')} Results saved to {args.output}")


if __name__ == "__main__":
    main()
