"""
Warehouse Order Fulfillment Environment
========================================
An OpenAI Gym-compatible RL environment where an agent optimizes
the assignment of incoming orders to warehouse workers to minimize
total fulfillment time and maximize worker utilization.

Sequential Decision Making:
  - At each step the agent picks which pending order to assign next.
  - The chosen order is routed to the least-busy available worker.

Trade-offs:
  - Assigning a long order to an already-loaded worker increases
    latency. Picking a short order first may clear the queue faster
    but might leave heavy orders for later.

Dynamic Environment:
  - New orders arrive stochastically between steps,
    keeping the queue unpredictable.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, Optional


class WarehouseOrderFulfillmentEnv(gym.Env):
    """
    Warehouse Order Fulfillment Environment.

    Observation (flat vector of length 1 + 2*max_workers + 2*max_queue):
        [0]                              : number of pending orders (normalized)
        [1 .. max_workers]               : worker busy-time remaining (normalized)
        [max_workers+1 .. 2*max_workers] : worker utilization so far (0-1)
        [2*max_workers+1 .. 2*mw+mq]    : processing time of each queue slot (0 if empty)
        [2*mw+mq+1 .. 2*mw+2*mq]       : wait time of each queue slot (normalized)

    Action (Discrete):
        index into the pending-order queue → assign that order to the
        worker with the least remaining busy time.

    Reward:
        +1.0  per order completed (scaled by speed bonus)
        -0.1  per time-step with idle workers while orders wait
        -0.05 * (queue_length / max_queue)  congestion penalty
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        num_workers: int = 4,
        max_queue: int = 20,
        max_orders: int = 50,
        max_steps: int = 200,
        order_time_range: tuple[int, int] = (1, 8),
        new_order_prob: float = 0.4,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        # ---- Config ----
        self.num_workers = num_workers
        self.max_queue = max_queue
        self.max_orders = max_orders
        self.max_steps = max_steps
        self.order_time_range = order_time_range
        self.new_order_prob = new_order_prob
        self.render_mode = render_mode

        # ---- Spaces ----
        obs_size = 1 + 2 * num_workers + 2 * max_queue
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        # Action = index of order in the queue to assign next
        # 0 .. max_queue-1  → pick that queue slot
        # last action        → "no-op / wait"
        self.action_space = spaces.Discrete(max_queue + 1)

        # ---- RNG ----
        self._np_random: np.random.Generator | None = None
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        # ---- Metrics accumulators ----
        self._metrics: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        elif self._np_random is None:
            self._np_random = np.random.default_rng()

        # Workers: remaining processing time
        self.worker_busy = np.zeros(self.num_workers, dtype=np.float32)
        # Workers: total time spent working (for utilization)
        self.worker_work_time = np.zeros(self.num_workers, dtype=np.float32)

        # Order queue: each entry = processing time required (0 = empty slot)
        self.queue_proc_time = np.zeros(self.max_queue, dtype=np.float32)
        self.queue_wait_time = np.zeros(self.max_queue, dtype=np.float32)

        # Seed queue with initial orders
        q_limit = max(4, self.max_queue // 2 + 1)
        n_initial = self._np_random.integers(min(3, q_limit - 1), q_limit)
        for i in range(n_initial):
            self.queue_proc_time[i] = float(
                self._np_random.integers(
                    self.order_time_range[0], self.order_time_range[1] + 1
                )
            )

        # Counters
        self.current_step = 0
        self.total_orders_generated = int(n_initial)
        self.orders_completed = 0
        self.total_fulfillment_time = 0.0
        self.total_wait_time = 0.0

        # Metrics
        self._metrics = {
            "orders_completed": 0,
            "avg_fulfillment_time": 0.0,
            "worker_utilization": 0.0,
            "total_reward": 0.0,
        }

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"Invalid action {action}"

        reward = 0.0
        self.current_step += 1

        # ------ 1. Process action ------
        is_noop = action == self.max_queue
        valid_assign = False

        if not is_noop and self.queue_proc_time[action] > 0:
            # Find least-busy worker
            worker_id = int(np.argmin(self.worker_busy))
            proc_time = self.queue_proc_time[action]
            wait = self.queue_wait_time[action]

            # Assign order
            self.worker_busy[worker_id] += proc_time
            self.worker_work_time[worker_id] += proc_time

            # Record fulfillment = wait + processing
            fulfillment = wait + proc_time
            self.total_fulfillment_time += fulfillment
            self.total_wait_time += wait
            self.orders_completed += 1

            # Clear queue slot
            self.queue_proc_time[action] = 0.0
            self.queue_wait_time[action] = 0.0

            # Compact queue (shift remaining orders left)
            self._compact_queue()

            valid_assign = True

            # Reward: completion bonus with speed bonus
            speed_bonus = max(0.0, 1.0 - (fulfillment / (self.order_time_range[1] * 3)))
            reward += 1.0 + 0.5 * speed_bonus

        elif not is_noop:
            # Tried to assign an empty slot → small penalty
            reward -= 0.3

        # ------ 2. Advance time by 1 tick ------
        self.worker_busy = np.maximum(self.worker_busy - 1.0, 0.0)

        # Increment wait time for pending orders
        pending_mask = self.queue_proc_time > 0
        self.queue_wait_time[pending_mask] += 1.0

        # ------ 3. Stochastic new order arrival ------
        if self.total_orders_generated < self.max_orders:
            if self._np_random.random() < self.new_order_prob:
                slot = self._first_empty_slot()
                if slot is not None:
                    pt = float(
                        self._np_random.integers(
                            self.order_time_range[0],
                            self.order_time_range[1] + 1,
                        )
                    )
                    self.queue_proc_time[slot] = pt
                    self.queue_wait_time[slot] = 0.0
                    self.total_orders_generated += 1

        # ------ 4. Penalties ------
        queue_len = int(np.sum(self.queue_proc_time > 0))
        idle_workers = int(np.sum(self.worker_busy == 0))

        # Idle-worker penalty only when there are pending orders
        if queue_len > 0 and idle_workers > 0:
            reward -= 0.1 * idle_workers

        # Queue congestion penalty
        reward -= 0.05 * (queue_len / max(self.max_queue, 1))

        # ------ 5. Termination ------
        all_done = (
            self.total_orders_generated >= self.max_orders
            and queue_len == 0
            and np.all(self.worker_busy == 0)
        )
        truncated = self.current_step >= self.max_steps
        terminated = all_done

        # Bonus for finishing all orders
        if terminated and not truncated:
            reward += 5.0

        # ------ 6. Metrics ------
        self._metrics["orders_completed"] = self.orders_completed
        self._metrics["avg_fulfillment_time"] = (
            self.total_fulfillment_time / max(self.orders_completed, 1)
        )
        self._metrics["worker_utilization"] = float(
            np.mean(self.worker_work_time) / max(self.current_step, 1)
        )
        self._metrics["total_reward"] = self._metrics.get("total_reward", 0.0) + reward

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        output = self._render_text()
        if self.render_mode == "ansi":
            return output
        elif self.render_mode == "human":
            print(output)

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        max_time = float(self.order_time_range[1])
        norm_steps = float(max(self.max_steps, 1))

        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        queue_len = int(np.sum(self.queue_proc_time > 0))

        # Pending count (normalized)
        obs[0] = queue_len / max(self.max_queue, 1)

        # Worker busy time (normalized)
        offset = 1
        obs[offset : offset + self.num_workers] = np.clip(
            self.worker_busy / (max_time * 2), 0.0, 1.0
        )

        # Worker utilization
        offset += self.num_workers
        obs[offset : offset + self.num_workers] = np.clip(
            self.worker_work_time / max(self.current_step, 1), 0.0, 1.0
        )

        # Queue processing times (normalized)
        offset += self.num_workers
        obs[offset : offset + self.max_queue] = np.clip(
            self.queue_proc_time / max_time, 0.0, 1.0
        )

        # Queue wait times (normalized)
        offset += self.max_queue
        obs[offset : offset + self.max_queue] = np.clip(
            self.queue_wait_time / norm_steps, 0.0, 1.0
        )

        return obs

    def _get_info(self) -> dict:
        return {
            "step": self.current_step,
            "orders_completed": self.orders_completed,
            "orders_generated": self.total_orders_generated,
            "queue_length": int(np.sum(self.queue_proc_time > 0)),
            "avg_fulfillment_time": float(
                self.total_fulfillment_time / max(self.orders_completed, 1)
            ),
            "worker_utilization": float(
                np.mean(self.worker_work_time) / max(self.current_step, 1)
            ),
            "worker_busy_times": self.worker_busy.tolist(),
        }

    def _first_empty_slot(self) -> Optional[int]:
        for i in range(self.max_queue):
            if self.queue_proc_time[i] == 0:
                return i
        return None

    def _compact_queue(self):
        """Shift non-zero entries to the front of the queue."""
        mask = self.queue_proc_time > 0
        proc = self.queue_proc_time[mask]
        wait = self.queue_wait_time[mask]
        self.queue_proc_time[:] = 0.0
        self.queue_wait_time[:] = 0.0
        self.queue_proc_time[: len(proc)] = proc
        self.queue_wait_time[: len(wait)] = wait

    def _render_text(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append(f"  WAREHOUSE ORDER FULFILLMENT  *  Step {self.current_step}/{self.max_steps}")
        lines.append("=" * 60)

        # Workers
        lines.append("\n  Workers:")
        for i in range(self.num_workers):
            busy = self.worker_busy[i]
            util = self.worker_work_time[i] / max(self.current_step, 1)
            status = f"BUSY ({busy:.0f} ticks left)" if busy > 0 else "IDLE"
            bar = "#" * int(util * 20) + "-" * (20 - int(util * 20))
            lines.append(
                f"    Worker {i}: [{bar}] {util*100:5.1f}%  {status}"
            )

        # Queue
        queue_len = int(np.sum(self.queue_proc_time > 0))
        lines.append(f"\n  Order Queue ({queue_len}/{self.max_queue}):")
        for i in range(min(queue_len, 10)):
            pt = self.queue_proc_time[i]
            wt = self.queue_wait_time[i]
            lines.append(
                f"    [{i:2d}]  proc_time={pt:.0f}  wait={wt:.0f}"
            )
        if queue_len > 10:
            lines.append(f"    ... and {queue_len - 10} more")

        # Metrics
        avg_ft = self.total_fulfillment_time / max(self.orders_completed, 1)
        lines.append(f"\n  Completed: {self.orders_completed}/{self.total_orders_generated}")
        lines.append(f"  Avg Fulfillment Time: {avg_ft:.2f}")
        lines.append(f"  Total Reward: {self._metrics.get('total_reward', 0):.2f}")
        lines.append("=" * 60)
        return "\n".join(lines)
