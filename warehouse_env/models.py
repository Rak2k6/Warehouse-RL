"""
Pydantic Models for the Warehouse Order Fulfillment Environment
================================================================
Typed data models conforming to the OpenEnv specification.
Used for type-safe communication between client, server, and graders.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── Action ───────────────────────────────────────────────────────────

class WarehouseAction(BaseModel):
    """Action to take in the warehouse environment.

    order_index: 0..max_queue-1 to assign a specific order,
                 or max_queue for no-op (wait).
    """
    order_index: int = Field(
        ...,
        ge=0,
        description="Index of the order in the queue to assign (0..max_queue-1), "
                    "or max_queue for no-op/wait.",
    )


# ── Reward Breakdown ─────────────────────────────────────────────────

class WarehouseReward(BaseModel):
    """Detailed breakdown of the reward signal."""
    completion_reward: float = Field(0.0, description="Reward for completing an order")
    speed_bonus: float = Field(0.0, description="Bonus for fast fulfillment")
    priority_bonus: float = Field(0.0, description="Bonus for priority order completion")
    idle_penalty: float = Field(0.0, description="Penalty for idle workers when queue has orders")
    congestion_penalty: float = Field(0.0, description="Penalty for queue congestion")
    wait_time_penalty: float = Field(0.0, description="Penalty for long-waiting orders")
    finish_bonus: float = Field(0.0, description="Bonus for completing all orders")
    total: float = Field(0.0, description="Sum of all reward components")


# ── Observation ──────────────────────────────────────────────────────

class WarehouseObservation(BaseModel):
    """Observation returned by the environment after each step."""
    pending_count: float = Field(
        ..., ge=0.0, le=1.0,
        description="Normalized count of pending orders in queue",
    )
    worker_busy: list[float] = Field(
        ..., description="Normalized remaining busy time per worker",
    )
    worker_utilization: list[float] = Field(
        ..., description="Cumulative utilization ratio per worker (0-1)",
    )
    queue_proc_times: list[float] = Field(
        ..., description="Normalized processing time per queue slot",
    )
    queue_wait_times: list[float] = Field(
        ..., description="Normalized wait time per queue slot",
    )
    queue_priorities: list[float] = Field(
        ..., description="Priority flag per queue slot (0 or 1)",
    )
    raw_vector: Optional[list[float]] = Field(
        None, description="Raw flat observation vector for RL agents",
    )

    @classmethod
    def from_obs_vector(
        cls,
        obs_vec: list[float],
        num_workers: int,
        max_queue: int,
    ) -> "WarehouseObservation":
        """Construct from the flat observation vector."""
        pending_count = obs_vec[0]

        w_start = 1
        worker_busy = obs_vec[w_start : w_start + num_workers]

        u_start = w_start + num_workers
        worker_utilization = obs_vec[u_start : u_start + num_workers]

        q_start = u_start + num_workers
        queue_proc_times = obs_vec[q_start : q_start + max_queue]

        wt_start = q_start + max_queue
        queue_wait_times = obs_vec[wt_start : wt_start + max_queue]

        p_start = wt_start + max_queue
        queue_priorities = obs_vec[p_start : p_start + max_queue]

        return cls(
            pending_count=pending_count,
            worker_busy=worker_busy,
            worker_utilization=worker_utilization,
            queue_proc_times=queue_proc_times,
            queue_wait_times=queue_wait_times,
            queue_priorities=queue_priorities,
            raw_vector=obs_vec,
        )


# ── State (full internal snapshot) ───────────────────────────────────

class WarehouseState(BaseModel):
    """Full internal state of the warehouse environment.

    Returned by the state() method for debugging and grading.
    """
    # Episode metadata
    episode_id: str = Field("", description="Unique episode identifier")
    step_count: int = Field(0, description="Current step number")
    mode: str = Field("normal", description="Scenario mode (normal/rush/low)")

    # Worker state
    worker_busy: list[float] = Field(
        ..., description="Remaining busy time per worker",
    )
    worker_work_time: list[float] = Field(
        ..., description="Total work time accumulated per worker",
    )

    # Queue state
    queue_proc_time: list[float] = Field(
        ..., description="Processing time per queue slot (0 = empty)",
    )
    queue_wait_time: list[float] = Field(
        ..., description="Accumulated wait time per queue slot",
    )
    queue_priority: list[float] = Field(
        ..., description="Priority flag per queue slot",
    )

    # Counters
    total_orders_generated: int = Field(0, description="Total orders generated so far")
    orders_completed: int = Field(0, description="Total orders completed")
    priority_orders_completed: int = Field(0, description="Priority orders completed")
    total_fulfillment_time: float = Field(0.0, description="Sum of all fulfillment times")
    total_wait_time: float = Field(0.0, description="Sum of all wait times")
    cumulative_reward: float = Field(0.0, description="Cumulative reward this episode")

    # Configuration
    num_workers: int = Field(4, description="Number of workers")
    max_queue: int = Field(20, description="Maximum queue capacity")
    max_orders: int = Field(50, description="Maximum orders per episode")
    max_steps: int = Field(200, description="Maximum steps per episode")


# ── Episode Result (for graders) ─────────────────────────────────────

class EpisodeResult(BaseModel):
    """Summary of a completed episode, passed to task graders."""
    orders_completed: int = 0
    orders_generated: int = 0
    priority_orders_completed: int = 0
    avg_fulfillment_time: float = 0.0
    worker_utilization: float = 0.0
    total_reward: float = 0.0
    steps: int = 0
    mode: str = "normal"
    terminated: bool = False
    truncated: bool = False
    queue_overflow_count: int = Field(
        0, description="Number of times an order was dropped due to full queue",
    )
