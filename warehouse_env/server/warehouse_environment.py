"""
Warehouse RL Environment Server
================================
MCP-compatible server for the Warehouse Order Fulfillment Environment.
Exposes Gym environment actions as Model Context Protocol (MCP) tools.
"""

import numpy as np
from typing import Any, Optional
from uuid import uuid4
from fastmcp import FastMCP

try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    # Standalone for direct testing
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State

from ..envs.warehouse_env import WarehouseOrderFulfillmentEnv
from ..models import WarehouseState


class WarehouseEnvironment(MCPEnvironment):
    """
    A warehouse order fulfillment environment served via MCP.

    Exposes tools:
    - assign_order(order_id): Pick a specific slot from the queue.
    - wait_step(): Advance time without assigning an order.

    Supports scenario modes: normal, rush, low.
    """

    def __init__(self, **env_kwargs):
        """Initialize with internal Gym environment and FastMCP tools."""
        self.gym_env = WarehouseOrderFulfillmentEnv(**env_kwargs)

        mcp = FastMCP("warehouse_env")

        @mcp.tool
        def assign_order(order_id: int) -> dict:
            """
            Assign a specific order from the queue to the next available worker.

            Args:
                order_id: The index (0 to max_queue-1) of the order in the current queue.

            Returns:
                Dictionary with processing results, reward breakdown, and updated state.
            """
            obs, reward, terminated, truncated, info = self.gym_env.step(order_id)
            self._state.step_count += 1
            return {
                "reward": float(reward),
                "observation": obs.tolist(),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": info,
                "decision_reason": info.get("decision_reason", ""),
                "reward_breakdown": info.get("reward_breakdown", {}),
            }

        @mcp.tool
        def wait_step() -> dict:
            """
            Advance time by 1 tick without assigning any new orders.
            Useful when workers are busy or queue is empty.

            Returns:
                Dictionary with environment response, reward breakdown, and metrics.
            """
            obs, reward, terminated, truncated, info = self.gym_env.step(
                self.gym_env.max_queue
            )
            self._state.step_count += 1
            return {
                "reward": float(reward),
                "observation": obs.tolist(),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": info,
                "decision_reason": info.get("decision_reason", ""),
                "reward_breakdown": info.get("reward_breakdown", {}),
            }

        super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Reset the internal Gym env and return as OpenEnv Observation."""
        obs_vec, info = self.gym_env.reset(seed=seed)
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status": "ready",
                "info": info,
                "mode": self.gym_env.mode,
                "description": (
                    "Warehouse environment reset. "
                    f"Mode: {self.gym_env.mode}. "
                    "Queue and workers initialized."
                ),
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Handle legacy actions if needed (routes to MCP core standard).
        """
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": "Direct actions not supported. Use CallToolAction for MCP tools."
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Expose step with state tracking."""
        return super().step(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        """Return current episode and step info."""
        return self._state

    def get_full_state(self) -> WarehouseState:
        """Return full internal state as a Pydantic model.

        This provides the complete environment snapshot for debugging,
        grading, and the OpenEnv state() specification.
        """
        internal = self.gym_env.get_state()
        return WarehouseState(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            mode=internal["mode"],
            worker_busy=internal["worker_busy"],
            worker_work_time=internal["worker_work_time"],
            queue_proc_time=internal["queue_proc_time"],
            queue_wait_time=internal["queue_wait_time"],
            queue_priority=internal["queue_priority"],
            total_orders_generated=internal["total_orders_generated"],
            orders_completed=internal["orders_completed"],
            priority_orders_completed=internal["priority_orders_completed"],
            total_fulfillment_time=internal["total_fulfillment_time"],
            total_wait_time=internal["total_wait_time"],
            cumulative_reward=internal["cumulative_reward"],
            num_workers=internal["num_workers"],
            max_queue=internal["max_queue"],
            max_orders=internal["max_orders"],
            max_steps=internal["max_steps"],
        )
