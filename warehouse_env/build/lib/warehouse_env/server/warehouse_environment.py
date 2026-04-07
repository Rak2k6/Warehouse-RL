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


class WarehouseEnvironment(MCPEnvironment):
    """
    A warehouse order fulfillment environment served via MCP.
    
    Exposes tools:
    - assign_order(order_id): Pick a specific slot from the queue.
    - wait_step(): Advance time without assigning an order.
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
                Dictionary with processing results and updated state metrics.
            """
            # Perform action and get Gym observation/reward
            # We map order_id to the Discrete action space
            obs, reward, terminated, truncated, info = self.gym_env.step(order_id)
            return {
                "reward": float(reward),
                "done": bool(terminated or truncated),
                "info": info,
                "status": "Order assigned successfully" if reward > 0 else "Invalid assignment / Empty slot"
            }

        @mcp.tool
        def wait_step() -> dict:
            """
            Advance time by 1 tick without assigning any new orders.
            Useful when workers are busy or queue is empty.
            
            Returns:
                Dictionary with environment response and current metrics.
            """
            # Action = max_queue is the "no-op" in our Gym environment
            obs, reward, terminated, truncated, info = self.gym_env.step(self.gym_env.max_queue)
            return {
                "reward": float(reward),
                "done": bool(terminated or truncated),
                "info": info,
                "status": "Time advanced by 1 tick"
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
                "description": "Warehouse environment reset. Queue and workers initialized."
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
            metadata={"error": "Direct actions not supported. Use CallToolAction for MCP tools."},
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Expose step with state tracking."""
        self._state.step_count += 1
        return super().step(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        """Return current episode and step info."""
        return self._state
