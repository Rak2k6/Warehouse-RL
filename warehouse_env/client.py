"""
Warehouse Environment Client
=============================
Standard client library for connecting to the Warehouse RL server via MCP.
Inherits all functionality from MCPToolClient.
"""

from openenv.core.mcp_client import MCPToolClient

class WarehouseEnv(MCPToolClient):
    """
    Client for interacting with the Warehouse Order Fulfillment Environment.

    Supports discovering and calling MCP tools:
    - list_tools(): Discover environment tools
    - call_tool("assign_order", order_id=0): Assign index 0 from queue
    - call_tool("wait_step"): Advance time without any action
    
    Example usage:
    >>> with WarehouseEnv(base_url="http://localhost:8000") as env:
    >>>     env.reset()
    >>>     tools = env.list_tools()
    >>>     # call assign_order tool
    >>>     res = env.call_tool("assign_order", order_id=0)
    >>>     print(res)
    """
    pass # Managed by MCPToolClient
