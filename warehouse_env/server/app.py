"""
FastAPI application for the Warehouse Environment.
Exposes the WarehouseEnvironment over HTTP/WebSocket MCP endpoints.
"""

import uvicorn
import os
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from .warehouse_environment import WarehouseEnvironment

# Create the app instance.
# env_name="warehouse_env" is used for the MCP tool prefix
app = create_app(
    WarehouseEnvironment, 
    CallToolAction, 
    CallToolObservation, 
    env_name="warehouse_env"
)

def main():
    """Start-up script for direct execution."""
    print("🚀 Starting Warehouse RL Server on http://0.0.0.0:7860")
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
if __name__ == "__main__":
    main()
