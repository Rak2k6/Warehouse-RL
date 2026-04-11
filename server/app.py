"""
FastAPI application for the Warehouse Environment.
Exposes the WarehouseEnvironment over HTTP/WebSocket MCP endpoints.
"""

import uvicorn
import os
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from server.warehouse_environment import WarehouseEnvironment

# Create the app instance.
# env_name="warehouse_env" is used for the MCP tool prefix
app = create_app(
    WarehouseEnvironment, 
    CallToolAction, 
    CallToolObservation, 
    env_name="warehouse_env"
)

@app.get("/")
def root():
    return {"message": "Warehouse RL OpenEnv is running 🚀"}

@app.get("/full_state")
def get_full_state_endpoint():
    """Return the full internal state of the environment.
    
    Renamed to /full_state to avoid naming collision with OpenEnv's built-in /state.
    """
    if hasattr(app, "env_server") and hasattr(app.env_server, "gym_env"):
        return app.env_server.gym_env.get_full_state_dict()
    return {"error": "Environment server instance not found or gym_env missing"}
    
def main():
    """Start-up script for direct execution."""
    print("🚀 Starting Warehouse RL Server on http://0.0.0.0:7860")
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
