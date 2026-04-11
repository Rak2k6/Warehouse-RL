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

# Global manual instance for direct HTTP validation routing
env_instance = None

# Strip out OpenEnv's restrictive paths so our custom routes take priority
app.router.routes = [route for route in app.router.routes if getattr(route, "path", "") not in ["/reset", "/step", "/state"]]

@app.get("/")
def root():
    return {"message": "Warehouse RL OpenEnv is running"}

import json
import numpy as np
from fastapi import Request
from fastapi.responses import JSONResponse
import traceback

def deep_serialize(obj):
    """Recursively convert all custom and numpy types to base Python primitives."""
    if isinstance(obj, dict):
        return {k: deep_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [deep_serialize(v) for v in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    elif type(obj).__name__.startswith("float"):
        return float(obj)
    elif type(obj).__name__.startswith("int") or type(obj).__name__.startswith("uint"):
        return int(obj)
    elif type(obj).__name__ == "ndarray" or hasattr(obj, "tolist"):
        return obj.tolist()
    else:
        return obj

@app.middleware("http")
async def intercept_openenv_rest_routes(request: Request, call_next):
    if request.url.path == "/reset" and request.method == "POST":
        try:
            raw_body = await request.body()
            body = json.loads(raw_body) if raw_body else {}
        except Exception:
            body = {}
        
        try:
            global env_instance

            if env_instance is None:
                env_instance = WarehouseEnvironment()

            obs = env_instance.reset(
                episode_id=body.get("episode_id") if isinstance(body, dict) else None
            )
            data = obs.model_dump() if hasattr(obs, "model_dump") else obs
            clean_data = deep_serialize(data)
            return JSONResponse(status_code=200, content=clean_data)
        except Exception as e:
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e)})

    elif request.url.path == "/step" and request.method == "POST":
        try:
            if env_instance is None:
                return JSONResponse(status_code=400, content={"error": "Environment not initialized. Call /reset first."})
            raw_body = await request.body()
            body = json.loads(raw_body) if raw_body else {}
        except Exception:
            body = {}
            
        try:
            raw_action = env_instance.gym_env.max_queue
            if isinstance(body, dict):
                raw_action = body.get("action", env_instance.gym_env.max_queue)
                
            if isinstance(raw_action, dict):
                action = int(raw_action.get("order_id", env_instance.gym_env.max_queue))
            else:
                action = int(raw_action)

            obs = env_instance._step_impl(action=action)
            data = obs.model_dump() if hasattr(obs, "model_dump") else obs
            clean_data = deep_serialize(data)
            return JSONResponse(status_code=200, content=clean_data)
        except Exception as e:
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e)})
            
    elif request.url.path == "/state" and request.method == "GET":
        try:
            if env_instance is None:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Environment not initialized. Call /reset first."}
                )

            data = env_instance.gym_env.get_full_state_dict()
            clean_data = deep_serialize(data)
            return JSONResponse(status_code=200, content=clean_data)
        except Exception as e:
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e)})
            
    # For all other routes, pass through normally
    return await call_next(request)
    
def main():
    """Start-up script for direct execution."""
    print("Starting Warehouse RL Server on http://0.0.0.0:7860")
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
