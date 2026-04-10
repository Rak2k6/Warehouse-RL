import os
import sys
import json
import asyncio
import traceback
from typing import Any, List, Optional
from contextlib import asynccontextmanager

# Delay heavy imports so startup never crashes
np = None
try:
    import numpy as np
except Exception as e:
    print(f"[WARN] numpy not available: {e}. Coercion will use fallback lists.")

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ── Environment Variables ───────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN")

# Dummy definitions
def default_llm_chat(*args, **kwargs): return None
def default_parse_action(*args, **kwargs): return None

client = None
model_name_deployed = MODEL_NAME
api_disabled = False
llm_chat = default_llm_chat
parse_action = default_parse_action

def safe_import_llm():
    try:
        from warehouse_env.llm_client import get_llm_client as glc, llm_chat as lc, parse_action as pa
        return glc, lc, pa
    except Exception as e:
        print(f"[WARN] LLM module unavailable → fallback mode. Reason: {e}")
        def dummy_get(): return None, "fallback", ""
        return dummy_get, default_llm_chat, default_parse_action

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, model_name_deployed, api_disabled, llm_chat, parse_action

    # 🚨 FORCE SAFE MODE IN HF
    if os.getenv("SPACE_ID"):
        print("[INFO] HF detected → disabling LLM for stability")
        client = None
        api_disabled = True
        yield
        return

    try:
        get_llm, chat_fn, parse_fn = safe_import_llm()

        if get_llm is None:
            api_disabled = True
        else:
            c, m, _ = get_llm()
            client = c
            model_name_deployed = m
            llm_chat = chat_fn
            parse_action = parse_fn

            if client is None:
                api_disabled = True

    except Exception as e:
        print(f"[ERROR] Failed to init LLM client: {e}")
        api_disabled = True
    
    yield

# FastAPI Application
app = FastAPI(title="Warehouse RL Inference Service", version="1.0.0", lifespan=lifespan)

def extract_state(obs: list[float], W: int = 4, Q: int = 20):
    """Extracts state arrays from the flat observation."""
    try:
        offset = 1
        worker_busy = obs[offset : offset + W]
        offset += W
        worker_util = obs[offset : offset + W]
        offset += W
        queue_proc = obs[offset : offset + Q]
        offset += Q
        queue_wait = obs[offset : offset + Q]
        offset += Q
        queue_prio = obs[offset : offset + Q]
        return worker_busy, worker_util, queue_proc, queue_wait, queue_prio
    except Exception:
        return [0]*W, [0]*W, [0]*Q, [0]*Q, [0]*Q

def get_heuristic_action(obs: list[float], W: int = 4, Q: int = 20) -> int:
    """Fastest-first + Priority heuristic."""
    try:
        _, _, queue_proc, _, queue_prio = extract_state(obs, W, Q)
        valid = [i for i, p in enumerate(queue_proc) if p > 0]
        if not valid:
            return Q  # explicit action safe-fallback
        return min(valid, key=lambda x: (-queue_prio[x], queue_proc[x]))
    except Exception as e:
        print(f"[ERROR] Heuristic fallback failed: {e}")
        return Q

def get_llm_action_sync(obs: list[float], W: int = 4, Q: int = 20) -> int:
    """Synchronous LLM fetching logic."""
    global api_disabled
    if api_disabled or client is None:
        return get_heuristic_action(obs, W, Q)
        
    try:
        worker_busy, worker_util, queue_proc, queue_wait, queue_prio = extract_state(obs, W, Q)
        
        valid_actions = [i for i, p in enumerate(queue_proc) if p > 0]
        valid_actions.append(Q) # Include fallback as valid
        
        if len(valid_actions) <= 1:
            return get_heuristic_action(obs, W, Q)
            
        queue_items = []
        for i in range(Q):
            if queue_proc[i] > 0:
                prio = " [PRIORITY]" if queue_prio[i] > 0.5 else ""
                queue_items.append(
                    f"  slot {i}: proc_time={float(queue_proc[i])*8:.1f}, wait={float(queue_wait[i])*200:.0f}{prio}"
                )
                
        workers = []
        for i in range(W):
            status = f"busy({float(worker_busy[i])*16:.1f}t)" if float(worker_busy[i]) > 0 else "idle"
            workers.append(f"  W{i}: {status}, util={float(worker_util[i]):.1%}")
            
        queue_str = "\n".join(queue_items) if queue_items else "  (empty)"
        worker_str = "\n".join(workers)
        
        prompt = f"""You are an AI controlling a warehouse.

CURRENT STATE:
- Queue ({len(queue_items)}/{Q} slots used):
{queue_str}
- Workers:
{worker_str}

VALID ACTIONS: {valid_actions}
- Actions 0-{Q-1}: Assign order at that queue slot to least-busy worker
- Action {Q}: Wait/no-op

GOAL: Minimize fulfillment time & maximize worker utilization.
STRATEGY: Prefer priority orders first, then shortest processing time.

Respond with ONLY a single integer representing your chosen action."""

        # Attempt LLM call
        text = llm_chat(client, model_name_deployed, prompt, temperature=0.0, max_tokens=10)
        action = parse_action(text, valid_actions)
        if action is not None:
            return action
            
        print("[WARN] Invalid LLM response, permanently falling back to heuristic.")
        api_disabled = True
        return get_heuristic_action(obs, W, Q)
        
    except Exception as e:
        print(f"[ERROR] LLM prediction sync block failed: {e}")
        api_disabled = True
        return get_heuristic_action(obs, W, Q)

@app.get("/")
def health_check():
    """Returns HF Spaces compatible health check"""
    return {"status": "ok", "mode": "llm" if not api_disabled else "heuristic"}

@app.post("/predict")
async def predict_action(request: Request):
    """Predict standard OpenEnv action given an observation (Max call time ≤ 1.5s)."""
    import time
    start_time = time.time()
    
    global api_disabled
    fallback_action = 20  # Safe fallback action

    try:
        # ---- Safe Request Parsing Core Requirement ----
        try:
            raw_body = await request.body()
        except Exception:
            raw_body = b""
            
        try:
            body_text = raw_body.decode('utf-8')
            data = json.loads(body_text) if body_text else {}
        except Exception:
            data = {}
            
        # Extract observation robustly
        obs = None
        if isinstance(data, dict):
            obs = data.get("observation", data.get("obs"))
        elif isinstance(data, list):
            obs = data
            
        # Coerce/Validate format safely
        if np is not None and isinstance(obs, np.ndarray):
            try:
                obs = obs.tolist()
            except Exception:
                pass
                
        if not isinstance(obs, list):
            return JSONResponse(content={"action": fallback_action})
            
        # If API is already disabled, bypass wrapper
        if api_disabled:
            action = get_heuristic_action(obs)
            return JSONResponse(content={"action": int(action)})

        # ---- LLM Timeout Enforcement Requirement ----
        try:
            action = await asyncio.wait_for(
                asyncio.to_thread(get_llm_action_sync, obs), 
                timeout=1.4
            )
        except asyncio.TimeoutError:
            print("[WARN] LLM API Timeout exceeded 1.4s! Disabling LLM for session.")
            api_disabled = True
            action = get_heuristic_action(obs)
        except Exception as e:
            print(f"[WARN] LLM Thread error: {e}. Disabling LLM for session.")
            api_disabled = True
            action = get_heuristic_action(obs)
            
        if time.time() - start_time > 2:
            return JSONResponse(content={"action": fallback_action})
            
        return JSONResponse(content={"action": int(action)})
        
    except Exception as e:
        print(f"[CRITICAL] Unhandled endpoint error caught gracefully: {e}")
        try:
            traceback.print_exc()
        except:
            pass
        return JSONResponse(content={"action": 20})

def main():
    print("🚀 Starting Inference Service on 0.0.0.0:7860...")
    try:
        uvicorn.run(app, host="0.0.0.0", port=7860, log_level="warning")
    except Exception as e:
        print(f"[CRITICAL] Uvicorn failed to start: {e}")

if __name__ == "__main__":
    main()
