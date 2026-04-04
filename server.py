from fastapi import FastAPI
from pydantic import BaseModel
from environment import TrafficEnv

app = FastAPI()
env = TrafficEnv()

class ActionParams(BaseModel):
    action: int

@app.post("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(params: ActionParams):
    return env.step(params.action)
