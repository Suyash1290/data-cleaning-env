from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from models.core import Action
from env.environment import DataCleaningEnv

app = FastAPI()
env_state = {}

class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: int = 42

print("APP STARTED SUCCESSFULLY")

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset(req: ResetRequest):
    env = DataCleaningEnv(difficulty=req.task_id)
    obs = env.reset(seed=req.seed)
    env_state["env"] = env
    return obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()

@app.post("/step")
def step(action: Action):
    if "env" not in env_state:
        raise HTTPException(status_code=400, detail="Environment not reset")
    env = env_state["env"]
    obs, rew, done = env.step(action)
    return {
        "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs.dict(),
        "reward": rew.model_dump() if hasattr(rew, "model_dump") else rew.dict(),
        "done": done
    }
