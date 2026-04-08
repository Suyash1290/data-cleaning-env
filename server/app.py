from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI()

print("APP STARTED SUCCESSFULLY")

# Hold state globally
env_state = {}

class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: int = 42

# Optional payload handler for strict evaluator pipelines requiring no body
from typing import Optional

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    if req is None:
        req = ResetRequest()
        
    # Lazy initializations to dramatically accelerate HF docker startup times.
    from env.environment import DataCleaningEnv
    
    env = DataCleaningEnv(difficulty=req.task_id)
    obs = env.reset(seed=req.seed)
    env_state["env"] = env
    return obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()

@app.post("/step")
def step(action_payload: dict):
    from models.core import Action
    # Cast dictionary back to Action model 
    action = Action(**action_payload)
    
    if "env" not in env_state:
        raise HTTPException(status_code=400, detail="Environment not reset")
    env = env_state["env"]
    obs, rew, done = env.step(action)
    return {
        "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs.dict(),
        "reward": rew.model_dump() if hasattr(rew, "model_dump") else rew.dict(),
        "done": done,
        "info": {}
    }

@app.get("/state")
def state():
    if "env" not in env_state or not getattr(env_state["env"], "state", None):
        raise HTTPException(status_code=400, detail="Environment not reset")
    st = env_state["env"].state
    return st.model_dump() if hasattr(st, "model_dump") else st.dict()

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
