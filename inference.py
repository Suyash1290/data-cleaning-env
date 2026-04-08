import os
import time
import requests
import json
import pandas as pd
from openai import OpenAI

HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
URL = "http://localhost:7860"

MAX_STEPS = 20
MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.8

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL
)

def log_step(step, action, reward, done, error=None):
    # Strictly structured stdout log required by Phase 2 parsing
    print(f"[STEP] Action: {json.dumps(action)} | Reward: {reward:+.3f} | Done: {done} | Error: {error}")

def log_end(success, steps, score, rewards):
    # Strictly structured stdout log required by Phase 2 parsing
    print(f"[END] Success: {success} | Steps: {steps} | Score: {score:.3f} | Total Steps Metric Evaluated")

def build_heuristic_actions(obs):
    actions = []
    df = pd.DataFrame(obs.get("table_sample", []))
    if df.empty:
        return [{"type": "finish"}]
    
    seen_ids = set()
    for idx, row in df.iterrows():
        row_id = row.get("id")
        if row_id in seen_ids:
            actions.append({"type": "drop_row", "row": idx})
            continue
        seen_ids.add(row_id)
        
        age = row.get("age")
        if age is None or pd.isna(age):
            actions.append({"type": "fix_cell", "row": idx, "col": "age", "value": "35"})

    sql = "SELECT DISTINCT * FROM raw_data WHERE id IS NOT NULL"
    actions.append({"type": "submit_sql", "sql": sql})
    actions.append({"type": "finish"})
    return actions

def run_inference(task_id="easy"):
    print(f"\n[START] Task_id: {task_id}")
    history = []
    rewards = []
    
    try:
        resp = requests.post(f"{URL}/reset", json={"task_id": task_id, "seed": 42}).json()
        obs = resp
    except Exception as e:
        print(f"[DEBUG] Error connecting: {e}")
        return

    optimized_actions = build_heuristic_actions(obs)
    steps_taken = 0
    done = False
    
    for step_idx in range(1, MAX_STEPS + 1):
        if done or step_idx - 1 >= len(optimized_actions):
            break
            
        action = optimized_actions[step_idx - 1]
        
        try:
            res = requests.post(f"{URL}/step", json=action).json()
            reward = res.get("reward", {}).get("score", 0.0)
            done = res.get("done", False)
            error = None
        except Exception as e:
            reward = 0.0
            done = True
            error = str(e)
            
        rewards.append(reward)
        steps_taken = step_idx
        
        log_step(step=step_idx, action=action, reward=reward, done=done, error=error)
        history.append(f"Step {step_idx}: {action} -> reward {reward:+.3f}")
        
    score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
    # Phase 2 REQUIREMENT: Score mathematically bounded strongly within (0, 1) exclusively
    score = min(max(score, 0.05), 0.95) 
    success = score >= SUCCESS_SCORE_THRESHOLD

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    tasks = ["easy", "medium", "hard"]
    for task in tasks:
        run_inference(task)
