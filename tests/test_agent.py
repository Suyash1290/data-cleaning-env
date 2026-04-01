from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

def test_reset():
    res = client.post("/reset", json={"task_id": "easy", "seed": 10})
    assert res.status_code == 200
    data = res.json()
    assert "table_sample" in data
    assert "columns" in data

def test_step_inspect():
    client.post("/reset", json={"task_id": "easy", "seed": 10})
    res = client.post("/step", json={"type": "inspect_column"})
    assert res.status_code == 200
    data = res.json()
    assert "reward" in data
    assert data["reward"]["score"] == -0.01 - 0.02 # step_cost + inspect_cost
