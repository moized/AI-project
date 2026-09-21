from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_chat():
    response = client.post("/chat", json={"query": "architecture"})
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "answer" in data
    assert "sources" in data
