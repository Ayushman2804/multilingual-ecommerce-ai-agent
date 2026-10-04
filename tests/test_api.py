"""Phase 6 Test Suite: FastAPI Endpoints, SSE Streaming, and Chat UI."""
import sys
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "ja" in data["supported_languages"]


def test_chat_ui_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Global Support AI Agent" in response.text
    assert "<!DOCTYPE html>" in response.text


def test_api_chat_endpoint_english(client):
    payload = {"message": "Where is my package for order ORD-1001?"}
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["language"] == "en"
    assert data["intent"] == "order_lookup"
    assert "ORD-1001" in data["response"]
    assert data["latency_ms"] < 250.0


def test_api_chat_endpoint_spanish_policy(client):
    # Using explicit unicode escape to prevent Windows console cp1252 mangling
    payload = {"message": "\u00bfCu\u00e1l es la pol\u00edtica de devoluciones?"}
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["language"] == "es"
    assert "30" in data["response"] or "devoluci" in data["response"].lower()


def test_api_chat_streaming(client):
    payload = {"message": "Track order ORD-1002"}
    response = client.post("/api/chat/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events = []
    for line in response.text.split("\n"):
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    assert len(events) >= 3  # start + tokens + done
    assert events[0]["type"] == "start"
    assert events[-1]["type"] == "done"
    assert events[-1]["latency_ms"] > 0


if __name__ == "__main__":
    pytest.main(["-v", str(Path(__file__))])
