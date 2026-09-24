from backend.main import app
from fastapi.testclient import TestClient


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_openapi_available():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["title"] == "AI Research Assistant API"


def test_documents_endpoint():
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    assert isinstance(response.json()["documents"], list)


def test_upload_rejects_unsupported_extension():
    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "blocked.exe",
                b"not a supported document",
                "application/octet-stream",
            )
        },
    )
    assert response.status_code == 415
