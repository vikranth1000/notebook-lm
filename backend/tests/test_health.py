from fastapi.testclient import TestClient

from notebooklm_backend.app import create_app


def test_healthcheck():
    client = TestClient(create_app())
    response = client.get("/api/healthz")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "detail" in data

