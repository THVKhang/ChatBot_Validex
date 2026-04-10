from dataclasses import replace

from fastapi.testclient import TestClient

from app import report_store
from app.api_server import app
from app.config import settings


client = TestClient(app)


def test_health_includes_runtime_details():
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "runtime" in payload
    assert "retrieval_mode" in payload["runtime"]
    assert "generation_mode" in payload["runtime"]


def test_metrics_endpoint_schema():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    payload = response.json()
    assert "chat_requests_total" in payload
    assert "chat_errors_total" in payload
    assert "latency" in payload


def test_reports_api_roundtrip(tmp_path, monkeypatch):
    reports_path = tmp_path / "reports.json"
    monkeypatch.setattr(report_store, "settings", replace(settings, reports_path=str(reports_path)))

    create_response = client.post(
        "/api/reports",
        json={
            "session_id": "session-1",
            "prompt": "Generate compliance post",
            "generated": {
                "title": "Compliance Guide",
                "outline": ["Step 1", "Step 2"],
                "draft": "Draft body",
                "sources_used": ["doc_01"],
            },
        },
    )

    assert create_response.status_code == 200
    report = create_response.json()["report"]

    list_response = client.get("/api/reports")
    assert list_response.status_code == 200
    reports = list_response.json()["reports"]
    assert reports
    assert reports[0]["id"] == report["id"]

    detail_response = client.get(f"/api/reports/{report['id']}")
    assert detail_response.status_code == 200
    assert detail_response.json()["report"]["title"] == "Compliance Guide"


def test_reports_api_404(tmp_path, monkeypatch):
    reports_path = tmp_path / "reports.json"
    monkeypatch.setattr(report_store, "settings", replace(settings, reports_path=str(reports_path)))

    response = client.get("/api/reports/non-existent-id")
    assert response.status_code == 404


def test_reports_api_delete_roundtrip(tmp_path, monkeypatch):
    reports_path = tmp_path / "reports.json"
    monkeypatch.setattr(report_store, "settings", replace(settings, reports_path=str(reports_path)))

    create_response = client.post(
        "/api/reports",
        json={
            "session_id": "session-2",
            "prompt": "Generate policy summary",
            "generated": {
                "title": "Policy Summary",
                "outline": ["A", "B"],
                "draft": "Draft text",
                "sources_used": ["doc_03"],
            },
        },
    )
    report_id = create_response.json()["report"]["id"]

    delete_response = client.delete(f"/api/reports/{report_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["status"] == "deleted"

    detail_response = client.get(f"/api/reports/{report_id}")
    assert detail_response.status_code == 404


def test_reports_api_delete_404(tmp_path, monkeypatch):
    reports_path = tmp_path / "reports.json"
    monkeypatch.setattr(report_store, "settings", replace(settings, reports_path=str(reports_path)))

    response = client.delete("/api/reports/non-existent-id")
    assert response.status_code == 404
