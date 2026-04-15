from dataclasses import replace

from fastapi.testclient import TestClient

from app import api_server
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
    assert report["status"] == "Draft"

    list_response = client.get("/api/reports")
    assert list_response.status_code == 200
    reports = list_response.json()["reports"]
    assert reports
    assert reports[0]["id"] == report["id"]

    detail_response = client.get(f"/api/reports/{report['id']}")
    assert detail_response.status_code == 200
    assert detail_response.json()["report"]["title"] == "Compliance Guide"
    assert detail_response.json()["report"]["status"] == "Draft"


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


def test_reports_api_status_update_flow(tmp_path, monkeypatch):
    reports_path = tmp_path / "reports.json"
    monkeypatch.setattr(report_store, "settings", replace(settings, reports_path=str(reports_path)))

    create_response = client.post(
        "/api/reports",
        json={
            "session_id": "session-3",
            "prompt": "Generate report",
            "generated": {
                "title": "Report Title",
                "outline": ["A"],
                "draft": "Draft",
                "sources_used": ["doc_10"],
            },
        },
    )
    report_id = create_response.json()["report"]["id"]

    reviewed_response = client.patch(
        f"/api/reports/{report_id}/status",
        json={"status": "Reviewed"},
    )
    approved_response = client.patch(
        f"/api/reports/{report_id}/status",
        json={"status": "Approved"},
    )

    assert reviewed_response.status_code == 200
    assert reviewed_response.json()["report"]["status"] == "Reviewed"
    assert approved_response.status_code == 200
    assert approved_response.json()["report"]["status"] == "Approved"


def test_reports_api_status_update_conflict(tmp_path, monkeypatch):
    reports_path = tmp_path / "reports.json"
    monkeypatch.setattr(report_store, "settings", replace(settings, reports_path=str(reports_path)))

    create_response = client.post(
        "/api/reports",
        json={
            "session_id": "session-4",
            "prompt": "Generate report",
            "generated": {
                "title": "Report Title",
                "outline": ["A"],
                "draft": "Draft",
                "sources_used": ["doc_11"],
            },
        },
    )
    report_id = create_response.json()["report"]["id"]

    conflict_response = client.patch(
        f"/api/reports/{report_id}/status",
        json={"status": "Approved"},
    )

    assert conflict_response.status_code == 409


def test_reports_api_publish_requires_approved_status(tmp_path, monkeypatch):
    reports_path = tmp_path / "reports.json"
    monkeypatch.setattr(report_store, "settings", replace(settings, reports_path=str(reports_path)))

    create_response = client.post(
        "/api/reports",
        json={
            "session_id": "session-p1",
            "prompt": "Generate report",
            "generated": {
                "title": "Report Title",
                "outline": ["A"],
                "draft": "Draft",
                "sources_used": ["doc_12"],
                "sections": [
                    {
                        "heading": "Section A",
                        "body": "Body content",
                        "image_url": "https://example.com/a.jpg",
                        "image_alt": "Section A image",
                    }
                ],
            },
        },
    )
    report_id = create_response.json()["report"]["id"]

    publish_response = client.post(f"/api/reports/{report_id}/publish")

    assert publish_response.status_code == 409


def test_reports_api_publish_returns_markdown_and_html(tmp_path, monkeypatch):
    reports_path = tmp_path / "reports.json"
    monkeypatch.setattr(report_store, "settings", replace(settings, reports_path=str(reports_path)))

    create_response = client.post(
        "/api/reports",
        json={
            "session_id": "session-p2",
            "prompt": "Generate report",
            "generated": {
                "title": "SEO Report",
                "outline": ["A"],
                "draft": "Draft body",
                "sources_used": ["doc_13"],
                "sections": [
                    {
                        "heading": "Section A",
                        "body": "Body content",
                        "image_url": "https://example.com/a.jpg",
                        "image_alt": "Section A image",
                    }
                ],
            },
        },
    )
    report_id = create_response.json()["report"]["id"]

    client.patch(f"/api/reports/{report_id}/status", json={"status": "Reviewed"})
    client.patch(f"/api/reports/{report_id}/status", json={"status": "Approved"})

    markdown_response = client.post(f"/api/reports/{report_id}/publish")
    html_response = client.post(f"/api/reports/{report_id}/publish?output_format=html")

    assert markdown_response.status_code == 200
    markdown_payload = markdown_response.json()
    assert markdown_payload["output"]["format"] == "markdown"
    assert "meta_description:" in markdown_payload["output"]["content"]
    assert "![Section A image](https://example.com/a.jpg)" in markdown_payload["output"]["content"]

    assert html_response.status_code == 200
    html_payload = html_response.json()
    assert html_payload["output"]["format"] == "html"
    assert "<meta name=\"description\"" in html_payload["output"]["content"]
    assert "alt=\"Section A image\"" in html_payload["output"]["content"]


def test_source_analytics_endpoint_returns_payload(monkeypatch):
    payload = {
        "table": "rag_blog_chunks",
        "total_chunks": 60,
        "total_sources": 9,
        "topics": [
            {"topic": "police_check", "chunks": 50},
            {"topic": "privacy", "chunks": 10},
        ],
        "authority_bands": [
            {"band": "0.90-1.00", "sources": 5, "chunks": 40},
            {"band": "0.80-0.89", "sources": 4, "chunks": 20},
        ],
        "generated_at": "2026-04-15T03:00:00+00:00",
    }

    monkeypatch.setattr(api_server, "fetch_source_analytics", lambda: payload)

    response = client.get("/api/source-analytics")

    assert response.status_code == 200
    result = response.json()
    assert result["table"] == "rag_blog_chunks"
    assert result["total_chunks"] == 60
    assert result["topics"][0]["topic"] == "police_check"


def test_source_analytics_endpoint_handles_runtime_error(monkeypatch):
    def _raise_runtime_error():
        raise RuntimeError("DATABASE_URL is required")

    monkeypatch.setattr(api_server, "fetch_source_analytics", _raise_runtime_error)

    response = client.get("/api/source-analytics")

    assert response.status_code == 503


def test_knowledge_health_endpoint_returns_payload(monkeypatch):
    payload = {
        "table": "validex_knowledge",
        "total_chunks": 108,
        "has_embedding_provider": True,
        "genuine_chunks": 12,
        "fake_chunks": 96,
        "other_chunks": 0,
        "genuine_percent": 11.11,
        "fake_percent": 88.89,
        "other_percent": 0.0,
        "provider_breakdown": [
            {"provider": "fake", "chunks": 96},
            {"provider": "google", "chunks": 12},
        ],
        "ready_for_retrieval": True,
        "generated_at": "2026-04-15T04:00:00+00:00",
    }

    monkeypatch.setattr(api_server, "fetch_knowledge_health", lambda: payload)

    response = client.get("/api/knowledge/health")

    assert response.status_code == 200
    result = response.json()
    assert result["table"] == "validex_knowledge"
    assert result["total_chunks"] == 108
    assert result["genuine_percent"] == 11.11
    assert result["ready_for_retrieval"] is True


def test_knowledge_health_endpoint_handles_runtime_error(monkeypatch):
    def _raise_runtime_error():
        raise RuntimeError("DATABASE_URL is required")

    monkeypatch.setattr(api_server, "fetch_knowledge_health", _raise_runtime_error)

    response = client.get("/api/knowledge/health")

    assert response.status_code == 503
