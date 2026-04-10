from dataclasses import replace

from app import report_store
from app.config import settings


def test_report_store_save_list_get(tmp_path, monkeypatch):
    reports_path = tmp_path / "reports.json"
    monkeypatch.setattr(report_store, "settings", replace(settings, reports_path=str(reports_path)))

    saved = report_store.save_report(
        prompt="Write a blog about police checks",
        title="Police Checks in Hiring",
        outline=["Why checks matter", "Compliance"],
        draft="Sample draft",
        sources_used=["doc_01"],
        sections=[
            {
                "heading": "Why checks matter",
                "body": "Sample body",
                "image_url": "https://example.com/image.jpg",
                "image_alt": "sample",
            }
        ],
        session_id="s1",
    )

    listed = report_store.list_reports()
    fetched = report_store.get_report(saved["id"])

    assert saved["id"]
    assert len(listed) == 1
    assert listed[0]["id"] == saved["id"]
    assert fetched is not None
    assert fetched["title"] == "Police Checks in Hiring"
    assert fetched["sections"]


def test_report_store_delete(tmp_path, monkeypatch):
    reports_path = tmp_path / "reports.json"
    monkeypatch.setattr(report_store, "settings", replace(settings, reports_path=str(reports_path)))

    saved = report_store.save_report(
        prompt="Write a blog about compliance",
        title="Compliance Report",
        outline=["Context"],
        draft="Body",
        sources_used=["doc_02"],
        sections=[],
        session_id="s2",
    )

    deleted = report_store.delete_report(saved["id"])
    missing = report_store.get_report(saved["id"])

    assert deleted is True
    assert missing is None
