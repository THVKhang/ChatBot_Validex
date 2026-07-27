"""Unit tests for the export endpoint (/api/chat/export) supporting DOCX, PDF, and HTML formats."""

import pytest
from fastapi.testclient import TestClient
from app.api_server import app

client = TestClient(app)

SAMPLE_MARKDOWN = """# Australian Police Check Guide

This is an introduction to background checks in Australia.

## Identity Requirements

Key considerations include:
- **Foreign Passport**: Primary document worth 70 points
- **Bank Statement**: Secondary document worth 25 points

> Note: Police checks are point-in-time snapshot documents.

| Document Type | Category | Points |
| --- | --- | --- |
| Passport | Primary | 70 |
| Bank Statement | Secondary | 25 |

### Summary and Next Steps

Contact Validex for automated compliance checks.
"""


def test_export_docx_returns_valid_document():
    response = client.post(
        "/api/chat/export",
        json={"markdown": SAMPLE_MARKDOWN, "format": "docx"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert len(response.content) > 1000
    assert response.content[:4] == b'PK\x03\x04'  # Zip archive header for docx


def test_export_pdf_returns_valid_pdf():
    response = client.post(
        "/api/chat/export",
        json={"markdown": SAMPLE_MARKDOWN, "format": "pdf"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert len(response.content) > 1000
    assert response.content.startswith(b'%PDF')  # PDF magic number header


def test_export_html_returns_valid_html():
    response = client.post(
        "/api/chat/export",
        json={"markdown": SAMPLE_MARKDOWN, "format": "html"}
    )
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<h1>Australian Police Check Guide</h1>" in response.text
    assert "<strong>Foreign Passport</strong>" in response.text


def test_export_invalid_format_returns_400():
    response = client.post(
        "/api/chat/export",
        json={"markdown": SAMPLE_MARKDOWN, "format": "invalid_format"}
    )
    assert response.status_code == 400
