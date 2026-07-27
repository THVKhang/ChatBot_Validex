"""Tests for Knowledge Gap Analysis tools and Gov.au Crawler."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open


# ═══════════════════════════════════════════════════════════════════════
# Test Coverage Matrix
# ═══════════════════════════════════════════════════════════════════════

class TestCoverageMatrix:
    def test_matrix_with_real_chunks(self, tmp_path):
        """Test coverage matrix generation with sample JSONL data."""
        chunks_file = tmp_path / "test_chunks.jsonl"
        records = [
            {"doc_id": "d1", "text": "How to get a police check in NSW", "title": "Police Check NSW", "jurisdiction": "NSW"},
            {"doc_id": "d2", "text": "Working with children check Victoria", "title": "WWCC VIC", "jurisdiction": "VIC"},
            {"doc_id": "d3", "text": "NDIS worker screening requirements", "title": "NDIS Screening", "jurisdiction": "Commonwealth"},
            {"doc_id": "d4", "text": "Spent convictions scheme under Crimes Act", "title": "Spent Convictions", "jurisdiction": "Commonwealth"},
            {"doc_id": "d5", "text": "100 point identity check requirements", "title": "100 Point ID", "jurisdiction": "Commonwealth"},
        ]
        chunks_file.write_text(
            "\n".join(json.dumps(r) for r in records),
            encoding="utf-8",
        )

        from app.knowledge_gap_analyzer import run_coverage_matrix
        report = run_coverage_matrix(chunks_path=str(chunks_file))

        assert report["total_chunks"] == 5
        assert "matrix" in report
        assert "blind_spots" in report

        matrix = report["matrix"]
        # NSW should have police_check
        assert matrix["police_check"]["NSW"] >= 1
        # VIC should have wwcc
        assert matrix["wwcc"]["VIC"] >= 1
        # Commonwealth should have ndis
        assert matrix["ndis"]["Commonwealth"] >= 1
        # QLD should be empty for everything
        assert matrix["police_check"]["QLD"] == 0
        assert matrix["wwcc"]["QLD"] == 0

    def test_matrix_empty_file(self, tmp_path):
        """Test coverage matrix with empty file."""
        chunks_file = tmp_path / "empty.jsonl"
        chunks_file.write_text("", encoding="utf-8")

        from app.knowledge_gap_analyzer import run_coverage_matrix
        report = run_coverage_matrix(chunks_path=str(chunks_file))

        assert report["total_chunks"] == 0
        assert report["empty_cells"] == report["total_cells"]

    def test_matrix_missing_file(self):
        """Test coverage matrix with missing file."""
        from app.knowledge_gap_analyzer import run_coverage_matrix
        report = run_coverage_matrix(chunks_path="/nonexistent/path.jsonl")
        assert "error" in report


class TestJurisdictionNormalization:
    def test_normalize_common_jurisdictions(self):
        from app.knowledge_gap_analyzer import _normalize_jurisdiction

        assert _normalize_jurisdiction("Commonwealth") == "Commonwealth"
        assert _normalize_jurisdiction("NSW") == "NSW"
        assert _normalize_jurisdiction("VIC") == "VIC"
        assert _normalize_jurisdiction("New South Wales") == "NSW"
        assert _normalize_jurisdiction("VICTORIA") == "VIC"
        assert _normalize_jurisdiction("Queensland") == "QLD"
        assert _normalize_jurisdiction("Western Australia") == "WA"
        assert _normalize_jurisdiction("CTH") == "Commonwealth"
        assert _normalize_jurisdiction("") == "Commonwealth"  # fallback


# ═══════════════════════════════════════════════════════════════════════
# Test ML Rejection Analysis
# ═══════════════════════════════════════════════════════════════════════

class TestMLRejectionAnalysis:
    def test_rejection_analysis_with_data(self, tmp_path):
        """Test ML rejection analyzer with sample training data."""
        data_file = tmp_path / "training.jsonl"
        records = [
            {
                "labels": {"quality_class": "high", "label_source": "heuristic"},
                "metadata": {"topic": "police check"},
                "features": {"nli_contradictions": 0.1},
            },
            {
                "labels": {"quality_class": "low", "label_source": "ml_gate_reject"},
                "metadata": {"topic": "spent convictions QLD", "rejection_reason": "Low faithfulness"},
                "features": {"nli_contradictions": 0.5},
            },
            {
                "labels": {"quality_class": "low", "label_source": "ml_gate_reject"},
                "metadata": {"topic": "WWCC Blue Card QLD", "rejection_reason": "Insufficient evidence"},
                "features": {"nli_contradictions": 0.7},
            },
            {
                "labels": {"quality_class": "low", "label_source": "heuristic"},
                "metadata": {"topic": "police check", "rejection_reason": "Too short"},
                "features": {"nli_contradictions": 0.2},
            },
        ]
        data_file.write_text(
            "\n".join(json.dumps(r) for r in records),
            encoding="utf-8",
        )

        from app.knowledge_gap_analyzer import run_ml_rejection_analysis
        report = run_ml_rejection_analysis(data_path=str(data_file))

        assert report["total_training_records"] == 4
        assert report["total_rejected"] == 3
        assert report["ml_gate_rejected"] == 2
        assert "spent convictions QLD" in report["top_rejected_topics"]
        assert len(report["high_nli_contradiction_topics"]) >= 2

    def test_rejection_analysis_missing_file(self):
        """Test ML rejection analyzer with missing file."""
        from app.knowledge_gap_analyzer import run_ml_rejection_analysis
        report = run_ml_rejection_analysis(data_path="/nonexistent/training.jsonl")
        assert "error" in report


# ═══════════════════════════════════════════════════════════════════════
# Test Prompt Sweep (with mocks)
# ═══════════════════════════════════════════════════════════════════════

class TestPromptSweep:
    def test_load_topics(self, tmp_path):
        """Test loading edge-case topics from JSON file."""
        topics_file = tmp_path / "topics.json"
        topics_file.write_text(json.dumps(["topic 1", "topic 2", "topic 3"]), encoding="utf-8")

        from app.knowledge_gap_analyzer import _load_topics
        topics = _load_topics(str(topics_file))

        assert len(topics) == 3
        assert topics[0] == "topic 1"

    def test_load_topics_missing_file(self):
        """Test loading topics from missing file."""
        from app.knowledge_gap_analyzer import _load_topics
        topics = _load_topics("/nonexistent/topics.json")
        assert topics == []

    def test_sweep_identifies_blind_spots(self, tmp_path):
        """Test that sweep correctly identifies blind spots (low similarity)."""
        topics_file = tmp_path / "topics.json"
        topics_file.write_text(json.dumps(["covered topic", "blind spot topic"]), encoding="utf-8")

        mock_pipeline = MagicMock()
        mock_pipeline._pgvector_connection_dsn.return_value = "postgresql://mock"
        mock_pipeline._query_embedding.return_value = [0.1] * 768

        mock_repo = MagicMock()

        # First topic: high similarity (covered)
        # Second topic: low similarity (blind spot)
        call_count = [0]
        def mock_hybrid_search(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return [{"similarity": 0.85, "title": "Good Match", "chunk_id": "c1", "doc_id": "d1",
                          "content": "...", "source_url": "", "source_domain": "", "source_type": "webpage",
                          "topic": "compliance", "region": "AU", "authority_score": 0.8, "approved": True,
                          "rrf_score": 0.03, "parent_id": None, "jurisdiction": "", "act_name": "",
                          "section_ref": "", "parent_context": "", "status": "in_force"}]
            else:
                return [{"similarity": 0.35, "title": "Bad Match", "chunk_id": "c2", "doc_id": "d2",
                          "content": "...", "source_url": "", "source_domain": "", "source_type": "webpage",
                          "topic": "compliance", "region": "AU", "authority_score": 0.5, "approved": True,
                          "rrf_score": 0.01, "parent_id": None, "jurisdiction": "", "act_name": "",
                          "section_ref": "", "parent_context": "", "status": "in_force"}]

        mock_repo.hybrid_search.side_effect = mock_hybrid_search

        with patch("app.langchain_pipeline.pipeline", mock_pipeline), \
             patch("app.vector_repository.PGVectorRepository", return_value=mock_repo):
            from app.knowledge_gap_analyzer import run_prompt_sweep
            report = run_prompt_sweep(topics_path=str(topics_file), threshold=0.65)

        assert report["total_topics"] == 2
        assert report["covered_count"] == 1
        assert report["blind_spot_count"] == 1
        assert report["blind_spots"][0]["topic"] == "blind spot topic"
        assert report["covered"][0]["topic"] == "covered topic"


# ═══════════════════════════════════════════════════════════════════════
# Test Gov.au Crawler
# ═══════════════════════════════════════════════════════════════════════

class TestGovCrawler:
    def test_html_to_markdown_headings(self):
        """Test that headings are converted correctly."""
        from app.gov_crawler import _regex_html_to_markdown
        html = "<h1>Title</h1><h2>Section</h2><p>Content here.</p>"
        md = _regex_html_to_markdown(html)

        assert "# Title" in md
        assert "## Section" in md
        assert "Content here." in md

    def test_html_to_markdown_tables(self):
        """Test that HTML tables are converted to Markdown tables."""
        from app.gov_crawler import _regex_html_to_markdown
        html = """
        <table>
          <tr><th>Document</th><th>Points</th></tr>
          <tr><td>Birth Certificate</td><td>70</td></tr>
          <tr><td>Passport</td><td>70</td></tr>
        </table>
        """
        md = _regex_html_to_markdown(html)

        assert "| Document | Points |" in md
        assert "| --- | --- |" in md
        assert "| Birth Certificate | 70 |" in md
        assert "| Passport | 70 |" in md

    def test_html_to_markdown_lists(self):
        """Test that HTML lists are converted."""
        from app.gov_crawler import _regex_html_to_markdown
        html = "<ul><li>Item 1</li><li>Item 2</li></ul>"
        md = _regex_html_to_markdown(html)

        assert "- Item 1" in md
        assert "- Item 2" in md

    def test_generate_doc_id_deterministic(self):
        """Test that doc_id generation is deterministic."""
        from app.gov_crawler import _generate_doc_id
        url = "https://www.acic.gov.au/our-services/national-police-checking-service"
        id1 = _generate_doc_id(url)
        id2 = _generate_doc_id(url)
        assert id1 == id2
        assert id1.startswith("gov_")

    def test_generate_doc_id_unique_for_different_urls(self):
        """Test that different URLs produce different doc_ids."""
        from app.gov_crawler import _generate_doc_id
        id1 = _generate_doc_id("https://www.acic.gov.au/page-1")
        id2 = _generate_doc_id("https://www.acic.gov.au/page-2")
        assert id1 != id2

    def test_url_to_jsonl_record_schema(self):
        """Test that JSONL record has all required fields for ingest_pgvector.py."""
        from app.gov_crawler import _url_to_jsonl_record
        record = _url_to_jsonl_record(
            url="https://www.acic.gov.au/test",
            markdown_text="## Test Content\n\nThis is test content.",
            source_meta={
                "jurisdiction": "Commonwealth",
                "topic": "compliance",
                "source_type": "webpage",
                "title": "ACIC Test",
            },
        )

        # Check all required fields exist
        required_fields = [
            "doc_id", "chunk_id", "source_url", "source_domain",
            "source_type", "topic", "region", "title",
            "authority_score", "approved", "text",
            "jurisdiction", "act_name", "section_ref", "parent_context",
        ]
        for field in required_fields:
            assert field in record, f"Missing field: {field}"

        assert record["source_domain"] == "www.acic.gov.au"
        assert record["jurisdiction"] == "Commonwealth"
        assert record["authority_score"] == 0.95
        assert record["approved"] is True
        assert "Test Content" in record["text"]

    def test_load_golden_sources(self, tmp_path):
        """Test loading golden sources from JSON file."""
        sources_file = tmp_path / "sources.json"
        sources_file.write_text(json.dumps({
            "federal": [
                {"url": "https://www.acic.gov.au/test", "jurisdiction": "Commonwealth"}
            ],
            "state_specific": [
                {"url": "https://www.ocg.nsw.gov.au/", "jurisdiction": "NSW"}
            ],
        }), encoding="utf-8")

        from app.gov_crawler import load_golden_sources
        sources = load_golden_sources(str(sources_file))

        assert len(sources) == 2
        assert sources[0]["group"] == "federal"
        assert sources[1]["group"] == "state_specific"

    def test_clean_markdown(self):
        """Test markdown cleanup removes excessive blank lines."""
        from app.gov_crawler import _clean_markdown
        dirty = "\n\n\n\n# Title\n\n\n\n\nContent\n\n\n\n"
        clean = _clean_markdown(dirty)

        assert clean.startswith("# Title")
        assert "\n\n\n" not in clean  # No triple newlines
