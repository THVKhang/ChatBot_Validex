import json
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from app.ingest_pgvector import build_parent_child_chunks, _chunk_hash
from app.langchain_pipeline import LangChainRAGPipeline

def test_build_parent_child_chunks_non_legal():
    # 1. Non-legal document containing headers and a table
    raw_records = [
        {
            "doc_id": "doc_test_1",
            "chunk_id": "chunk_test_1_1",
            "title": "Validex Guide",
            "text": "This is the introduction text. It describes the overall process.",
            "source_url": "http://example.com/guide",
        },
        {
            "doc_id": "doc_test_1",
            "chunk_id": "chunk_test_1_2",
            "title": "Validex Guide",
            "text": "## Document Categories\n\nHere are the categories of identity documents:\n\n| Category | Points | Examples |\n|---|---|---|\n| Commencement | 70 | Birth certificate, Passport |\n| Primary | 40 | Driver licence, Medicare |\n\nThis is additional text under the table section.",
            "source_url": "http://example.com/guide",
        }
    ]

    # Mock the LLM call for table summarization
    with patch("app.ingest_pgvector.summarize_table_via_llm") as mock_summarize:
        mock_summarize.return_value = "This table outlines points for Commencement (70 pts) and Primary (40 pts) documents."
        
        children, parents = build_parent_child_chunks(raw_records, "test_table")
        
        # We expect children:
        # - Introduction child
        # - Section body child
        # - Table summary child
        assert len(children) >= 3
        assert len(parents) >= 3
        
        # Verify parent-child linkage
        child_ids = [c["chunk_id"] for c in children]
        parent_ids = [p["parent_id"] for p in parents]
        
        for c in children:
            assert c["parent_id"] in parent_ids
            # Check context injection
            assert c["text"].startswith("[Validex Guide >")
            
        # Verify table parent contains raw markdown
        table_parents = [p for p in parents if "Commencement | 70" in p["content"]]
        assert len(table_parents) == 1
        
        # Verify table child contains LLM summary
        table_children = [c for c in children if "table" in c["chunk_id"]]
        assert len(table_children) == 1
        assert "70 pts" in table_children[0]["text"]


def test_build_parent_child_chunks_legal():
    # 2. Legal document with Crimes Act structure
    raw_records = [
        {
            "doc_id": "doc_test_legal",
            "chunk_id": "chunk_test_legal_1",
            "title": "Crimes Act 1914",
            "text": "Part VIIC — Spent Convictions\nDivision 3 — Exclusions\nSection 85ZM  Spent convictions scheme\n(1) Under this section, a conviction is spent if the waiting period has expired.\n(2) The waiting period is 10 years for adult convictions.",
            "source_url": "http://example.com/legal",
            "act_name": "Crimes Act 1914 (Cth)",
            "jurisdiction": "Commonwealth",
        }
    ]
    
    children, parents = build_parent_child_chunks(raw_records, "test_table")
    
    assert len(children) >= 1
    assert len(parents) >= 1
    
    # Check legal breadcrumbs context injection
    breadcrumb = children[0]["text"]
    assert "Crimes Act 1914 (Cth)" in breadcrumb
    assert "Part VIIC" in breadcrumb
    assert "Division 3" in breadcrumb
    assert "Section 85ZM" in breadcrumb
    assert children[0]["parent_id"] == parents[0]["parent_id"]
    assert "Section 85ZM" in parents[0]["content"]


def test_retrieve_routing_simple_vs_complex():
    pipeline = LangChainRAGPipeline()
    
    # Mock connection check to always return True
    pipeline._pgvector_connection_dsn = MagicMock(return_value="postgresql://mock")
    pipeline._query_embedding = MagicMock(return_value=[0.1] * 768)
    
    # Mock hybrid search rows
    mock_rows = [
        {
            "chunk_id": "child_1",
            "doc_id": "doc_1",
            "content": "[Guide > Sec 1] Child chunk short content",
            "source_url": "http://example.com",
            "source_domain": "example.com",
            "source_type": "webpage",
            "topic": "guide",
            "region": "AU",
            "title": "Guide",
            "authority_score": 0.8,
            "approved": True,
            "similarity": 0.9,
            "rrf_score": 0.03,
            "parent_id": "parent_1",
            "jurisdiction": "Commonwealth",
            "act_name": "",
            "section_ref": "",
            "parent_context": "Guide > Sec 1",
            "status": "in_force",
        }
    ]
    
    with patch("app.vector_repository.PGVectorRepository") as mock_repo_class:
        mock_repo = MagicMock()
        mock_repo.hybrid_search.return_value = mock_rows
        mock_repo.get_parent_content.return_value = "### Parent Chunk Full Content - 2000 tokens of details..."
        mock_repo_class.return_value = mock_repo
        
        # Test Simple routing (should return child content)
        bundle_simple = pipeline._retrieve({"effective_topic": "How to verify identity", "retrieval_top_k": 1, "complexity_level": "simple"})
        assert len(bundle_simple.documents) == 1
        assert bundle_simple.documents[0].page_content == "[Guide > Sec 1] Child chunk short content"
        
        # Test Complex routing (should swap content to parent content)
        bundle_complex = pipeline._retrieve({"effective_topic": "How to verify identity", "retrieval_top_k": 1, "complexity_level": "complex"})
        assert len(bundle_complex.documents) == 1
        assert bundle_complex.documents[0].page_content == "### Parent Chunk Full Content - 2000 tokens of details..."
        mock_repo.get_parent_content.assert_called_with("validex_knowledge", "parent_1")
