import pytest
from langchain_core.documents import Document
from app.langchain_pipeline import pipeline, MISSING_INTERNAL_DATA_TEXT
from app.generator import GeneratedBlog

def test_pipeline_accuracy_missing_data_fallbacks():
    """Verify that the pipeline correctly uses fallback or appends the missing data warning

    when no relevant context documents are available.
    """
    res = pipeline.run("Write an article about cooking pasta in Rome.")
    assert res is not None
    assert "runtime" in res
    assert "generated" in res
    
    draft = res["generated"].get("draft", "")
    gen_mode = res["runtime"].get("generation_mode", "fallback")
    
    # Either the pipeline detected out_of_domain/low_confidence and ran in fallback mode,
    # or the draft correctly notes that internal data is missing.
    assert gen_mode in ["fallback", "hybrid_fallback"] or MISSING_INTERNAL_DATA_TEXT in draft


def test_pipeline_citation_accuracy():
    """Verify that when context documents are supplied, the pipeline correctly appends

    citations in the expected '[Nguồn: doc_id | URL: source_url]' format.
    """
    gen_blog = GeneratedBlog(
        title="Validex System Overview",
        outline=["Architecture"],
        draft="# Validex System Overview\n\n## Architecture\n\nWe provide background screening.",
        sources_used=[],
        sections=[
            GeneratedBlog.Section(
                heading="Architecture",
                body="We provide background screening.",
                image_url="https://example.com/img.jpg",
                image_alt="Sample"
            )
        ]
    )
    test_docs = [
        Document(
            page_content="Validex offers automated check solutions.",
            metadata={"doc_id": "doc_test_101", "source_url": "https://validex.com.au"}
        )
    ]
    
    refined = pipeline._enforce_grounding_and_citations(gen_blog, test_docs)
    assert refined is not None
    assert "[Nguồn: doc_test_101 | URL: https://validex.com.au]" in refined.draft
    assert "## Danh mục nguồn tham khảo" in refined.draft
    assert "- [Nguồn: doc_test_101 | URL: https://validex.com.au]" in refined.draft


def test_pipeline_accuracy_no_hallucinations_fictional():
    """Verify that asking a fictional question leads to fallback or missing data note."""
    res = pipeline.run("Describe the new 2026 Australian cyber tax law introduced by ACIC.")
    assert res is not None
    assert "runtime" in res
    
    draft = res["generated"].get("draft", "")
    gen_mode = res["runtime"].get("generation_mode", "fallback")
    
    assert gen_mode in ["fallback", "hybrid_fallback"] or MISSING_INTERNAL_DATA_TEXT in draft
