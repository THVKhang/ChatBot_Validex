from langchain_core.documents import Document

from app.generator import GeneratedBlog
from app.langchain_pipeline import LangChainRAGPipeline
from app.langchain_pipeline import MISSING_INTERNAL_DATA_TEXT
from app.langchain_pipeline import SOURCE_LINE_PREFIX


def test_enforce_grounding_adds_doc_url_page_citations():
    pipeline = LangChainRAGPipeline()
    generated = GeneratedBlog(
        title="Police check guidance",
        outline=["Section"],
        draft="# Police check guidance\n\n## Section\n\nSome body text.",
        sources_used=[],
        sections=[
            GeneratedBlog.Section(
                heading="Section",
                body="Some body text.",
                image_url="https://example.com/image.jpg",
                image_alt="sample",
            )
        ],
    )
    docs = [
        Document(
            page_content="Official police check guidance for Australia.",
            metadata={
                "doc_id": "doc_01",
                "source_url": "https://example.gov.au/police-check",
                "page": "3",
            },
        )
    ]

    result = pipeline._enforce_grounding_and_citations(generated, docs)

    assert result.sections
    assert SOURCE_LINE_PREFIX in result.sections[0].body
    assert "[Nguồn: doc_01 | URL: https://example.gov.au/police-check]" in result.sections[0].body
    assert "## Danh mục nguồn tham khảo" in result.draft


def test_enforce_grounding_uses_missing_data_sentence_without_docs():
    pipeline = LangChainRAGPipeline()
    generated = GeneratedBlog(
        title="Fallback",
        outline=["Section"],
        draft="# Fallback",
        sources_used=[],
        sections=[
            GeneratedBlog.Section(
                heading="Section",
                body="",
                image_url="https://example.com/image.jpg",
                image_alt="sample",
            )
        ],
    )

    result = pipeline._enforce_grounding_and_citations(generated, [])

    assert MISSING_INTERNAL_DATA_TEXT in result.sections[0].body
    assert MISSING_INTERNAL_DATA_TEXT in result.draft
