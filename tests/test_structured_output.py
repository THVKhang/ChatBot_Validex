from dataclasses import replace

import app.langchain_pipeline as langchain_pipeline_module
from app.config import settings
from app.langchain_pipeline import LangChainRAGPipeline
from app.parser import parse_user_input
from langchain_core.documents import Document


class _FakeStructuredInvoker:
    def invoke(self, _instruction: str):
        return {
            "title": "5 Reasons Validex Delivers Better Police Check Workflows in Australia",
            "introduction": "Australian employers need reliable, compliant screening workflows. This guide explains practical ways to improve consistency and reduce onboarding friction across police check operations.",
            "sections": [
                {
                    "header": "1. Compliance-First Process Design",
                    "content": "Start with policy-aligned workflow checkpoints so every request follows a consistent, auditable standard for background checks.",
                    "image_search_keyword": "australia compliance workflow",
                },
                {
                    "header": "2. Faster Candidate Experience",
                    "content": "Use clear status updates and required-document guidance to reduce delay and confusion for applicants and HR teams.",
                    "image_search_keyword": "candidate onboarding australia",
                },
                {
                    "header": "3. Better Risk Visibility",
                    "content": "Track turnaround time, exception patterns, and source quality to improve decision confidence and operational performance.",
                    "image_search_keyword": "risk dashboard compliance",
                },
            ],
            "conclusion": "With structured process controls and transparent communication, organisations can improve both compliance confidence and hiring velocity.",
            "meta_tags": "validex, australia, police check, compliance, onboarding",
        }


class _FakeLLM:
    def with_structured_output(self, _schema):
        return _FakeStructuredInvoker()


def test_generate_with_llm_uses_structured_output_schema(monkeypatch):
    pipeline = LangChainRAGPipeline()
    pipeline._llm = _FakeLLM()

    local_settings = replace(
        settings,
        use_live_llm=True,
        use_structured_output=True,
        use_unsplash_images=True,
        unsplash_access_key="dummy-key",
    )
    monkeypatch.setattr(langchain_pipeline_module, "settings", local_settings)
    monkeypatch.setattr(
        pipeline,
        "_search_unsplash_image",
        lambda keyword: (f"https://images.example/{keyword.replace(' ', '-')}.jpg", f"{keyword} alt"),
    )

    parsed = parse_user_input("Write an 800 words blog about police check compliance for employer")
    docs = [
        Document(
            page_content="Official AU compliance context for police check workflows.",
            metadata={"doc_id": "doc_001", "score": 92, "semantic_score": 0.82},
        )
    ]

    generated = pipeline._generate_with_llm(parsed, docs, previous_draft=None)

    assert generated is not None
    assert generated.title.startswith("5 Reasons Validex")
    assert len(generated.sections) == 3
    assert generated.sections[0].image_url.startswith("https://images.example/")
    assert "## Introduction" in generated.draft
    assert "## Conclusion" in generated.draft
    assert generated.sources_used == ["doc_001"]


def test_resolve_section_image_falls_back_without_unsplash_key(monkeypatch):
    pipeline = LangChainRAGPipeline()
    local_settings = replace(
        settings,
        use_unsplash_images=True,
        unsplash_access_key="",
    )
    monkeypatch.setattr(langchain_pipeline_module, "settings", local_settings)

    image_url, image_alt = pipeline._resolve_section_image(
        topic="police check australia",
        image_keyword="compliance workflow",
        heading="Compliance Workflow",
    )

    assert "picsum.photos/seed" in image_url
    assert image_alt == "Compliance Workflow illustration"


def test_unsplash_image_tool_returns_json_payload(monkeypatch):
    pipeline = LangChainRAGPipeline()
    local_settings = replace(
        settings,
        use_unsplash_images=True,
        unsplash_access_key="dummy-key",
    )
    monkeypatch.setattr(langchain_pipeline_module, "settings", local_settings)
    monkeypatch.setattr(
        pipeline,
        "_search_unsplash_image",
        lambda _keyword: ("https://images.example/photo.jpg", "smart lock photo"),
    )

    output = pipeline._tool_unsplash_image_search("smart lock security")

    assert "https://images.example/photo.jpg" in output
    assert "smart lock security" in output
