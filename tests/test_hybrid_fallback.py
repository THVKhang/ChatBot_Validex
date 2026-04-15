from dataclasses import replace

import app.langchain_pipeline as langchain_pipeline_module
from app.config import settings
from app.langchain_pipeline import LangChainRAGPipeline
from app.langchain_pipeline import RetrievalBundle
from app.main import process_prompt
from app.retriever import RetrievalDecision
from app.session_manager import SessionManager


def test_hybrid_fallback_out_of_domain_includes_warning_and_runtime_flag(monkeypatch):
    local_settings = replace(
        settings,
        use_pgvector_retrieval=False,
        use_pinecone_retrieval=False,
        allow_hybrid_fallback=True,
    )
    monkeypatch.setattr(langchain_pipeline_module, "settings", local_settings)

    def fake_retrieve_with_guard(*_args, **_kwargs):
        return RetrievalDecision(
            docs=[],
            status="out_of_domain",
            confidence=0.0,
            top_score=0,
            reason="forced out of domain for test",
        )

    monkeypatch.setattr(langchain_pipeline_module, "retrieve_with_guard", fake_retrieve_with_guard)

    payload = process_prompt("Write a travel guide about Paris", SessionManager())

    assert payload["retrieval_meta"]["status"] == "out_of_domain"
    assert payload["runtime"]["external_knowledge_used"] is True
    assert payload["generated"]["draft"].startswith(settings.hybrid_warning_text)
    assert payload["generated"]["sources_used"] == []


def test_hybrid_fallback_low_confidence_includes_warning_and_runtime_flag(monkeypatch):
    local_settings = replace(
        settings,
        use_pgvector_retrieval=False,
        use_pinecone_retrieval=False,
        allow_hybrid_fallback=True,
    )
    monkeypatch.setattr(langchain_pipeline_module, "settings", local_settings)

    def fake_retrieve_with_guard(*_args, **_kwargs):
        return RetrievalDecision(
            docs=[],
            status="low_confidence",
            confidence=0.12,
            top_score=1,
            reason="forced low confidence for test",
        )

    monkeypatch.setattr(langchain_pipeline_module, "retrieve_with_guard", fake_retrieve_with_guard)

    payload = process_prompt("Explain police check requirements", SessionManager())

    assert payload["retrieval_meta"]["status"] == "low_confidence"
    assert payload["runtime"]["external_knowledge_used"] is True
    assert payload["generated"]["draft"].startswith(settings.hybrid_warning_text)
    assert payload["generated"]["sources_used"] == []


def test_strict_mode_disables_hybrid_fallback(monkeypatch):
    strict_settings = replace(
        settings,
        use_pgvector_retrieval=False,
        use_pinecone_retrieval=False,
        allow_hybrid_fallback=False,
    )
    monkeypatch.setattr(langchain_pipeline_module, "settings", strict_settings)

    def fake_retrieve_with_guard(*_args, **_kwargs):
        return RetrievalDecision(
            docs=[],
            status="out_of_domain",
            confidence=0.0,
            top_score=0,
            reason="forced out of domain for strict-mode test",
        )

    monkeypatch.setattr(langchain_pipeline_module, "retrieve_with_guard", fake_retrieve_with_guard)

    payload = process_prompt("Write a travel guide about Paris", SessionManager())

    assert payload["retrieval_meta"]["status"] == "out_of_domain"
    assert payload["runtime"]["external_knowledge_used"] is False
    assert payload["generated"]["title"] == "Need More Context"


def test_pgvector_non_fake_guard_skips_local_fallback(monkeypatch):
    guarded_settings = replace(
        settings,
        use_pgvector_retrieval=True,
        use_pinecone_retrieval=False,
        pgvector_require_non_fake_embeddings=True,
    )
    monkeypatch.setattr(langchain_pipeline_module, "settings", guarded_settings)

    pipeline = LangChainRAGPipeline()
    monkeypatch.setattr(pipeline, "_pgvector_connection_dsn", lambda: "postgresql://example")
    monkeypatch.setattr(
        pipeline,
        "_retrieve_from_pgvector",
        lambda _query, _top_k: RetrievalBundle(
            decision=RetrievalDecision(
                docs=[],
                status="no_match",
                confidence=0.0,
                top_score=0,
                reason="no pgvector match",
            ),
            documents=[],
        ),
    )

    state = {"local_called": False}

    def fake_local(_query, _top_k):
        state["local_called"] = True
        return RetrievalBundle(
            decision=RetrievalDecision(
                docs=[
                    langchain_pipeline_module.RetrievedDoc(
                        doc_id="doc_local",
                        score=9,
                        content="fallback text",
                        semantic_score=0.8,
                    )
                ],
                status="ok",
                confidence=0.8,
                top_score=9,
                reason="local retrieval",
            ),
            documents=[],
        )

    monkeypatch.setattr(pipeline, "_retrieve_from_local_guard", fake_local)

    bundle = pipeline._retrieve({"effective_topic": "police check", "retrieval_top_k": 4})

    assert bundle.decision.status == "no_match"
    assert state["local_called"] is False
