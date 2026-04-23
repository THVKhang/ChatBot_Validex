from dataclasses import replace

import pytest

import app.langchain_pipeline as langchain_pipeline_module
from app.config import settings
from app.main import process_prompt
from app.retriever import RetrievalDecision
from app.retriever import RetrievedDoc
from app.session_manager import SessionManager


@pytest.mark.parametrize(
    "prompt,expected_profile,min_k,max_k",
    [
        ("Write a 400 chu blog about police check", "short", 3, 4),
        ("Write a 800 words blog about police check", "medium", 6, 8),
        ("Write a 1200 words detailed blog about police check", "long", 10, 12),
    ],
)
def test_dynamic_top_k_matches_length_profile(monkeypatch, prompt, expected_profile, min_k, max_k):
    local_settings = replace(
        settings,
        use_pgvector_retrieval=False,
        use_pinecone_retrieval=False,
        enforce_quality_gate=False,
    )
    monkeypatch.setattr(langchain_pipeline_module, "settings", local_settings)

    captured: dict[str, int] = {}

    def fake_retrieve_with_guard(_query, _data_dir, top_k=3, *_args, **_kwargs):
        captured["top_k"] = top_k
        docs = [
            RetrievedDoc(
                doc_id=f"doc_{index + 1}",
                score=10,
                content="police check compliance hiring workflow " * 80,
                semantic_score=0.75,
            )
            for index in range(top_k)
        ]
        return RetrievalDecision(
            docs=docs,
            status="ok",
            confidence=0.8,
            top_score=10,
            reason="retrieval successful",
        )

    monkeypatch.setattr(langchain_pipeline_module, "retrieve_with_guard", fake_retrieve_with_guard)

    payload = process_prompt(prompt, SessionManager())
    budget = payload["runtime"]["token_budget"]

    assert budget["length_profile"] == expected_profile
    assert min_k * 3 <= captured["top_k"] <= max_k * 3
    assert captured["top_k"] == budget["recommended_top_k"] * 3
    assert budget["input_tokens_target_min"] < budget["input_tokens_target_max"]


def test_runtime_token_budget_includes_input_output_estimates(monkeypatch):
    local_settings = replace(
        settings,
        use_pgvector_retrieval=False,
        use_pinecone_retrieval=False,
        enforce_quality_gate=False,
    )
    monkeypatch.setattr(langchain_pipeline_module, "settings", local_settings)

    def fake_retrieve_with_guard(_query, _data_dir, top_k=3, *_args, **_kwargs):
        docs = [
            RetrievedDoc(
                doc_id=f"doc_{index + 1}",
                score=9,
                content="verified source context for police check process " * 120,
                semantic_score=0.72,
            )
            for index in range(top_k)
        ]
        return RetrievalDecision(
            docs=docs,
            status="ok",
            confidence=0.82,
            top_score=12,
            reason="retrieval successful",
        )

    monkeypatch.setattr(langchain_pipeline_module, "retrieve_with_guard", fake_retrieve_with_guard)

    payload = process_prompt("Write a medium blog about police check", SessionManager())
    budget = payload["runtime"]["token_budget"]

    assert budget["output_tokens_target"] == settings.output_tokens_medium
    assert budget["output_tokens_estimated"] > 0
    assert budget["input_tokens_estimated"] > 0
    assert budget["retrieved_docs"] >= budget["context_docs_used"] >= 1
