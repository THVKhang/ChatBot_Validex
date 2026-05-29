from dataclasses import replace

import app.langchain_pipeline as langchain_pipeline_module
import app.llm.provider as provider_module
from app.config import settings
from app.langchain_pipeline import LangChainRAGPipeline
import langchain_google_genai


class _FakeGoogleChat:
    def __init__(self, model, google_api_key, temperature=0.0, **_kwargs):
        self.model = model
        self.google_api_key = google_api_key
        self.temperature = temperature


class _FakeGoogleEmbeddings:
    def __init__(self, model, google_api_key=None, **_kwargs):
        self.model = model
        self.google_api_key = google_api_key

    def embed_query(self, _text):
        return [1.0, 2.0, 3.0]


def test_build_llm_prefers_google_provider(monkeypatch):
    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", _FakeGoogleChat)

    local_settings = replace(
        settings,
        use_live_llm=True,
        llm_provider="google",
        google_api_key="google-key",
        model_name="gpt-4o-mini",
        google_model_name="models/gemini-2.5-flash",
        use_pinecone_retrieval=False,
    )
    monkeypatch.setattr(langchain_pipeline_module, "settings", local_settings)

    pipeline = LangChainRAGPipeline()

    from app.llm.provider import ResilientLLM
    assert isinstance(pipeline._llm, ResilientLLM)
    assert isinstance(pipeline._llm.primary, _FakeGoogleChat)
    assert pipeline._llm.primary.model == "models/gemini-2.5-flash"


def test_build_embedding_prefers_google_provider(monkeypatch):
    monkeypatch.setattr(langchain_pipeline_module, "GoogleGenerativeAIEmbeddings", _FakeGoogleEmbeddings)

    local_settings = replace(
        settings,
        use_live_llm=False,
        embedding_provider="google",
        google_api_key="google-key",
        embedding_model="text-embedding-3-small",
        google_embedding_model="models/gemini-embedding-001",
        use_pinecone_retrieval=False,
    )
    monkeypatch.setattr(langchain_pipeline_module, "settings", local_settings)

    pipeline = LangChainRAGPipeline()

    assert isinstance(pipeline._embedding_model, _FakeGoogleEmbeddings)
    assert pipeline._embedding_model.model == "models/gemini-embedding-001"
    assert pipeline._query_embedding("hello") == [1.0, 2.0, 3.0]


def test_google_embedding_model_overrides_legacy_embedding_model(monkeypatch):
    monkeypatch.setattr(langchain_pipeline_module, "GoogleGenerativeAIEmbeddings", _FakeGoogleEmbeddings)

    local_settings = replace(
        settings,
        use_live_llm=False,
        embedding_provider="google",
        google_api_key="google-key",
        embedding_model="models/text-embedding-004",
        google_embedding_model="models/gemini-embedding-001",
        use_pinecone_retrieval=False,
    )
    monkeypatch.setattr(langchain_pipeline_module, "settings", local_settings)

    pipeline = LangChainRAGPipeline()

    assert isinstance(pipeline._embedding_model, _FakeGoogleEmbeddings)
    assert pipeline._embedding_model.model == "models/gemini-embedding-001"
