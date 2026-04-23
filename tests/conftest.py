"""Shared pytest fixtures for all ChatBot Validex tests."""
import pytest
from app.langchain_pipeline import pipeline


@pytest.fixture(autouse=True)
def _reset_circuit_breaker():
    """Reset circuit breaker state before each test so state doesn't leak."""
    pipeline._cb_consecutive_failures = 0
    pipeline._cb_open_until = 0.0
    yield
    pipeline._cb_consecutive_failures = 0
    pipeline._cb_open_until = 0.0

@pytest.fixture(autouse=True)
def _mock_duckduckgo_search(monkeypatch):
    """Disable duckduckgo web search fallback in tests to avoid live network calls and side effects."""
    class DummyDDGS:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def text(self, *args, **kwargs): return []
    import sys
    if "duckduckgo_search" in sys.modules:
        monkeypatch.setattr("duckduckgo_search.DDGS", DummyDDGS)
    else:
        class DummyDDGSModule:
            DDGS = DummyDDGS
        sys.modules["duckduckgo_search"] = DummyDDGSModule
