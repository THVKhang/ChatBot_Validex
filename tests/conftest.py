import os
os.environ["USE_LIVE_LLM"] = "0"
os.environ["ENABLE_PROMPT_GUARD"] = "1"
os.environ["USE_RATE_LIMIT"] = "0"
os.environ["JWT_SECRET_KEY"] = "test_secret_key_for_validex_chatbot_security_audit_12345"
os.environ["CACHE_ENABLED"] = "0"

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
def _reset_settings_cache():
    from app.config import settings
    object.__setattr__(settings, "cache_enabled", False)
    yield
    object.__setattr__(settings, "cache_enabled", False)

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
