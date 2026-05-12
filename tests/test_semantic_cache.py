import pytest
from app.semantic_cache import semantic_cache

def test_semantic_cache_flow(monkeypatch):
    # Mock the embedding generation
    def mock_get_embedding(self, text):
        # Return a dummy vector
        return [0.1] * 1536
    
    monkeypatch.setattr("app.semantic_cache.PgSemanticCache._get_embedding", mock_get_embedding)
    
    prompt = "Test prompt for semantic caching 123"
    response = {
        "parsed": {"topic": "test", "intent": "create_blog"},
        "generated": {"draft": "This is a test draft."}
    }
    
    # 1. Save to cache
    semantic_cache.save_cache(prompt, response)
    
    # 2. Search for exact same prompt (should hit)
    cached = semantic_cache.search_cache(prompt)
    assert cached is not None
    assert cached["generated"]["draft"] == "This is a test draft."
    assert cached.get("_semantic_cache_hit") is True
    
    # 3. Search for completely different prompt (should miss if embedding was different, but since we mock embedding to be exactly the same, it will hit)
    # So we'll mock the embedding to return something different
    def mock_get_embedding_diff(self, text):
        return [-0.1] * 1536
    monkeypatch.setattr("app.semantic_cache.PgSemanticCache._get_embedding", mock_get_embedding_diff)
    
    missed = semantic_cache.search_cache("Completely different prompt")
    assert missed is None
