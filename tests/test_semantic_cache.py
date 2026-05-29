import pytest
from app.semantic_cache import semantic_cache

def test_semantic_cache_flow(monkeypatch):
    import json
    from app.config import settings
    
    # Force settings.cache_enabled to True for this test using object.__setattr__
    object.__setattr__(settings, "cache_enabled", True)
    monkeypatch.setattr(semantic_cache, "dsn", "postgresql://fake_dsn")

    # Mock the embedding generation
    def mock_get_embedding(self, text):
        # Return a dummy vector
        return [0.1] * 1536
    
    monkeypatch.setattr("app.semantic_cache.PgSemanticCache._get_embedding", mock_get_embedding)
    
    in_memory_db = []

    class MockCursor:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def execute(self, query, params=None):
            if "INSERT INTO" in query:
                # params is (prompt, embedding_str, json.dumps(clean_response))
                in_memory_db.append({
                    "prompt": params[0],
                    "embedding": json.loads(params[1]),  # converts '[0.1, 0.1...]' to list
                    "response": json.loads(params[2])
                })
            elif "SELECT" in query:
                # params is (embedding_str, embedding_str, distance_threshold)
                self.row = None
                if in_memory_db:
                    # check if the embedding matches target
                    emb_param = params[0]
                    if "[-0.1" in emb_param:
                        # opposite vector, exceeds distance threshold
                        self.row = None
                    else:
                        item = in_memory_db[0]
                        self.row = (item["response"], 1.0)
            
        def fetchone(self):
            return getattr(self, "row", None)

    class MockConnection:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def cursor(self): return MockCursor()
        def commit(self): pass

    import psycopg
    monkeypatch.setattr(psycopg, "connect", lambda dsn: MockConnection())

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
