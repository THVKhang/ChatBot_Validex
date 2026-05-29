import sys; sys.path.insert(0,'.')
import os
import psycopg
from app.semantic_cache import PgSemanticCache
from app.local_semantics import get_embedding

def test_cache():
    dsn = os.environ.get('DATABASE_URL')
    if not dsn:
        from dotenv import load_dotenv
        load_dotenv()
        dsn = os.environ.get('DATABASE_URL')

    if not dsn:
        print("No DATABASE_URL found. Skipping test.")
        return

    # Enable cache for test
    from unittest.mock import patch
    with patch('app.config.settings.cache_enabled', True):
        cache = PgSemanticCache(threshold=0.8)
        cache.dsn = dsn

        print('1. Testing embedding generation...')
        emb = cache._get_embedding('Test prompt for semantic cache')
        assert len(emb) == 384, f'Expected 384 dims, got {len(emb)}'

        print('2. Inserting into semantic cache table...')
        with psycopg.connect(cache.dsn) as conn:
            with conn.cursor() as cur:
                emb_str = f'[{",".join(str(x) for x in emb)}]'
                cur.execute(
                    'INSERT INTO validex_semantic_cache (prompt_text, prompt_embedding, generated_response) VALUES (%s, %s::vector, %s) RETURNING id',
                    ('Test prompt for semantic cache', emb_str, '{"test": "success"}')
                )
                conn.commit()

        print('3. Searching cache...')
        result = cache.search_cache('Test prompt for semantic cache')
        assert result is not None, "Cache miss! Expected cache hit."
        assert result['test'] == 'success'

        print('\nSemantic Cache with 384-dim Local Model is WORKING!')

if __name__ == '__main__':
    test_cache()
