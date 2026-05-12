"""Semantic caching layer using PostgreSQL pgvector."""

import json
import logging
from typing import Any
import os

import psycopg

from app.config import settings

logger = logging.getLogger(__name__)

class PgSemanticCache:
    """Semantic cache using PostgreSQL pgvector for fast similarity search."""
    
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.dsn = os.environ.get("DATABASE_URL")
        
    def _get_embedding(self, text: str) -> list[float] | None:
        """Get embedding for the prompt text using the configured provider."""
        try:
            from app.ingest_pgvector import get_embeddings_model
            embedding_model, _ = get_embeddings_model()
            if not embedding_model:
                return None
            return embedding_model.embed_query(text)
        except Exception as exc:
            logger.warning("SemanticCache failed to generate embedding: %s", exc)
            return None

    def search_cache(self, prompt: str) -> dict[str, Any] | None:
        """Search the semantic cache for a highly similar prompt."""
        if not self.dsn or not settings.cache_enabled:
            return None
            
        embedding = self._get_embedding(prompt)
        if not embedding:
            return None
            
        try:
            with psycopg.connect(self.dsn) as conn:
                with conn.cursor() as cur:
                    # Calculate cosine distance using <=> operator.
                    # Cosine similarity = 1 - cosine_distance.
                    # So we want 1 - (embedding <=> prompt_embedding) >= threshold.
                    # Or equivalently: embedding <=> prompt_embedding <= 1 - threshold.
                    distance_threshold = 1.0 - self.threshold
                    
                    # Convert embedding list to string format for pgvector '[1.0, 2.0, ...]'
                    embedding_str = f"[{','.join(str(x) for x in embedding)}]"
                    
                    cur.execute(
                        """
                        SELECT generated_response, 1 - (prompt_embedding <=> %s::vector) AS similarity
                        FROM validex_semantic_cache
                        WHERE prompt_embedding <=> %s::vector <= %s
                        ORDER BY similarity DESC
                        LIMIT 1
                        """,
                        (embedding_str, embedding_str, distance_threshold)
                    )
                    row = cur.fetchone()
                    if row:
                        response_json, similarity = row
                        logger.info("Semantic Cache HIT: similarity=%.3f for prompt='%s'", similarity, prompt[:50])
                        
                        # Set a flag indicating this came from the semantic cache
                        if isinstance(response_json, dict):
                            response_json["_semantic_cache_hit"] = True
                            
                        return response_json
                        
            logger.debug("Semantic Cache MISS for prompt='%s'", prompt[:50])
            return None
            
        except Exception as exc:
            logger.error("SemanticCache search failed: %s", exc)
            return None

    def save_cache(self, prompt: str, response: dict[str, Any]) -> None:
        """Save a generated response to the semantic cache."""
        if not self.dsn or not settings.cache_enabled:
            return
            
        # Clean the response from cache flags before saving
        clean_response = dict(response)
        clean_response.pop("_semantic_cache_hit", None)
        
        # In a background task or non-blocking way ideally, but for now blocking
        embedding = self._get_embedding(prompt)
        if not embedding:
            return
            
        try:
            with psycopg.connect(self.dsn) as conn:
                with conn.cursor() as cur:
                    embedding_str = f"[{','.join(str(x) for x in embedding)}]"
                    
                    cur.execute(
                        """
                        INSERT INTO validex_semantic_cache (prompt_text, prompt_embedding, generated_response)
                        VALUES (%s, %s::vector, %s)
                        """,
                        (prompt, embedding_str, json.dumps(clean_response))
                    )
                conn.commit()
            logger.info("Semantic Cache SAVED for prompt='%s'", prompt[:50])
        except Exception as exc:
            logger.error("SemanticCache save failed: %s", exc)

semantic_cache = PgSemanticCache(threshold=0.95)
