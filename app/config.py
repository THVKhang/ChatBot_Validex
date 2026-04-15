from dataclasses import dataclass
import os
from dotenv import load_dotenv


load_dotenv()


def _normalize_pgvector_table(value: str | None) -> str:
    normalized = str(value or "").strip()
    if not normalized or normalized == "rag_blog_chunks":
        return "validex_knowledge"
    return normalized


@dataclass(frozen=True)
class Settings:
    model_name: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    google_model_name: str = os.getenv("GOOGLE_MODEL_NAME", "models/gemini-2.5-flash")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    google_embedding_model: str = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
    llm_provider: str = os.getenv("LLM_PROVIDER", "auto")
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "auto")
    top_k: int = int(os.getenv("TOP_K", "3"))
    output_tokens_short: int = int(os.getenv("OUTPUT_TOKENS_SHORT", "600"))
    output_tokens_medium: int = int(os.getenv("OUTPUT_TOKENS_MEDIUM", "1200"))
    output_tokens_long: int = int(os.getenv("OUTPUT_TOKENS_LONG", "1800"))
    input_output_ratio_min: float = float(os.getenv("INPUT_OUTPUT_RATIO_MIN", "1.5"))
    input_output_ratio_max: float = float(os.getenv("INPUT_OUTPUT_RATIO_MAX", "2.0"))
    chunk_token_estimate: int = int(os.getenv("CHUNK_TOKEN_ESTIMATE", "260"))
    top_k_short_min: int = int(os.getenv("TOP_K_SHORT_MIN", "3"))
    top_k_short_max: int = int(os.getenv("TOP_K_SHORT_MAX", "4"))
    top_k_medium_min: int = int(os.getenv("TOP_K_MEDIUM_MIN", "6"))
    top_k_medium_max: int = int(os.getenv("TOP_K_MEDIUM_MAX", "8"))
    top_k_long_min: int = int(os.getenv("TOP_K_LONG_MIN", "10"))
    top_k_long_max: int = int(os.getenv("TOP_K_LONG_MAX", "12"))
    data_processed_dir: str = os.getenv("DATA_PROCESSED_DIR", "data/processed")
    metadata_path: str = os.getenv("METADATA_PATH", "data/metadata/documents.json")
    reports_path: str = os.getenv("REPORTS_PATH", "data/reports/reports.json")
    min_top_score: int = int(os.getenv("MIN_TOP_SCORE", "3"))
    min_confidence: float = float(os.getenv("MIN_CONFIDENCE", "0.35"))
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "")
    pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE", "default")
    pgvector_table: str = _normalize_pgvector_table(os.getenv("PGVECTOR_TABLE", "validex_knowledge"))
    use_pgvector_retrieval: bool = os.getenv("USE_PGVECTOR_RETRIEVAL", "1") == "1"
    pgvector_require_non_fake_embeddings: bool = os.getenv("PGVECTOR_REQUIRE_NON_FAKE_EMBEDDINGS", "1") == "1"
    pgvector_min_similarity: float = float(os.getenv("PGVECTOR_MIN_SIMILARITY", "0.15"))
    allow_fake_embeddings: bool = os.getenv("ALLOW_FAKE_EMBEDDINGS", "0") == "1"
    fake_embedding_dim: int = int(os.getenv("FAKE_EMBEDDING_DIM", "1536"))
    use_live_llm: bool = os.getenv("USE_LIVE_LLM", "0") == "1"
    use_structured_output: bool = os.getenv("USE_STRUCTURED_OUTPUT", "1") == "1"
    use_pinecone_retrieval: bool = os.getenv("USE_PINECONE_RETRIEVAL", "0") == "1"
    use_agentic_rag: bool = os.getenv("USE_AGENTIC_RAG", "0") == "1"
    use_unsplash_images: bool = os.getenv("USE_UNSPLASH_IMAGES", "1") == "1"
    unsplash_access_key: str = os.getenv("UNSPLASH_ACCESS_KEY", "")
    unsplash_api_base: str = os.getenv("UNSPLASH_API_BASE", "https://api.unsplash.com")
    unsplash_timeout_seconds: int = int(os.getenv("UNSPLASH_TIMEOUT_SECONDS", "8"))
    allow_hybrid_fallback: bool = os.getenv("ALLOW_HYBRID_FALLBACK", "1") == "1"
    hybrid_warning_text: str = os.getenv(
        "HYBRID_WARNING_TEXT",
        "Note: Parts of this article were generated using external knowledge and may not be grounded in the internal knowledge base.",
    )
    validex_website_url: str = os.getenv("VALIDEX_WEBSITE_URL", "")
    agent_max_iterations: int = int(os.getenv("AGENT_MAX_ITERATIONS", "4"))
    enforce_quality_gate: bool = os.getenv("ENFORCE_QUALITY_GATE", "1") == "1"
    min_sections: int = int(os.getenv("MIN_SECTIONS", "3"))
    min_sources_used: int = int(os.getenv("MIN_SOURCES_USED", "1"))
    min_draft_chars: int = int(os.getenv("MIN_DRAFT_CHARS", "700"))
    use_rate_limit: bool = os.getenv("USE_RATE_LIMIT", "1") == "1"
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
    use_redis_rate_limit: bool = os.getenv("USE_REDIS_RATE_LIMIT", "0") == "1"
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    metrics_window_size: int = int(os.getenv("METRICS_WINDOW_SIZE", "200"))
    tool_allowed_domains: str = os.getenv(
        "TOOL_ALLOWED_DOMAINS",
        "validex.com.au,www.validex.com.au,acic.gov.au,www.acic.gov.au,afp.gov.au,www.afp.gov.au,oaic.gov.au,www.oaic.gov.au",
    )


settings = Settings()
