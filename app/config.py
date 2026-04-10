from dataclasses import dataclass
import os
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    model_name: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    top_k: int = int(os.getenv("TOP_K", "3"))
    data_processed_dir: str = os.getenv("DATA_PROCESSED_DIR", "data/processed")
    metadata_path: str = os.getenv("METADATA_PATH", "data/metadata/documents.json")
    reports_path: str = os.getenv("REPORTS_PATH", "data/reports/reports.json")
    min_top_score: int = int(os.getenv("MIN_TOP_SCORE", "3"))
    min_confidence: float = float(os.getenv("MIN_CONFIDENCE", "0.35"))
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "")
    pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE", "default")
    use_live_llm: bool = os.getenv("USE_LIVE_LLM", "0") == "1"
    use_pinecone_retrieval: bool = os.getenv("USE_PINECONE_RETRIEVAL", "0") == "1"
    use_agentic_rag: bool = os.getenv("USE_AGENTIC_RAG", "0") == "1"
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
