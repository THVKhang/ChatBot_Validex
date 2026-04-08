from dataclasses import dataclass
import os
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    model_name: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    top_k: int = int(os.getenv("TOP_K", "3"))
    data_processed_dir: str = os.getenv("DATA_PROCESSED_DIR", "data/processed")
    metadata_path: str = os.getenv("METADATA_PATH", "data/metadata/documents.json")
    min_top_score: int = int(os.getenv("MIN_TOP_SCORE", "3"))
    min_confidence: float = float(os.getenv("MIN_CONFIDENCE", "0.35"))


settings = Settings()
