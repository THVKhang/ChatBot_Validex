"""Local Semantic Evaluation using SentenceTransformers.

Provides a fast, zero-token local embedding model to evaluate semantic similarity
between topics, sentences, and documents without calling OpenAI APIs.
"""

import logging
import threading
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)

# Global singleton to hold the model in memory
_MODEL = None
_MODEL_LOCK = threading.Lock()


def get_model() -> Any:
    """Lazy load the sentence transformer model to save memory if unused."""
    global _MODEL
    if _MODEL is None:
        # Serialise loading so concurrent requests don't each build a model.
        with _MODEL_LOCK:
            if _MODEL is not None:
                return _MODEL
            return _load_model_locked()
    return _MODEL


def _load_model_locked() -> Any:
    """Build the singleton. Callers must hold ``_MODEL_LOCK``."""
    global _MODEL
    try:
        from sentence_transformers import SentenceTransformer
        import os

        # Prioritize fine-tuned Validex model
        finetuned_path = os.path.join("data", "models", "bge-base-finetuned-validex")
        if os.path.isdir(finetuned_path) and os.path.isfile(os.path.join(finetuned_path, "config.json")):
            logger.info("Loading FINE-TUNED Validex Semantic Model (%s)...", finetuned_path)
            _MODEL = SentenceTransformer(finetuned_path)
        else:
            logger.info("Loading fallback Local Semantic Model (all-MiniLM-L6-v2)...")
            _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Local Semantic Model loaded successfully.")
    except ImportError:
        logger.error("Failed to load sentence_transformers. Please install it.")
        raise
    return _MODEL


def get_embedding(text: str) -> np.ndarray:
    """Generate embedding for a single string."""
    return get_model().encode(text)


def get_embeddings(texts: list[str]) -> np.ndarray:
    """Generate embeddings for a list of strings."""
    return get_model().encode(texts)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    # Dot product divided by the product of their L2 norms
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def batch_cosine_similarity(v_target: np.ndarray, v_list: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a target vector and a list/matrix of vectors.
    
    Uses sentence_transformers optimized util if available, otherwise numpy broadcast.
    """
    try:
        from sentence_transformers import util
        # util.cos_sim returns a matrix of shape (1, len(v_list)), we want the 1D array
        return util.cos_sim(v_target, v_list)[0].numpy()
    except Exception:
        # Fallback manual numpy calculation
        norm_target = np.linalg.norm(v_target)
        norms_list = np.linalg.norm(v_list, axis=1)
        # Avoid division by zero
        valid = (norm_target > 0) & (norms_list > 0)
        sims = np.zeros(len(v_list))
        if valid.any():
            sims[valid] = np.dot(v_list[valid], v_target) / (norm_target * norms_list[valid])
        return sims

# Core Australian legal context vector for scoring source relevance
REFERENCE_CONTEXT = (
    "Australian national police check, criminal history screening, "
    "working with children check (WWCC), NDIS worker clearance, "
    "Aged Care screening, right to work in Australia, work visa background checks, "
    "Privacy Act 1988 compliance, spent convictions scheme, "
    "employment background verification."
)

_REFERENCE_EMBEDDING = None

def get_reference_embedding() -> np.ndarray:
    """Get the embedding for the Australian legal reference context."""
    global _REFERENCE_EMBEDDING
    if _REFERENCE_EMBEDDING is None:
        _REFERENCE_EMBEDDING = get_embedding(REFERENCE_CONTEXT)
    return _REFERENCE_EMBEDDING
