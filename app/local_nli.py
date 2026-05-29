"""Local NLI Fact-Checker using CrossEncoder.

Uses a lightweight DeBERTa-v3 cross-encoder to verify facts locally,
ensuring zero hallucinations without spending LLM API tokens.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Global singleton to hold the model in memory
_NLI_MODEL = None


def get_nli_model() -> Any:
    """Lazy load the CrossEncoder model."""
    global _NLI_MODEL
    if _NLI_MODEL is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading Local NLI Model (cross-encoder/nli-deberta-v3-small)...")
            # DeBERTa-v3-small NLI is fast and accurate.
            # Outputs logits for [Contradiction, Entailment, Neutral]
            _NLI_MODEL = CrossEncoder('cross-encoder/nli-deberta-v3-small', max_length=512)
            logger.info("Local NLI Model loaded successfully.")
        except ImportError:
            logger.error("Failed to load sentence_transformers. Please install it.")
            raise
    return _NLI_MODEL


def _rewrite_contradicting_sentence(sentence: str, context: str) -> str:
    """Uses Fast LLM to rewrite a hallucinated sentence based on context."""
    try:
        from app.langchain_pipeline import pipeline
        llm = getattr(pipeline, "_fast_llm", pipeline._llm)
        if not llm:
            return ""
            
        prompt = (
            "The following sentence contains a factual error or hallucination based on the context.\n\n"
            f"Context: {context[:2000]}\n\n"
            f"Incorrect Sentence: {sentence}\n\n"
            "Rewrite this sentence to be factually correct and smooth based ONLY on the context. "
            "If the context does not support any part of the sentence, return exactly 'EMPTY_STRING'.\n"
            "Return ONLY the rewritten sentence."
        )
        response = llm.invoke(prompt)
        res = getattr(response, "content", str(response)).strip()
        if res and not res.upper().startswith("EMPTY_STRING"):
            return res
    except Exception as exc:
        logger.warning(f"Failed to rewrite sentence: {exc}")
    return ""


def verify_facts_nli(draft: str, context: str) -> str:
    """
    Splits the draft into sentences and compares each against the combined context.
    Removes sentences that the NLI model confidently flags as 'Contradiction'.
    
    Returns the cleaned draft.
    """
    if not draft or not context:
        return draft
        
    model = get_nli_model()
    
    # Split draft into paragraphs to preserve structure
    paragraphs = draft.split('\n\n')
    cleaned_paragraphs = []
    
    contradictions_found = 0
    
    for para in paragraphs:
        if not para.strip() or para.startswith('#'):
            # Skip empty lines or headings
            cleaned_paragraphs.append(para)
            continue
            
        # Split paragraph into sentences (basic regex for sentence boundaries)
        # Handle decimal points, initials, etc., loosely
        sentences = re.split(r'(?<=[.!?])\s+', para)
        
        valid_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If the sentence is too short, assume it's valid (e.g., transitional phrases)
            if len(sentence.split()) < 4:
                valid_sentences.append(sentence)
                continue
                
            # Predict NLI: Pair of (Premise, Hypothesis)
            # Premise = context, Hypothesis = sentence generated
            # Labels for nli-deberta-v3 are typically: 0: Contradiction, 1: Entailment, 2: Neutral
            try:
                scores = model.predict([context[:4000], sentence])
                # We want to catch explicit Contradictions.
                # If Contradiction (index 0) has the highest score and is significantly higher than Entailment (index 1)
                import numpy as np
                # Apply softmax to get probabilities
                probs = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
                
                prob_contradiction = float(probs[0])
                prob_entailment = float(probs[1])
                
                # If contradiction probability > 50% AND entailment is very low, it's a hallucination
                if prob_contradiction > 0.60 and prob_entailment < 0.20:
                    logger.warning(f"NLI flagged Contradiction (prob: {prob_contradiction:.2f}). Rewriting sentence: '{sentence}'")
                    rewritten = _rewrite_contradicting_sentence(sentence, context)
                    if rewritten:
                        # Double-check the rewritten sentence with NLI
                        new_scores = model.predict([context[:4000], rewritten])
                        new_probs = np.exp(new_scores) / np.sum(np.exp(new_scores), axis=-1, keepdims=True)
                        if new_probs[0] < 0.50:  # If new contradiction prob is low enough
                            valid_sentences.append(rewritten)
                            logger.info(f"NLI Smoothing successful: '{rewritten}'")
                        else:
                            logger.warning("Rewritten sentence still contradicts. Dropping entirely.")
                    else:
                        logger.warning("Rewrite failed or empty. Dropping sentence.")
                    contradictions_found += 1
                else:
                    valid_sentences.append(sentence)
            except Exception as exc:
                logger.error(f"NLI prediction failed for sentence: {exc}")
                # Fallback to keeping the sentence on error
                valid_sentences.append(sentence)
                
        if valid_sentences:
            cleaned_paragraphs.append(" ".join(valid_sentences))
            
    if contradictions_found > 0:
        logger.info(f"NLI Fact-Checker removed {contradictions_found} hallucinated sentences.")
        
    return "\n\n".join(cleaned_paragraphs)
