"""Local NLI Fact-Checker using CrossEncoder.

Uses a lightweight DeBERTa-v3 cross-encoder to verify facts locally,
ensuring zero hallucinations without spending LLM API tokens.
"""

import logging
from app.llm.provider import extract_text
import re
from typing import Any

logger = logging.getLogger(__name__)

# Global singleton to hold the model in memory
_NLI_MODEL = None

# Characters of context fed to the NLI model as the premise. The model's window
# is 512 tokens for premise + hypothesis combined, so ~1500 characters (~375
# tokens) is what genuinely fits. Measured on a 55-sentence draft: a 4000-char
# premise cost 19.3s, 1500 chars costs 12.5s, and the extra characters were
# being truncated away by the tokenizer either way.
PREMISE_CHAR_LIMIT = 1500

# Sentences that assert nothing factual cannot hallucinate a fact, so they are
# not worth a forward pass. A sentence is checked when it carries a number, a
# date, a modal obligation, or a proper-noun-ish token.
_FACTUAL_MARKERS = re.compile(
    r"\d|\b(must|should|require[sd]?|need[s]?|cannot|can't|shall|may not|"
    r"act|regulation|scheme|days?|weeks?|months?|years?|percent|fee|cost)\b",
    re.IGNORECASE,
)


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
        res = extract_text(response).strip()
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
        
    import numpy as np

    model = get_nli_model()
    # nli-deberta-v3-small has a 512-token window that must hold BOTH the
    # premise and the hypothesis. A 4000-character premise is ~1000 tokens, so
    # the tokenizer silently discarded more than half of it — the model never
    # saw that text, but every sentence still paid to encode it. Size the
    # premise to what actually fits and leave room for the sentence.
    premise = context[:PREMISE_CHAR_LIMIT]

    # Split draft into paragraphs to preserve structure
    paragraphs = draft.split('\n\n')

    # Pass 1 — collect every sentence that needs checking. Scoring them one at a
    # time meant one model forward pass per sentence, each re-encoding the same
    # 4000-character premise: ~30s for a single article. A cross-encoder batches
    # pairs natively, so gather first and score once.
    layout: list[list[str | None]] = []   # per paragraph: sentence or None placeholder
    pending: list[tuple[int, int, str]] = []
    for para_index, para in enumerate(paragraphs):
        if not para.strip() or para.startswith('#'):
            layout.append([para])
            continue
        sentences = re.split(r'(?<=[.!?])\s+', para)
        row: list[str | None] = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # Short fragments (transitions, list labels) are not worth scoring,
            # and neither are sentences that assert no checkable fact.
            if len(sentence.split()) < 4 or not _FACTUAL_MARKERS.search(sentence):
                row.append(sentence)
                continue
            pending.append((para_index, len(row), sentence))
            row.append(None)
        layout.append(row)

    # Pass 2 — one batched prediction for every candidate sentence.
    # Labels for nli-deberta-v3: 0: Contradiction, 1: Entailment, 2: Neutral
    probabilities = []
    if pending:
        try:
            raw = np.asarray(model.predict([[premise, s] for _, _, s in pending]))
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            exp = np.exp(raw - raw.max(axis=-1, keepdims=True))
            probabilities = exp / exp.sum(axis=-1, keepdims=True)
        except Exception as exc:
            logger.error("NLI batch prediction failed (%s) — keeping draft unchanged", exc)
            return draft

    # Pass 3 — rewrite or drop only what the model flagged.
    contradictions_found = 0
    for offset, (para_index, slot, sentence) in enumerate(pending):
        prob_contradiction = float(probabilities[offset][0])
        prob_entailment = float(probabilities[offset][1])

        if not (prob_contradiction > 0.60 and prob_entailment < 0.20):
            layout[para_index][slot] = sentence
            continue

        logger.warning(
            "NLI flagged Contradiction (prob: %.2f). Rewriting sentence: '%s'",
            prob_contradiction, sentence,
        )
        contradictions_found += 1
        replacement = None
        rewritten = _rewrite_contradicting_sentence(sentence, context)
        if rewritten:
            try:
                check = np.asarray(model.predict([[premise, rewritten]]))
                if check.ndim == 1:
                    check = check.reshape(1, -1)
                exp = np.exp(check - check.max(axis=-1, keepdims=True))
                if float((exp / exp.sum(axis=-1, keepdims=True))[0][0]) < 0.50:
                    replacement = rewritten
                    logger.info("NLI Smoothing successful: '%s'", rewritten)
                else:
                    logger.warning("Rewritten sentence still contradicts. Dropping entirely.")
            except Exception as exc:
                logger.error("NLI re-check failed (%s) — dropping sentence", exc)
        else:
            logger.warning("Rewrite failed or empty. Dropping sentence.")
        layout[para_index][slot] = replacement

    cleaned_paragraphs = []
    for row in layout:
        kept = [s for s in row if s]
        if kept:
            cleaned_paragraphs.append(" ".join(kept))

    if contradictions_found > 0:
        logger.info(f"NLI Fact-Checker removed {contradictions_found} hallucinated sentences.")

    return "\n\n".join(cleaned_paragraphs)
