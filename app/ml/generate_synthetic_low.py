"""Generate Synthetic Low-Quality Data.

Uses the pipeline's fast LLM (Gemini/Groq) to generate flawed, off-topic,
and hallucinated blog content to train the ML Quality Gate to recognize bad inputs.
"""

from __future__ import annotations

import json
import logging
import random
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TRAINING_DATA_PATH = "data/ml/training_data.jsonl"

BAD_PROMPT_TEMPLATE = """
You are an AI trained to generate BAD, FLAWED, and UNACCEPTABLE blog content to test a Quality Assurance system.
Given the topic: "{topic}"

Please write a short blog section (about 100 to 150 words) that contains ONE of the following fatal flaws:
1. Hallucination: Make up fake facts, fake fees (e.g., claiming a police check costs $10,000), or fake Australian laws.
2. Contamination: Start with the topic, but halfway through, completely change the subject to something irrelevant like cooking recipes or WWCC (Working With Children Check).
3. Repetition & Fluff: Write in massive walls of text. Repeat the exact same sentence 4 times. Use extreme corporate fluff without giving any actual information.

Output ONLY the raw text of the bad blog post. No pleasantries, no markdown code block formatting (like ```).
"""

topics = [
    "How to apply for a National Police Check",
    "Privacy Act obligations for employers",
    "Identity documents required for an Australian police check",
    "Spent convictions scheme NSW",
    "NDIS Worker Screening Check guide",
    "WWCC vs Police Check in Victoria",
    "Police check processing times in Australia"
]


def generate_synthetic_low():
    # Import pipeline to reuse initialized resilient fast LLM (handles Groq and Google fallbacks automatically)
    try:
        from app.langchain_pipeline import pipeline
        llm = pipeline._fast_llm or pipeline._llm
        if not llm:
            raise RuntimeError("LLM not initialized")
    except Exception as exc:
        logger.error("Could not load LLM from pipeline: %s. Falling back to direct model initialization...", exc)
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            google_key = os.getenv("GOOGLE_API_KEY", "")
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=google_key, temperature=0.9)
        except Exception as e:
            logger.error("Direct initialization failed: %s", e)
            return

    from langchain_core.prompts import PromptTemplate
    from app.ml.ml_data_collector import extract_features
    
    prompt = PromptTemplate.from_template(BAD_PROMPT_TEMPLATE)

    synthetic_data = []

    logger.info("Starting generation of 50 synthetic low-quality samples...")
    for i in range(50):
        topic = random.choice(topics)
        try:
            rendered = prompt.format(topic=topic)
            response = llm.invoke(rendered)
            content = getattr(response, "content", str(response)).strip()
            
            # Construct dummy GraphState to pass to extract_features
            dummy_state = {
                "draft": content,
                "parsed": {"topic": topic},
                "retrieved_docs": [],
                "retrieval_meta": {},
                "revision_count": 0,
                "retrieval_attempts": 0
            }
            features = extract_features(dummy_state)
            
            # Inject distinct low-quality features to help classifier learn boundaries:
            features["nli_contradictions"] = random.randint(3, 8)  # High contradictions
            features["draft_fk_grade"] = round(random.uniform(15.0, 22.0), 2)  # Hard to read
            features["draft_reading_ease"] = round(random.uniform(0.0, 15.0), 2)  # Low reading ease
            
            # Form matching training_data.jsonl structure
            sample = {
                "features": features,
                "labels": {
                    "quality_class": "low",
                    "faithfulness_score": round(random.uniform(0.0, 0.25), 2),
                    "label_source": "synthetic"
                }
            }
            synthetic_data.append(sample)
            logger.info("Generated sample %d/50", i + 1)
        except Exception as e:
            logger.error("Error generating sample %d: %s", i + 1, e)

    if not synthetic_data:
        logger.error("No samples generated.")
        return

    # Append to existing training data
    out_path = Path(TRAINING_DATA_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with out_path.open("a", encoding="utf-8") as f:
        for item in synthetic_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("Success! Appended %d low-quality samples to %s", len(synthetic_data), TRAINING_DATA_PATH)


if __name__ == "__main__":
    generate_synthetic_low()
