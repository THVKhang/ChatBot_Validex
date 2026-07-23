"""DSPy-style Offline Prompt Optimizer.

Evaluates different prompt variations for the blog section generator against
the golden test set to select the optimal prompt format.

Optimizes for:
1. Readability (Grade 8-10 level, short sentences, active voice)
2. CTA Compliance (Contains validex.com.au)
3. Structural constraint compliance (no intro fillers, BLUF compliance)
4. Semantic Jaccard overlap with golden answers
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from app.config import settings
from app.parser import ParsedPrompt
from app.utils import tokenize

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_PATH = "data/benchmark/generation_queries.json"
GOLDENS_PATH = "data/goldens/golden_answers.json"
REPORT_OUTPUT_PATH = "data/ml/prompt_optimization_report.json"

# ── Prompt Candidates ───────────────────────────────────────────────

# Prompt A: Original prompt
PROMPT_A = """Validex Blog Editor. Section: ## {heading}
Query: {raw_prompt}
Topic: {topic}

SCOPE: Focus on {focus}. OFF-LIMITS: {other_sections}

RULES:
{role_rule}
- Start with active subject + strong verb (e.g., 'The APIN protocol encrypts...').
- Address the user's specific scenario with concrete details and examples.
- Write 3-4 paragraphs separated by blank lines. Mix prose with bullet lists where helpful.
- READABILITY (CRITICAL): Write at a Grade 8-10 reading level. Use short sentences (15-20 words max). Prefer common everyday words over technical jargon. Break complex ideas into simple steps. Avoid passive voice. Avoid long subordinate clauses.
- Use **bold** for key terms, legislation, and important concepts.
- For legal content, cite Act names and Section numbers from context.
- Output body only, no heading.
{domain_pivot}
<context>
{context_text}
</context>"""

# Prompt B: Readability-Optimized prompt
PROMPT_B = """You are a professional technical writer specializing in clear, readable Australian compliance guides.
Write the section content for: ## {heading}
For the blog topic: {topic}

Context to base your writing on:
<context>
{context_text}
</context>

SCOPE:
- Strictly focus on: {focus}
- Do NOT mention or overlap with: {other_sections}

WRITING RULES:
{role_rule}
- Do NOT include the section heading in the output.
- Write in active voice. Start sentences with active subjects and strong verbs.
- Write at a Grade 8 reading level. Use short, punchy sentences (maximum 15 words).
- Avoid technical jargon, corporate buzzwords, and long parenthetical clauses. Explain concepts using everyday English.
- Use **bold** highlighting ONLY for official Australian Acts, specific sections, or critical compliance dates.
{domain_pivot}"""

# Prompt C: Structure-Optimized prompt (Strict BLUF and CTA focus)
PROMPT_C = """Validex Compliance Expert. Write ONLY the body text for section: ## {heading}
Topic: {topic} (User Query: {raw_prompt})

Available Context:
{context_text}

Scope Boundary:
Focus solely on '{focus}'. Avoid discussing: {other_sections}.

STRICT FORMATTING AND QUALITY RULES:
1. {role_rule}
2. No introduction filler. Do NOT start with 'This section...', 'Here we...', or 'To begin with...'. Start directly with the core message.
3. Write 3-4 short paragraphs (under 60 words each). Use a bulleted list for complex steps.
4. Citing legal sources is mandatory: use **bold** format for all cited Australian laws, regulations, and sections.
5. For conclusion sections, you MUST end with a clear Call-To-Action directing the reader to visit validex.com.au.
6. READABILITY: Sentences must be under 18 words. Avoid passive voice.
{domain_pivot}"""


# ── Metrics and Scorers ─────────────────────────────────────────────

def _get_sentence_length_score(text: str) -> float:
    """Score readability based on average sentence length. Higher is better (shorter)."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return 1.0
    words = text.split()
    avg_len = len(words) / len(sentences)
    # Target: 12-18 words per sentence. Penalty for longer.
    if avg_len <= 15:
        return 1.0
    elif avg_len > 25:
        return 0.0
    else:
        return 1.0 - ((avg_len - 15) / 10)


def _check_intro_filler(text: str) -> float:
    """Penalize starting with conversational fluff/filler phrases."""
    text_lower = text.lower().strip()
    fluff_patterns = [
        "this section", "here we", "in this section", "to begin", 
        "firstly", "in conclusion", "to summarize", "this paragraph"
    ]
    for pattern in fluff_patterns:
        if text_lower.startswith(pattern):
            return 0.0
    return 1.0


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    tokens_a = set(tokenize(text_a))
    tokens_b = set(tokenize(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a.intersection(tokens_b)) / max(1, len(tokens_a.union(tokens_b)))


def evaluate_text(text: str, golden_text: str, is_conclusion: bool) -> dict[str, float]:
    """Score the generated text across multiple prompt criteria."""
    jaccard = _jaccard_similarity(text, golden_text)
    sentence_len = _get_sentence_length_score(text)
    intro_check = _check_intro_filler(text)
    
    # CTA check for conclusions
    cta_score = 1.0
    if is_conclusion:
        cta_score = 1.0 if "validex.com.au" in text.lower() else 0.0
        
    weighted_score = (
        jaccard * 0.3 +
        sentence_len * 0.3 +
        intro_check * 0.2 +
        cta_score * 0.2
    )
    
    return {
        "score": round(weighted_score, 4),
        "jaccard": round(jaccard, 4),
        "readability": round(sentence_len, 4),
        "no_fluff": round(intro_check, 4),
        "cta_match": round(cta_score, 4)
    }


# ── Optimizer Runner ────────────────────────────────────────────────

def run_prompt_optimization() -> dict[str, Any]:
    """Evaluate candidate templates on the benchmark query set using live LLM."""
    if not Path(BENCHMARK_PATH).exists() or not Path(GOLDENS_PATH).exists():
        logger.error("Benchmark or golden files missing.")
        return {}
        
    benchmark_cases = json.loads(Path(BENCHMARK_PATH).read_text(encoding="utf-8"))
    golden_cases = json.loads(Path(GOLDENS_PATH).read_text(encoding="utf-8"))
    golden_map = {g["id"]: g for g in golden_cases}
    
    # Import pipeline to run generation
    try:
        from app.langchain_pipeline import pipeline
        llm = pipeline._fast_llm or pipeline._llm
        if not llm:
            raise RuntimeError("LLM not initialized in pipeline.")
    except Exception as exc:
        logger.error("Could not import/initialize LLM: %s", exc)
        return {}
        
    candidates = {
        "Prompt_A": PROMPT_A,
        "Prompt_B": PROMPT_B,
        "Prompt_C": PROMPT_C,
    }
    
    results = {name: [] for name in candidates}
    
    for case in benchmark_cases:
        prompt_text = case["prompt"]
        golden_id = case["golden_id"]
        golden = golden_map.get(golden_id)
        if not golden:
            continue
            
        golden_text = golden["golden_draft"]
        is_conclusion = "checklist" not in prompt_text.lower()
        
        # Setup minimal scope/topic for rendering
        parsed = ParsedPrompt(
            raw_prompt=prompt_text,
            topic=golden["title"],
            intent="create_blog",
            tone="clear_professional",
            audience="general audience",
            length="medium"
        )
        heading = "Conclusion" if is_conclusion else "Employer Checklist"
        scope = {
            "focus": "police check requirements and validity criteria",
            "forbidden_overlap": ["candidate registration", "billing setup"]
        }
        context_text = golden_text  # Use golden text as context for clean generation
        role_rule = "- Focus on direct scenarios.\n"
        domain_pivot = ""
        
        for name, template in candidates.items():
            rendered = template.format(
                heading=heading,
                raw_prompt=parsed.raw_prompt,
                topic=parsed.topic,
                focus=scope["focus"],
                other_sections=", ".join(scope["forbidden_overlap"]),
                role_rule=role_rule,
                domain_pivot=domain_pivot,
                context_text=context_text
            )
            
            try:
                response = llm.invoke(rendered)
                gen_text = getattr(response, "content", str(response)).strip()
                scores = evaluate_text(gen_text, golden_text, is_conclusion)
                scores["case_id"] = case["id"]
                scores["generated_text"] = gen_text
                results[name].append(scores)
            except Exception as e:
                logger.warning("Generation failed for %s on case %s: %s", name, case["id"], e)
                
    # Calculate average scores
    summary = {}
    best_candidate = ""
    best_score = -1.0
    
    for name, case_results in results.items():
        if not case_results:
            continue
        avg_score = sum(r["score"] for r in case_results) / len(case_results)
        avg_jaccard = sum(r["jaccard"] for r in case_results) / len(case_results)
        avg_readability = sum(r["readability"] for r in case_results) / len(case_results)
        avg_fluff = sum(r["no_fluff"] for r in case_results) / len(case_results)
        
        summary[name] = {
            "average_overall": round(avg_score, 4),
            "average_jaccard": round(avg_jaccard, 4),
            "average_readability": round(avg_readability, 4),
            "average_no_fluff": round(avg_fluff, 4),
        }
        
        if avg_score > best_score:
            best_score = avg_score
            best_candidate = name
            
    report = {
        "best_candidate": best_candidate,
        "best_score": best_score,
        "summary": summary,
        "details": results
    }
    
    Path(REPORT_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(REPORT_OUTPUT_PATH).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Optimization report saved to %s", REPORT_OUTPUT_PATH)
    
    return report


if __name__ == "__main__":
    run_prompt_optimization()
