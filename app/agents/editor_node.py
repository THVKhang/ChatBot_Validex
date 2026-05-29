"""LLM-Powered Editor Node — hybrid quality gate with rubric scoring + SEO + readability.

Three evaluation layers:
  Layer 0: Code-based structural checks (0 LLM tokens)
  Layer 1: SEO + Readability checks (0 LLM tokens)  [NEW]
  Layer 2: Fast-accept for strong drafts (0 LLM tokens)
  Layer 3: LLM rubric scoring for ambiguous cases (~200 tokens)
"""
import json
import logging
import re
from typing import Any
from collections import Counter

from app.graph_state import GraphState
from app.langchain_pipeline import pipeline
from app.ab_test_logger import log_prompt_evaluation
from app.config import settings

logger = logging.getLogger(__name__)


# ── Flesch-Kincaid Readability (code-based, 0 LLM tokens) ──────────

def _count_syllables(word: str) -> int:
    """Approximate syllable count for English words."""
    word = word.lower().strip()
    if len(word) <= 3:
        return 1
    # Remove trailing 'e' (silent e)
    if word.endswith('e'):
        word = word[:-1]
    # Count vowel groups
    count = len(re.findall(r'[aeiouy]+', word))
    return max(1, count)


def _remove_quotes(text: str) -> str:
    """Remove text inside double or single quotes."""
    text = re.sub(r'"[^"]*"', '', text)
    text = re.sub(r"'[^']*'", '', text)
    return text


def _flesch_kincaid_grade(text: str) -> float:
    """Calculate Flesch-Kincaid Grade Level. Lower is easier to read.
    Target: 8-12 for general audience blogs.
    """
    clean_text = _remove_quotes(text)
    sentences = re.split(r'[.!?]+', clean_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if not sentences:
        return 0.0

    words = re.findall(r'[a-zA-Z]+', clean_text)
    if not words:
        return 0.0

    total_syllables = sum(_count_syllables(w) for w in words)
    avg_sentence_length = len(words) / len(sentences)
    avg_syllables_per_word = total_syllables / len(words)

    # Flesch-Kincaid Grade Level formula
    grade = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59
    return round(max(0, grade), 1)


def _flesch_reading_ease(text: str) -> float:
    """Calculate Flesch Reading Ease. Higher is easier. Target: 50-70 for blogs."""
    clean_text = _remove_quotes(text)
    sentences = re.split(r'[.!?]+', clean_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if not sentences:
        return 0.0

    words = re.findall(r'[a-zA-Z]+', clean_text)
    if not words:
        return 0.0

    total_syllables = sum(_count_syllables(w) for w in words)
    avg_sentence_length = len(words) / len(sentences)
    avg_syllables_per_word = total_syllables / len(words)

    score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    return round(max(0, min(100, score)), 1)


# ── SEO Checks (code-based, 0 LLM tokens) ──────────────────────────

def _check_seo(draft: str, topic: str) -> list[str]:
    """Run SEO quality checks. Returns list of issue strings."""
    issues = []
    topic_words = set(re.findall(r'\w+', topic.lower()))
    draft_lower = draft.lower()
    draft_words = re.findall(r'\w+', draft_lower)
    total_words = len(draft_words)

    if total_words == 0:
        return ["S01:empty-draft"]

    # S01: Keyword density (topic words should appear 1-3% of total)
    if topic_words:
        topic_mentions = sum(1 for w in draft_words if w in topic_words)
        density = topic_mentions / total_words * 100
        if density < 0.5:
            issues.append(f"S01:low-keyword-density ({density:.1f}%, target 1-3%)")
        elif density > 5.0:
            issues.append(f"S02:keyword-stuffing ({density:.1f}%, target 1-3%)")

    # S03: Heading hierarchy (should have H2s, optionally H3s)
    h1_count = len(re.findall(r'^# [^#]', draft, re.MULTILINE))
    h2_count = len(re.findall(r'^## [^#]', draft, re.MULTILINE))
    h3_count = len(re.findall(r'^### [^#]', draft, re.MULTILINE))

    if h1_count > 1:
        issues.append(f"S03:multiple-h1 ({h1_count} H1 tags, should be max 1)")
    if h2_count < 2:
        issues.append(f"S04:insufficient-h2 ({h2_count} sections, need >=3)")

    # S05: Topic mention in first paragraph
    first_para = draft[:500].split('\n\n')[0] if draft else ""
    first_para_lower = first_para.lower()
    topic_in_intro = any(w in first_para_lower for w in topic_words if len(w) > 3)
    if not topic_in_intro:
        issues.append("S05:topic-not-in-intro (main topic should appear in first paragraph)")

    # S06: Meta-length check — sections shouldn't be too short
    sections = re.split(r'^## ', draft, flags=re.MULTILINE)
    short_sections = [s for s in sections[1:] if len(s.split()) < 50]
    if short_sections and len(short_sections) > len(sections) // 2:
        issues.append(f"S06:thin-sections ({len(short_sections)} sections under 50 words)")

    return issues


# ── LLM Evaluation (uses _editor_llm for temperature divergence) ────

def _get_editor_llm():
    """Get the Editor-specific LLM (different model/temperature from Writer)."""
    try:
        from app.langchain_pipeline import pipeline
    except Exception:
        return None
    # Prefer dedicated editor LLM (breaks Debate Agent Problem)
    editor = getattr(pipeline, "_editor_llm", None)
    if editor is not None:
        return editor
    # Fallback to primary LLM
    return pipeline._llm


def _compare_drafts(old_draft: str, new_draft: str, topic: str) -> bool:
    """Comparison Gate: Only accept revision if it's measurably better.

    Uses the Editor LLM to compare old vs new draft.
    Returns True if new_draft is better, False if old_draft should be kept.
    """
    llm = _get_editor_llm()
    if llm is None or not old_draft or not new_draft:
        return True  # Can't compare, accept the revision

    prompt = (
        f"Compare these two blog drafts about '{topic}' and decide which is BETTER overall.\n"
        f"Consider: accuracy, coherence, completeness, readability, and engagement.\n\n"
        f"DRAFT A (original):\n{old_draft[:1200]}\n\n"
        f"DRAFT B (revised):\n{new_draft[:1200]}\n\n"
        "Reply with EXACTLY one line: 'A' or 'B' followed by a brief reason.\n"
        "Example: B — better flow and more specific technical details"
    )

    try:
        response = llm.invoke(prompt)
        raw = getattr(response, "content", str(response)).strip()
        choice = raw.strip().upper()[:1]
        logger.info(f"Comparison Gate: chose Draft {'B (revised)' if choice == 'B' else 'A (original)'} — {raw[:80]}")
        return choice == "B"
    except Exception as exc:
        logger.warning(f"Comparison Gate failed: {exc}, accepting revision by default")
        return True


def _llm_evaluate_draft(draft: str, parsed: dict) -> dict:
    """Use Editor LLM to evaluate draft quality with detailed rubric.

    NOTE: Uses _editor_llm (different temperature/model from Writer) to break
    the Debate Agent Problem — the evaluator has a genuinely different perspective.

    Returns dict with scores and verdict.
    """
    llm = _get_editor_llm()
    if llm is None:
        return {"verdict": "ACCEPT", "overall": 7, "feedback": ""}

    topic = parsed.get("topic", "unknown")
    audience = parsed.get("audience", "general audience")
    tone = parsed.get("tone", "professional")

    prompt = (
        f"Rate this blog draft (topic: {topic}, audience: {audience}, tone: {tone}).\n\n"
        f"Draft:\n{draft[:1500]}\n\n"
        "Score each dimension 1-10, then give verdict.\n"
        "Return EXACTLY 3 lines:\n"
        "Line 1: relevance_score|coherence_score|factuality_score\n"
        "Line 2: verdict (ACCEPT if avg>=7, REVISE if 4-6, REJECT if <4)\n"
        "Line 3: brief issues (or 'none')\n\n"
        "Example:\n"
        "8|7|8\n"
        "ACCEPT\n"
        "none\n\n"
        "Example:\n"
        "5|4|6\n"
        "REVISE\n"
        "repetitive intro, missing conclusion"
    )

    try:
        response = llm.invoke(prompt)
        raw = getattr(response, "content", str(response)).strip()

        lines = [l.strip() for l in raw.strip().split('\n') if l.strip()]

        if len(lines) >= 2:
            # Parse scores from line 1
            scores_line = lines[0].strip('`"\' ')
            score_parts = scores_line.split('|')
            if len(score_parts) >= 3:
                try:
                    relevance = int(score_parts[0].strip())
                    coherence = int(score_parts[1].strip())
                    factuality = int(score_parts[2].strip())
                    overall = round((relevance + coherence + factuality) / 3)

                    verdict = lines[1].strip().upper()
                    if verdict not in ("ACCEPT", "REVISE", "REJECT"):
                        verdict = "ACCEPT" if overall >= 7 else ("REVISE" if overall >= 4 else "REJECT")

                    issues_text = lines[2] if len(lines) >= 3 else "none"
                    issues = [i.strip() for i in issues_text.split(',') if i.strip() and i.strip() != 'none']

                    logger.info(
                        f"Editor LLM: rel={relevance} coh={coherence} fact={factuality} "
                        f"overall={overall} verdict={verdict}"
                    )
                    return {
                        "relevance": relevance,
                        "coherence": coherence,
                        "factuality": factuality,
                        "overall": overall,
                        "verdict": verdict,
                        "issues": issues,
                    }
                except (ValueError, IndexError):
                    pass

        # Fallback: try old pipe-delimited format
        clean = raw.strip('`"\' \n')
        parts = clean.split('|')
        if len(parts) >= 5:
            try:
                result = {
                    "relevance": int(parts[0].strip()),
                    "coherence": int(parts[1].strip()),
                    "overall": int(parts[2].strip()),
                    "verdict": parts[3].strip().upper(),
                    "issues": [i.strip() for i in parts[4].split(',') if i.strip() and i.strip() != 'none'],
                }
                if result["verdict"] in ("ACCEPT", "REVISE", "REJECT"):
                    return result
            except (ValueError, IndexError):
                pass

    except Exception as exc:
        logger.warning(f"LLM editor evaluation failed: {exc}")

    return {"verdict": "ACCEPT", "overall": 7, "feedback": ""}


# ── Main Editor Node ────────────────────────────────────────────────

from app.agents.base import BaseAgentNode

class EditorAgentNode(BaseAgentNode):
    def execute(self, state: GraphState) -> GraphState:
        """Hybrid editor: code quality gate + SEO/readability + LLM for ambiguous cases."""
        logger.info("Executing Hybrid Editor Node")
    
        draft = state.get("draft", "")
        parsed = state.get("parsed", {})
        revision_count = state.get("revision_count", 0)
        loop_step = state.get("loop_step", 0) + 1
    
        word_count = len(draft.split())
        heading_count = draft.count('## ')
        feedback_items = []
    
        # ── Layer 0: Code-based structural checks (0 LLM tokens) ──────
        length_req = parsed.get("length", "medium")
        if length_req == "long" and word_count < 400:
            feedback_items.append("E03:too-short (min 400 words for 'long')")
        elif length_req == "short" and word_count > 300:
            feedback_items.append("E07:too-long (max 300 words for 'short')")
    
        if parsed.get("intent") == "create_blog" and heading_count < 2:
            feedback_items.append("E04:insufficient-headings (need ## sections)")
    
        # Paragraph balance check
        paragraphs = [p.strip() for p in draft.split('\n\n') if p.strip() and not p.strip().startswith('#')]
        if paragraphs:
            short_paras = [p for p in paragraphs if len(p.split()) < 15]
            if len(short_paras) > len(paragraphs) * 0.5:
                feedback_items.append("E05:thin-paragraphs (multiple sections too short)")
    
        # Repetition detection
        words = draft.lower().split()
        if len(words) > 50:
            ngrams = [' '.join(words[i:i+4]) for i in range(len(words)-3)]
            repeated = [t for t, c in Counter(ngrams).items() if c >= 4 and len(t) > 12]
            if repeated:
                feedback_items.append(f"E01:repetition ({len(repeated)} phrases repeated 4+ times)")
    
        # Conclusion check
        if parsed.get("intent") == "create_blog" and 'conclusion' not in draft.lower():
            feedback_items.append("E06:no-conclusion")
    
        # ── Layer 1: SEO + Readability checks (0 LLM tokens) [NEW] ────
        seo_issues = _check_seo(draft, parsed.get("topic", ""))
        if seo_issues:
            logger.info(f"Editor SEO: {seo_issues}")
            # Only add SEO issues on first pass (don't block revisions for SEO)
            if revision_count == 0:
                feedback_items.extend(seo_issues)
    
        # Readability check (English only — FK is not valid for other languages)
        detected_language = parsed.get("language", "en")
        if word_count > 50 and detected_language == "en":
            fk_grade = _flesch_kincaid_grade(draft)
            reading_ease = _flesch_reading_ease(draft)
            logger.info(f"Editor Readability: FK Grade={fk_grade}, Reading Ease={reading_ease}")
    
            max_iters = getattr(settings, "agent_max_iterations", 4)
            fk_limit = 12 if revision_count < max_iters - 1 else 14
            
            if fk_grade > fk_limit:
                if revision_count >= max_iters - 1:
                    logger.warning(f"Draft accepted with high FK Grade {fk_grade} due to max iterations. Needs Human Review.")
                    # We do not append to feedback_items to allow it to pass, but flag it
                    # Assuming draft is modified or meta_tags updated elsewhere. 
                    # Since we can't easily modify meta_tags here, we just let it pass Layer 1.
                else:
                    feedback_items.append(f"R01:too-complex (FK Grade {fk_grade}, target 8-{fk_limit})")
                    
            if reading_ease < 30:
                if revision_count >= max_iters - 1:
                    logger.warning(f"Draft accepted with low Reading Ease {reading_ease} due to max iterations.")
                else:
                    feedback_items.append(f"R02:poor-readability (Flesch {reading_ease}, target 50-70)")
    
        # ── Layer 2: Fast-accept for strong drafts (0 LLM tokens) ─────
        # If code checks found clear issues, skip fast-accept
        if feedback_items:
            logger.info("Editor: code-gate found issues (0 LLM tokens): %s", feedback_items)
            quality_score = None
        # Strong draft with no structural issues → accept immediately
        elif word_count > 500 and heading_count >= 4 and parsed.get("intent") == "create_blog":
            logger.info("Editor: code-gate ACCEPTED draft (0 LLM tokens, %d words, %d headings)", word_count, heading_count)
            log_prompt_evaluation(
                prompt_version="v4-hybrid-seo",
                topic=parsed.get("topic", "unknown"),
                editor_verdict="ACCEPTED",
                structural_issues=[],
                total_tokens_used=0
            )
            return {
                "editor_feedback": None,
                "revision_count": revision_count + 1,
                "quality_gate_blocked": False,
                "loop_step": loop_step,
            }
        # ── Layer 2.5: Comparison Gate (on revisions) ───────────────────
        if revision_count > 0 and not feedback_items:
            previous_draft = state.get("previous_draft")
            if previous_draft and draft:
                is_better = _compare_drafts(previous_draft, draft, parsed.get("topic", ""))
                if not is_better:
                    logger.info("Comparison Gate: revision NOT better than original, keeping original")
                    # Accept original and stop revision loop
                    log_prompt_evaluation(
                        prompt_version="v4-hybrid-seo",
                        topic=parsed.get("topic", "unknown"),
                        editor_verdict="ACCEPTED_GATE",
                        structural_issues=[],
                        total_tokens_used=0
                    )
                    return {
                        "editor_feedback": None,
                        "revision_count": revision_count + 1,
                        "quality_gate_blocked": False,
                        "draft": previous_draft,  # Restore original draft
                        "loop_step": loop_step,
                    }
    
        # ── Layer 3: LLM rubric for ambiguous cases ────────────────────
        else:
            quality_score = None
            if draft and revision_count < 2:
                evaluation = _llm_evaluate_draft(draft, parsed)
                quality_score = evaluation.get("overall", 7)
                verdict = evaluation.get("verdict", "ACCEPT")
    
                if verdict == "REVISE" and revision_count < 2:
                    issues = evaluation.get("issues", [])
                    llm_feedback = evaluation.get("feedback", "") or ", ".join(issues) if issues else ""
                    if llm_feedback:
                        feedback_parts = [f"Quality review (score {quality_score}/10)"]
                        if evaluation.get("relevance"):
                            feedback_parts.append(f"relevance={evaluation['relevance']}")
                        if evaluation.get("coherence"):
                            feedback_parts.append(f"coherence={evaluation['coherence']}")
                        if evaluation.get("factuality"):
                            feedback_parts.append(f"factuality={evaluation['factuality']}")
                        feedback_parts.append(f": {llm_feedback}")
                        feedback_items.append(" ".join(feedback_parts))
                elif verdict == "REJECT" and revision_count < 2:
                    issues = evaluation.get("issues", [])
                    llm_feedback = evaluation.get("feedback", "") or ", ".join(issues) if issues else ""
                    feedback_items.append(f"Quality too low (score {quality_score}/10): {llm_feedback}")
    
        # --- Decision ---
        if feedback_items and revision_count < 3:
            combined_feedback = " | ".join(feedback_items)
            logger.warning(f"Editor rejected draft: {combined_feedback}")
    
            # Determine strictness of rejection for logging
            has_structural = any(i.startswith("E") for i in feedback_items)
            has_seo = any(i.startswith("S") for i in feedback_items)
            has_readability = any(i.startswith("R") for i in feedback_items)
    
            if has_structural:
                verdict = "REJECTED"
            elif has_seo or has_readability:
                verdict = "REJECTED_QUALITY"
            else:
                verdict = "REJECTED_LLM"
    
            structural_codes = [i.split(":")[0] for i in feedback_items if i[0] in "ESR"]
    
            log_prompt_evaluation(
                prompt_version="v4-hybrid-seo",
                topic=parsed.get("topic", "unknown"),
                editor_verdict=verdict,
                structural_issues=structural_codes,
                total_tokens_used=0
            )
    
            return {
                "editor_feedback": combined_feedback,
                "revision_count": revision_count + 1,
                "quality_gate_blocked": bool(quality_score and quality_score < 5),
                "loop_step": loop_step,
            }
    
        # Accept
        if quality_score:
            logger.info(f"Editor accepted draft (quality={quality_score}/10)")
        else:
            logger.info("Editor accepted draft")
    
        log_prompt_evaluation(
            prompt_version="v4-hybrid-seo",
            topic=parsed.get("topic", "unknown"),
            editor_verdict="LLM_PASSED" if quality_score else "ACCEPTED",
            structural_issues=[],
            total_tokens_used=0
        )
    
        return {
            "editor_feedback": None,
            "revision_count": revision_count + 1,
            "quality_gate_blocked": False,
            "loop_step": loop_step,
        }

editor_node = EditorAgentNode()

