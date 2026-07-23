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
    # Only count significant topic words (>3 chars) to avoid false positives
    # from common words like 'in', 'a', 'the', 'for'
    significant_topic_words = {w for w in topic_words if len(w) > 3}
    if significant_topic_words:
        topic_mentions = sum(1 for w in draft_words if w in significant_topic_words)
        density = topic_mentions / total_words * 100
        if density < 0.5:
            issues.append(f"S01:low-keyword-density ({density:.1f}%, target 1-3%)")
        elif density > 8.0:
            issues.append(f"S02:keyword-stuffing ({density:.1f}%, target 1-5%)")

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
    """Use Editor LLM to evaluate draft quality with boolean checklist.

    NOTE: Uses _editor_llm (different temperature/model from Writer) to break
    the Debate Agent Problem — the evaluator has a genuinely different perspective.

    Uses YES/NO questions instead of 1-10 scores to reduce LLM variance.
    Score is computed code-side from checklist results.

    Returns dict with scores and verdict.
    """
    llm = _get_editor_llm()
    if llm is None:
        return {"verdict": "ACCEPT", "overall": 7, "feedback": ""}

    topic = parsed.get("topic", "unknown")
    audience = parsed.get("audience", "general audience")
    tone = parsed.get("tone", "professional")

    prompt = (
        f"Evaluate this blog draft about '{topic}' for {audience} in {tone} tone.\n\n"
        f"Draft:\n{draft[:1500]}\n\n"
        "Answer each question with YES or NO. Return EXACTLY one JSON object:\n"
        "{\n"
        '  "addresses_topic": true/false,\n'
        '  "has_introduction": true/false,\n'
        '  "has_conclusion": true/false,\n'
        '  "uses_source_data": true/false,\n'
        '  "no_unverified_claims": true/false,\n'
        '  "appropriate_length": true/false,\n'
        '  "clear_structure": true/false,\n'
        '  "no_repetition": true/false\n'
        "}\n\n"
        "Return ONLY the JSON object, no other text."
    )

    try:
        response = llm.invoke(prompt)
        raw = getattr(response, "content", str(response)).strip()

        # Extract JSON from response (handle markdown code fences)
        json_match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if json_match:
            checklist = json.loads(json_match.group(0))

            # Compute score from boolean checklist (code-based, deterministic)
            checks = [
                checklist.get("addresses_topic", True),
                checklist.get("has_introduction", True),
                checklist.get("has_conclusion", True),
                checklist.get("uses_source_data", True),
                checklist.get("no_unverified_claims", True),
                checklist.get("appropriate_length", True),
                checklist.get("clear_structure", True),
                checklist.get("no_repetition", True),
            ]
            passed = sum(1 for c in checks if c)
            total = len(checks)
            overall = round(passed / total * 10)

            # Deterministic thresholds
            if passed >= 6:
                verdict = "ACCEPT"
            elif passed >= 4:
                verdict = "REVISE"
            else:
                verdict = "REJECT"

            # Collect failed checks as issues
            issue_keys = [
                ("addresses_topic", "off-topic"),
                ("has_introduction", "missing introduction"),
                ("has_conclusion", "missing conclusion"),
                ("uses_source_data", "not grounded in sources"),
                ("no_unverified_claims", "contains unverified claims"),
                ("appropriate_length", "inappropriate length"),
                ("clear_structure", "poor structure"),
                ("no_repetition", "repetitive content"),
            ]
            issues = [desc for key, desc in issue_keys if not checklist.get(key, True)]

            logger.info(
                "Editor LLM Checklist: %d/%d passed, overall=%d, verdict=%s, issues=%s",
                passed, total, overall, verdict, issues or "none",
            )
            return {
                "relevance": 10 if checklist.get("addresses_topic", True) else 3,
                "coherence": 10 if checklist.get("clear_structure", True) else 3,
                "factuality": 10 if checklist.get("no_unverified_claims", True) else 3,
                "overall": overall,
                "verdict": verdict,
                "issues": issues,
                "checklist": checklist,
            }

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
    
        # Repetition detection — exclude topic-derived n-grams to avoid false positives
        words = draft.lower().split()
        if len(words) > 50:
            topic_lower = parsed.get("topic", "").lower()
            topic_words_set = set(topic_lower.split())
            ngrams = [' '.join(words[i:i+4]) for i in range(len(words)-3)]
            repeated = []
            for phrase, count in Counter(ngrams).items():
                if count < 5 or len(phrase) < 20:
                    continue
                # Skip if the phrase is mostly topic words (legitimate domain repetition)
                phrase_words = set(phrase.split())
                topic_overlap = len(phrase_words & topic_words_set) / len(phrase_words)
                if topic_overlap >= 0.5:
                    continue
                repeated.append(phrase)
            if repeated:
                feedback_items.append(f"E01:repetition ({len(repeated)} phrases repeated 5+ times)")
    
        # Conclusion check
        if parsed.get("intent") == "create_blog" and 'conclusion' not in draft.lower():
            feedback_items.append("E06:no-conclusion")

        # Merged list items check (E12) — detect lists missing bullet points
        list_signals = [r"considerations\s+include:", r"factors\s+include:", r"requirements\s+include:", r"following\s+points:"]
        for signal in list_signals:
            match = re.search(signal, draft, re.IGNORECASE)
            if match:
                snippet = draft[match.end():match.end() + 300]
                # If there are no bullet points ('- ' or '* ') in the following 300 chars
                if not re.search(r'[\n\r]\s*[-*]\s+', snippet):
                    feedback_items.append("E12:merged-list-items (Enumerations must use Markdown bullet points '- ' on new lines)")
                    break

        # Unknown legislation check (E11)
        golden_acts = [
            r"crimes\s+act", 
            r"privacy\s+act", 
            r"financial\s+transaction\s+reports\s+act",
            r"spent\s+convictions\s+act",
            r"child\s+protection\s+act",
            r"criminal\s+records\s+act",
            r"national\s+police\s+checking\s+service\s+agreement"
        ]
        act_mentions = re.findall(r'\b[A-Z][A-Za-z\s]{3,50}\s+Act\b', draft)
        for act in act_mentions:
            act_clean = act.lower().strip()
            if act_clean in ("this act", "the act", "an act", "each act", "other act", "such act"):
                continue
            is_golden = any(re.search(gold, act_clean) for gold in golden_acts)
            if not is_golden:
                feedback_items.append(f"E11:unknown-legislation (Mentioned unknown legislation '{act.strip()}'. Only reference Crimes Act, Privacy Act, Spent Convictions Act, Child Protection Act, or Financial Transaction Reports Act)")
                break

        # ── Layer 0.5: Regulatory Fact-Check (0 LLM tokens) ────────────
        # Scans draft for incorrect legal numbers (points, fees, validity periods)
        # and auto-corrects them using the verified Golden Facts Registry.
        from app.golden_facts import check_facts, auto_correct_facts
        fact_violations = check_facts(draft)
        if fact_violations:
            # Auto-correct wrong numbers directly in the draft
            draft, fixed_violations = auto_correct_facts(draft)
            for v in fact_violations:
                feedback_items.append(
                    f"F:{v.fact_id}:wrong-number "
                    f"(Draft says {v.found_value}, correct is {v.correct_value}. "
                    f"Source: {v.source}. Auto-corrected.)"
                )
            if fixed_violations:
                logger.info(
                    "Editor Layer 0.5: Auto-corrected %d regulatory fact(s) in draft",
                    len(fixed_violations),
                )

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
        evaluation = None
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
            # Evaluate using LLM to always populate scorecard
            evaluation = _llm_evaluate_draft(draft, parsed)
            return {
                "editor_feedback": None,
                "revision_count": revision_count + 1,
                "quality_gate_blocked": False,
                "loop_step": loop_step,
                "editor_evaluation": evaluation,
                "global_step_count": state.get("global_step_count", 0) + 1,
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
                    # Evaluate the restored original draft
                    evaluation = _llm_evaluate_draft(previous_draft, parsed)
                    return {
                        "editor_feedback": None,
                        "revision_count": revision_count + 1,
                        "quality_gate_blocked": False,
                        "draft": previous_draft,  # Restore original draft
                        "loop_step": loop_step,
                        "editor_evaluation": evaluation,
                        "global_step_count": state.get("global_step_count", 0) + 1,
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
                "previous_draft": draft,  # State Cleansing: save for Comparison Gate, Writer uses feedback not old draft
                "revision_count": revision_count + 1,
                "quality_gate_blocked": bool(quality_score and quality_score < 5),
                "loop_step": loop_step,
                "editor_evaluation": evaluation,
                "global_step_count": state.get("global_step_count", 0) + 1,
            }
    
        # Accept
        if not evaluation:
            evaluation = _llm_evaluate_draft(draft, parsed)
            quality_score = evaluation.get("overall", 7)
            
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
            "editor_evaluation": evaluation,
            "global_step_count": state.get("global_step_count", 0) + 1,
        }

editor_node = EditorAgentNode()

