"""LLM-Powered Editor Node — evaluates draft quality with rubric scoring."""
import json
import logging
import re
from app.graph_state import GraphState

logger = logging.getLogger(__name__)


def _llm_evaluate_draft(draft: str, parsed: dict) -> dict:
    """Use LLM to evaluate draft quality on a 1-10 rubric.
    
    Returns dict with scores and verdict.
    """
    try:
        from app.langchain_pipeline import pipeline
    except Exception:
        return {"verdict": "ACCEPT", "overall": 7, "feedback": ""}

    if pipeline._llm is None:
        return {"verdict": "ACCEPT", "overall": 7, "feedback": ""}

    topic = parsed.get("topic", "unknown")
    audience = parsed.get("audience", "general audience")
    tone = parsed.get("tone", "professional")
    length = parsed.get("length", "medium")

    prompt = (
        "You are an editorial quality reviewer. Evaluate this blog draft on 5 criteria.\n\n"
        f"Topic: {topic}\n"
        f"Target audience: {audience}\n"
        f"Expected tone: {tone}\n"
        f"Expected length: {length}\n\n"
        f"Draft:\n{draft[:3000]}\n\n"
        "Rate each criterion 1-10 and provide a verdict.\n"
        "Return ONLY a JSON object:\n"
        "{\n"
        '  "relevance": <1-10>,       // Is content on-topic?\n'
        '  "accuracy": <1-10>,        // Are claims supported with citations?\n'
        '  "coherence": <1-10>,       // Do sections flow logically?\n'
        '  "completeness": <1-10>,    // Are key aspects covered?\n'
        '  "tone_match": <1-10>,      // Does tone match the audience?\n'
        '  "overall": <1-10>,         // Overall quality\n'
        '  "feedback": "<specific improvement suggestions>",\n'
        '  "verdict": "ACCEPT" or "REVISE" or "REJECT"\n'
        "}\n\n"
        "Use ACCEPT if overall >= 7, REVISE if 4-6, REJECT if < 4."
    )

    try:
        response = pipeline._llm.invoke(prompt)
        raw = getattr(response, "content", str(response))
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            result = json.loads(match.group(0))
            if "verdict" in result and "overall" in result:
                logger.info(
                    f"Editor LLM evaluation: overall={result['overall']}, "
                    f"verdict={result['verdict']}"
                )
                return result
    except Exception as exc:
        logger.warning(f"LLM editor evaluation failed: {exc}")

    return {"verdict": "ACCEPT", "overall": 7, "feedback": ""}


def editor_node(state: GraphState) -> GraphState:
    """LLM-powered editor: evaluates draft quality with rubric scoring."""
    logger.info("Executing Smart Editor Node")
    
    draft = state.get("draft", "")
    parsed = state.get("parsed", {})
    revision_count = state.get("revision_count", 0)
    
    word_count = len(draft.split())
    feedback_items = []
    
    # --- Rule-based checks (safety net) ---
    length_req = parsed.get("length", "medium")
    if length_req == "long" and word_count < 400:
        feedback_items.append("Draft is too short for 'long' requirement. Expand with more details (min 400 words).")
    elif length_req == "short" and word_count > 300:
        feedback_items.append("Draft is too long for 'short' requirement. Make it more concise (max 300 words).")
        
    if parsed.get("intent") == "create_blog" and "##" not in draft:
        feedback_items.append("Draft is missing markdown headings (##). Add proper section structure.")

    # --- LLM evaluation (if no rule-based issues) ---
    quality_score = None
    if not feedback_items and draft and revision_count < 2:
        evaluation = _llm_evaluate_draft(draft, parsed)
        quality_score = evaluation.get("overall", 7)
        verdict = evaluation.get("verdict", "ACCEPT")
        
        if verdict == "REVISE" and revision_count < 2:
            llm_feedback = evaluation.get("feedback", "")
            if llm_feedback:
                feedback_items.append(f"Quality review (score {quality_score}/10): {llm_feedback}")
        elif verdict == "REJECT" and revision_count < 2:
            llm_feedback = evaluation.get("feedback", "")
            feedback_items.append(f"Quality too low (score {quality_score}/10): {llm_feedback}")
    
    # --- Decision ---
    if feedback_items and revision_count < 3:
        combined_feedback = " ".join(feedback_items)
        logger.warning(f"Editor rejected draft: {combined_feedback}")
        return {
            "editor_feedback": combined_feedback,
            "revision_count": revision_count + 1,
            "quality_gate_blocked": bool(quality_score and quality_score < 5),
        }
        
    # Accept
    if quality_score:
        logger.info(f"Editor accepted draft (quality={quality_score}/10)")
    else:
        logger.info("Editor accepted draft")
        
    return {
        "editor_feedback": None,
        "revision_count": revision_count + 1,
        "quality_gate_blocked": False,
    }
