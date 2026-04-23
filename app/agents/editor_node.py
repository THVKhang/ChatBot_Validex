import logging
from app.graph_state import GraphState

logger = logging.getLogger(__name__)

def editor_node(state: GraphState) -> GraphState:
    """Agent that reviews the draft and checks compliance and quality."""
    logger.info("Executing Editor Node")
    
    draft = state.get("draft", "")
    parsed = state.get("parsed", {})
    revision_count = state.get("revision_count", 0)
    
    word_count = len(draft.split())
    
    feedback = []
    
    # Check length compliance
    length_req = parsed.get("length", "medium")
    if length_req == "long" and word_count < 400:
        feedback.append("The draft is too short for a 'long' requirement. Please expand the content with more details, examples, and comprehensive explanations (minimum 400 words).")
    elif length_req == "short" and word_count > 300:
        feedback.append("The draft is too long for a 'short' requirement. Please summarize and make it more concise (maximum 300 words).")
        
    # Check if there are no sections when creating a blog
    if parsed.get("intent") == "create_blog" and "##" not in draft and not state.get("quality_gate_blocked"):
        feedback.append("The draft is missing proper markdown headings (##). Please structure the blog with clear sections.")
        
    if feedback and revision_count < 2:
        # Reject draft
        combined_feedback = " ".join(feedback)
        logger.warning(f"Editor rejected draft: {combined_feedback}")
        return {
            "editor_feedback": combined_feedback,
            "revision_count": revision_count + 1
        }
        
    # Accept draft or max revisions reached
    if feedback:
        logger.warning("Editor accepted draft despite flaws due to max revision limit.")
    else:
        logger.info("Editor accepted draft.")
        
    return {
        "editor_feedback": None,
        "revision_count": revision_count + 1
    }
