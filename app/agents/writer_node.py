import logging
from langchain_core.documents import Document
from app.graph_state import GraphState
from app.langchain_pipeline import pipeline
from app.parser import ParsedPrompt
from app.generator import GeneratedBlog

logger = logging.getLogger(__name__)

def writer_node(state: GraphState) -> GraphState:
    """Agent that writes the blog draft based on parsed intent and retrieved docs."""
    logger.info(f"Executing Writer Node (Revision {state.get('revision_count', 0)})")
    
    parsed_dict = state["parsed"]
    parsed = ParsedPrompt(
        raw_prompt=state["prompt"],
        intent=parsed_dict["intent"],
        topic=parsed_dict["topic"],
        audience=parsed_dict["audience"],
        tone=parsed_dict["tone"],
        length=parsed_dict["length"],
        custom_instructions=parsed_dict.get("context_note", "")
    )
    
    # Convert retrieved docs to Document objects for pipeline
    retrieved_docs = state.get("retrieved_docs", [])
    docs = [
        Document(
            page_content=d["content"],
            metadata={
                "doc_id": d["doc_id"],
                "score": d["score"],
                "source": d.get("source", "")
            }
        )
        for d in retrieved_docs
    ]
    
    # If there is editor feedback, prepend it to custom_instructions so LLM fixes it
    feedback = state.get("editor_feedback")
    if feedback:
        logger.info(f"Writer incorporating editor feedback: {feedback}")
        if parsed.custom_instructions:
            parsed.custom_instructions = f"EDITOR FEEDBACK: {feedback}\n\nORIGINAL INSTRUCTIONS: {parsed.custom_instructions}"
        else:
            parsed.custom_instructions = f"EDITOR FEEDBACK: {feedback}"
    
    # Generate draft using the existing LLM generator
    previous_draft = None # We could get this from session.latest_turn() if needed
    
    # Get previous draft if this is a rewrite/shorten request
    if parsed.intent in {"rewrite", "shorten"}:
        last_turn = state["session"].latest_turn()
        if last_turn:
            previous_draft = last_turn.generated_draft
            
    # Also pass the draft from previous revision if the Editor rejected it
    if state.get("draft") and state.get("revision_count", 0) > 0:
        previous_draft = state["draft"]
        
    llm_trace = {}
    generated: GeneratedBlog | None = pipeline._generate_with_llm(parsed, docs, previous_draft, llm_trace=llm_trace)
    
    if not generated:
        # Fallback to templates if LLM fails
        logger.warning("LLM Generation failed, using fallback templates")
        generated = pipeline._generate_with_fallback(parsed, docs, previous_draft)
        
    return {
        "title": generated.title,
        "outline": generated.outline,
        "draft": generated.draft,
        "sources_used": generated.sources_used
    }
