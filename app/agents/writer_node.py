"""Multi-Stage Writer Node — Plan → Draft → Self-Review pipeline."""
import json
import logging
import re
from langchain_core.documents import Document
from app.graph_state import GraphState
from app.langchain_pipeline import pipeline
from app.parser import ParsedPrompt
from app.generator import GeneratedBlog

logger = logging.getLogger(__name__)


def _plan_outline(parsed: ParsedPrompt, docs: list[Document]) -> list[dict]:
    """Stage 1: LLM generates a detailed outline with key points per section."""
    if pipeline._llm is None:
        return []

    doc_summaries = "\n".join([
        f"- [{d.metadata.get('title', 'Source')}]: {d.page_content[:200]}"
        for d in docs[:5]
    ])

    prompt = (
        f"You are planning a blog article about: {parsed.topic}\n"
        f"Audience: {parsed.audience} | Tone: {parsed.tone} | Length: {parsed.length}\n\n"
        "Available source material:\n"
        f"{doc_summaries}\n\n"
        "Create a detailed outline for this blog. Return ONLY a JSON array where each item has:\n"
        '- "heading": section heading\n'
        '- "key_points": array of 2-3 key points to cover\n'
        '- "relevant_sources": which sources to cite\n\n'
        "Include 4-6 sections (Introduction, 2-4 body sections, Conclusion).\n"
        "Make headings specific and compelling, not generic.\n"
        "Return ONLY the JSON array."
    )

    try:
        response = pipeline._llm.invoke(prompt)
        raw = getattr(response, "content", str(response))
        match = re.search(r"\[[\s\S]*\]", raw)
        if match:
            outline = json.loads(match.group(0))
            if isinstance(outline, list) and len(outline) >= 3:
                logger.info(f"Writer Plan: generated outline with {len(outline)} sections")
                return outline
    except Exception as exc:
        logger.warning(f"Writer Plan failed: {exc}")

    return []


def _self_review(draft: str, parsed: ParsedPrompt, docs: list[Document]) -> str:
    """Stage 3: LLM self-reviews and improves the draft."""
    if pipeline._llm is None or not draft:
        return draft

    doc_titles = [d.metadata.get("title", "Source") for d in docs[:5]]

    prompt = (
        "You are an editorial reviewer. Review this blog draft and improve it.\n\n"
        "Check for:\n"
        "1. ACCURACY: Are all claims supported by the available sources? Remove unsupported claims.\n"
        "2. CITATIONS: Does every factual statement have a [Source: ...] citation?\n"
        "3. COHERENCE: Do sections flow logically? Are transitions smooth?\n"
        "4. COMPLETENESS: Are all key aspects of the topic covered?\n"
        "5. TONE: Is the tone consistent and appropriate for the target audience?\n\n"
        f"Topic: {parsed.topic}\n"
        f"Audience: {parsed.audience}\n"
        f"Tone: {parsed.tone}\n"
        f"Available sources: {', '.join(doc_titles)}\n\n"
        f"Draft to review:\n{draft[:4000]}\n\n"
        "Return the IMPROVED version of the draft. Keep the same markdown format.\n"
        "Fix any issues you find. Do NOT add conversational commentary.\n"
        "Output ONLY the improved markdown blog post."
    )

    try:
        response = pipeline._llm.invoke(prompt)
        improved = getattr(response, "content", str(response)).strip()
        # Basic validation
        if len(improved) > len(draft) * 0.5 and "##" in improved:
            logger.info(f"Writer Self-Review: improved draft ({len(draft)} → {len(improved)} chars)")
            return improved
        logger.warning("Self-review output failed validation, keeping original")
    except Exception as exc:
        logger.warning(f"Writer Self-Review failed: {exc}")

    return draft


def writer_node(state: GraphState) -> GraphState:
    """Multi-stage writer: Plan → Draft → Self-Review."""
    logger.info(f"Executing Smart Writer Node (Revision {state.get('revision_count', 0)})")
    
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
    
    # Convert retrieved docs to Document objects
    retrieved_docs = state.get("retrieved_docs", [])
    docs = [
        Document(
            page_content=d["content"],
            metadata={
                "doc_id": d["doc_id"],
                "score": d["score"],
                "source": d.get("source", ""),
                "title": d.get("title", ""),
                "source_url": d.get("source_url", ""),
            }
        )
        for d in retrieved_docs
    ]
    
    # Handle editor feedback
    feedback = state.get("editor_feedback")
    if feedback:
        logger.info(f"Writer incorporating editor feedback: {feedback}")
        if parsed.custom_instructions:
            parsed.custom_instructions = f"EDITOR FEEDBACK: {feedback}\n\nORIGINAL INSTRUCTIONS: {parsed.custom_instructions}"
        else:
            parsed.custom_instructions = f"EDITOR FEEDBACK: {feedback}"
    
    # Get previous draft for continuation mode
    previous_draft = None
    last_turn = state["session"].latest_turn()
    if last_turn and last_turn.generated_draft:
        previous_draft = last_turn.generated_draft
        logger.info(f"Continuation mode: using previous draft ({len(previous_draft)} chars)")
    
    if state.get("draft") and state.get("revision_count", 0) > 0:
        previous_draft = state["draft"]
    
    # --- Stage 1: PLAN (only for new blogs, skip for rewrites) ---
    outline_plan = []
    if parsed.intent == "create_blog" and not feedback:
        outline_plan = _plan_outline(parsed, docs)
        if outline_plan:
            # Inject outline into custom_instructions so the LLM follows it
            outline_text = "\n".join([
                f"## {item.get('heading', 'Section')}\n"
                f"  Key points: {', '.join(item.get('key_points', []))}"
                for item in outline_plan
            ])
            plan_instruction = f"\n\nFOLLOW THIS OUTLINE STRUCTURE:\n{outline_text}"
            parsed.custom_instructions = (parsed.custom_instructions or "") + plan_instruction
    
    # --- Stage 2: DRAFT ---
    llm_trace = {}
    generated: GeneratedBlog | None = pipeline._generate_with_llm(
        parsed, docs, previous_draft, llm_trace=llm_trace
    )
    
    if not generated:
        logger.warning("LLM Generation failed, using fallback templates")
        generated = pipeline._generate_with_fallback(parsed, docs, previous_draft)
        return {
            "title": generated.title,
            "outline": generated.outline,
            "draft": generated.draft,
            "sources_used": generated.sources_used
        }
    
    # --- Stage 3: SELF-REVIEW (only for new blogs, skip for quick edits) ---
    final_draft = generated.draft
    if parsed.intent == "create_blog" and not feedback and len(final_draft) > 500:
        final_draft = _self_review(final_draft, parsed, docs)
    
    return {
        "title": generated.title,
        "outline": generated.outline,
        "draft": final_draft,
        "sources_used": generated.sources_used
    }
