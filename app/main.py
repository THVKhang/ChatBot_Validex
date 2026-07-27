from app.langchain_pipeline import pipeline
import uuid
import re
import logging
from app.session_manager import SessionManager

_logger = logging.getLogger(__name__)

# ── Shared Post-Processing — used by both /api/chat and /api/chat/stream ──

_SYSTEM_PROMPT_FINGERPRINTS = [
    "CONFIDENTIALITY RULE",
    "YOUR ADAPTIVE IDENTITY",
    "Validex Australian Expert Writer",
    "JURISDICTION ISOLATION RULE",
    "HR-TO-TECH TRANSLATION RULE",
    "ANTI-REPETITION GUARDRAIL",
    "TOPIC ISOLATION RULE",
    "GROUNDING RULES",
    "CRITICAL FACTUAL CONSTRAINTS",
    "Intent-Adaptive Australian Expert",
    "OVERRIDE]: If Custom Instructions",
]


def _sanitize_ui_artifacts(text: str) -> str:
    """Clean web scraping UI artifacts, HTML button labels, and status badges.
    
    Catches full toolbar strings like:
      'edit_note description Word picture_as_pdf PDF html HTML refresh bookmark thumb_up thumb_down'
    that leak from gov website scraping via collect_au_sources.py.
    """
    if not text:
        return text
    
    # ── Pass 1a: Remove entire trailing UI toolbar blocks (multi-line format) ──
    # These are scraped navigation/toolbar elements that appear as a block
    cleaned = re.sub(
        r'(?:edit_note|picture_as_pdf|smart_toy|thumb_up|thumb_down)\s*'
        r'(?:description|Word|PDF|html|HTML|refresh|bookmark|thumb_up|thumb_down|edit_note|picture_as_pdf|smart_toy|person|\s)*$',
        '',
        text,
        flags=re.MULTILINE
    )
    
    # ── Pass 1b: Remove inline toolbar chains (-- description html refresh ...) ──
    # Older format: '-- description html refresh bookmark thumb_up thumb_down'
    cleaned = re.sub(
        r'--\s*(?:description|html|refresh|bookmark|thumb_up|thumb_down|share|like|dislike|rating|standard\s+pipeline|complex\s+pipeline)(?:\s+(?:description|html|refresh|bookmark|thumb_up|thumb_down|share|like|dislike|rating|standard\s+pipeline|complex\s+pipeline))*',
        '',
        cleaned,
        flags=re.IGNORECASE
    )
    
    # ── Pass 2: Remove individual Material Design icon names ──
    # These are Angular Material / Google icon names that appear when scraping web UIs
    # NOTE: 'description' omitted — too common in English prose. Caught by Pass 1 in toolbar context.
    cleaned = re.sub(
        r'\b(?:edit_note|picture_as_pdf|smart_toy|content_copy|open_in_new|'
        r'thumb_up|thumb_down|bookmark_border|'
        r'more_vert|more_horiz|'
        r'arrow_back|arrow_forward|navigate_next|navigate_before|'
        r'standard\s+pipeline|complex\s+pipeline)\b',
        '',
        cleaned,
        flags=re.IGNORECASE
    )
    
    # ── Pass 3: Remove standalone 'Word' / 'PDF' / 'HTML' tokens on their own line ──
    # These are export button labels, NOT real content words
    # Only remove when they appear as isolated tokens (not part of sentences)
    cleaned = re.sub(r'^\s*(?:Word|PDF|HTML)\s*$', '', cleaned, flags=re.MULTILINE)
    
    # ── Pass 4: Clean "References" section if it only contains UI junk ──
    # Pattern: "References\n\n[UI artifacts]" at end of text
    cleaned = re.sub(
        r'\n+References\s*\n+\s*$',
        '',
        cleaned
    )
    
    # ── Pass 5: Remove emoji status badges ──
    cleaned = re.sub(r'[\U0001F7E0-\U0001F7E4\U0001F44D\U0001F44E]', '', cleaned)
    
    # ── Pass 6: Clean up orphaned artifacts ──
    cleaned = re.sub(r'\n\s*--\s*\n', '\n', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Collapse 3+ blank lines → 2
    cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
    
    return cleaned.strip()


def sanitize_payload(payload: dict) -> dict:
    """Sanitize a generated payload: file paths, system prompt leaks, SEO analysis.

    This function MUST be called on every payload before it reaches the client,
    regardless of whether the request was streaming or non-streaming.
    """
    generated = payload.get("generated", {})
    draft = generated.get("draft", "")

    # 1. File Path & UI Artifact Sanitization
    draft = re.sub(r'file://[^\s\]\)]+', 'https://www.validex.com.au', draft)
    draft = re.sub(r'[A-Z]:\\\\[^\s\]\)]+', '', draft)
    draft = re.sub(r'[A-Z]:/Users/[^\s\]\)]+', '', draft)
    draft = _sanitize_ui_artifacts(draft)
    generated["draft"] = draft

    # 2. Sanitize sources_used — strip file:// paths & UI artifacts
    sources = generated.get("sources_used", [])
    sanitized_sources = []
    for src in sources:
        src_str = str(src)
        if "file://" in src_str or src_str.startswith("C:"):
            parts = src_str.replace("\\", "/").split("/")
            src_str = parts[-1] if parts else src_str
        src_str = _sanitize_ui_artifacts(src_str)
        if src_str:
            sanitized_sources.append(src_str)
    generated["sources_used"] = sanitized_sources

    # 3. System Prompt Leak Detection (CRITICAL SECURITY)
    draft = generated["draft"]
    leaked_count = sum(1 for fp in _SYSTEM_PROMPT_FINGERPRINTS if fp.lower() in draft.lower())
    if leaked_count >= 2:
        _logger.critical(
            "SECURITY: System prompt leak detected in output (%d fingerprints). Scrubbing draft.", leaked_count
        )
        generated["draft"] = (
            "# Content Generation\n\n"
            "I am a blog content specialist for Validex. "
            "I can help you create professional content about Australian police checks, "
            "background screening, and compliance topics.\n\n"
            "Please provide a topic and I'll generate a publication-ready article for you."
        )
        generated["title"] = "Content Generation"

    # 4. Native Python SEO Analysis (0 Tokens)
    from app.seo_optimizer import run_seo_analysis
    generated["seo"] = run_seo_analysis(
        title=generated.get("title", ""),
        draft=generated.get("draft", "")
    )

    payload["generated"] = generated
    return payload

def process_prompt(prompt: str, session: SessionManager, *, request_id: str | None = None, session_id: str | None = None, from_api: bool = False) -> dict:
    from app.graph import multi_agent_graph
    from app.semantic_cache import semantic_cache
    
    # Check Semantic Cache before doing any heavy lifting
    cached_payload = semantic_cache.search_cache(prompt)
    if cached_payload:
        _logger.info("SEMANTIC CACHE HIT! Bypassing LangGraph.")
        session.add_turn(
            prompt,
            "",
            parsed_intent=cached_payload.get("parsed", {}).get("intent", ""),
            parsed_topic=cached_payload.get("parsed", {}).get("topic", ""),
            generated_draft=cached_payload.get("generated", {}).get("draft", ""),
        )
        return cached_payload

    # Initialize the LangGraph state
    initial_state = {
        "prompt": prompt,
        "session": session,
        "request_id": request_id,
        "revision_count": 0,
        "loop_step": 0,
        "global_step_count": 0,
        "from_api": from_api
    }
    
    # Execute the Graph
    config = {
        "configurable": {
            "thread_id": str(request_id or uuid.uuid4())
        },
        "metadata": {
            "request_id": request_id,
            "session_id": session_id,
        },
        "tags": ["api_non_streaming"] if session_id else ["cli_mode"]
    }
    final_state = multi_agent_graph.invoke(initial_state, config=config)
    
    # Format the payload for backward compatibility with frontend/tests
    parsed_length = final_state.get("parsed", {}).get("length", "medium")
    token_plan = pipeline._token_budget_plan(parsed_length)
    
    retrieved_docs = final_state.get("retrieved_docs", [])
    doc_tokens = []
    for doc in retrieved_docs:
        content = ""
        if hasattr(doc, "page_content"):
            content = doc.page_content
        elif hasattr(doc, "content"):
            content = doc.content
        elif isinstance(doc, dict):
            content = doc.get("content", "") or doc.get("page_content", "")
        doc_tokens.append(pipeline._estimate_token_count(content))
    estimated_input_tokens = sum(doc_tokens)
    input_budget_sufficient = estimated_input_tokens >= token_plan.input_tokens_min
    estimated_output_tokens = pipeline._estimate_token_count(final_state.get("draft", ""))

    # Construct retrieval_meta
    retrieval_meta = final_state.get("retrieval_meta", {
        "status": "ok",
        "confidence": 1.0,
        "top_score": 100,
        "reason": "default decision status"
    })

    # Determine if external knowledge was used
    settings = pipeline.settings
    ret_status = retrieval_meta.get("status", "ok")
    external_knowledge_used = False
    if ret_status in {"low_confidence", "out_of_domain", "no_match", "web_search"}:
        if settings.allow_hybrid_fallback:
            external_knowledge_used = True

    formatted_retrieved = []
    for i, doc in enumerate(retrieved_docs):
        doc_id = f"doc_{i}"
        score = 0
        content = ""
        if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
            doc_id = doc.metadata.get("doc_id", doc_id)
            score = doc.metadata.get("score", 0)
        elif isinstance(doc, dict):
            m = doc.get("metadata", {}) or {}
            doc_id = doc.get("doc_id") or m.get("doc_id") or doc_id
            score = doc.get("score") or m.get("score") or 0

        if hasattr(doc, "page_content"):
            content = doc.page_content
        elif hasattr(doc, "content"):
            content = doc.content
        elif isinstance(doc, dict):
            content = doc.get("content", "") or doc.get("page_content", "")

        formatted_retrieved.append({
            "doc_id": doc_id,
            "score": score,
            "content": content,
            "snippet": content[:200]
        })

    payload = {
        "parsed": final_state.get("parsed", {}),
        "retrieved": formatted_retrieved,
        "retrieval_meta": retrieval_meta,
        "generated": {
            "title": final_state.get("title", ""),
            "outline": final_state.get("outline", []),
            "draft": final_state.get("draft", ""),
            "sources_used": final_state.get("sources_used", []),
            "evaluation": final_state.get("editor_evaluation") or {
                "relevance": 9,
                "coherence": 9,
                "factuality": 9,
                "overall": 9,
                "verdict": "ACCEPT",
                "issues": []
            }
        },
        # Map quality gate status
        "runtime": {
            "quality_gate_blocked": bool(final_state.get("editor_feedback")),
            "generation_mode": "multi-agent",
            "retrieval_mode": "hybrid",
            "external_knowledge_used": external_knowledge_used,
            "token_budget": {
                "length_profile": token_plan.length_profile,
                "output_tokens_target": token_plan.output_tokens,
                "output_tokens_estimated": estimated_output_tokens,
                "input_tokens_target_min": token_plan.input_tokens_min,
                "input_tokens_target": token_plan.input_tokens_target,
                "input_tokens_target_max": token_plan.input_tokens_max,
                "input_tokens_estimated": estimated_input_tokens,
                "input_budget_sufficient": input_budget_sufficient,
                "recommended_top_k": token_plan.recommended_top_k,
                "retrieved_docs": len(retrieved_docs),
                "context_docs_used": len(retrieved_docs),
            }
        }
    }


    payload = sanitize_payload(payload)
    
    # Save the generated response to Semantic Cache for future identical queries
    semantic_cache.save_cache(prompt, payload)

    session.add_turn(
        prompt,
        "",  # Keep compact storage for API/CLI shared path.
        parsed_intent=payload["parsed"].get("intent", ""),
        parsed_topic=payload["parsed"].get("topic", ""),
        generated_draft=payload["generated"]["draft"],
    )
    return payload


def run_once(prompt: str, session: SessionManager) -> str:
    payload = process_prompt(prompt, session)
    parsed = payload["parsed"]
    retrieved = payload["retrieved"]
    generated = payload["generated"]

    parsed_block = [
        "=== PARSED ===",
        f"intent: {parsed['intent']}",
        f"topic: {parsed['topic']}",
        f"audience: {parsed['audience']}",
        f"tone: {parsed['tone']}",
        f"length: {parsed['length']}",
    ]
    if parsed["context_note"]:
        parsed_block.append(parsed["context_note"])

    retrieved_block = ["=== RETRIEVED TOP DOCS ==="]
    if retrieved:
        retrieved_block.extend([f"- {item['doc_id']} (score={item['score']})" for item in retrieved])
    else:
        retrieved_block.append("- no relevant docs")

    result = "\n".join([
        *parsed_block,
        "",
        *retrieved_block,
        "",
        "=== GENERATED TITLE ===",
        generated["title"],
        "",
        "=== GENERATED OUTLINE ===",
        *[f"- {item}" for item in generated["outline"]],
        "",
        "=== GENERATED DRAFT ===",
        generated["draft"],
        "",
        "=== SOURCES USED ===",
        *(generated["sources_used"] or ["- none"]),
    ])

    # Keep rendered output for CLI history preview.
    if session.turns:
        session.turns[-1].assistant_output = result
    return result


def main() -> None:
    session = SessionManager()
    print("AI Blog Generator Prototype")
    print("Type 'exit' to quit.\n")

    while True:
        prompt = input("Prompt: ").strip()
        if not prompt:
            continue
        if prompt.lower() == "exit":
            break

        output = run_once(prompt, session)
        print(output)
        print("\n---\n")


if __name__ == "__main__":
    main()
