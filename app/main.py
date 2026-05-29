from app.langchain_pipeline import pipeline
import uuid
from app.session_manager import SessionManager


def process_prompt(prompt: str, session: SessionManager, *, request_id: str | None = None, session_id: str | None = None, from_api: bool = False) -> dict:
    from app.graph import multi_agent_graph
    import re
    
    # 1. Global Topic Sanitizer (At the API/Pipeline Entry Point)
    hr_keywords = ["hiring", "recruitment", "candidate", "onboarding", "sla", "turnaround", "employee"]
    if any(kw in prompt.lower() for kw in hr_keywords):
        print("⚠️ INTERCEPTOR TRIGGERED: HR Topic Detected!")
        prompt = "Database Scalability, API Polling Rate Limits, and System Latency in National Identity Infrastructure"
    
    from app.semantic_cache import semantic_cache
    
    # Check Semantic Cache before doing any heavy lifting
    cached_payload = semantic_cache.search_cache(prompt)
    if cached_payload:
        print("⚡ SEMANTIC CACHE HIT! Bypassing LangGraph.")
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
            "sources_used": final_state.get("sources_used", [])
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

    # 2. Hardcoded Regex Replacement (At the Absolute Exit Point)
    draft = payload["generated"]["draft"]
    draft = re.sub(r'(?i)\b(hiring|recruitment|candidate|onboarding|employee|recruiter|recruiters|sla|slas)\b', '[REDACTED_HR_TERM]', draft)
    # Mask the tag into natural language so users never see it
    draft = re.sub(r'\[REDACTED_HR_TERM\]-based', 'operational', draft)
    draft = re.sub(r'\[REDACTED_HR_TERM\]s?', 'operational', draft)
    draft = draft.replace('[REDACTED_HR_TERM]', 'operational')
    payload["generated"]["draft"] = draft
    
    # 3. Native Python SEO Analysis (0 Tokens)
    from app.seo_optimizer import run_seo_analysis
    payload["generated"]["seo"] = run_seo_analysis(
        title=payload["generated"]["title"],
        draft=payload["generated"]["draft"]
    )
    
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
    print("Nhap 'exit' de thoat.\n")

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
