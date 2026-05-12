from app.langchain_pipeline import pipeline
from app.session_manager import SessionManager


def process_prompt(prompt: str, session: SessionManager, *, request_id: str | None = None) -> dict:
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
        "revision_count": 0
    }
    
    # Execute the Graph
    final_state = multi_agent_graph.invoke(initial_state)
    
    # Format the payload for backward compatibility with frontend/tests
    payload = {
        "parsed": final_state.get("parsed", {}),
        "retrieved": final_state.get("retrieved_docs", []),
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
            "retrieval_mode": "hybrid"
        }
    }

    # 2. Hardcoded Regex Replacement (At the Absolute Exit Point)
    draft = payload["generated"]["draft"]
    draft = re.sub(r'(?i)\b(hiring|recruitment|candidate|onboarding|employee|recruiter|recruiters|sla|slas)\b', '[REDACTED_HR_TERM]', draft)
    payload["generated"]["draft"] = draft
    
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
