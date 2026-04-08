from dataclasses import replace

from app.config import settings
from app.generator import generate_blog_output
from app.parser import parse_user_input
from app.retriever import retrieve_with_guard
from app.session_manager import SessionManager


def process_prompt(prompt: str, session: SessionManager) -> dict:
    parsed = parse_user_input(prompt)

    context_note = ""
    effective_parsed = parsed
    previous_draft = None
    last_turn = session.latest_turn()

    if parsed.intent in {"rewrite", "shorten"} and parsed.topic == "current draft" and last_turn:
        effective_parsed = replace(parsed, topic=last_turn.parsed_topic or "current draft")
        previous_draft = last_turn.generated_draft or None
        context_note = "context: using previous draft"

    decision = retrieve_with_guard(
        effective_parsed.topic,
        settings.data_processed_dir,
        settings.top_k,
        settings.metadata_path,
        settings.min_top_score,
        settings.min_confidence,
    )
    docs = decision.docs

    if decision.status == "ok":
        generated = generate_blog_output(effective_parsed, docs, previous_draft=previous_draft)
    else:
        reason_map = {
            "out_of_domain": "Query hien tai nam ngoai pham vi dataset RAG hien co.",
            "low_confidence": "Do lien quan retrieval qua thap de tao draft an toan.",
            "no_match": "Khong tim thay tai lieu phu hop trong knowledge base.",
            "no_data": "He thong chua co du lieu processed de retrieval.",
        }
        generated = {
            "title": "Need More Context",
            "outline": [
                "Xac dinh lai chu de trong pham vi dataset",
                "Bo sung tai lieu lien quan vao data/raw",
                "Chay ingest de cap nhat data/processed va metadata",
                "Gui lai prompt cu the hon",
            ],
            "draft": (
                "Toi chua the tao draft dang tin cay cho prompt nay. "
                f"Ly do: {reason_map.get(decision.status, decision.reason)} "
                "Hay thu prompt cu the hon trong domain police check/recruitment/compliance, "
                "hoac cap nhat dataset RAG truoc khi tao noi dung."
            ),
            "sources_used": [],
        }

    payload = {
        "parsed": {
            "intent": parsed.intent,
            "topic": effective_parsed.topic,
            "audience": parsed.audience,
            "tone": parsed.tone,
            "length": parsed.length,
            "context_note": context_note,
        },
        "retrieved": [
            {
                "doc_id": doc.doc_id,
                "score": doc.score,
                "semantic_score": round(doc.semantic_score, 3),
                "snippet": doc.content[:220],
            }
            for doc in docs
        ],
        "retrieval_meta": {
            "status": decision.status,
            "confidence": round(decision.confidence, 3),
            "top_score": decision.top_score,
            "reason": decision.reason,
        },
        "generated": {
            "title": generated["title"] if isinstance(generated, dict) else generated.title,
            "outline": generated["outline"] if isinstance(generated, dict) else generated.outline,
            "draft": generated["draft"] if isinstance(generated, dict) else generated.draft,
            "sources_used": generated["sources_used"] if isinstance(generated, dict) else generated.sources_used,
        },
    }

    session.add_turn(
        prompt,
        "",  # Keep compact storage for API/CLI shared path.
        parsed_intent=parsed.intent,
        parsed_topic=effective_parsed.topic,
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
