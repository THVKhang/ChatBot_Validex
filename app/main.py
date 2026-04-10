from app.langchain_pipeline import pipeline
from app.session_manager import SessionManager


def process_prompt(prompt: str, session: SessionManager) -> dict:
    previous_topic = None
    previous_draft = None
    last_turn = session.latest_turn()
    if last_turn:
        previous_topic = last_turn.parsed_topic or None
        previous_draft = last_turn.generated_draft or None

    payload = pipeline.run(
        prompt,
        previous_turn_topic=previous_topic,
        previous_draft=previous_draft,
    )

    session.add_turn(
        prompt,
        "",  # Keep compact storage for API/CLI shared path.
        parsed_intent=payload["parsed"]["intent"],
        parsed_topic=payload["parsed"]["topic"],
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
