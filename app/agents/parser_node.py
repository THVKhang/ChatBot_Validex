"""Smart Parser Node — uses LLM for intent detection with regex fallback."""
import json
import logging
from app.llm.provider import extract_text
import re
from app.graph_state import GraphState
from app.parser import parse_prompt, ParsedPrompt

logger = logging.getLogger(__name__)


def _llm_parse_intent(prompt: str) -> dict | None:
    """Use LLM to parse intent, topic, tone, audience, length from prompt.
    
    Returns a dict with parsed fields, or None if LLM fails.
    """
    try:
        from app.langchain_pipeline import pipeline
    except Exception:
        return None

    llm_to_use = getattr(pipeline, "_fast_llm", pipeline._llm) or pipeline._llm
    if llm_to_use is None:
        return None

    # Only send the first line (user message) — strip editorial settings
    user_message = prompt.split("\n")[0].strip()
    if not user_message:
        return None

    system = (
        "Parse user intent for a blog system.\n"
        "Return ONLY one line: intent|topic|tone|audience|length\n"
        "Example: create_blog|police check|professional|general audience|medium"
    )

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        response = llm_to_use.invoke([
            SystemMessage(content=system),
            HumanMessage(content=f'"{user_message}"'),
        ])
        raw = extract_text(response).strip()
        
        # Try pipe-delimited parse first
        # Strip any surrounding quotes/backticks
        clean = raw.strip('`"\' \n')
        parts = clean.split('|')
        if len(parts) >= 5:
            parsed = {
                "intent": parts[0].strip(),
                "topic": parts[1].strip(),
                "tone": parts[2].strip(),
                "audience": parts[3].strip(),
                "length": parts[4].strip(),
            }
            if parsed["intent"] in ("create_blog", "rewrite", "shorten") and parsed["topic"]:
                logger.info(f"LLM parser (pipe): intent={parsed['intent']}, topic={parsed['topic']}")
                return parsed
        
        # Fallback: try JSON parse
        match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            if "intent" in parsed and "topic" in parsed:
                logger.info(f"LLM parser (json fallback): intent={parsed['intent']}, topic={parsed['topic']}")
                return parsed
    except Exception as exc:
        logger.warning(f"LLM parse failed: {exc}")

    return None


from app.agents.base import BaseAgentNode

class ParserAgentNode(BaseAgentNode):
    def execute(self, state: GraphState) -> GraphState:
        """Agent that parses the raw prompt to extract intent, topic, audience, tone, length."""
        logger.info("Executing Parser Node")
        
        prompt = state["prompt"]
        session = state["session"]
        
        # --- Regex-first: only call LLM when regex result is ambiguous ---
        regex_parsed = parse_prompt(prompt)
        needs_llm = (not regex_parsed.topic or regex_parsed.topic == "current draft")
        
        if needs_llm:
            llm_result = _llm_parse_intent(prompt)
        else:
            llm_result = None
        
        if llm_result:
            parsed_prompt = ParsedPrompt(
                raw_prompt=prompt,
                intent=llm_result.get("intent", regex_parsed.intent),
                topic=llm_result.get("topic", regex_parsed.topic),
                audience=regex_parsed.audience,
                tone=llm_result.get("tone", regex_parsed.tone),
                length=llm_result.get("length", regex_parsed.length),
                language=regex_parsed.language,
                target_sections=regex_parsed.target_sections,
                target_images=regex_parsed.target_images,
                custom_instructions=regex_parsed.custom_instructions,
                modifiers=regex_parsed.modifiers,
            )
            logger.info("Parser: LLM assisted (regex was ambiguous)")
        else:
            parsed_prompt = regex_parsed
            logger.info("Parser: regex-only (no LLM needed)")
        
        # --- Continuation Mode ---
        # If this session already has a previous turn with a generated draft,
        # treat ALL follow-up prompts as refinements of the existing draft
        # (like ChatGPT/Gemini: same chat = refine, new chat = fresh start).
        last_turn = session.latest_turn()
        has_previous_draft = last_turn and last_turn.generated_draft
        edit_instruction = None  # Will be set if this is an edit request
        
        if has_previous_draft:
            # Preserve the original topic from the first turn in this session
            first_topic = session.turns[0].parsed_topic if session.turns[0].parsed_topic else last_turn.parsed_topic
            if first_topic and parsed_prompt.topic in ("current draft", ""):
                parsed_prompt.topic = first_topic
                logger.info(f"Continuation mode: keeping topic '{first_topic}' from session")
            
            # Force rewrite intent so the writer includes the previous draft as context
            if parsed_prompt.intent == "create_blog":
                parsed_prompt.intent = "rewrite"
                logger.info("Continuation mode: overriding intent to 'rewrite'")
            
            # Capture the user's edit instruction for routing
            user_instruction = prompt.split("\n")[0].strip()
            if user_instruction:
                edit_instruction = user_instruction
                logger.info(f"Edit intent detected: '{edit_instruction}' — will skip RAG")
            
            # Inject the user's new instruction as custom_instructions for the LLM
            if user_instruction and parsed_prompt.custom_instructions:
                parsed_prompt.custom_instructions = f"USER EDIT REQUEST: {user_instruction}\n\n{parsed_prompt.custom_instructions}"
            elif user_instruction:
                parsed_prompt.custom_instructions = f"USER EDIT REQUEST: {user_instruction}"
        else:
            # First message in session — resolve pronoun references if needed
            if parsed_prompt.topic == "current draft" or parsed_prompt.intent in {"rewrite", "shorten"}:
                if last_turn and last_turn.parsed_topic:
                    parsed_prompt.topic = last_turn.parsed_topic
                    logger.info(f"Resolved topic from session: {parsed_prompt.topic}")
                
        # Add extracted data to state
        context_note = parsed_prompt.custom_instructions
        if parsed_prompt.intent in {"rewrite", "shorten"} and has_previous_draft:
            context_note = "context: using previous draft"

        return {
            "parsed": {
                "intent": parsed_prompt.intent,
                "topic": parsed_prompt.topic,
                "audience": parsed_prompt.audience,
                "tone": parsed_prompt.tone,
                "length": parsed_prompt.length,
                "language": parsed_prompt.language,
                "target_sections": parsed_prompt.target_sections,
                "target_images": parsed_prompt.target_images,
                "context_note": context_note
            },
            "edit_instruction": edit_instruction,
        }

parser_node = ParserAgentNode()

