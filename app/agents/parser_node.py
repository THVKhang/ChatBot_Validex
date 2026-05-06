"""Smart Parser Node — uses LLM for intent detection with regex fallback."""
import json
import logging
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

    if pipeline._llm is None:
        return None

    # Only send the first line (user message) — strip editorial settings
    user_message = prompt.split("\n")[0].strip()
    if not user_message:
        return None

    system = (
        "You are a prompt parser for a blog generation system. "
        "Analyze the user's message and extract structured metadata.\n\n"
        "Return ONLY a JSON object with these fields:\n"
        '- "intent": one of "create_blog", "rewrite", "shorten"\n'
        '  - "create_blog" = user wants a NEW blog on a topic\n'
        '  - "rewrite" = user wants to EDIT/MODIFY/ADD TO an existing draft\n'
        '  - "shorten" = user wants to make content shorter/more concise\n'
        '- "topic": the main subject (e.g. "police check in Australia")\n'
        '  - If the user is editing an existing draft, set topic to the subject being discussed\n'
        '  - If the user uses pronouns like "this", "it", "the blog", set topic to "current draft"\n'
        '- "tone": one of "professional", "friendly", "casual", "clear_professional"\n'
        '- "audience": target audience (e.g. "HR professionals", "general audience")\n'
        '- "length": one of "short", "medium", "long"\n\n'
        "Examples:\n"
        '- "children policy in australia" → {"intent":"create_blog","topic":"children policy in australia","tone":"clear_professional","audience":"general audience","length":"medium"}\n'
        '- "add 2 images about this topic" → {"intent":"rewrite","topic":"current draft","tone":"clear_professional","audience":"general audience","length":"medium"}\n'
        '- "make the conclusion stronger" → {"intent":"rewrite","topic":"current draft","tone":"clear_professional","audience":"general audience","length":"medium"}\n'
        '- "make it shorter" → {"intent":"shorten","topic":"current draft","tone":"clear_professional","audience":"general audience","length":"short"}\n'
    )

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        response = pipeline._llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=f"Parse this user message: \"{user_message}\""),
        ])
        raw = getattr(response, "content", str(response))
        # Extract JSON from response
        match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            # Validate required fields
            if "intent" in parsed and "topic" in parsed:
                logger.info(f"LLM parser result: intent={parsed['intent']}, topic={parsed['topic']}")
                return parsed
    except Exception as exc:
        logger.warning(f"LLM parse failed: {exc}")

    return None


def parser_node(state: GraphState) -> GraphState:
    """Agent that parses the raw prompt to extract intent, topic, audience, tone, length."""
    logger.info("Executing Parser Node")
    
    prompt = state["prompt"]
    session = state["session"]
    
    # --- Try LLM Parser first, fallback to regex ---
    llm_result = _llm_parse_intent(prompt)
    
    if llm_result:
        # Use LLM result, merge with editorial settings from regex parser
        regex_parsed = parse_prompt(prompt)
        parsed_prompt = ParsedPrompt(
            raw_prompt=prompt,
            intent=llm_result.get("intent", regex_parsed.intent),
            topic=llm_result.get("topic", regex_parsed.topic),
            audience=regex_parsed.audience,  # Keep regex for editorial settings detection
            tone=llm_result.get("tone", regex_parsed.tone),
            length=llm_result.get("length", regex_parsed.length),
            custom_instructions=regex_parsed.custom_instructions,
            modifiers=regex_parsed.modifiers,
        )
        logger.info("Parser: using LLM result with regex editorial settings")
    else:
        parsed_prompt = parse_prompt(prompt)
        logger.info("Parser: using regex fallback")
    
    # --- Continuation Mode ---
    # If this session already has a previous turn with a generated draft,
    # treat ALL follow-up prompts as refinements of the existing draft
    # (like ChatGPT/Gemini: same chat = refine, new chat = fresh start).
    last_turn = session.latest_turn()
    has_previous_draft = last_turn and last_turn.generated_draft
    
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
        
        # Inject the user's new instruction as custom_instructions for the LLM
        user_instruction = prompt.split("\n")[0].strip()
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
    return {
        "parsed": {
            "intent": parsed_prompt.intent,
            "topic": parsed_prompt.topic,
            "audience": parsed_prompt.audience,
            "tone": parsed_prompt.tone,
            "length": parsed_prompt.length,
            "context_note": parsed_prompt.custom_instructions
        }
    }
