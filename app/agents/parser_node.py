import logging
from app.graph_state import GraphState
from app.parser import parse_prompt

logger = logging.getLogger(__name__)

def parser_node(state: GraphState) -> GraphState:
    """Agent that parses the raw prompt to extract intent, topic, audience, tone, length."""
    logger.info("Executing Parser Node")
    
    prompt = state["prompt"]
    session = state["session"]
    
    # Allow prompt injection from previous session context if rewriting/shortening
    parsed_prompt = parse_prompt(prompt)
    
    # Handle "rewrite" / "shorten" logic by looking at previous turn
    if parsed_prompt.intent in {"rewrite", "shorten"}:
        last_turn = session.latest_turn()
        if last_turn and last_turn.parsed_topic:
            parsed_prompt.topic = last_turn.parsed_topic
            
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
