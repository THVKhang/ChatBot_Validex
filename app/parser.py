from dataclasses import dataclass
import re


@dataclass
class ParsedPrompt:
    raw_prompt: str
    intent: str
    topic: str
    tone: str
    audience: str
    length: str


DEFAULT_INTENT = "create_blog"
DEFAULT_AUDIENCE = "general audience"
DEFAULT_TONE = "clear_professional"
DEFAULT_LENGTH = "medium"


def _detect_intent(prompt_lower: str) -> str:
    if any(kw in prompt_lower for kw in ["rewrite", "viet lai", "re-write"]):
        return "rewrite"
    if any(kw in prompt_lower for kw in ["shorter", "rut gon", "ngan hon", "tom tat"]):
        return "shorten"
    return DEFAULT_INTENT


def _detect_tone(prompt_lower: str) -> str:
    if any(kw in prompt_lower for kw in ["professional", "chuyen nghiep"]):
        return "professional"
    if any(kw in prompt_lower for kw in ["friendly", "than thien"]):
        return "friendly"
    if any(kw in prompt_lower for kw in ["casual", "tu nhien"]):
        return "casual"
    return DEFAULT_TONE


def _detect_length(prompt_lower: str) -> str:
    if any(kw in prompt_lower for kw in ["short", "shorter", "ngan", "ngan gon", "rut gon"]):
        return "short"
    if any(kw in prompt_lower for kw in ["long", "dai", "chi tiet"]):
        return "long"
    return DEFAULT_LENGTH


def _detect_audience(prompt_lower: str) -> str:
    audience_keywords = [
        "job seekers",
        "first-time job applicants",
        "employer",
        "hr",
        "backoffice",
        "nguoi moi",
    ]
    for keyword in audience_keywords:
        if keyword in prompt_lower:
            return keyword

    match_for = re.search(r"for\s+([a-zA-Z\-\s]{3,40})", prompt_lower)
    if match_for:
        return match_for.group(1).strip()

    return DEFAULT_AUDIENCE


def _extract_topic(prompt: str, prompt_lower: str) -> str:
    markers = ["about", "ve"]
    for marker in markers:
        pattern = rf"{marker}\s+(.+)$"
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            topic = match.group(1).strip(" .,!?:;\n\t")
            if topic:
                return topic

    # For prompts like "Make it shorter", keep a stable fallback.
    return "current draft"


def _clean_topic_text(topic: str) -> str:
    cleaned = topic.strip(" .,!?:;\n\t")

    # Remove trailing tone constraints, e.g. "in a clear and professional tone".
    cleaned = re.sub(
        r",?\s*in\s+(a\s+)?(clear|professional|friendly|casual)[\w\s-]*tone\.?$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Remove trailing audience constraints if they are explicit audience keywords.
    cleaned = re.sub(
        r"\s+for\s+(job seekers|first-time job applicants|employer|hr|backoffice|nguoi moi)\.?$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = cleaned.strip(" .,!?:;\n\t")
    return cleaned or "current draft"


def parse_prompt(prompt: str) -> ParsedPrompt:
    prompt_lower = prompt.lower()

    intent = _detect_intent(prompt_lower)
    tone = _detect_tone(prompt_lower)
    audience = _detect_audience(prompt_lower)
    length = _detect_length(prompt_lower)
    topic = _clean_topic_text(_extract_topic(prompt, prompt_lower))

    return ParsedPrompt(
        raw_prompt=prompt,
        intent=intent,
        topic=topic,
        tone=tone,
        audience=audience,
        length=length,
    )


def parse_user_input(text: str) -> ParsedPrompt:
    """Week 2 parser entrypoint for free-form chat input."""
    return parse_prompt(text)
