from dataclasses import dataclass, field
import re


@dataclass
class ParsedPrompt:
    raw_prompt: str
    intent: str
    topic: str
    tone: str
    audience: str
    length: str
    custom_instructions: str = ""
    modifiers: dict = field(default_factory=dict)


DEFAULT_INTENT = "create_blog"
DEFAULT_AUDIENCE = "general audience"
DEFAULT_TONE = "clear_professional"
DEFAULT_LENGTH = "medium"


def _detect_intent(prompt_lower: str) -> str:
    # Use word-boundary regex to avoid 'editorial' matching 'edit'
    edit_keywords = [r"\bedit\b", r"\brevise\b", r"\bupdate\b", r"\badjust\b", r"\bfix\b", r"\bsua\b", r"\bchinh sua\b", r"\bchỉnh sửa\b"]
    if any(re.search(kw, prompt_lower) for kw in edit_keywords):
        return "rewrite"

    image_edit_patterns = [
        r"\b(?:keep|remove|reduce|limit|retain|only|show|use|make|set|change|add)\b.*\b(?:image|images|picture|pictures|photo|photos)\b",
        r"\b(?:image|images|picture|pictures|photo|photos)\b.*\b(?:keep|remove|reduce|limit|retain|only|show|use|make|set|change|add)\b",
        r"\b(?:chi|chỉ|giu|giữ|bo|bỏ|xoa|xóa|giam|giảm|them|thêm)\b.*\b(?:anh|ảnh|hinh|hình)\b",
        r"\bmake\s+it\s+(?:\d+|one|two|three)\s+(?:image|images|picture|pictures|photo|photos)\b",
        r"\badd\s+\d+\s+(?:image|images|picture|pictures|photo|photos)\b",
    ]
    if any(re.search(pattern, prompt_lower) for pattern in image_edit_patterns):
        return "rewrite"

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
    word_target_match = re.search(r"\b(\d{1,3}(?:,\d{3})+|\d{3,4})\s*(chu|tu|words?)\b", prompt_lower)
    if word_target_match:
        raw_value = word_target_match.group(1).replace(",", "")
        try:
            target_words = int(raw_value)
            if target_words <= 500:
                return "short"
            if target_words <= 1000:
                return "medium"
            return "long"
        except ValueError:
            pass

    if any(kw in prompt_lower for kw in ["short", "shorter", "ngan", "ngan gon", "rut gon"]):
        return "short"
    if any(kw in prompt_lower for kw in ["long", "dai", "chi tiet"]):
        return "long"
    return DEFAULT_LENGTH


def _detect_audience(prompt_lower: str) -> str:
    configured_match = re.search(r"target_audience\s*:\s*([^\r\n]+)", prompt_lower)
    if configured_match:
        configured_value = configured_match.group(1).strip(" .,!?:;\n\t")
        if configured_value:
            return configured_value

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


# Pronoun/reference patterns that indicate the user is referring to the previous topic
_REFERENCE_PATTERNS = re.compile(
    r"^(this\s+topic|this|it|the\s+topic|the\s+blog|the\s+draft|the\s+article|bai\s+nay|bài\s+này|no|nó)$",
    re.IGNORECASE,
)


def _extract_topic(prompt: str, prompt_lower: str, intent: str) -> str:
    markers = ["about", "regarding", "ve", "về"]

    # Frontend sends additional config lines after the first prompt sentence,
    # so extract marker content up to the first line break.
    for marker in markers:
        pattern = rf"\b{marker}\s+([^\r\n]+)"
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            topic = match.group(1).strip(" .,!?:;\n\t")
            # Resolve pronoun references like "this topic", "it", "this"
            if topic and _REFERENCE_PATTERNS.match(topic):
                return "current draft"
            if topic:
                return topic

    # Follow-up prompts like "make it shorter" should keep previous topic context.
    if intent in {"shorten", "rewrite"}:
        return "current draft"

    # Fallback for create-blog prompts without explicit markers.
    # Use the first line as topic after removing common command prefixes.
    # Accept short keywords (>= 2 chars) so users can just type a keyword.
    first_line = prompt.splitlines()[0].strip() if prompt.strip() else ""
    if first_line:
        raw_first_line = first_line
        first_line = re.sub(
            r"^(write|create|generate|draft|viet|viết|soan|soạn)\s+",
            "",
            first_line,
            flags=re.IGNORECASE,
        )
        first_line = re.sub(
            r"^(a\s+|an\s+)?(blog|article|post|bai\s+blog|bài\s+blog)\s+",
            "",
            first_line,
            flags=re.IGNORECASE,
        )
        first_line = first_line.strip(" .,!?:;\n\t")
        if len(first_line) >= 2:
            return first_line
        # If stripping removed too much, use the raw first line as the topic.
        raw_first_line = raw_first_line.strip(" .,!?:;\n\t")
        if len(raw_first_line) >= 2:
            return raw_first_line

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


_URL_PATTERN = re.compile(r"https?://[^\s,\"'<>]+", re.IGNORECASE)
_IMAGE_EXT_PATTERN = re.compile(r"\.(jpe?g|png|webp|gif|bmp|svg)(\?[^\s]*)?$", re.IGNORECASE)


def _extract_modifiers(prompt: str) -> dict:
    """Extract structured modifiers from the prompt text."""
    modifiers: dict = {}

    # Detect all URLs
    urls = _URL_PATTERN.findall(prompt)
    web_urls = []
    image_urls = []
    for url in urls:
        if _IMAGE_EXT_PATTERN.search(url):
            image_urls.append(url)
        else:
            web_urls.append(url)

    if web_urls:
        modifiers["scrape_urls"] = web_urls
    if image_urls:
        modifiers["ocr_urls"] = image_urls

    # Detect explicit image count: "3 images", "2 ảnh"
    img_count_match = re.search(r"(\d+)\s*(?:images?|ảnh|anh|hình|hinh|pictures?|photos?)", prompt, re.IGNORECASE)
    if img_count_match:
        modifiers["image_count"] = int(img_count_match.group(1))

    return modifiers


def parse_prompt(prompt: str) -> ParsedPrompt:
    prompt_lower = prompt.lower()

    intent = _detect_intent(prompt_lower)
    tone = _detect_tone(prompt_lower)
    audience = _detect_audience(prompt_lower)
    length = _detect_length(prompt_lower)
    topic = _clean_topic_text(_extract_topic(prompt, prompt_lower, intent))
    modifiers = _extract_modifiers(prompt)

    return ParsedPrompt(
        raw_prompt=prompt,
        intent=intent,
        topic=topic,
        tone=tone,
        audience=audience,
        length=length,
        modifiers=modifiers,
    )


def parse_user_input(text: str) -> ParsedPrompt:
    """Week 2 parser entrypoint for free-form chat input."""
    return parse_prompt(text)
