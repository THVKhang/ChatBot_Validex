import re


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    return re.findall(r"[a-zA-Z0-9]+", normalized)
