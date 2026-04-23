"""Prompt validation and sanitization layer.

Guards against injection attacks, excessively long inputs, and
malformed prompts before they reach the LLM pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.config import settings


@dataclass
class PromptValidationResult:
    """Result of prompt validation."""
    is_valid: bool
    cleaned_prompt: str
    warnings: list[str] = field(default_factory=list)
    rejection_reason: str | None = None


# Patterns that suggest prompt injection attempts
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?|context)", re.IGNORECASE),
    re.compile(r"(disregard|forget|override)\s+(all\s+)?(previous|above|prior|your)\s+(instructions?|prompts?|rules?|context|guidelines)", re.IGNORECASE),
    re.compile(r"^system\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(assistant|user|human)\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"you\s+are\s+now\s+(a|an|the)\s+", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be|you're)\s+", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if\s+)?(you\s+are|a|an)\s+", re.IGNORECASE),
    re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),
    re.compile(r"(\[INST\]|\[/INST\]|<\|im_start\|>|<\|im_end\|>|<\|system\|>)", re.IGNORECASE),
]

# Control characters that shouldn't appear in prompts (except normal whitespace)
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Excessive whitespace normalization
_EXCESSIVE_WHITESPACE = re.compile(r"[ \t]{10,}")
_EXCESSIVE_NEWLINES = re.compile(r"\n{4,}")


def validate_prompt(text: str | None) -> PromptValidationResult:
    """Validate and sanitize a user prompt.

    Returns a PromptValidationResult with the cleaned text and any
    warnings or rejection reasons.
    """
    if not settings.enable_prompt_guard:
        return PromptValidationResult(
            is_valid=True,
            cleaned_prompt=str(text or ""),
        )

    # --- Empty check ---
    if not text or not text.strip():
        return PromptValidationResult(
            is_valid=False,
            cleaned_prompt="",
            rejection_reason="Prompt cannot be empty.",
        )

    cleaned = text.strip()

    # --- Length check ---
    max_length = max(10, settings.max_prompt_length)
    if len(cleaned) > max_length:
        return PromptValidationResult(
            is_valid=False,
            cleaned_prompt=cleaned[:max_length],
            rejection_reason=f"Prompt exceeds maximum length of {max_length} characters ({len(cleaned)} given).",
        )

    # --- Control character removal ---
    warnings: list[str] = []
    control_matches = _CONTROL_CHAR_PATTERN.findall(cleaned)
    if control_matches:
        cleaned = _CONTROL_CHAR_PATTERN.sub("", cleaned)
        warnings.append(f"Removed {len(control_matches)} control character(s).")

    # --- Normalize excessive whitespace ---
    cleaned = _EXCESSIVE_WHITESPACE.sub("  ", cleaned)
    cleaned = _EXCESSIVE_NEWLINES.sub("\n\n\n", cleaned)

    # --- Injection pattern detection ---
    injection_matches: list[str] = []
    for pattern in _INJECTION_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            injection_matches.append(match.group(0).strip()[:60])

    if injection_matches:
        return PromptValidationResult(
            is_valid=False,
            cleaned_prompt=cleaned,
            warnings=warnings,
            rejection_reason=(
                f"Prompt contains potentially unsafe pattern(s): "
                f"{', '.join(repr(m) for m in injection_matches[:3])}. "
                f"Please rephrase your request."
            ),
        )

    # --- Final empty check after cleaning ---
    if not cleaned.strip():
        return PromptValidationResult(
            is_valid=False,
            cleaned_prompt="",
            rejection_reason="Prompt is empty after sanitization.",
        )

    return PromptValidationResult(
        is_valid=True,
        cleaned_prompt=cleaned,
        warnings=warnings,
    )
