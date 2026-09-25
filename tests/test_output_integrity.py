"""Tests for the qualities a published article must have.

Every case here corresponds to a defect that reached generated output while the
existing suite stayed green:
  - sections ending mid-sentence ("...building trust and maintaining")
  - the H1 being the raw user instruction, truncated on a comma
  - generated headings containing template debris ("Overview of a professional
    blog post for")
  - acronyms title-cased into "Nsw" / "Ndis"
"""

import re

import pytest

from app.generator import _build_topic_aware_outline, format_title
from app.langchain_pipeline import (
    _summarize_section_for_history,
    _trim_to_last_complete_sentence,
)
from app.main import _repair_truncated_sections
from app.parser import parse_user_input

REAL_PROMPT = (
    "Write a professional blog post for Australian employers explaining "
    "what a National Police Check is, who needs one, and how long it takes to process."
)


def _section_bodies(markdown: str) -> dict[str, str]:
    parts = re.split(r"(?m)^##\s+(.+?)\s*$", markdown)
    return {
        parts[i].strip(): parts[i + 1].strip()
        for i in range(1, len(parts) - 1, 2)
    }


def _ends_cleanly(body: str) -> bool:
    """Bullets, headings and table rows may end without terminal punctuation."""
    if not body.strip():
        return True
    last = body.rstrip().rsplit("\n", 1)[-1].strip()
    if last.startswith(("-", "*", "#", "|", ">")):
        return True
    return last.endswith((".", "!", "?", ":", "]", "**"))


class TestTrimToLastCompleteSentence:
    def test_drops_trailing_half_sentence(self):
        text = "Employers must act early. This part was cut off mid"
        assert _trim_to_last_complete_sentence(text) == "Employers must act early."

    def test_keeps_complete_text_untouched(self):
        text = "Checks take one to three days."
        assert _trim_to_last_complete_sentence(text) == text

    def test_preserves_bullet_lists(self):
        text = "Key points:\n- Verify identity documents\n- Keep consent on file"
        assert _trim_to_last_complete_sentence(text) == text

    def test_leaves_text_with_no_sentence_end_alone(self):
        text = "No terminal punctuation at all"
        assert _trim_to_last_complete_sentence(text) == text

    def test_handles_empty(self):
        assert _trim_to_last_complete_sentence("") == ""


class TestRepairTruncatedSections:
    def test_repairs_every_truncated_section(self):
        draft = (
            "# Police Checks\n\n"
            "## Introduction\n"
            "A police check is a criminal history check. It supports safe hiring.\n\n"
            "## What This Means\n"
            "Requirements vary by jurisdiction and industry, and\n\n"
            "## Conclusion\n"
            "Start early. This safeguards your workforce and builds trust while maintaining\n"
        )
        assert sum(not _ends_cleanly(b) for b in _section_bodies(draft).values()) == 2

        fixed = _repair_truncated_sections(draft)
        assert all(_ends_cleanly(b) for b in _section_bodies(fixed).values())
        # Repair must not silently empty a section.
        assert "police check is a criminal history check" in fixed.lower()

    def test_leaves_a_clean_draft_unchanged_in_substance(self):
        draft = (
            "# Title\n\n## Introduction\nOne complete sentence here.\n\n"
            "## Steps\n- First step\n- Second step\n"
        )
        fixed = _repair_truncated_sections(draft)
        assert "One complete sentence here." in fixed
        assert "- Second step" in fixed

    def test_empty_draft_is_safe(self):
        assert _repair_truncated_sections("") == ""


class TestTopicExtraction:
    def test_topic_is_the_subject_not_the_instruction(self):
        topic = parse_user_input(REAL_PROMPT).topic
        assert not topic.lower().startswith("a professional blog post")
        assert "blog post" not in topic.lower()
        assert "national police check" in topic.lower()

    @pytest.mark.parametrize("prompt,expected", [
        ("Write a blog about spent convictions in NSW", "spent convictions in NSW"),
        ("Create an article on NDIS worker screening", "NDIS worker screening"),
    ])
    def test_marker_prompts_extract_cleanly(self, prompt, expected):
        assert parse_user_input(prompt).topic == expected


class TestGeneratedHeadings:
    def test_headings_contain_no_template_debris(self):
        parsed = parse_user_input(REAL_PROMPT)
        outline = _build_topic_aware_outline(parsed)
        joined = " ".join(outline).lower()
        for debris in ("blog post", "professional blog", "a professional"):
            assert debris not in joined, f"heading debris {debris!r} in {outline}"

    def test_headings_stay_short(self):
        parsed = parse_user_input(REAL_PROMPT)
        for heading in _build_topic_aware_outline(parsed):
            assert len(heading.split()) <= 10, f"heading too long: {heading!r}"


class TestFormatTitle:
    def test_title_is_not_the_raw_prompt(self):
        title = format_title(parse_user_input(REAL_PROMPT).topic)
        assert not title.lower().startswith("a professional blog post")
        assert not title.rstrip().endswith(",")

    @pytest.mark.parametrize("topic,expected", [
        ("spent convictions in NSW", "Spent Convictions in NSW"),
        ("NDIS worker screening", "NDIS Worker Screening"),
        ("police check for HR teams", "Police Check for HR Teams"),
    ])
    def test_acronyms_keep_their_casing(self, topic, expected):
        assert format_title(topic) == expected


class TestSectionHistoryDigest:
    def test_digest_is_much_smaller_than_the_section(self):
        section = "## Processing Times\n\n" + ("Checks take one to three days. " * 60)
        digest = _summarize_section_for_history(section)
        assert len(digest) < len(section) / 4
        assert "Processing Times" in digest
