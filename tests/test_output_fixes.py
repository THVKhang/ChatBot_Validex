"""Unit tests for the 4 production output quality fixes."""

import pytest
from app.main import _sanitize_ui_artifacts
from app.generator import _build_topic_aware_outline
from app.parser import ParsedPrompt
from app.agents.editor_node import editor_node


def test_sanitize_ui_artifacts():
    raw_text = "References:\n- Source 1 | URL: https://example.com -- description html refresh bookmark thumb_up thumb_down 🟢 Standard Pipeline"
    cleaned = _sanitize_ui_artifacts(raw_text)
    assert "thumb_up" not in cleaned
    assert "description" not in cleaned
    assert "Standard Pipeline" not in cleaned
    assert "🟢" not in cleaned
    assert "Source 1" in cleaned


test_cases = [
    ("Do police checks expire?", "Police Check Expiry & Validity"),
    ("do police checks expire", "Police Check Expiry & Validity"),
]

def test_topic_normalization():
    parsed = ParsedPrompt(
        raw_prompt="Do police checks expire?",
        intent="informational",
        topic="Do police checks expire?",
        tone="clear_professional",
        audience="general audience",
        length="medium",
    )
    outline = _build_topic_aware_outline(parsed)
    assert any("Expiry & Validity" in heading for heading in outline)
    assert not any("Understanding Do police checks expire?" in heading for heading in outline)


def test_editor_e12_merged_list_items_detection():
    # Draft with a merged list missing bullet points
    merged_draft = (
        "# Title\n\n"
        "## Introduction\nIntro text\n\n"
        "Some key considerations include: Providing accurate information Withholding misleading info can have consequences Understanding the rules\n\n"
        "## Conclusion and Next Steps\nConclusion text"
    )
    state = {
        "draft": merged_draft,
        "parsed": {"intent": "create_blog", "topic": "police check"},
        "revision_count": 0,
        "loop_step": 0,
        "global_step_count": 0,
    }
    result = editor_node.execute(state)
    feedback = result.get("editor_feedback", "")
    assert "E12:merged-list-items" in feedback
