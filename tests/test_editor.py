import pytest
from app.agents.editor_node import editor_node

def test_editor_unknown_legislation_rejected():
    # Draft with a hallucinated act
    draft = "## Introduction\nThis blog post outlines requirements. The rules are specified in the Australian Citizenship Act 2007."
    state = {
        "draft": draft,
        "parsed": {"topic": "police check", "length": "short", "intent": "create_blog"},
        "revision_count": 0,
        "loop_step": 0,
    }
    
    result = editor_node.execute(state)
    assert result["editor_feedback"] is not None
    assert "E11:unknown-legislation" in result["editor_feedback"]
    assert "Australian Citizenship Act" in result["editor_feedback"]


def test_editor_golden_legislation_accepted(monkeypatch):
    # Draft with a golden act (Crimes Act)
    draft = "## Introduction\nThis outlines spent convictions. The Crimes Act 1914 governs this check.\n## Conclusion\nWe are done."
    state = {
        "draft": draft,
        "parsed": {"topic": "police check", "length": "short", "intent": "create_blog"},
        "revision_count": 0,
        "loop_step": 0,
    }
    
    # Mock LLM evaluation to avoid live api dependency
    import app.agents.editor_node as editor_node_module
    monkeypatch.setattr(editor_node_module, "_llm_evaluate_draft", lambda _d, _p: {"verdict": "ACCEPT", "overall": 8})
    
    result = editor_node.execute(state)
    # Should accept draft and not have any E11 feedback
    if result["editor_feedback"]:
        assert "E11:unknown-legislation" not in result["editor_feedback"]
