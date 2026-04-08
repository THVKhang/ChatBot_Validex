from app.main import run_once
from app.session_manager import SessionManager


def test_run_once_uses_previous_draft_context_for_shorten():
    session = SessionManager()

    first_output = run_once("Write a blog about police check for job seekers", session)
    assert "=== GENERATED DRAFT ===" in first_output

    second_output = run_once("Make it shorter", session)
    assert "context: using previous draft" in second_output
    assert "=== GENERATED DRAFT ===" in second_output


def test_run_once_out_of_domain_returns_safe_message():
    session = SessionManager()
    output = run_once("Write a travel guide about Paris", session)
    assert "Need More Context" in output
    assert "Ly do:" in output
