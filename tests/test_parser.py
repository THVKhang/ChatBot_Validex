import pytest

from app.parser import parse_prompt, parse_user_input


def test_parse_prompt_rewrite_intent():
    parsed = parse_prompt("Viet lai bai nay theo giong chuyen nghiep hon")
    assert parsed.intent == "rewrite"
    assert parsed.tone == "professional"


def test_parse_prompt_topic_marker():
    parsed = parse_prompt("Viet cho toi mot blog ve police check la gi")
    assert "police check" in parsed.topic.lower()


def test_parse_user_input_entrypoint():
    parsed = parse_user_input("Viet blog ve police check cho employer")
    assert parsed.topic


@pytest.mark.parametrize(
    "prompt,expected_intent,expected_length,expected_audience",
    [
        ("Write a blog about police checks for job seekers", "create_blog", "medium", "job seekers"),
        ("Make it shorter", "shorten", "short", "general audience"),
        ("Rewrite in a more professional tone", "rewrite", "medium", "general audience"),
        ("Viet blog ve police check cho employer", "create_blog", "medium", "employer"),
        ("Rut gon bai nay", "shorten", "short", "general audience"),
        ("Viet lai theo giong chuyen nghiep", "rewrite", "medium", "general audience"),
        ("Write a long blog about police check for HR", "create_blog", "long", "hr"),
        ("Create blog about onboarding for backoffice", "create_blog", "medium", "backoffice"),
        ("Write about police check", "create_blog", "medium", "general audience"),
        ("Rewrite this for first-time job applicants", "rewrite", "medium", "first-time job applicants"),
    ],
)
def test_parser_with_10_prompt_samples(prompt, expected_intent, expected_length, expected_audience):
    parsed = parse_user_input(prompt)
    assert parsed.intent == expected_intent
    assert parsed.length == expected_length
    assert parsed.audience == expected_audience


def test_parser_default_tone_when_missing():
    parsed = parse_user_input("Write a blog about police checks")
    assert parsed.tone == "clear_professional"


def test_parser_cleans_topic_with_tone_clause():
    parsed = parse_user_input(
        "Write a blog about what a police check is for first-time job applicants, in a clear and professional tone"
    )
    assert "professional tone" not in parsed.topic.lower()


def test_parser_cleans_topic_with_audience_suffix():
    parsed = parse_user_input("Write a blog about police checks for job seekers")
    assert parsed.topic.lower() == "police checks"
