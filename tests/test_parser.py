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


def test_parser_extracts_topic_from_regarding_marker():
    parsed = parse_user_input(
        "Generate an in-depth editorial regarding police check procedures and background verification policies in the modern workplace."
    )
    assert "police check procedures" in parsed.topic.lower()


def test_parser_extracts_topic_from_multiline_configured_prompt():
    parsed = parse_user_input(
        "Generate an in-depth editorial regarding police check procedures and background verification policies in the modern workplace.\n\n"
        "Editorial settings:\n"
        "- tone: Professional\n"
        "- target_word_count: 800 Words\n"
        "- target_audience: HR Professionals"
    )
    assert "police check procedures" in parsed.topic.lower()


def test_parser_extracts_audience_from_multiline_configured_prompt():
    parsed = parse_user_input(
        "Generate an in-depth editorial regarding police check procedures and background verification policies in the modern workplace.\n\n"
        "Editorial settings:\n"
        "- tone: Professional\n"
        "- target_word_count: 800 Words\n"
        "- target_audience: HR Professionals"
    )
    assert parsed.audience == "hr professionals"


@pytest.mark.parametrize(
    "prompt",
    [
        "make it 1 picture",
        "Keep only 1 picture for blog",
        "Remove all images from this draft",
        "Chi giu 1 anh cho bai blog",
    ],
)
def test_parser_treats_image_edit_prompts_as_rewrite(prompt):
    parsed = parse_user_input(prompt)
    assert parsed.intent == "rewrite"


@pytest.mark.parametrize(
    "prompt,expected_length",
    [
        ("Write a 400 chu blog about police check", "short"),
        ("Viet bai 800 tu ve police check cho employer", "medium"),
        ("Write 1,200 words about police check compliance", "long"),
    ],
)
def test_parser_detects_length_from_word_target(prompt, expected_length):
    parsed = parse_user_input(prompt)
    assert parsed.length == expected_length
