from app.generator import format_title, generate_blog_output, generate_draft, generate_outline
from app.parser import ParsedPrompt
from app.retriever import RetrievedDoc


def test_generate_outline_has_sections():
    parsed = ParsedPrompt(
        raw_prompt="x",
        intent="generate",
        topic="police check",
        tone="neutral",
        audience="general",
        length="short",
    )
    outline = generate_outline(parsed)
    assert len(outline) == 4


def test_generate_draft_contains_topic():
    parsed = ParsedPrompt(
        raw_prompt="x",
        intent="generate",
        topic="police check",
        tone="neutral",
        audience="general",
        length="short",
    )
    text = generate_draft(parsed, docs=[])
    assert "police check" in text.lower()


def test_generate_blog_output_schema():
    parsed = ParsedPrompt(
        raw_prompt="x",
        intent="create_blog",
        topic="police check",
        tone="clear_professional",
        audience="job seekers",
        length="medium",
    )
    docs = [RetrievedDoc(doc_id="doc_01", score=3, content="police check context")]
    output = generate_blog_output(parsed, docs)

    assert output.title
    assert isinstance(output.outline, list) and len(output.outline) > 0
    assert isinstance(output.draft, str) and len(output.draft) > 0
    assert output.sources_used == ["doc_01"]


def test_format_title_compacts_current_draft():
    assert format_title("current draft") == "Current Draft Update"


def test_generate_blog_output_uses_clean_title():
    parsed = ParsedPrompt(
        raw_prompt="x",
        intent="create_blog",
        topic="what a police check is for first-time job applicants",
        tone="professional",
        audience="first-time job applicants",
        length="medium",
    )
    output = generate_blog_output(parsed, docs=[])
    assert "What a Police Check Is" in output.title


def test_generate_blog_output_length_modes():
    short_parsed = ParsedPrompt(
        raw_prompt="x",
        intent="create_blog",
        topic="police check",
        tone="professional",
        audience="job seekers",
        length="short",
    )
    long_parsed = ParsedPrompt(
        raw_prompt="x",
        intent="create_blog",
        topic="police check",
        tone="professional",
        audience="job seekers",
        length="long",
    )

    short_output = generate_blog_output(short_parsed, docs=[])
    long_output = generate_blog_output(long_parsed, docs=[])
    assert len(long_output.draft) > len(short_output.draft)


def test_generate_blog_output_respects_shorten_intent():
    parsed = ParsedPrompt(
        raw_prompt="x",
        intent="shorten",
        topic="police check",
        tone="professional",
        audience="job seekers",
        length="medium",
    )
    output = generate_blog_output(parsed, docs=[])
    assert "rut gon" in output.draft.lower()


def test_generate_blog_output_respects_rewrite_intent():
    parsed = ParsedPrompt(
        raw_prompt="x",
        intent="rewrite",
        topic="police check",
        tone="professional",
        audience="job seekers",
        length="medium",
    )
    output = generate_blog_output(parsed, docs=[])
    assert "rewrite" in output.draft.lower()
