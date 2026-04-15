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
    assert len(outline) >= 4
    assert outline[0] == "Introduction"


def test_generate_outline_police_topic_includes_step_sections():
    parsed = ParsedPrompt(
        raw_prompt="Write a step-by-step guide about police check application",
        intent="generate",
        topic="police check application",
        tone="neutral",
        audience="general",
        length="medium",
    )
    outline = generate_outline(parsed)
    assert any(item.lower().startswith("step 1") for item in outline)
    assert any("validex" in item.lower() for item in outline)


def test_generate_outline_police_topic_editorial_prompt_avoids_step_sections():
    parsed = ParsedPrompt(
        raw_prompt="Generate an editorial regarding police check procedures and background verification policies in the modern workplace",
        intent="generate",
        topic="police check procedures and background verification policies in the modern workplace",
        tone="neutral",
        audience="general",
        length="medium",
    )
    outline = generate_outline(parsed)
    assert not any(item.lower().startswith("step") for item in outline)
    assert any("understanding the national police check" in item.lower() for item in outline)


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
    assert "shortened version" in output.draft.lower()


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
    assert "rewritten" in output.draft.lower()


def test_generate_blog_output_hides_doc_ids_and_grounding_labels_in_section_body():
    parsed = ParsedPrompt(
        raw_prompt="x",
        intent="create_blog",
        topic="police check",
        tone="professional",
        audience="hr professionals",
        length="medium",
    )
    docs = [
        RetrievedDoc(doc_id="doc_01", score=9, content="Police checks are commonly required for sensitive workplace roles."),
        RetrievedDoc(doc_id="doc_02", score=8, content="Candidate communication should include expected screening timelines and required identity documents."),
    ]

    output = generate_blog_output(parsed, docs)
    assert output.sections
    first_body = output.sections[0].body
    assert "[doc_" not in first_body
    assert "grounding points from retrieved context" not in first_body.lower()
