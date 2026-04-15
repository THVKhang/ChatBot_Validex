from app.langchain_pipeline import LangChainRAGPipeline


def test_apply_prompt_edit_constraints_limits_to_single_image():
    pipeline = LangChainRAGPipeline()
    payload = {
        "draft": "# Title\n\n## A\n\n![A](https://img/a.jpg)\n\nBody A\n\n## B\n\n![B](https://img/b.jpg)\n\nBody B",
        "sections": [
            {"heading": "A", "body": "Body A", "image_url": "https://img/a.jpg", "image_alt": "A"},
            {"heading": "B", "body": "Body B", "image_url": "https://img/b.jpg", "image_alt": "B"},
        ],
    }

    updated = pipeline._apply_prompt_edit_constraints("Keep only 1 picture for blog", payload)

    kept_urls = [item.get("image_url") for item in updated["sections"] if item.get("image_url")]
    assert len(kept_urls) == 1
    assert updated["draft"].count("![") == 1


def test_apply_prompt_edit_constraints_removes_all_images():
    pipeline = LangChainRAGPipeline()
    payload = {
        "draft": "# Title\n\n## A\n\n![A](https://img/a.jpg)\n\nBody A",
        "sections": [
            {"heading": "A", "body": "Body A", "image_url": "https://img/a.jpg", "image_alt": "A"},
        ],
    }

    updated = pipeline._apply_prompt_edit_constraints("Remove all images from this draft", payload)

    assert not updated["sections"][0]["image_url"]
    assert "![" not in updated["draft"]


def test_apply_prompt_edit_constraints_skips_context_section_image():
    pipeline = LangChainRAGPipeline()
    payload = {
        "draft": (
            "# Title\n\n"
            "## Context from Previous Draft\n\n"
            "![context](https://img/context.jpg)\n\n"
            "Context body\n\n"
            "## A\n\n"
            "![A](https://img/a.jpg)\n\n"
            "Body A\n\n"
            "## B\n\n"
            "![B](https://img/b.jpg)\n\n"
            "Body B"
        ),
        "sections": [
            {
                "heading": "Context from Previous Draft",
                "body": "Context body",
                "image_url": "https://img/context.jpg",
                "image_alt": "context",
            },
            {"heading": "A", "body": "Body A", "image_url": "https://img/a.jpg", "image_alt": "A"},
            {"heading": "B", "body": "Body B", "image_url": "https://img/b.jpg", "image_alt": "B"},
        ],
    }

    updated = pipeline._apply_prompt_edit_constraints("make it 1 picture", payload)

    assert updated["sections"][0]["image_url"] == ""
    kept_urls = [item.get("image_url") for item in updated["sections"] if item.get("image_url")]
    assert kept_urls == ["https://img/a.jpg"]
    assert "![context]" not in updated["draft"]
    assert "![A](https://img/a.jpg)" in updated["draft"]
    assert "![B](https://img/b.jpg)" not in updated["draft"]