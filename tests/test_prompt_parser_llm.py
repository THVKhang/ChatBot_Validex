from app.langchain_pipeline import LangChainRAGPipeline


class _FakePromptParserInvoker:
    def __init__(self, payload):
        self._payload = payload

    def invoke(self, _instruction: str):
        return self._payload


class _FakeLLMWithParser:
    def __init__(self, payload):
        self._payload = payload

    def with_structured_output(self, _schema):
        return _FakePromptParserInvoker(self._payload)


def test_pipeline_parse_uses_llm_when_available():
    pipeline = LangChainRAGPipeline()
    pipeline._llm = _FakeLLMWithParser(
        {
            "intent": "rewrite",
            "topic": "current draft",
            "tone": "professional",
            "audience": "hr professionals",
            "length": "medium",
        }
    )

    parsed = pipeline._parse({"prompt": "Keep only 1 picture for blog"})

    assert parsed.intent == "rewrite"
    assert parsed.topic == "current draft"
    assert parsed.audience == "hr professionals"


def test_pipeline_parse_falls_back_when_llm_unavailable():
    pipeline = LangChainRAGPipeline()
    pipeline._llm = None

    parsed = pipeline._parse({"prompt": "Write a blog about police checks for job seekers"})

    assert parsed.intent == "create_blog"
    assert "police check" in parsed.topic.lower()


def test_pipeline_parse_overrides_llm_create_for_image_edit_prompt():
    pipeline = LangChainRAGPipeline()
    pipeline._llm = _FakeLLMWithParser(
        {
            "intent": "create_blog",
            "topic": "police check",
            "tone": "professional",
            "audience": "general audience",
            "length": "medium",
        }
    )

    parsed = pipeline._parse({"prompt": "make it 1 picture"})

    assert parsed.intent == "rewrite"
    assert parsed.topic == "current draft"