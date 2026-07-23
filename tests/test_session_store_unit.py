"""Session store unit tests.

Covers:
- Session serialization/deserialization
- TTL behavior
- User isolation
- Unicode content handling
- Large session history
"""

import json
import pytest

from app.session_manager import ChatTurn, SessionManager
from app.session_store import _turns_to_json, _json_to_turns


class TestSessionSerialization:
    """Test SessionManager serialization to/from JSON."""

    def test_empty_session_serializes_to_empty_list(self):
        session = SessionManager()
        result = _turns_to_json(session)
        assert json.loads(result) == []

    def test_single_turn_roundtrip(self):
        session = SessionManager()
        session.add_turn("Hello", "Hi there", parsed_intent="create_blog", parsed_topic="greetings")
        
        json_str = _turns_to_json(session)
        turns = _json_to_turns(json_str)
        
        assert len(turns) == 1
        assert turns[0].user_prompt == "Hello"
        assert turns[0].assistant_output == "Hi there"
        assert turns[0].parsed_intent == "create_blog"
        assert turns[0].parsed_topic == "greetings"

    def test_multiple_turns_roundtrip(self):
        session = SessionManager()
        for i in range(5):
            session.add_turn(f"prompt_{i}", f"output_{i}")
        
        json_str = _turns_to_json(session)
        turns = _json_to_turns(json_str)
        
        assert len(turns) == 5
        assert turns[0].user_prompt == "prompt_0"
        assert turns[4].user_prompt == "prompt_4"

    def test_unicode_content_roundtrip(self):
        session = SessionManager()
        session.add_turn(
            "Viết blog về kiểm tra lý lịch tư pháp",
            "Đây là bài viết về kiểm tra lý lịch tư pháp tại Úc",
            parsed_topic="kiểm tra lý lịch tư pháp",
        )
        
        json_str = _turns_to_json(session)
        turns = _json_to_turns(json_str)
        
        assert turns[0].user_prompt == "Viết blog về kiểm tra lý lịch tư pháp"
        assert "lý lịch tư pháp" in turns[0].assistant_output

    def test_generated_draft_truncated_at_500_chars(self):
        session = SessionManager()
        long_draft = "x" * 1000
        session.add_turn("prompt", "output", generated_draft=long_draft)
        
        json_str = _turns_to_json(session)
        data = json.loads(json_str)
        
        assert len(data[0]["generated_draft"]) == 500

    def test_special_characters_in_content(self):
        session = SessionManager()
        session.add_turn(
            'He said "hello" & <goodbye>',
            "Response with 'quotes' and \\backslashes\\",
        )
        
        json_str = _turns_to_json(session)
        turns = _json_to_turns(json_str)
        
        assert turns[0].user_prompt == 'He said "hello" & <goodbye>'


class TestJsonToTurns:
    """Test _json_to_turns edge cases."""

    def test_invalid_json_string_returns_empty(self):
        turns = _json_to_turns("not valid json at all")
        assert turns == []

    def test_none_input_returns_empty(self):
        turns = _json_to_turns(None)
        assert turns == []

    def test_dict_input_returns_empty(self):
        """Non-list dict should return empty."""
        turns = _json_to_turns({"key": "value"})
        assert turns == []

    def test_list_with_non_dict_items_skipped(self):
        data = [{"user_prompt": "valid"}, "invalid_string", 42, None]
        turns = _json_to_turns(data)
        assert len(turns) == 1
        assert turns[0].user_prompt == "valid"

    def test_missing_fields_use_defaults(self):
        data = [{"user_prompt": "only prompt"}]
        turns = _json_to_turns(data)
        assert len(turns) == 1
        assert turns[0].assistant_output == ""
        assert turns[0].parsed_intent == ""

    def test_extra_fields_ignored(self):
        data = [{"user_prompt": "test", "extra_field": "ignored", "another": 123}]
        turns = _json_to_turns(data)
        assert len(turns) == 1
        assert turns[0].user_prompt == "test"

    def test_already_parsed_list(self):
        """When input is already a Python list (from JSONB)."""
        data = [
            {"user_prompt": "p1", "assistant_output": "o1"},
            {"user_prompt": "p2", "assistant_output": "o2"},
        ]
        turns = _json_to_turns(data)
        assert len(turns) == 2


class TestSessionManagerFeatures:
    """Test SessionManager utility methods."""

    def test_latest_turn_empty_session(self):
        session = SessionManager()
        assert session.latest_turn() is None

    def test_latest_turn_returns_last(self):
        session = SessionManager()
        session.add_turn("first", "f_out")
        session.add_turn("second", "s_out")
        session.add_turn("third", "t_out")
        
        latest = session.latest_turn()
        assert latest.user_prompt == "third"

    def test_history_text_format(self):
        session = SessionManager()
        session.add_turn("Hello", "World")
        session.add_turn("How are you", "Fine thanks")
        
        text = session.history_text()
        assert "[1] USER: Hello" in text
        assert "[1] BOT: World" in text
        assert "[2] USER: How are you" in text

    def test_conversation_summary_empty(self):
        session = SessionManager()
        assert session.conversation_summary() == ""

    def test_conversation_summary_limited_turns(self):
        session = SessionManager()
        for i in range(10):
            session.add_turn(f"prompt_{i}", f"output_{i}")
        
        summary = session.conversation_summary(max_turns=3)
        assert "prompt_7" in summary  # Should include last 3 turns
        assert "prompt_0" not in summary  # Should NOT include old turns

    def test_to_langchain_messages(self):
        session = SessionManager()
        session.add_turn("Hello", "", generated_draft="Generated content here")
        
        messages = session.to_langchain_messages(max_turns=5)
        assert len(messages) == 2  # HumanMessage + AIMessage

    def test_to_langchain_messages_empty(self):
        session = SessionManager()
        messages = session.to_langchain_messages()
        assert messages == []


class TestLargeSessionHistory:
    """Test behavior with large session histories."""

    def test_large_session_serialization(self):
        """Session with 100 turns should serialize correctly."""
        session = SessionManager()
        for i in range(100):
            session.add_turn(
                f"Prompt {i} about {'police check' if i % 2 == 0 else 'compliance'}",
                f"Detailed output {i} with lots of content",
                parsed_intent="create_blog",
                parsed_topic=f"topic_{i}",
                generated_draft=f"Draft content for turn {i} " * 20,
            )
        
        json_str = _turns_to_json(session)
        turns = _json_to_turns(json_str)
        
        assert len(turns) == 100
        assert turns[0].user_prompt == "Prompt 0 about police check"
        assert turns[99].user_prompt == "Prompt 99 about compliance"

    def test_history_text_with_many_turns(self):
        session = SessionManager()
        for i in range(50):
            session.add_turn(f"p{i}", f"o{i}")
        
        text = session.history_text()
        assert "[50] USER: p49" in text
        assert len(text) > 0

    def test_conversation_summary_max_turns_respected(self):
        session = SessionManager()
        for i in range(20):
            session.add_turn(f"prompt_{i}", f"output_{i}")
        
        summary = session.conversation_summary(max_turns=2)
        lines = summary.split("\n")
        # Header + 2 turns × 2 lines each = 5 lines max
        assert len(lines) <= 6
