from dataclasses import dataclass, field


@dataclass
class ChatTurn:
    user_prompt: str
    assistant_output: str
    parsed_intent: str = ""
    parsed_topic: str = ""
    generated_draft: str = ""


@dataclass
class SessionManager:
    turns: list[ChatTurn] = field(default_factory=list)

    def add_turn(
        self,
        user_prompt: str,
        assistant_output: str,
        parsed_intent: str = "",
        parsed_topic: str = "",
        generated_draft: str = "",
    ) -> None:
        self.turns.append(
            ChatTurn(
                user_prompt=user_prompt,
                assistant_output=assistant_output,
                parsed_intent=parsed_intent,
                parsed_topic=parsed_topic,
                generated_draft=generated_draft,
            )
        )

    def latest_turn(self) -> ChatTurn | None:
        if not self.turns:
            return None
        return self.turns[-1]

    def history_text(self) -> str:
        lines: list[str] = []
        for idx, turn in enumerate(self.turns, start=1):
            lines.append(f"[{idx}] USER: {turn.user_prompt}")
            lines.append(f"[{idx}] BOT: {turn.assistant_output[:120]}")
        return "\n".join(lines)

    def to_langchain_messages(self, max_turns: int = 5) -> list:
        """Convert recent chat history to LangChain message objects.

        Returns a list of HumanMessage/AIMessage pairs from the most
        recent ``max_turns`` turns, suitable for injecting into an LLM
        prompt as conversation context.
        """
        try:
            from langchain_core.messages import HumanMessage, AIMessage
        except ImportError:  # pragma: no cover
            return []

        recent = self.turns[-max_turns:] if max_turns > 0 else self.turns
        messages: list = []
        for turn in recent:
            messages.append(HumanMessage(content=turn.user_prompt))
            # Use a concise summary rather than the full draft to save tokens
            summary = turn.generated_draft[:300] if turn.generated_draft else turn.assistant_output[:300]
            if summary:
                messages.append(AIMessage(content=summary))
        return messages

    def conversation_summary(self, max_turns: int = 5) -> str:
        """Return a compact text summary of recent conversation for prompt injection."""
        recent = self.turns[-max_turns:] if max_turns > 0 else self.turns
        if not recent:
            return ""
        lines: list[str] = ["=== Conversation History ==="]
        for idx, turn in enumerate(recent, start=1):
            lines.append(f"User [{idx}]: {turn.user_prompt[:200]}")
            draft_preview = turn.generated_draft[:200] if turn.generated_draft else turn.assistant_output[:200]
            if draft_preview:
                lines.append(f"Assistant [{idx}]: {draft_preview}")
        return "\n".join(lines)
