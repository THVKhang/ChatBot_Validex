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
