"""Session memory manager with sliding window and conversation state."""
from typing import Dict, List, Any, Optional
import datetime
from pydantic import BaseModel, Field


class MessageTurn(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str
    language: str = "en"
    intent: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())


class ConversationSession(BaseModel):
    session_id: str
    user_language: str = "en"
    turns: List[MessageTurn] = Field(default_factory=list)
    frustration_score: float = 0.0
    failed_attempts: int = 0
    is_escalated: bool = False
    escalation_reason: Optional[str] = None
    order_ids_referenced: List[str] = Field(default_factory=list)

    def add_turn(self, role: str, content: str, language: str = "en", intent: Optional[str] = None):
        turn = MessageTurn(role=role, content=content, language=language, intent=intent)
        self.turns.append(turn)
        # Keep sliding window of latest 20 turns
        if len(self.turns) > 20:
            self.turns = self.turns[-20:]

    def get_history_text(self) -> str:
        """Returns formatted conversation history."""
        lines = []
        for t in self.turns:
            lines.append(f"{t.role.upper()} ({t.language}): {t.content}")
        return "\n".join(lines)


class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, ConversationSession] = {}

    def get_or_create(self, session_id: str, default_lang: str = "en") -> ConversationSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationSession(
                session_id=session_id,
                user_language=default_lang,
            )
        return self._sessions[session_id]

    def clear(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]


session_manager = SessionManager()
