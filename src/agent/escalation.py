"""Human escalation module with automated conversation summarization."""
import uuid
import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from src.agent.memory import ConversationSession


class HandoffTicket(BaseModel):
    ticket_id: str
    session_id: str
    created_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    language: str
    frustration_score: float
    escalation_reason: str
    referenced_orders: List[str]
    conversation_summary: str
    recommended_action: str


def generate_handoff_ticket(session: ConversationSession, reason: str) -> HandoffTicket:
    """Creates a structured human agent handoff ticket with conversation summary."""
    ticket_id = f"ESC-{uuid.uuid4().hex[:6].upper()}"
    
    # Generate automated conversation summary
    user_turns = [t.content for t in session.turns if t.role == "user"]
    last_user_query = user_turns[-1] if user_turns else "None"
    
    # Determine recommended action
    if session.order_ids_referenced:
        rec_action = f"Check logistics & refund status for order(s): {', '.join(session.order_ids_referenced)}. Customer requires immediate resolution."
    elif "return" in reason.lower() or "refund" in reason.lower():
        rec_action = "Review manual return override policy and authorize return label."
    else:
        rec_action = "Contact customer via live chat/email to resolve complex inquiry."

    summary = (
        f"Customer ({session.user_language.upper()}) escalated after {len(session.turns)} turn(s). "
        f"Trigger: {reason}. "
        f"Last customer message: '{last_user_query}'. "
        f"Overall Frustration Level: {session.frustration_score:.2f}/1.00."
    )

    ticket = HandoffTicket(
        ticket_id=ticket_id,
        session_id=session.session_id,
        language=session.user_language,
        frustration_score=session.frustration_score,
        escalation_reason=reason,
        referenced_orders=list(set(session.order_ids_referenced)),
        conversation_summary=summary,
        recommended_action=rec_action,
    )
    return ticket
