"""Phase 4 Test Suite: Session Memory, Sentiment Thresholding, and Human Escalation."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools.db import init_mock_db
from src.agent.memory import session_manager
from src.agent.sentiment import compute_sentiment_score
from src.agent.escalation import generate_handoff_ticket
from src.agent.core import MultilingualCSAgent
from src.rag.indexer import KnowledgeBaseIndexer
from src.rag.retriever import MultilingualRetriever

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def agent():
    init_mock_db()
    faq_dir = PROJECT_ROOT / "data" / "faq"
    index_dir = PROJECT_ROOT / "data" / "index"
    indexer = KnowledgeBaseIndexer(faq_dir=faq_dir, index_dir=index_dir)
    indexer.load_index()
    retriever = MultilingualRetriever(indexer=indexer)
    return MultilingualCSAgent(retriever=retriever)


def test_sentiment_scoring():
    # Calm queries
    assert compute_sentiment_score("Can you check my order status please?", "en") < 0.2
    assert compute_sentiment_score("Hola, ¿dónde está mi pedido?", "es") < 0.2

    # High frustration queries
    frustrated_en = compute_sentiment_score("This is RIDICULOUS!! You are a complete scam, refund me NOW!!", "en")
    assert frustrated_en >= 0.60

    frustrated_es = compute_sentiment_score("¡¡Esto es una estafa inaceptable!! ¡Exijo una solución ya!", "es")
    assert frustrated_es >= 0.50

    frustrated_ja = compute_sentiment_score("最悪の対応です。完全に詐欺です！！", "ja")
    assert frustrated_ja >= 0.50


def test_multi_turn_session_memory(agent):
    session_id = "test_sess_001"
    
    # Turn 1: Ask about shipping
    res1 = agent.process_message("How long does standard shipping take?", session_id=session_id)
    assert res1["session_id"] == session_id
    assert res1["is_escalated"] is False

    # Turn 2: Mention an order
    res2 = agent.process_message("Where is ORD-1001?", session_id=session_id)
    assert res2["session_id"] == session_id
    assert "ORD-1001" in res2["response"]

    # Verify session retains memory
    sess = session_manager.get_or_create(session_id)
    assert len(sess.turns) == 4  # 2 user turns + 2 assistant turns
    assert "ORD-1001" in sess.order_ids_referenced


def test_sentiment_escalation_and_handoff_ticket(agent):
    session_id = "test_sess_frustrated"

    # Turn 1: Normal query
    agent.process_message("Where is my package for ORD-1004?", session_id=session_id)

    # Turn 2: Angry response that trips frustration threshold
    res2 = agent.process_message(
        "THIS IS RIDICULOUS!! My package is 45 days late, you are a scam! Give me my money back NOW!!",
        session_id=session_id,
    )

    assert res2["is_escalated"] is True
    assert res2["handoff_ticket"] is not None
    ticket = res2["handoff_ticket"]
    assert ticket["ticket_id"].startswith("ESC-")
    assert "ORD-1004" in ticket["referenced_orders"]
    assert ticket["frustration_score"] >= 0.60
    assert "Customer" in ticket["conversation_summary"]
    assert "Check logistics" in ticket["recommended_action"]


def test_japanese_escalation_handoff(agent):
    session_id = "test_sess_ja_esc"
    res = agent.process_message("全く使い物になりません。今すぐ人間の担当者に代わってください！！", session_id=session_id)
    assert res["language"] == "ja"
    assert res["is_escalated"] is True
    assert res["handoff_ticket"] is not None
    assert res["handoff_ticket"]["language"] == "ja"


if __name__ == "__main__":
    pytest.main(["-v", str(Path(__file__))])
