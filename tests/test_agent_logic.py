"""Phase 3 Test Suite: Agent Logic, Multilingual Routing, and Tool Calling."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools.db import init_mock_db
from src.tools.order_tools import lookup_order, request_return_or_refund, lookup_product
from src.agent.lang_detector import detect_language
from src.agent.router import route_intent, Intent
from src.agent.core import MultilingualCSAgent
from src.rag.embeddings import MultilingualEmbeddingService
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


def test_language_detection():
    assert detect_language("Where is my package?") == "en"
    assert detect_language("¿Dónde está mi pedido?") == "es"
    assert detect_language("Bonjour, où est ma commande ?") == "fr"
    assert detect_language("Wo ist meine Lieferung?") == "de"
    assert detect_language("注文した荷物はどこですか？") == "ja"


def test_intent_routing():
    assert route_intent("Track ORD-1001")["intent"] == Intent.ORDER_LOOKUP
    assert route_intent("I want to return ORD-1001")["intent"] == Intent.RETURN_REQUEST
    assert route_intent("Do you sell noise-canceling headphones?")["intent"] == Intent.PRODUCT_INQUIRY
    assert route_intent("What is your refund policy?")["intent"] == Intent.FAQ
    assert route_intent("This is terrible, I want to talk to a human manager!")["intent"] == Intent.ESCALATION


def test_order_lookup_tool():
    res = lookup_order("ORD-1001")
    assert res["found"] is True
    assert res["status"] == "delivered"
    assert res["customer_name"] == "John Doe"

    not_found = lookup_order("ORD-9999")
    assert not_found["found"] is False


def test_refund_tool_within_and_outside_30_days():
    # ORD-1001 was delivered 12 days ago -> eligible
    res_valid = request_return_or_refund("ORD-1001", "Does not fit")
    assert res_valid["success"] is True
    assert "refund_id" in res_valid

    # ORD-1004 was delivered 45 days ago -> ineligible
    res_expired = request_return_or_refund("ORD-1004", "Changed mind")
    assert res_expired["success"] is False
    assert "exceeding the 30-day" in res_expired["error"]


def test_multilingual_agent_execution(agent):
    # English Order Lookup
    res_en = agent.process_message("Where is my order ORD-1001?")
    assert res_en["language"] == "en"
    assert "ORD-1001" in res_en["response"]
    assert res_en["latency_ms"] < 200.0

    # Spanish Order Lookup
    res_es = agent.process_message("¿Dónde está mi pedido ORD-1002?")
    assert res_es["language"] == "es"
    assert "ORD-1002" in res_es["response"]

    # French Return Request
    res_fr = agent.process_message("Je voudrais retourner ORD-1005")
    assert res_fr["language"] == "fr"
    assert "ORD-1005" in res_fr["response"] or "Retour" in res_fr["response"]

    # German Escalation
    res_de = agent.process_message("Ich will sofort mit einem menschlichen Mitarbeiter sprechen!")
    assert res_de["is_escalated"] is True
    assert "Kundendienstmitarbeiter" in res_de["response"]

    # Japanese Catalog Search
    res_ja = agent.process_message("モニターの在庫はありますか？")
    assert res_ja["language"] == "ja"
    assert res_ja["intent"] == "product_inquiry"
    assert "4K" in res_ja["response"]


if __name__ == "__main__":
    pytest.main(["-v", str(Path(__file__))])
