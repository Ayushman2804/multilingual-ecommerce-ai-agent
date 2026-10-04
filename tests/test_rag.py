"""Phase 2 Test Suite: Knowledge Base, Embeddings, FAISS Index, and Retrieval Citations."""
import sys
from pathlib import Path
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.embeddings import MultilingualEmbeddingService
from src.rag.indexer import KnowledgeBaseIndexer
from src.rag.retriever import MultilingualRetriever

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FAQ_DIR = PROJECT_ROOT / "data" / "faq"
INDEX_DIR = PROJECT_ROOT / "data" / "index"


@pytest.fixture(scope="module")
def retriever():
    # Use fallback embeddings for lightning-fast test execution if needed
    embedding_service = MultilingualEmbeddingService()
    indexer = KnowledgeBaseIndexer(faq_dir=FAQ_DIR, index_dir=INDEX_DIR, embedding_service=embedding_service)
    indexer.build_and_save_index()
    return MultilingualRetriever(indexer=indexer, top_k=3)


def test_index_corpus_size(retriever):
    """Ensures all 5 languages are loaded and indexed."""
    docs = retriever.indexer.documents
    assert len(docs) >= 15, f"Expected at least 15 FAQ docs across 5 languages, found {len(docs)}"
    
    languages = {d["language"] for d in docs}
    expected = {"en", "es", "fr", "de", "ja"}
    assert expected.issubset(languages), f"Missing languages. Found: {languages}"


def test_retrieval_english(retriever):
    res = retriever.retrieve("What is your return policy for damaged items?", language="en")
    assert res["latency_ms"] < 100.0, f"Retrieval too slow: {res['latency_ms']}ms"
    assert len(res["results"]) > 0
    top_doc = res["results"][0]
    assert "return" in top_doc["category"].lower()
    assert "source_url" in top_doc
    assert "[SOURCE:" in res["context_prompt"]


def test_retrieval_spanish(retriever):
    res = retriever.retrieve("¿Cuánto tarda el envío exprés?", language="es")
    assert len(res["results"]) > 0
    top_doc = res["results"][0]
    assert top_doc["language"] == "es"
    assert "shipping" in top_doc["category"].lower() or "envios" in top_doc["source_url"]


def test_retrieval_french(retriever):
    res = retriever.retrieve("Est-ce que les téléphones sont sous garantie ?", language="fr")
    assert len(res["results"]) > 0
    top_doc = res["results"][0]
    assert top_doc["language"] == "fr"
    assert "warranty" in top_doc["category"].lower()


def test_retrieval_german(retriever):
    res = retriever.retrieve("Welche Zahlungsmethoden werden akzeptiert?", language="de")
    assert len(res["results"]) > 0
    top_doc = res["results"][0]
    assert top_doc["language"] == "de"
    assert "payments" in top_doc["category"].lower()


def test_retrieval_japanese(retriever):
    res = retriever.retrieve("返金はいつ口座に反映されますか？", language="ja")
    assert len(res["results"]) > 0
    top_doc = res["results"][0]
    assert top_doc["language"] == "ja"
    assert "payments" in top_doc["category"].lower() or "returns" in top_doc["category"].lower()


def test_retrieval_citations_format(retriever):
    res = retriever.retrieve("How long is warranty coverage?", language="en")
    prompt = res["context_prompt"]
    assert "[SOURCE:" in prompt
    assert "URL: https://" in prompt


if __name__ == "__main__":
    pytest.main(["-v", str(Path(__file__))])
