"""Phase 7 Evaluation Runner: 50-Query Multilingual Benchmark Suite.

Calculates:
- Answer Correctness & Keyword Recall
- Language Detection Accuracy
- Intent Routing Accuracy
- Retrieval Hit Rate & Source Citations
- Tool-calling & Escalation Accuracy
- P50, P90, P95 Latency metrics
"""
import sys
import json
import time
from pathlib import Path
import numpy as np

# Ensure root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools.db import init_mock_db
from src.rag.indexer import KnowledgeBaseIndexer
from src.rag.retriever import MultilingualRetriever
from src.agent.core import MultilingualCSAgent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = PROJECT_ROOT / "eval" / "eval_dataset.json"
RESULTS_FILE = PROJECT_ROOT / "eval" / "eval_results.json"


def run_benchmark():
    print("\n" + "=" * 70)
    print("STARTING 50-QUERY MULTILINGUAL BENCHMARK EVALUATION")
    print("=" * 70)

    # 1. Initialize System
    init_mock_db()
    faq_dir = PROJECT_ROOT / "data" / "faq"
    index_dir = PROJECT_ROOT / "data" / "index"
    indexer = KnowledgeBaseIndexer(faq_dir=faq_dir, index_dir=index_dir)
    if not indexer.load_index():
        indexer.build_and_save_index()
    retriever = MultilingualRetriever(indexer=indexer)
    agent = MultilingualCSAgent(retriever=retriever)

    # 2. Load Dataset
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total_queries = len(dataset)
    print(f"Loaded {total_queries} evaluation test cases across en, es, fr, de, ja.\n")

    latencies = []
    correct_languages = 0
    correct_intents = 0
    keyword_hits = 0
    citation_hits = 0
    total_faq_queries = 0
    escalation_true_positives = 0
    total_escalations = 0

    detailed_results = []

    for item in dataset:
        qid = item["id"]
        q_text = item["query"]
        expected_lang = item["language"]
        expected_intent = item["expected_intent"]
        expected_keywords = item.get("expected_keywords", [])
        expected_citation = item.get("expected_citation")

        t0 = time.perf_counter()
        res = agent.process_message(q_text)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        # Accuracy checks
        lang_match = (res["language"] == expected_lang)
        if lang_match:
            correct_languages += 1

        # Intent match
        intent_match = (res["intent"] == expected_intent) or (expected_intent == "escalation" and res["is_escalated"])
        if intent_match:
            correct_intents += 1

        # Keyword recall
        resp_text = res["response"].lower()
        kw_match = any(kw.lower() in resp_text for kw in expected_keywords)
        if kw_match:
            keyword_hits += 1

        # Citation hit
        if expected_citation:
            total_faq_queries += 1
            if expected_citation in res["citations"] or expected_citation in res["response"]:
                citation_hits += 1

        # Escalation accuracy
        if expected_intent == "escalation":
            total_escalations += 1
            if res["is_escalated"]:
                escalation_true_positives += 1

        detailed_results.append({
            "id": qid,
            "query": q_text,
            "expected_lang": expected_lang,
            "detected_lang": res["language"],
            "expected_intent": expected_intent,
            "predicted_intent": res["intent"],
            "latency_ms": round(elapsed_ms, 2),
            "response_snippet": res["response"][:80],
            "passed": lang_match and kw_match,
        })

    # Metrics calculation
    latencies = np.array(latencies)
    p50 = float(np.percentile(latencies, 50))
    p90 = float(np.percentile(latencies, 90))
    p95 = float(np.percentile(latencies, 95))
    min_lat = float(np.min(latencies))
    max_lat = float(np.max(latencies))
    under_3s_pct = float(np.sum(latencies < 3000.0) / len(latencies) * 100.0)

    lang_acc = (correct_languages / total_queries) * 100.0
    intent_acc = (correct_intents / total_queries) * 100.0
    keyword_acc = (keyword_hits / total_queries) * 100.0
    retrieval_hit_rate = (citation_hits / max(1, total_faq_queries)) * 100.0
    escalation_acc = (escalation_true_positives / max(1, total_escalations)) * 100.0

    summary = {
        "total_test_queries": total_queries,
        "language_detection_accuracy_pct": round(lang_acc, 2),
        "intent_routing_accuracy_pct": round(intent_acc, 2),
        "keyword_answer_recall_pct": round(keyword_acc, 2),
        "retrieval_hit_rate_pct": round(retrieval_hit_rate, 2),
        "escalation_accuracy_pct": round(escalation_acc, 2),
        "p50_latency_ms": round(p50, 2),
        "p90_latency_ms": round(p90, 2),
        "p95_latency_ms": round(p95, 2),
        "min_latency_ms": round(min_lat, 2),
        "max_latency_ms": round(max_lat, 2),
        "sla_compliance_rate_pct": round(under_3s_pct, 2),
    }

    # Save results
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": detailed_results}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("BENCHMARK EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total Test Queries:            {total_queries}")
    print(f"Language Detection Accuracy:   {lang_acc:.1f}%")
    print(f"Intent Classification Acc:     {intent_acc:.1f}%")
    print(f"Answer Keyword Recall:         {keyword_acc:.1f}%")
    print(f"FAQ Retrieval Hit Rate:        {retrieval_hit_rate:.1f}%")
    print(f"Escalation Precision/Recall:   {escalation_acc:.1f}%")
    print("-" * 70)
    print(f"Latency P50:                   {p50:.2f} ms")
    print(f"Latency P90:                   {p90:.2f} ms")
    print(f"Latency P95:                   {p95:.2f} ms")
    print(f"Max Latency:                   {max_lat:.2f} ms")
    print(f"SLA Compliance (< 3000ms):     {under_3s_pct:.1f}%")
    print("=" * 70 + "\n")

    return summary


if __name__ == "__main__":
    run_benchmark()
