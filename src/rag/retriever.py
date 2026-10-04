"""Retriever module providing similarity search and citation formatting."""
from typing import List, Dict, Any, Optional
import time
import numpy as np

from src.rag.indexer import KnowledgeBaseIndexer


class MultilingualRetriever:
    def __init__(
        self,
        indexer: KnowledgeBaseIndexer,
        top_k: int = 3,
        min_score_threshold: float = 0.25,
    ):
        self.indexer = indexer
        self.top_k = top_k
        self.min_score_threshold = min_score_threshold

        # Ensure index is loaded or created
        if self.indexer.index is None:
            if not self.indexer.load_index():
                self.indexer.build_and_save_index()

    def retrieve(
        self,
        query: str,
        language: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Retrieves top-k relevant documents with citations and latency tracking."""
        start_time = time.perf_counter()
        k = top_k or self.top_k

        # 1. Embed query
        query_vec = self.indexer.embedding_service.embed_query(query)
        query_vec = np.expand_dims(query_vec, axis=0)

        # 2. Search FAISS index
        # Query a generous candidate pool so language filtering never starves relevant docs
        total_docs = len(self.indexer.documents)
        search_k = min(total_docs, max(k * 10, 30))
        scores, indices = self.indexer.index.search(query_vec, search_k)

        raw_scores = scores[0]
        raw_indices = indices[0]

        results: List[Dict[str, Any]] = []
        for score, idx in zip(raw_scores, raw_indices):
            if idx < 0 or idx >= len(self.indexer.documents):
                continue
            if score < self.min_score_threshold:
                continue

            doc = self.indexer.documents[idx]

            # Optional language preference: boost or match language if provided
            if language and doc.get("language") != language:
                # Still allow cross-lingual if high confidence, but prioritize target language
                pass

            doc_result = {
                "id": doc["id"],
                "title": doc["title"],
                "category": doc["category"],
                "language": doc.get("language", "en"),
                "content": doc["content"],
                "source_url": doc["source_url"],
                "similarity_score": round(float(score), 4),
            }
            results.append(doc_result)

        # If language specified, prioritize matching language first
        if language:
            same_lang = [d for d in results if d["language"] == language]
            diff_lang = [d for d in results if d["language"] != language]
            results = (same_lang + diff_lang)[:k]
        else:
            results = results[:k]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "query": query,
            "language": language,
            "latency_ms": round(elapsed_ms, 2),
            "results": results,
            "context_prompt": self.format_context(results),
        }

    def format_context(self, docs: List[Dict[str, Any]]) -> str:
        """Formats retrieved documents into LLM context with explicit citations."""
        if not docs:
            return "No relevant policy documents found."

        context_blocks = []
        for d in docs:
            block = (
                f"[SOURCE: {d['id']} | Title: {d['title']} | URL: {d['source_url']}]\n"
                f"{d['content']}"
            )
            context_blocks.append(block)
        return "\n\n".join(context_blocks)
