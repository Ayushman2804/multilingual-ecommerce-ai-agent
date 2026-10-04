"""FAISS Indexer for Multilingual Support Corpus.

Loads FAQ corpora, generates embeddings, and serializes in-process FAISS IndexFlatIP.
"""
from typing import List, Dict, Any, Optional
import json
import logging
from pathlib import Path
import faiss
import numpy as np

from src.rag.embeddings import MultilingualEmbeddingService

logger = logging.getLogger(__name__)


class KnowledgeBaseIndexer:
    def __init__(
        self,
        faq_dir: Path,
        index_dir: Path,
        embedding_service: Optional[MultilingualEmbeddingService] = None,
    ):
        self.faq_dir = Path(faq_dir)
        self.index_dir = Path(index_dir)
        self.embedding_service = embedding_service or MultilingualEmbeddingService()
        self.index_file = self.index_dir / "faq_index.bin"
        self.metadata_file = self.index_dir / "documents.json"
        
        self.index: Optional[faiss.Index] = None
        self.documents: List[Dict[str, Any]] = []

    def load_corpus(self) -> List[Dict[str, Any]]:
        """Loads and flattens all language FAQ files from faq_dir."""
        documents = []
        for file_path in self.faq_dir.glob("faq_*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    items = json.load(f)
                    for item in items:
                        # Composite text representation for embedding
                        item["embed_text"] = f"Title: {item['title']}\nCategory: {item['category']}\nContent: {item['content']}"
                        documents.append(item)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
        self.documents = documents
        return documents

    def build_and_save_index(self) -> faiss.Index:
        """Embeds corpus and builds FAISS IndexFlatIP (cosine similarity)."""
        if not self.documents:
            self.load_corpus()

        if not self.documents:
            raise ValueError(f"No documents found in {self.faq_dir}")

        texts = [doc["embed_text"] for doc in self.documents]
        logger.info(f"Embedding {len(texts)} FAQ documents...")
        vectors = self.embedding_service.embed_texts(texts)

        # IndexFlatIP expects normalized vectors for cosine similarity
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        self.index = index

        # Persist index and metadata
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_file))
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

        logger.info(f"Successfully built and persisted FAISS index with {index.ntotal} vectors to {self.index_file}")
        return index

    def load_index(self) -> bool:
        """Loads index and metadata from disk if available."""
        if self.index_file.exists() and self.metadata_file.exists():
            self.index = faiss.read_index(str(self.index_file))
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            logger.info(f"Loaded existing FAISS index with {self.index.ntotal} documents.")
            return True
        return False
