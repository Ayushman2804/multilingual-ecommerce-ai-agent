"""Multilingual embedding generator using sentence-transformers.

Optimized for sub-30ms CPU inference using paraphrase-multilingual-MiniLM-L12-v2.
Outputs L2-normalized vectors for direct cosine similarity computation in FAISS IndexFlatIP.
"""
from typing import List, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)


class MultilingualEmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self._model = None
        self._dimension = 384

    @property
    def dimension(self) -> int:
        return self._dimension

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading multilingual embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                if hasattr(self._model, "get_embedding_dimension"):
                    self._dimension = self._model.get_embedding_dimension()
                else:
                    self._dimension = self._model.get_sentence_embedding_dimension()
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer model ({e}). Using deterministic hash fallback.")
                self._model = "fallback"

    def embed_texts(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Embeds single string or list of strings into L2-normalized float32 numpy array."""
        if isinstance(texts, str):
            texts = [texts]

        self._load_model()

        if self._model != "fallback":
            # Real SentenceTransformer inference
            embeddings = self._model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embeddings.astype(np.float32)

        # Fast deterministic fallback for offline / test environments
        return self._generate_fallback_embeddings(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string."""
        return self.embed_texts([query])[0]

    def _generate_fallback_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generates deterministic pseudo-semantic normalized vectors for testing without internet."""
        import hashlib
        vectors = []
        for text in texts:
            words = text.lower().split()
            vec = np.zeros(self._dimension, dtype=np.float32)
            for i, word in enumerate(words):
                h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
                idx = h % self._dimension
                vec[idx] += 1.0 / (i + 1)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            else:
                vec[0] = 1.0
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)
