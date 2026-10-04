"""Configuration and environment settings for the Multilingual Customer Support Agent."""
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Multilingual CS AI Agent"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    
    # Supported Languages
    SUPPORTED_LANGUAGES: List[str] = ["en", "es", "fr", "de", "ja"]
    DEFAULT_LANGUAGE: str = "en"

    # Latency Targets (milliseconds)
    TARGET_TOTAL_LATENCY_MS: float = 3000.0
    TARGET_TTFT_MS: float = 600.0  # Time To First Token (streaming)
    MAX_RAG_LATENCY_MS: float = 120.0
    MAX_ROUTER_LATENCY_MS: float = 40.0
    MAX_TOOL_LATENCY_MS: float = 80.0

    # Embeddings & Vector DB
    # Tradeoff recommendation: 384-dim multilingual MiniLM is ~25ms on CPU vs >120ms for BGE-M3
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    FAISS_INDEX_TYPE: str = "IndexFlatIP"  # Inner product for normalized cosine similarity
    TOP_K_RETRIEVAL: int = 3
    SIMILARITY_THRESHOLD: float = 0.65

    # Model Defaults (Quantized Open Model or Hosted Fallback)
    PRIMARY_MODEL: str = "qwen2.5-7b-instruct"
    FALLBACK_MODEL: str = "gemini-1.5-flash"
    MODEL_TEMPERATURE: float = 0.1
    MAX_GENERATION_TOKENS: int = 250

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
