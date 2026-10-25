"""Configuration settings for Enterprise Hybrid RAG pipeline."""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable override support."""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Directory Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    CHROMA_PERSIST_DIR: Path = DATA_DIR / "chroma_db"
    BM25_INDEX_DIR: Path = DATA_DIR / "bm25_index"
    EVALUATION_DIR: Path = DATA_DIR / "evaluations"

    # ChromaDB Collection
    CHROMA_COLLECTION_NAME: str = "technical_docs_v1"

    # Embedding & Reranker Models
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Ingestion & Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    MIN_CHUNK_SIZE: int = 50

    # Hybrid Retrieval Defaults
    DEFAULT_ALPHA: float = 0.5  # 0.0 = pure sparse BM25, 1.0 = pure dense vector
    DEFAULT_TOP_K: int = 5
    DEFAULT_DENSE_TOP_K: int = 25
    DEFAULT_SPARSE_TOP_K: int = 25
    RRF_K: int = 60  # Constant for Reciprocal Rank Fusion

    # LLM Settings
    LLM_PROVIDER: str = "local"  # "local", "openai", "groq", "ollama"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    TEMPERATURE: float = 0.0

    # API Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000


settings = Settings()
