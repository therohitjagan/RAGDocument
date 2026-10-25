"""Ingestion package for document loading, synthetic corpus generation, and context-aware chunking."""

from ingestion.chunker import ContextAwareChunker
from ingestion.loaders import DocumentLoader
from ingestion.pipeline import IngestionPipeline
from ingestion.synthetic_corpus import generate_enterprise_corpus

__all__ = [
    "ContextAwareChunker",
    "DocumentLoader",
    "IngestionPipeline",
    "generate_enterprise_corpus",
]
