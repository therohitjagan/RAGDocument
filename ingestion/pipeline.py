"""High-throughput document ingestion pipeline orchestrator."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, List, Optional
from core.config import settings
from core.models import Chunk, Document
from ingestion.chunker import ContextAwareChunker
from ingestion.loaders import DocumentLoader

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Coordinates batch loading, context-aware chunking, and dual-index dispatch
    to dense (ChromaDB) and sparse (BM25) search engines.
    """

    def __init__(
        self,
        chunker: Optional[ContextAwareChunker] = None,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ):
        self.chunker = chunker or ContextAwareChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def process_documents(
        self,
        documents: List[Document],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Chunk]:
        """
        Deconstruct a list of documents into enriched context-aware chunks.
        """
        all_chunks: List[Chunk] = []
        total_docs = len(documents)

        start_time = time.time()
        for idx, doc in enumerate(documents):
            doc_chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(doc_chunks)

            if progress_callback and (idx % 100 == 0 or idx == total_docs - 1):
                progress_callback(idx + 1, total_docs)

        elapsed = time.time() - start_time
        logger.info(
            f"Successfully processed {total_docs} documents into {len(all_chunks)} "
            f"context-aware chunks in {elapsed:.2f}s."
        )
        return all_chunks

    def ingest_from_directory(
        self,
        directory_path: str | Path,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Chunk]:
        """
        Load all documents from a folder and chunk them.
        """
        docs = DocumentLoader.load_directory(directory_path)
        return self.process_documents(docs, progress_callback=progress_callback)
