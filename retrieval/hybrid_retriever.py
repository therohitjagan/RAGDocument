"""Unified Hybrid Retriever orchestrating ChromaDB, BM25, Fusion, and Reranking."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from core.config import settings
from core.models import (
    Chunk,
    RetrievalQuery,
    RetrievalStrategy,
    ScoredChunk,
)
from retrieval.dense import DenseChromaRetriever
from retrieval.fusion import HybridFusion
from retrieval.reranker import CrossEncoderReranker
from retrieval.sparse import SparseBM25Retriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Enterprise-grade Hybrid Retriever orchestrating:
    1. Pre-filtering by metadata (domain, doc_type, tags)
    2. Dense semantic retrieval via ChromaDB
    3. Sparse keyword search via BM25
    4. Convex combination or Reciprocal Rank Fusion (RRF)
    5. Cross-Encoder reranking for precision scoring
    """

    def __init__(
        self,
        dense_retriever: Optional[DenseChromaRetriever] = None,
        sparse_retriever: Optional[SparseBM25Retriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.dense = dense_retriever or DenseChromaRetriever()
        self.sparse = sparse_retriever or SparseBM25Retriever(index_dir=settings.BM25_INDEX_DIR)
        self.reranker = reranker or CrossEncoderReranker()

    def index_chunks(self, chunks: List[Chunk], append: bool = True) -> None:
        """
        Dual-index chunks into both ChromaDB and BM25 index stores.
        """
        logger.info(f"Indexing {len(chunks)} chunks into ChromaDB and BM25...")
        self.dense.add_chunks(chunks)
        self.sparse.index_chunks(chunks, append=append)
        self.sparse.save()
        logger.info("Indexing completed successfully.")

    def retrieve(self, query: RetrievalQuery) -> List[ScoredChunk]:
        """
        Execute retrieval based on the requested strategy and parameter set.
        """
        start_time = time.time()
        strategy = query.strategy
        q_text = query.query.strip()
        top_k = query.top_k
        meta_filter = query.metadata_filter

        if not q_text:
            return []

        # 1. PURE DENSE
        if strategy == RetrievalStrategy.DENSE:
            results = self.dense.search(
                query=q_text,
                top_k=top_k,
                metadata_filter=meta_filter,
            )
            return results

        # 2. PURE SPARSE (BM25)
        if strategy == RetrievalStrategy.SPARSE:
            results = self.sparse.search(
                query=q_text,
                top_k=top_k,
                metadata_filter=meta_filter,
            )
            return results

        # 3. HYBRID (Dense + Sparse Fusion)
        dense_candidates = self.dense.search(
            query=q_text,
            top_k=query.dense_top_k,
            metadata_filter=meta_filter,
        )
        sparse_candidates = self.sparse.search(
            query=q_text,
            top_k=query.sparse_top_k,
            metadata_filter=meta_filter,
        )

        fused_candidates = HybridFusion.convex_combination(
            dense_results=dense_candidates,
            sparse_results=sparse_candidates,
            alpha=query.alpha,
            top_k=max(top_k * 2, 20),
        )

        if strategy == RetrievalStrategy.HYBRID:
            return fused_candidates[:top_k]

        # 4. HYBRID + RERANK
        if strategy == RetrievalStrategy.HYBRID_RERANK:
            reranked = self.reranker.rerank(
                query=q_text,
                candidates=fused_candidates,
                top_k=top_k,
            )
            return reranked

        # Default fallback
        return fused_candidates[:top_k]
