"""FastAPI production server exposing enterprise Hybrid RAG endpoints."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.config import settings
from core.models import (
    Citation,
    Document,
    DocumentMetadata,
    RAGResponse,
    RetrievalQuery,
    RetrievalStrategy,
    ScoredChunk,
    StrategyBenchmarkResult,
)
from evaluation.benchmark_runner import BenchmarkRunner
from generation.synthesizer import GroundedSynthesizer
from ingestion.chunker import ContextAwareChunker
from retrieval.hybrid_retriever import HybridRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_api")

app = FastAPI(
    title="Enterprise Hybrid RAG API",
    description="Dense ChromaDB + Sparse BM25 + Cross-Encoder Reranking over 5,000+ Technical Documents",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton state components
retriever: Optional[HybridRetriever] = None
synthesizer: Optional[GroundedSynthesizer] = None
chunker: Optional[ContextAwareChunker] = None


def get_retriever() -> HybridRetriever:
    global retriever
    if retriever is None:
        retriever = HybridRetriever()
    return retriever


def get_synthesizer() -> GroundedSynthesizer:
    global synthesizer
    if synthesizer is None:
        synthesizer = GroundedSynthesizer()
    return synthesizer


def get_chunker() -> ContextAwareChunker:
    global chunker
    if chunker is None:
        chunker = ContextAwareChunker()
    return chunker


class IngestDocumentRequest(BaseModel):
    doc_id: str
    title: str
    content: str
    domain: str = "general"
    doc_type: str = "technical_doc"
    tags: List[str] = Field(default_factory=list)


@app.get("/health")
@app.get("/api/v1/health")
async def health_check():
    """System health check and index metrics."""
    r = get_retriever()
    dense_count = r.dense.count()
    sparse_count = r.sparse.count()
    return {
        "status": "healthy",
        "chroma_vectors": dense_count,
        "bm25_chunks": sparse_count,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "reranker_model": settings.RERANKER_MODEL_NAME,
        "llm_provider": settings.LLM_PROVIDER,
    }


@app.post("/api/v1/query", response_model=RAGResponse)
async def query_rag(request: RetrievalQuery):
    """
    Execute hybrid retrieval and grounded answer synthesis.
    """
    r = get_retriever()
    s = get_synthesizer()

    t_start = time.time()
    retrieved = r.retrieve(request)
    retrieval_ms = (time.time() - t_start) * 1000.0

    response = s.synthesize(
        query=request.query,
        retrieved_chunks=retrieved,
        strategy_used=request.strategy,
        retrieval_latency_ms=retrieval_ms,
    )
    response.metadata_filters_applied = request.metadata_filter
    return response


@app.post("/api/v1/ingest")
async def ingest_document(payload: IngestDocumentRequest):
    """
    Ingest and dual-index a new document on the fly.
    """
    c = get_chunker()
    r = get_retriever()

    meta = DocumentMetadata(
        doc_id=payload.doc_id,
        title=payload.title,
        domain=payload.domain,
        doc_type=payload.doc_type,
        tags=payload.tags,
    )
    doc = Document(id=payload.doc_id, content=payload.content, metadata=meta)
    chunks = c.chunk_document(doc)

    if not chunks:
        raise HTTPException(status_code=400, detail="Document could not be chunked.")

    r.index_chunks(chunks)

    return {
        "message": f"Successfully ingested and indexed document '{payload.doc_id}'",
        "chunks_created": len(chunks),
    }


@app.get("/api/v1/documents")
async def get_documents_summary():
    """Summary of indexed documents and domain distribution."""
    r = get_retriever()
    total_chunks = r.sparse.count()

    domains: Dict[str, int] = {}
    doc_types: Dict[str, int] = {}

    for c in r.sparse.chunks:
        m = c.metadata
        domains[m.domain] = domains.get(m.domain, 0) + 1
        doc_types[m.doc_type] = doc_types.get(m.doc_type, 0) + 1

    return {
        "total_chunks": total_chunks,
        "domain_distribution": domains,
        "doc_type_distribution": doc_types,
    }


@app.post("/api/v1/evaluate", response_model=List[StrategyBenchmarkResult])
async def trigger_evaluation(top_k: int = 5, alpha: float = 0.5):
    """
    Trigger RAGAS benchmark evaluation across all retrieval strategies.
    """
    r = get_retriever()
    runner = BenchmarkRunner(retriever=r)
    try:
        samples = runner.load_test_samples()
        if not samples:
            raise HTTPException(
                status_code=404,
                detail="Gold benchmark dataset not found. Please generate the corpus first.",
            )
        # Evaluate on up to 25 benchmark queries for responsive API execution
        results = runner.run_and_report(samples=samples[:25], top_k=top_k, alpha=alpha)
        return results
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
