"""Data models for Enterprise Hybrid RAG pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RetrievalStrategy(str, Enum):
    """Supported retrieval strategies for benchmarking and querying."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class DocumentMetadata(BaseModel):
    """Metadata attributes associated with an ingested technical document."""
    doc_id: str
    title: str
    doc_type: str = "technical_doc"  # e.g., "architecture_spec", "rfc", "runbook", "api_doc"
    domain: str = "infrastructure"  # e.g., "database", "security", "microservices", "networking"
    tags: List[str] = Field(default_factory=list)
    version: str = "1.0.0"
    created_at: Optional[str] = None
    source_path: Optional[str] = None
    author: Optional[str] = None


class Document(BaseModel):
    """Raw document entity before chunking."""
    id: str
    content: str
    metadata: DocumentMetadata


class ChunkMetadata(BaseModel):
    """Enriched metadata associated with each chunk for context-aware filtering."""
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_type: str
    domain: str
    tags: List[str] = Field(default_factory=list)
    section_path: str = ""  # Hierarchical breadcrumb, e.g., "Architecture > Storage Engine > Sharding"
    chunk_index: int = 0
    total_chunks: int = 1
    char_count: int = 0
    token_count: int = 0


class Chunk(BaseModel):
    """Context-aware chunk containing text, breadcrumbs, and enriched metadata."""
    id: str
    content: str  # Enriched chunk text (Context header + section content)
    raw_content: str  # Raw chunk text without prepended breadcrumbs
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None


class ScoredChunk(BaseModel):
    """Retrieved chunk annotated with individual and fused retrieval scores."""
    chunk: Chunk
    score: float  # Final ranking score
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None
    rank: int = 0


class RetrievalQuery(BaseModel):
    """Search request query payload with tunable parameters and metadata filters."""
    query: str
    top_k: int = 5
    dense_top_k: int = 25
    sparse_top_k: int = 25
    alpha: float = 0.5  # Convex combination weight: score = alpha * dense + (1 - alpha) * sparse
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RERANK
    metadata_filter: Optional[Dict[str, Any]] = None  # e.g. {"domain": "database", "doc_type": "rfc"}
    reranker_model: Optional[str] = None


class Citation(BaseModel):
    """Grounded reference citation for answer faithfulness."""
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    snippet: str
    confidence: float


class RAGResponse(BaseModel):
    """Final grounded Q&A response with citations, retrieved chunks, and telemetry."""
    query: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    retrieved_chunks: List[ScoredChunk] = Field(default_factory=list)
    strategy_used: RetrievalStrategy
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    metadata_filters_applied: Optional[Dict[str, Any]] = None


class EvaluationSample(BaseModel):
    """Gold-standard evaluation instance for RAGAS and benchmarking."""
    query: str
    ground_truth_answer: str
    ground_truth_doc_ids: List[str]
    domain: str = "general"


class StrategyBenchmarkResult(BaseModel):
    """Performance & accuracy metrics for a specific retrieval strategy."""
    strategy: str
    hit_rate: float  # Fraction of queries where ground truth document was in top-K
    context_recall: float
    context_precision: float
    faithfulness: float  # % of response claims grounded in context (RAGAS target: 95%)
    answer_relevance: float
    mean_latency_ms: float
    total_queries: int
