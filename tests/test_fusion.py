"""Unit tests for Hybrid Fusion (RRF and Convex Combination)."""

import pytest
from core.models import Chunk, ChunkMetadata, ScoredChunk
from retrieval.fusion import HybridFusion


def make_dummy_chunk(chunk_id: str, title: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        content=f"Content for {title}",
        raw_content=f"Raw {title}",
        metadata=ChunkMetadata(
            chunk_id=chunk_id,
            doc_id=f"doc_{chunk_id}",
            doc_title=title,
            doc_type="rfc",
            domain="systems",
        ),
    )


def test_reciprocal_rank_fusion():
    c1 = make_dummy_chunk("c1", "Doc 1")
    c2 = make_dummy_chunk("c2", "Doc 2")
    c3 = make_dummy_chunk("c3", "Doc 3")

    # Dense results: c1, c2, c3
    dense = [
        ScoredChunk(chunk=c1, score=0.95),
        ScoredChunk(chunk=c2, score=0.85),
        ScoredChunk(chunk=c3, score=0.75),
    ]

    # Sparse results: c2, c1
    sparse = [
        ScoredChunk(chunk=c2, score=12.5),
        ScoredChunk(chunk=c1, score=10.0),
    ]

    fused = HybridFusion.reciprocal_rank_fusion(dense, sparse, k=60, top_k=3)
    assert len(fused) == 3
    # c1 and c2 should both be scored high because they appear in both
    top_ids = [item.chunk.id for item in fused[:2]]
    assert "c1" in top_ids and "c2" in top_ids


def test_convex_combination_weighting():
    c1 = make_dummy_chunk("c1", "Doc 1")
    c2 = make_dummy_chunk("c2", "Doc 2")

    # Dense favors c1 strongly
    dense = [
        ScoredChunk(chunk=c1, score=1.0),
        ScoredChunk(chunk=c2, score=0.1),
    ]
    # Sparse favors c2 strongly
    sparse = [
        ScoredChunk(chunk=c1, score=1.0),
        ScoredChunk(chunk=c2, score=20.0),
    ]

    # With alpha = 0.9 (mostly dense), c1 should win
    fused_dense_favored = HybridFusion.convex_combination(dense, sparse, alpha=0.9, top_k=2)
    assert fused_dense_favored[0].chunk.id == "c1"

    # With alpha = 0.1 (mostly sparse), c2 should win
    fused_sparse_favored = HybridFusion.convex_combination(dense, sparse, alpha=0.1, top_k=2)
    assert fused_sparse_favored[0].chunk.id == "c2"
