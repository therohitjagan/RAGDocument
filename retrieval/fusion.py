"""Hybrid retrieval fusion algorithms: Reciprocal Rank Fusion (RRF) and Convex Combination."""

from __future__ import annotations

from typing import Dict, List, Optional
from core.models import Chunk, ScoredChunk


class HybridFusion:
    """
    Combines dense semantic vector retrieval and sparse keyword (BM25) retrieval results
    using either Reciprocal Rank Fusion (RRF) or normalized Alpha Convex Combination.
    """

    @staticmethod
    def _min_max_normalize(scores: List[float]) -> List[float]:
        """Normalize an arbitrary list of numeric scores to the range [0.0, 1.0]."""
        if not scores:
            return []
        min_s = min(scores)
        max_s = max(scores)
        if max_s - min_s < 1e-9:
            return [1.0 for _ in scores]
        return [(s - min_s) / (max_s - min_s) for s in scores]

    @classmethod
    def reciprocal_rank_fusion(
        cls,
        dense_results: List[ScoredChunk],
        sparse_results: List[ScoredChunk],
        k: int = 60,
        top_k: int = 10,
    ) -> List[ScoredChunk]:
        """
        Rank-based Reciprocal Rank Fusion (RRF):
        RRF_score(chunk) = 1 / (k + rank_dense) + 1 / (k + rank_sparse)
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Chunk] = {}
        dense_score_map: Dict[str, float] = {}
        sparse_score_map: Dict[str, float] = {}

        # Process dense rankings
        for rank, item in enumerate(dense_results, start=1):
            c_id = item.chunk.id
            chunk_map[c_id] = item.chunk
            dense_score_map[c_id] = item.score
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + (1.0 / (k + rank))

        # Process sparse rankings
        for rank, item in enumerate(sparse_results, start=1):
            c_id = item.chunk.id
            chunk_map[c_id] = item.chunk
            sparse_score_map[c_id] = item.score
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + (1.0 / (k + rank))

        # Build fused list
        fused: List[ScoredChunk] = []
        for c_id, score in rrf_scores.items():
            fused.append(
                ScoredChunk(
                    chunk=chunk_map[c_id],
                    score=score,
                    dense_score=dense_score_map.get(c_id),
                    sparse_score=sparse_score_map.get(c_id),
                )
            )

        fused.sort(key=lambda x: x.score, reverse=True)
        top_fused = fused[:top_k]

        for rank, item in enumerate(top_fused, start=1):
            item.rank = rank

        return top_fused

    @classmethod
    def convex_combination(
        cls,
        dense_results: List[ScoredChunk],
        sparse_results: List[ScoredChunk],
        alpha: float = 0.5,
        top_k: int = 10,
    ) -> List[ScoredChunk]:
        """
        Normalized Weighted Score Combination:
        Score = alpha * Normalized(Dense) + (1 - alpha) * Normalized(Sparse)
        where alpha=1.0 is pure dense, and alpha=0.0 is pure sparse.
        """
        alpha = max(0.0, min(1.0, alpha))
        chunk_map: Dict[str, Chunk] = {}
        dense_raw_scores: Dict[str, float] = {}
        sparse_raw_scores: Dict[str, float] = {}

        # Collect candidate IDs and raw scores
        for item in dense_results:
            c_id = item.chunk.id
            chunk_map[c_id] = item.chunk
            dense_raw_scores[c_id] = item.score

        for item in sparse_results:
            c_id = item.chunk.id
            chunk_map[c_id] = item.chunk
            sparse_raw_scores[c_id] = item.score

        all_ids = list(chunk_map.keys())
        if not all_ids:
            return []

        # Min-max normalize dense scores across candidates
        d_scores = [dense_raw_scores.get(cid, 0.0) for cid in all_ids]
        norm_d_scores = cls._min_max_normalize(d_scores)
        norm_d_map = dict(zip(all_ids, norm_d_scores))

        # Min-max normalize sparse scores across candidates
        s_scores = [sparse_raw_scores.get(cid, 0.0) for cid in all_ids]
        norm_s_scores = cls._min_max_normalize(s_scores)
        norm_s_map = dict(zip(all_ids, norm_s_scores))

        # Compute fused score
        fused: List[ScoredChunk] = []
        for cid in all_ids:
            norm_dense = norm_d_map.get(cid, 0.0)
            norm_sparse = norm_s_map.get(cid, 0.0)
            fused_score = (alpha * norm_dense) + ((1.0 - alpha) * norm_sparse)

            fused.append(
                ScoredChunk(
                    chunk=chunk_map[cid],
                    score=float(fused_score),
                    dense_score=dense_raw_scores.get(cid),
                    sparse_score=sparse_raw_scores.get(cid),
                )
            )

        fused.sort(key=lambda x: x.score, reverse=True)
        top_fused = fused[:top_k]

        for rank, item in enumerate(top_fused, start=1):
            item.rank = rank

        return top_fused
