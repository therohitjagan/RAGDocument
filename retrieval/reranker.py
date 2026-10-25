"""Cross-Encoder Reranker for second-stage precision re-ranking."""

from __future__ import annotations

import logging
import os
import re
from typing import List, Optional
from core.config import settings
from core.models import ScoredChunk

logger = logging.getLogger(__name__)


class FastCrossInteractionReranker:
    """
    High-precision heuristic cross-encoder fallback measuring:
    1. Exact keyword & technical identifier alignment (e.g., config keys, ports, algorithms)
    2. Query term proximity and consecutive term coverage
    3. Structural heading relevance (breadcrumb alignment)
    """

    @staticmethod
    def _score_pair(query: str, chunk_content: str, breadcrumb: str, doc_id: str = "") -> float:
        q_tokens = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 1]
        if not q_tokens:
            return 0.5

        c_text = chunk_content.lower()
        bc_text = breadcrumb.lower()

        # 1. Term coverage
        matched_tokens = [t for t in q_tokens if t in c_text]
        coverage = len(matched_tokens) / len(q_tokens)

        # 2. Heading alignment (heavy boost if query terms appear in heading/section breadcrumbs)
        bc_matches = [t for t in q_tokens if t in bc_text]
        bc_score = len(bc_matches) / len(q_tokens)

        # 3. Exact phrase match boost
        phrase_boost = 0.25 if query.lower() in c_text else 0.0

        # 4. Numeric / parameter match boost
        num_query = re.findall(r"\b\d+\b", query)
        num_matched = sum(1 for n in num_query if n in c_text)
        num_boost = 0.20 * (num_matched / max(1, len(num_query))) if num_query else 0.0

        # 5. Exact Doc ID match boost
        doc_id_boost = 0.35 if (doc_id and doc_id.lower() in query.lower()) else 0.0

        final_score = (0.40 * coverage) + (0.25 * bc_score) + phrase_boost + num_boost + doc_id_boost
        return min(1.0, max(0.0, final_score))

    def predict(self, pairs: List[List[str]]) -> List[float]:
        scores = []
        for q, doc in pairs:
            # Document starts with [DocID: ...] [Document: ...] [Section: ...]
            bc_match = re.search(r"\[Section:\s*(.*?)\]", doc)
            bc = bc_match.group(1) if bc_match else ""
            id_match = re.search(r"\[DocID:\s*(.*?)\]", doc)
            doc_id = id_match.group(1) if id_match else ""
            scores.append(self._score_pair(q, doc, bc, doc_id))
        return scores


class CrossEncoderReranker:
    """
    Reranks top candidate chunks using Cross-Encoder architectures
    to maximize answer faithfulness and retrieval precision.
    """

    def __init__(self, model_name: str = settings.RERANKER_MODEL_NAME):
        self.model_name = model_name
        self.cross_encoder = None
        self._init_model()

    def _init_model(self) -> None:
        """Attempt to load SentenceTransformers CrossEncoder or fallback."""
        enable_hf = os.environ.get("ENABLE_HF_DOWNLOADS", "false").lower() == "true"
        if enable_hf:
            try:
                from sentence_transformers import CrossEncoder
                self.cross_encoder = CrossEncoder(self.model_name)
                logger.info(f"Loaded CrossEncoder model: {self.model_name}")
                return
            except Exception as e:
                logger.warning(
                    f"SentenceTransformers CrossEncoder not loaded ({e}). Using FastCrossInteractionReranker fallback."
                )

        # Default high-performance, deterministic cross-encoder
        self.cross_encoder = FastCrossInteractionReranker()
        logger.info("Initialized FastCrossInteractionReranker (low-latency, zero-dependency).")

    def rerank(
        self,
        query: str,
        candidates: List[ScoredChunk],
        top_k: int = 5,
    ) -> List[ScoredChunk]:
        """
        Score query-candidate pairs with cross-attention and return top_k candidates.
        """
        if not candidates:
            return []

        # Prepare query-document pairs
        pairs = [[query, item.chunk.content] for item in candidates]

        try:
            scores = self.cross_encoder.predict(pairs)
        except Exception as e:
            logger.error(f"Error during cross-encoder prediction: {e}")
            return candidates[:top_k]

        reranked_results: List[ScoredChunk] = []
        for idx, item in enumerate(candidates):
            raw_score = float(scores[idx])
            # Normalize score to [0.0, 1.0] if logits
            norm_score = 1.0 / (1.0 + 2.71828 ** (-raw_score)) if (raw_score < 0.0 or raw_score > 1.0) else raw_score

            # Weighted ensemble of cross-encoder confidence (65%) and initial stage retrieval score (35%)
            blended_score = (0.65 * norm_score) + (0.35 * item.score)

            reranked_results.append(
                ScoredChunk(
                    chunk=item.chunk,
                    score=blended_score,
                    dense_score=item.dense_score,
                    sparse_score=item.sparse_score,
                    rerank_score=norm_score,
                )
            )

        # Sort descending by rerank score
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        top_results = reranked_results[:top_k]

        for rank, item in enumerate(top_results, start=1):
            item.rank = rank

        return top_results
