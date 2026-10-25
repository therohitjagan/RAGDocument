"""Sparse BM25 keyword retrieval engine with metadata pre-filtering support."""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from rank_bm25 import BM25Okapi
from core.models import Chunk, ScoredChunk

logger = logging.getLogger(__name__)


class SparseBM25Retriever:
    """
    BM25Okapi retriever supporting code/identifier tokenization,
    metadata filtering, and serialization.
    """

    # Common English and technical stopwords
    STOPWORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
        "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
        "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
        "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him",
        "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "it's",
        "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
        "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
        "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
        "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the",
        "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
        "they", "this", "those", "through", "to", "too", "under", "until", "up",
        "very", "was", "wasn't", "we", "were", "weren't", "what", "when", "where",
        "which", "while", "who", "whom", "why", "with", "won't", "would", "wouldn't",
        "you", "your", "yours", "yourself", "yourselves"
    }

    def __init__(self, index_dir: Optional[str | Path] = None):
        self.index_dir = Path(index_dir) if index_dir else None
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Chunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self.chunk_id_map: Dict[str, int] = {}
        if self.index_dir and (self.index_dir / "bm25_store.pkl").is_file():
            self.load()

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """
        Subword & identifier-aware tokenization:
        Splits on non-alphanumeric, decomposes CamelCase/snake_case,
        and filters out stopwords.
        """
        # Split camelCase into words (e.g. IngressProxy -> Ingress Proxy)
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        # Tokenize alphanumeric sequences
        tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        # Filter stopwords and 1-char tokens
        cleaned_tokens = [
            t for t in tokens if len(t) > 1 and t not in cls.STOPWORDS
        ]
        return cleaned_tokens

    def index_chunks(self, chunks: List[Chunk], append: bool = False) -> None:
        """Build or update the BM25 inverted index from a collection of chunks."""
        if not chunks:
            logger.warning("No chunks provided to index_chunks.")
            return

        if append and self.chunks:
            existing_ids = set(self.chunk_id_map.keys())
            new_unique = [c for c in chunks if c.id not in existing_ids]
            if not new_unique:
                return
            self.chunks.extend(new_unique)
        else:
            self.chunks = chunks

        self.tokenized_corpus = [self.tokenize(c.content) for c in self.chunks]
        self.chunk_id_map = {c.id: idx for idx, c in enumerate(self.chunks)}
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"Built BM25 index over {len(self.chunks)} chunks.")

    def search(
        self,
        query: str,
        top_k: int = 25,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[ScoredChunk]:
        """
        Execute BM25 keyword query with optional metadata pre-filtering.
        """
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        doc_scores = self.bm25.get_scores(query_tokens)

        # Match and score candidates
        scored_candidates: List[ScoredChunk] = []
        query_set = set(query_tokens)

        for idx, score in enumerate(doc_scores):
            # Check lexical term overlap
            matched_terms = query_set.intersection(self.tokenized_corpus[idx])
            if not matched_terms:
                continue

            # Ensure positive score even in small N corpora where standard Okapi log(1)=0
            positive_score = float(score) if score > 0.0 else float(len(matched_terms)) * 0.5

            chunk = self.chunks[idx]

            # Apply metadata filtering if specified
            if metadata_filter:
                match = True
                meta_dict = chunk.metadata.model_dump()
                for filter_k, filter_v in metadata_filter.items():
                    if filter_k == "tags":
                        if isinstance(filter_v, list):
                            if not any(tag in chunk.metadata.tags for tag in filter_v):
                                match = False
                                break
                        elif filter_v not in chunk.metadata.tags:
                            match = False
                            break
                    elif meta_dict.get(filter_k) != filter_v:
                        match = False
                        break
                if not match:
                    continue

            scored_candidates.append(
                ScoredChunk(
                    chunk=chunk,
                    score=positive_score,
                    sparse_score=positive_score,
                )
            )

        # Sort descending by BM25 score
        scored_candidates.sort(key=lambda x: x.score, reverse=True)
        top_results = scored_candidates[:top_k]

        for rank, sc in enumerate(top_results, start=1):
            sc.rank = rank

        return top_results

    def save(self, file_path: Optional[str | Path] = None) -> None:
        """Persist BM25 index and chunk metadata to disk."""
        save_path = Path(file_path or (self.index_dir / "bm25_store.pkl"))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": self.chunks,
            "tokenized_corpus": self.tokenized_corpus,
            "chunk_id_map": self.chunk_id_map,
        }
        with open(save_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved BM25 index to {save_path}")

    def load(self, file_path: Optional[str | Path] = None) -> bool:
        """Load BM25 index from disk."""
        load_path = Path(file_path or (self.index_dir / "bm25_store.pkl"))
        if not load_path.is_file():
            logger.warning(f"BM25 index file not found at {load_path}")
            return False

        with open(load_path, "rb") as f:
            data = pickle.load(f)

        self.chunks = data["chunks"]
        self.tokenized_corpus = data["tokenized_corpus"]
        self.chunk_id_map = data["chunk_id_map"]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"Loaded BM25 index with {len(self.chunks)} chunks from {load_path}")
        return True

    def count(self) -> int:
        """Return the number of indexed chunks."""
        return len(self.chunks)
