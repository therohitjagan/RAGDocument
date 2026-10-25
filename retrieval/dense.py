"""Dense semantic vector retrieval engine using ChromaDB and Sentence-Transformers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from core.config import settings
from core.models import Chunk, ChunkMetadata, ScoredChunk

logger = logging.getLogger(__name__)


try:
    from chromadb.api.types import EmbeddingFunction
except ImportError:
    class EmbeddingFunction:  # type: ignore
        pass


class LocalDenseEmbeddingFunction(EmbeddingFunction):
    """
    Fallback deterministic embedding generator (384-dim) for offline/lightweight environments
    using semantic hashing and subword frequency projection.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    @staticmethod
    def name() -> str:
        return "local_dense"

    def get_config(self) -> Dict[str, Any]:
        return {"dim": self.dim}

    @classmethod
    def build_from_config(cls, config: Dict[str, Any]) -> "LocalDenseEmbeddingFunction":
        return cls(dim=config.get("dim", 384))

    def __call__(self, input: List[str]) -> List[List[float]]:
        import zlib
        embeddings = []
        for text in input:
            vec = np.zeros(self.dim, dtype=np.float32)
            words = text.lower().split()
            for word in words:
                h = zlib.crc32(word.encode("utf-8")) % self.dim
                vec[h] += 1.0
                # Bigram / character n-grams for subword semantic capture
                if len(word) >= 3:
                    for i in range(len(word) - 2):
                        sh = zlib.crc32(word[i:i + 3].encode("utf-8")) % self.dim
                        vec[sh] += 0.5
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec.tolist())
        return embeddings


class DenseChromaRetriever:
    """
    ChromaDB persistent vector store wrapper for dense semantic search
    with metadata filtering and distance normalization.
    """

    def __init__(
        self,
        persist_dir: Optional[str | Path] = None,
        collection_name: str = settings.CHROMA_COLLECTION_NAME,
        model_name: str = settings.EMBEDDING_MODEL_NAME,
    ):
        self.persist_dir = Path(persist_dir or settings.CHROMA_PERSIST_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.model_name = model_name

        self.client = None
        self.collection = None
        self.embedding_fn = None
        self._init_chroma()

    def _init_chroma(self) -> None:
        """Initialize ChromaDB client and embedding model."""
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        # Choose embedding function
        enable_hf = False
        import os
        if os.environ.get("ENABLE_HF_DOWNLOADS", "false").lower() == "true":
            try:
                from chromadb.utils import embedding_functions
                self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.model_name
                )
                logger.info(f"Loaded SentenceTransformer embedding model: {self.model_name}")
                enable_hf = True
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer ({e}).")

        if not enable_hf:
            self.embedding_fn = LocalDenseEmbeddingFunction(dim=384)
            logger.info("Using LocalDenseEmbeddingFunction (384-dim, zero network latency).")

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"Initialized ChromaDB collection '{self.collection_name}' with {self.collection.count()} vectors."
        )

    def add_chunks(self, chunks: List[Chunk], batch_size: int = 250) -> None:
        """
        Ingest chunks into ChromaDB in batches with structured metadata.
        """
        if not chunks:
            return

        total = len(chunks)
        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c.id for c in batch]
            documents = [c.content for c in batch]
            metadatas = []

            for c in batch:
                meta = {
                    "chunk_id": c.metadata.chunk_id,
                    "doc_id": c.metadata.doc_id,
                    "doc_title": c.metadata.doc_title,
                    "doc_type": c.metadata.doc_type,
                    "domain": c.metadata.domain,
                    "tags": ",".join(c.metadata.tags),
                    "section_path": c.metadata.section_path,
                    "chunk_index": c.metadata.chunk_index,
                    "char_count": c.metadata.char_count,
                    "raw_content": c.raw_content[:1500],  # Stored for clean reconstruction
                }
                metadatas.append(meta)

            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
        logger.info(f"Upserted {total} chunks into ChromaDB collection '{self.collection_name}'.")

    def search(
        self,
        query: str,
        top_k: int = 25,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[ScoredChunk]:
        """
        Perform dense semantic vector retrieval against ChromaDB with metadata filtering.
        """
        if self.collection.count() == 0:
            return []

        # Convert simple filter into ChromaDB 'where' clause
        where_clause = None
        if metadata_filter:
            filter_clauses = []
            for k, v in metadata_filter.items():
                if k == "tags":
                    # Tag containment check in comma-separated string
                    filter_clauses.append({"tags": {"$contains": str(v)}})
                else:
                    filter_clauses.append({k: {"$eq": v}})

            if len(filter_clauses) == 1:
                where_clause = filter_clauses[0]
            elif len(filter_clauses) > 1:
                where_clause = {"$and": filter_clauses}

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        scored_chunks: List[ScoredChunk] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for idx, chunk_id in enumerate(ids):
            meta = metadatas[idx]
            dist = distances[idx]
            # In cosine space, distance is (1 - cosine_similarity). Score = 1 - dist.
            similarity = max(0.0, min(1.0, 1.0 - dist))

            tags = [t.strip() for t in meta.get("tags", "").split(",") if t.strip()]
            chunk_metadata = ChunkMetadata(
                chunk_id=meta.get("chunk_id", chunk_id),
                doc_id=meta.get("doc_id", ""),
                doc_title=meta.get("doc_title", ""),
                doc_type=meta.get("doc_type", "technical_doc"),
                domain=meta.get("domain", "general"),
                tags=tags,
                section_path=meta.get("section_path", ""),
                chunk_index=meta.get("chunk_index", 0),
                char_count=meta.get("char_count", len(docs[idx])),
            )

            chunk = Chunk(
                id=chunk_id,
                content=docs[idx],
                raw_content=meta.get("raw_content", docs[idx]),
                metadata=chunk_metadata,
            )

            scored_chunks.append(
                ScoredChunk(
                    chunk=chunk,
                    score=similarity,
                    dense_score=similarity,
                    rank=idx + 1,
                )
            )

        return scored_chunks

    def count(self) -> int:
        """Return total vectors in the collection."""
        return self.collection.count() if self.collection else 0

    def clear(self) -> None:
        """Wipe collection vectors."""
        if self.collection:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
