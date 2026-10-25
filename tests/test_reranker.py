"""Unit tests for Cross-Encoder Reranker."""

import pytest
from core.models import Chunk, ChunkMetadata, ScoredChunk
from retrieval.reranker import CrossEncoderReranker, FastCrossInteractionReranker


def test_fast_cross_interaction_reranker():
    reranker = FastCrossInteractionReranker()

    query = "What is the buffer capacity of Raft heartbeat?"
    doc_irrelevant = "[Document: Kafka] [Section: Broker] Kafka broker page cache diagnostics."
    doc_relevant = "[Document: Raft Consensus] [Section: Storage Engine > Buffer] The Raft heartbeat buffer capacity is 512MB."

    scores = reranker.predict([
        [query, doc_irrelevant],
        [query, doc_relevant],
    ])

    assert len(scores) == 2
    assert scores[1] > scores[0]


def test_cross_encoder_rerank_ordering():
    c_unrelated = Chunk(
        id="c_unrelated",
        content="[Document: Kubernetes] [Section: Networking] Calico network policy ingress isolation.",
        raw_content="Calico network policy ingress isolation.",
        metadata=ChunkMetadata(
            chunk_id="c_unrelated",
            doc_id="k8s_01",
            doc_title="Kubernetes",
            doc_type="runbook",
            domain="infrastructure",
        ),
    )

    c_target = Chunk(
        id="c_target",
        content="[Document: Raft Tuning] [Section: Heartbeat] Recommended buffer capacity is 1024MB for Raft heartbeat.",
        raw_content="Recommended buffer capacity is 1024MB for Raft heartbeat.",
        metadata=ChunkMetadata(
            chunk_id="c_target",
            doc_id="raft_01",
            doc_title="Raft Tuning",
            doc_type="rfc",
            domain="distributed_systems",
        ),
    )

    # Initial ranking where unrelated chunk had a slightly higher initial score
    candidates = [
        ScoredChunk(chunk=c_unrelated, score=0.80),
        ScoredChunk(chunk=c_target, score=0.75),
    ]

    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(
        query="What is the buffer capacity for Raft heartbeat?",
        candidates=candidates,
        top_k=2,
    )

    # Cross-encoder should promote c_target to Rank 1
    assert reranked[0].chunk.id == "c_target"
    assert reranked[0].rank == 1
