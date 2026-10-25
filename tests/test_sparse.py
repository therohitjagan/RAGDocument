"""Unit tests for Sparse BM25 Keyword Retriever."""

import pytest
from core.models import Chunk, ChunkMetadata
from retrieval.sparse import SparseBM25Retriever


def test_bm25_tokenization():
    raw_query = "KafkaConsumerGroup and IngressProxy on port 9092"
    tokens = SparseBM25Retriever.tokenize(raw_query)
    # Check camelCase splitting
    assert "kafka" in tokens
    assert "consumer" in tokens
    assert "group" in tokens
    assert "ingress" in tokens
    assert "proxy" in tokens
    assert "9092" in tokens
    assert "and" not in tokens  # Stopword removed


def test_bm25_indexing_and_metadata_filtering():
    c1 = Chunk(
        id="c1",
        content="[Document: Envoy Mesh] Envoy proxy handles mTLS certificates and upstream retries.",
        raw_content="Envoy proxy handles mTLS certificates and upstream retries.",
        metadata=ChunkMetadata(
            chunk_id="c1",
            doc_id="envoy_01",
            doc_title="Envoy Mesh",
            doc_type="rfc",
            domain="microservices",
            tags=["envoy", "mtls"],
        ),
    )

    c2 = Chunk(
        id="c2",
        content="[Document: Kafka Streams] Kafka consumer group rebalance uses cooperative sticky assignor.",
        raw_content="Kafka consumer group rebalance uses cooperative sticky assignor.",
        metadata=ChunkMetadata(
            chunk_id="c2",
            doc_id="kafka_01",
            doc_title="Kafka Streams",
            doc_type="runbook",
            domain="streaming",
            tags=["kafka", "streaming"],
        ),
    )

    bm25 = SparseBM25Retriever()
    bm25.index_chunks([c1, c2])
    assert bm25.count() == 2

    # Query without filter
    res = bm25.search("mTLS certificates Envoy", top_k=5)
    assert len(res) >= 1
    assert res[0].chunk.id == "c1"

    # Query with metadata filter matching domain
    res_filtered = bm25.search(
        "consumer group rebalance",
        top_k=5,
        metadata_filter={"domain": "streaming"},
    )
    assert len(res_filtered) == 1
    assert res_filtered[0].chunk.id == "c2"

    # Query with metadata filter excluding domain
    res_excluded = bm25.search(
        "consumer group rebalance",
        top_k=5,
        metadata_filter={"domain": "database"},
    )
    assert len(res_excluded) == 0
