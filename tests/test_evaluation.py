"""Unit tests for RAGAS evaluation metrics and synthesizer faithfulness."""

import pytest
from core.models import Chunk, ChunkMetadata, ScoredChunk
from evaluation.ragas_evaluator import RAGASEvaluator
from generation.synthesizer import GroundedSynthesizer
from retrieval.hybrid_retriever import HybridRetriever


def test_ragas_faithfulness_calculation():
    evaluator = RAGASEvaluator(retriever=None, synthesizer=None)

    c = Chunk(
        id="c1",
        content="The port is 9092 and buffer capacity is 512MB.",
        raw_content="The port is 9092 and buffer capacity is 512MB.",
        metadata=ChunkMetadata(
            chunk_id="c1",
            doc_id="doc1",
            doc_title="Kafka",
            doc_type="spec",
            domain="streaming",
        ),
    )
    context_chunks = [ScoredChunk(chunk=c, score=0.9)]

    # Grounded answer: contains matching numbers and facts
    grounded_answer = "The listener port is 9092 and the buffer capacity is 512MB."
    faithfulness_high = evaluator.evaluate_faithfulness(grounded_answer, context_chunks)
    assert faithfulness_high >= 0.95

    # Hallucinated answer: contains nonexistent numbers and facts
    hallucinated_answer = "The listener port is 443 and the buffer capacity is 8192MB with Redis clustering."
    faithfulness_low = evaluator.evaluate_faithfulness(hallucinated_answer, context_chunks)
    assert faithfulness_low < 0.50


def test_ragas_hit_rate_and_reciprocal_rank():
    evaluator = RAGASEvaluator(retriever=None, synthesizer=None)

    c1 = Chunk(
        id="c1",
        content="",
        raw_content="",
        metadata=ChunkMetadata(
            chunk_id="c1", doc_id="doc_a", doc_title="A", doc_type="rfc", domain="d"
        ),
    )
    c2 = Chunk(
        id="c2",
        content="",
        raw_content="",
        metadata=ChunkMetadata(
            chunk_id="c2", doc_id="doc_b", doc_title="B", doc_type="rfc", domain="d"
        ),
    )

    chunks = [
        ScoredChunk(chunk=c1, score=0.9, rank=1),
        ScoredChunk(chunk=c2, score=0.8, rank=2),
    ]

    # Target doc is doc_a (Rank 1)
    is_hit, rr = evaluator.evaluate_retrieval_hit(chunks, ["doc_a"])
    assert is_hit is True
    assert rr == 1.0

    # Target doc is doc_b (Rank 2)
    is_hit, rr = evaluator.evaluate_retrieval_hit(chunks, ["doc_b"])
    assert is_hit is True
    assert rr == 0.5

    # Target doc is doc_c (Not in results)
    is_hit, rr = evaluator.evaluate_retrieval_hit(chunks, ["doc_c"])
    assert is_hit is False
    assert rr == 0.0


def test_synthesizer_grounding_and_citations():
    synthesizer = GroundedSynthesizer(provider="local")

    c = Chunk(
        id="c_envoy",
        content="[Document: Envoy Architecture] Envoy listener timeout is set to 250ms with 3 retries.",
        raw_content="Envoy listener timeout is set to 250ms with 3 retries.",
        metadata=ChunkMetadata(
            chunk_id="c_envoy",
            doc_id="envoy_spec",
            doc_title="Envoy Architecture",
            doc_type="rfc",
            domain="microservices",
            section_path="Networking > Listener",
        ),
    )
    retrieved = [ScoredChunk(chunk=c, score=0.92, rank=1)]

    response = synthesizer.synthesize(
        query="What is the listener timeout for Envoy?",
        retrieved_chunks=retrieved,
    )

    assert "Envoy Architecture" in response.answer
    assert len(response.citations) >= 1
    assert response.citations[0].doc_id == "envoy_spec"
    assert response.citations[0].section == "Networking > Listener"
    assert response.retrieval_latency_ms >= 0
