"""Unit tests for Context-Aware Chunker."""

import pytest
from core.models import Document, DocumentMetadata
from ingestion.chunker import ContextAwareChunker


def test_context_aware_chunker_breadcrumbs_and_headers():
    doc_content = """# Distributed Consensus Architecture

---
domain: distributed_systems
doc_type: rfc
---

## Storage Engine

### Raft Log Compaction
The Raft log must be periodically compacted to prevent disk exhaustion.
Snapshot threshold is set to 10000 entries.

### Snapshot Restoration
Snapshots are transmitted over streaming gRPC connections.
"""

    metadata = DocumentMetadata(
        doc_id="rfc_consensus_01",
        title="Distributed Consensus Architecture",
        domain="distributed_systems",
        doc_type="rfc",
        tags=["raft", "consensus"],
    )
    doc = Document(id="rfc_consensus_01", content=doc_content, metadata=metadata)

    chunker = ContextAwareChunker(chunk_size=300, chunk_overlap=30)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2

    # Verify context-aware header injection
    for c in chunks:
        assert "[Document: Distributed Consensus Architecture]" in c.content
        assert "[Domain: distributed_systems | Type: rfc]" in c.content
        assert c.metadata.doc_id == "rfc_consensus_01"
        assert c.metadata.total_chunks == len(chunks)

    # Check breadcrumb tracking
    assert any("Raft Log Compaction" in c.metadata.section_path for c in chunks)
    assert any("Snapshot Restoration" in c.metadata.section_path for c in chunks)


def test_chunker_handles_empty_or_small_docs():
    metadata = DocumentMetadata(
        doc_id="small_01",
        title="Small Document",
    )
    doc = Document(
        id="small_01",
        content="This is a simple one-sentence documentation snippet for testing.",
        metadata=metadata,
    )
    chunker = ContextAwareChunker(chunk_size=500, min_chunk_size=10)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert "Small Document" in chunks[0].content
