"""CLI script to generate 5,000+ technical documents and initialize ChromaDB + BM25 indexes."""

import argparse
import sys
import time
from pathlib import Path

# Configure utf-8 standard stream encoding for cross-platform terminal compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import settings
from ingestion.chunker import ContextAwareChunker
from ingestion.pipeline import IngestionPipeline
from ingestion.synthetic_corpus import generate_enterprise_corpus
from retrieval.hybrid_retriever import HybridRetriever


def main():
    parser = argparse.ArgumentParser(description="Generate and index 5,000+ technical documents.")
    parser.add_argument(
        "--count",
        type=int,
        default=5000,
        help="Number of technical documents to generate (default: 5000)",
    )
    parser.add_argument(
        "--index-top",
        type=int,
        default=250,
        help="Number of core architecture documents to immediately index into ChromaDB & BM25 (default: 250)",
    )
    args = parser.parse_args()

    print(f"\n=================================================================")
    print(f"  GENERATING {args.count:,} ENTERPRISE TECHNICAL DOCUMENTS")
    print(f"=================================================================\n")

    t0 = time.time()
    docs, gold_samples = generate_enterprise_corpus(
        output_dir=settings.RAW_DATA_DIR,
        target_count=args.count,
        save_to_disk=True,
    )
    gen_time = time.time() - t0
    print(f"Generated {len(docs):,} documents across 8 engineering domains in {gen_time:.2f}s.")
    print(f"Saved raw markdown documents to: {settings.RAW_DATA_DIR}")
    print(f"Generated {len(gold_samples):,} gold RAGAS evaluation samples.")

    print(f"\nDeconstructing documents with Context-Aware Chunker and indexing...")
    chunker = ContextAwareChunker(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    pipeline = IngestionPipeline(chunker=chunker)

    # Process and dual-index the primary core documents
    docs_to_index = docs[: args.index_top]
    chunks = pipeline.process_documents(docs_to_index)
    print(f"Produced {len(chunks):,} enriched chunks with hierarchical breadcrumbs.")

    retriever = HybridRetriever()
    t_idx = time.time()
    retriever.index_chunks(chunks, append=False)
    idx_time = time.time() - t_idx
    print(f"Dual-indexing into ChromaDB and BM25 completed in {idx_time:.2f}s.")
    print(f"\nPipeline ready! Run `python scripts/run_benchmark.py` or launch the UI.")


if __name__ == "__main__":
    main()
