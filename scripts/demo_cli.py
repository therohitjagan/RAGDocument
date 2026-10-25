"""Interactive terminal CLI for querying the Enterprise Hybrid RAG pipeline."""

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

from core.models import RetrievalQuery, RetrievalStrategy
from generation.synthesizer import GroundedSynthesizer
from retrieval.hybrid_retriever import HybridRetriever


def main():
    parser = argparse.ArgumentParser(description="Query the Hybrid RAG pipeline in your terminal.")
    parser.add_argument("--query", type=str, help="Question to query the system with.")
    parser.add_argument(
        "--strategy",
        type=str,
        default="hybrid_rerank",
        choices=["dense", "sparse", "hybrid", "hybrid_rerank"],
        help="Retrieval strategy (default: hybrid_rerank)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top K results (default: 5)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha weight for hybrid (default: 0.5)")
    args = parser.parse_args()

    retriever = HybridRetriever()
    synthesizer = GroundedSynthesizer()

    query_text = args.query
    if not query_text:
        print("\nEntering Interactive Q&A Mode (Type 'exit' to quit):")
        while True:
            try:
                query_text = input("\n[Question] > ").strip()
                if not query_text or query_text.lower() in ("exit", "quit"):
                    break

                execute_query(retriever, synthesizer, query_text, args.strategy, args.top_k, args.alpha)
            except (KeyboardInterrupt, EOFError):
                break
    else:
        execute_query(retriever, synthesizer, query_text, args.strategy, args.top_k, args.alpha)


def execute_query(retriever, synthesizer, query_text: str, strategy: str, top_k: int, alpha: float):
    strategy_enum = RetrievalStrategy(strategy)
    query_obj = RetrievalQuery(
        query=query_text,
        strategy=strategy_enum,
        top_k=top_k,
        alpha=alpha,
    )

    t0 = time.time()
    retrieved = retriever.retrieve(query_obj)
    retrieval_ms = (time.time() - t0) * 1000.0

    resp = synthesizer.synthesize(
        query=query_text,
        retrieved_chunks=retrieved,
        strategy_used=strategy_enum,
        retrieval_latency_ms=retrieval_ms,
    )

    print("\n" + "=" * 60)
    print("📝 GROUNDED ANSWER:")
    print("=" * 60)
    print(resp.answer)

    print("\n" + "-" * 60)
    print(f"⏱️  Telemetry: Retrieval: {resp.retrieval_latency_ms:.1f}ms | Synthesis: {resp.generation_latency_ms:.1f}ms | Total: {resp.total_latency_ms:.1f}ms")
    print("-" * 60)

    print("\n📌 CITATIONS:")
    for idx, c in enumerate(resp.citations, 1):
        print(f"  [{idx}] {c.doc_title} (ID: {c.doc_id}) | Section: {c.section} (Confidence: {c.confidence:.1%})")
        print(f"      Snippet: {c.snippet}")

    print("\n🔬 RETRIEVED CHUNKS SCORE BREAKDOWN:")
    for sc in retrieved:
        d_s = f"{sc.dense_score:.3f}" if sc.dense_score is not None else "-"
        s_s = f"{sc.sparse_score:.3f}" if sc.sparse_score is not None else "-"
        r_s = f"{sc.rerank_score:.3f}" if sc.rerank_score is not None else "-"
        print(f"  Rank {sc.rank}: [{sc.chunk.id}] Final={sc.score:.4f} (Dense={d_s}, BM25={s_s}, Rerank={r_s}) - {sc.chunk.metadata.doc_title}")


if __name__ == "__main__":
    main()
