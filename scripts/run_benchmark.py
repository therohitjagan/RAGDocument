"""CLI script to run RAGAS benchmark evaluation across retrieval strategies."""

import argparse
import sys
from pathlib import Path

# Configure utf-8 standard stream encoding for cross-platform terminal compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import settings
from evaluation.benchmark_runner import BenchmarkRunner
from retrieval.hybrid_retriever import HybridRetriever


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the enterprise pipeline.")
    parser.add_argument(
        "--samples",
        type=int,
        default=25,
        help="Number of evaluation queries to run (default: 25)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K chunks for retrieval (default: 5)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Alpha weight for hybrid convex combination (default: 0.5)",
    )
    args = parser.parse_args()

    retriever = HybridRetriever()
    runner = BenchmarkRunner(retriever=retriever)

    test_samples = runner.load_test_samples()
    if not test_samples:
        print("ERROR: No benchmark test samples found.")
        print("Please generate the corpus first by running:")
        print("  python scripts/generate_corpus.py --count 5000")
        sys.exit(1)

    eval_subset = test_samples[: args.samples]
    results = runner.run_and_report(
        samples=eval_subset,
        top_k=args.top_k,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
