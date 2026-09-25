"""Run a small, transparent local retrieval baseline, without API calls."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from retrieval_graph.corpus import ZEPHYR_COMMIT
from retrieval_graph.local_index import LocalIndex

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = REPO_ROOT / "eval" / "retrieval_cases.json"


def evaluate(cases_path: Path, *, k: int = 5) -> dict:
    """Compare keyword, vector, and hybrid search against labeled source files."""
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if not cases or any(not case.get("relevant_paths") for case in cases):
        raise ValueError("Every evaluation case needs at least one relevant path")
    index = LocalIndex()
    summary: dict = {
        "set_name": cases_path.stem,
        "corpus_revision": ZEPHYR_COMMIT,
        "cases": len(cases),
        "k": k,
        "label_granularity": "source file",
        "note": "Small manually authored file-level relevance set; not an answer-quality benchmark. Interpret with the set's creation and tuning history.",
        "modes": {},
    }
    for mode in ("keyword", "vector", "hybrid"):
        rows = []
        for case in cases:
            start = time.perf_counter()
            hits = index.search(case["question"], k=k, mode=mode)
            latency_ms = (time.perf_counter() - start) * 1000
            relevant = set(case["relevant_paths"])
            paths = [str(hit.document.metadata["path"]) for hit in hits]
            found = relevant.intersection(paths)
            first_rank = next(
                (i for i, path in enumerate(paths, 1) if path in relevant), None
            )
            rows.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "relevant_paths": sorted(relevant),
                    "retrieved_paths": paths,
                    "recall_at_k": len(found) / len(relevant),
                    "reciprocal_rank": 1 / first_rank if first_rank else 0,
                    "latency_ms": round(latency_ms, 2),
                }
            )
        summary["modes"][mode] = {
            "recall_at_k": round(
                statistics.mean(row["recall_at_k"] for row in rows), 4
            ),
            "mrr_at_k": round(
                statistics.mean(row["reciprocal_rank"] for row in rows), 4
            ),
            "median_latency_ms": round(
                statistics.median(row["latency_ms"] for row in rows), 2
            ),
            "rows": rows,
        }
    return summary


def main() -> None:
    """Run the baseline and optionally persist the full per-case report."""
    parser = argparse.ArgumentParser(description="EmbedPilot retrieval baseline")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    report = evaluate(args.cases, k=args.k)
    for mode, result in report["modes"].items():
        print(
            f"{mode}: Recall@{args.k}={result['recall_at_k']:.4f}, MRR@{args.k}={result['mrr_at_k']:.4f}, median={result['median_latency_ms']:.2f} ms"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Full report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
