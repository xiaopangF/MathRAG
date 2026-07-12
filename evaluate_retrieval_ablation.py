"""
Run retrieval ablation experiments for MathRAG.

The script reuses ``evaluate_retrieval.py`` and compares a fixed set of
retrieval variants on the same eval dataset and index.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from evaluate_retrieval import evaluate_retrieval


DEFAULT_VARIANTS: dict[str, dict[str, Any]] = {
    "full": {
        "description": "Embedding + BM25 + RRF + Reranker",
    },
    "no_rrf": {
        "description": "Embedding + BM25 + Reranker, without RRF prior",
        "rrf_weight": 0.0,
    },
    "no_bm25": {
        "description": "Embedding + RRF prior + Reranker, without BM25 candidates",
        "use_hybrid_search": False,
    },
    "no_reranker": {
        "description": "Embedding + BM25 + RRF, without second-stage reranker",
        "use_reranker": False,
    },
    "narrow_recall": {
        "description": "Full pipeline with first-stage candidate depth reduced to 5 + 5",
        "top_k_embedding": 5,
        "top_k_bm25": 5,
    },
}


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep the score fields needed for an ablation summary."""
    return {
        "keyword": metrics["keyword_metrics"],
        "page": metrics["page_metrics"],
        "section": metrics["section_metrics"],
        "failed_counts": {
            "keyword": len(metrics.get("failed_cases", [])),
            "page": len(metrics.get("page_failed_cases", [])),
            "section": len(metrics.get("section_failed_cases", [])),
        },
    }


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 90)
    print("Retrieval Ablation Summary")
    print("=" * 90)
    print(
        "| Variant | Keyword R@1/3/5 | Keyword MRR | "
        "Page R@1/3/5 | Section R@1/3/5 | Misses k/p/s |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for item in results:
        metrics = item["metrics"]
        keyword = metrics["keyword"]
        page = metrics["page"]
        section = metrics["section"]
        misses = metrics["failed_counts"]
        print(
            f"| {item['name']} | "
            f"{pct(keyword['recall_at_1'])} / {pct(keyword['recall_at_3'])} / {pct(keyword['recall_at_5'])} | "
            f"{keyword['mrr']:.4f} | "
            f"{pct(page['recall_at_1'])} / {pct(page['recall_at_3'])} / {pct(page['recall_at_5'])} | "
            f"{pct(section['recall_at_1'])} / {pct(section['recall_at_3'])} / {pct(section['recall_at_5'])} | "
            f"{misses['keyword']} / {misses['page']} / {misses['section']} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MathRAG retrieval ablations.")
    parser.add_argument("--eval-path", default="data/eval/questions.grounded.dev.jsonl")
    parser.add_argument("--index-dir", default="data/faiss_index")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-k-embedding", type=int, default=20)
    parser.add_argument("--top-k-bm25", type=int, default=20)
    parser.add_argument("--rerank-batch-size", type=int, default=64)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rrf-weight", type=float, default=1.0)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument(
        "--variants",
        default="full,no_rrf,no_bm25,no_reranker,narrow_recall",
        help="Comma-separated variant names. Available: "
        + ", ".join(DEFAULT_VARIANTS),
    )
    parser.add_argument(
        "--output-json",
        default="reports/retrieval_ablation_grounded_dev.json",
    )
    parser.add_argument(
        "--include-full-metrics",
        action="store_true",
        help="Store full failed-case details for every variant.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(project_root)

    base_config = {
        "eval_path": args.eval_path,
        "index_dir": args.index_dir,
        "top_k": args.top_k,
        "top_k_embedding": args.top_k_embedding,
        "top_k_bm25": args.top_k_bm25,
        "rerank_batch_size": args.rerank_batch_size,
        "rrf_k": args.rrf_k,
        "rrf_weight": args.rrf_weight,
        "use_hybrid_search": True,
        "use_reranker": True,
        "use_gpu": args.use_gpu,
    }

    selected = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = [name for name in selected if name not in DEFAULT_VARIANTS]
    if unknown:
        print(f"Unknown variants: {', '.join(unknown)}")
        return 2

    results: list[dict[str, Any]] = []
    for name in selected:
        variant = DEFAULT_VARIANTS[name]
        config = deepcopy(base_config)
        config.update({k: v for k, v in variant.items() if k != "description"})

        print("\n" + "-" * 90)
        print(f"Running ablation: {name}")
        print(variant["description"])
        metrics = evaluate_retrieval(**config)

        result = {
            "name": name,
            "description": variant["description"],
            "config": config,
            "metrics": metrics if args.include_full_metrics else compact_metrics(metrics),
        }
        results.append(result)

    output = {
        "eval_path": args.eval_path,
        "index_dir": args.index_dir,
        "variants": results,
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nJSON result saved to: {output_path}")

    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
