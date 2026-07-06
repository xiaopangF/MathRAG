"""
MathRAG retrieval evaluation.

The script evaluates whether retrieved chunks contain expected keywords for each
question in a JSONL file, and reports Recall@K and MRR.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def load_eval_questions(path: str | Path) -> list[dict[str, Any]]:
    """Load retrieval evaluation questions from a JSONL file."""
    eval_path = Path(path)
    questions: list[dict[str, Any]] = []

    with eval_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{eval_path}:{line_no} 不是合法 JSON: {exc}") from exc

            if "question" not in item:
                raise ValueError(f"{eval_path}:{line_no} 缺少 question 字段")
            questions.append(item)

    return questions


def get_expected_keywords(item: dict[str, Any]) -> list[str]:
    """Prefer retrieval-specific keywords and fall back to answer keywords."""
    expected = item.get("expected_chunk_keywords") or item.get("expected_keywords") or []
    if isinstance(expected, str):
        return [expected]
    return [str(keyword) for keyword in expected if str(keyword).strip()]


def is_hit(content: str, expected_keywords: list[str]) -> bool:
    """Return True when content contains any expected keyword."""
    content_lower = content.lower()
    return any(keyword.lower() in content_lower for keyword in expected_keywords)


def evaluate_retrieval(
    eval_path: str | Path = "data/eval/questions.jsonl",
    index_dir: str | Path = "data/faiss_index",
    top_k: int = 5,
    top_k_embedding: int = 20,
    use_gpu: bool = False,
) -> dict[str, Any]:
    """Run retrieval evaluation and return metrics plus failed cases."""
    try:
        from src.retriever import MathRAGRetriever, RAGConfig
    except ImportError as exc:
        raise ImportError("无法导入 MathRAGRetriever，请检查 src.retriever 模块") from exc

    eval_path = Path(eval_path)
    index_dir = Path(index_dir)

    if not eval_path.exists():
        raise FileNotFoundError(f"评测文件不存在: {eval_path}")
    if not (index_dir / "index.faiss").exists():
        raise FileNotFoundError(f"FAISS 索引不存在: {index_dir / 'index.faiss'}")
    if not (index_dir / "chunks_meta.jsonl").exists():
        raise FileNotFoundError(f"索引元数据不存在: {index_dir / 'chunks_meta.jsonl'}")

    config = RAGConfig(
        faiss_index_dir=str(index_dir),
        top_k_embedding=max(top_k_embedding, top_k),
        top_k_rerank=top_k,
        use_gpu=use_gpu,
    )
    retriever = MathRAGRetriever(config)

    questions = load_eval_questions(eval_path)
    evaluated_count = 0
    skipped_count = 0
    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    reciprocal_ranks: list[float] = []
    failed_cases: list[dict[str, Any]] = []

    for item in questions:
        question = item["question"]
        expected = get_expected_keywords(item)
        if not expected:
            skipped_count += 1
            continue

        evaluated_count += 1
        results = retriever.retrieve(question, top_k=top_k)

        rank = None
        for i, result in enumerate(results, start=1):
            if is_hit(result.content, expected):
                rank = i
                break

        if rank == 1:
            hit_at_1 += 1
        if rank is not None and rank <= 3:
            hit_at_3 += 1
        if rank is not None and rank <= 5:
            hit_at_5 += 1

        reciprocal_ranks.append(1 / rank if rank is not None else 0)

        if rank is None:
            failed_cases.append(
                {
                    "id": item.get("id"),
                    "question": question,
                    "expected_keywords": expected,
                    "top_results": [
                        {
                            "rank": idx,
                            "title": result.title,
                            "score": result.rerank_score,
                            "preview": result.content[:160] + "...",
                        }
                        for idx, result in enumerate(results[:5], start=1)
                    ],
                }
            )

    metrics = {
        "eval_path": str(eval_path),
        "index_dir": str(index_dir),
        "question_count": len(questions),
        "evaluated_count": evaluated_count,
        "skipped_count": skipped_count,
        "top_k": top_k,
        "top_k_embedding": max(top_k_embedding, top_k),
        "recall_at_1": hit_at_1 / evaluated_count if evaluated_count else 0,
        "recall_at_3": hit_at_3 / evaluated_count if evaluated_count else 0,
        "recall_at_5": hit_at_5 / evaluated_count if evaluated_count else 0,
        "mrr": sum(reciprocal_ranks) / evaluated_count if evaluated_count else 0,
        "hits": {
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
            "hit_at_5": hit_at_5,
        },
        "failed_cases": failed_cases,
    }
    return metrics


def print_report(metrics: dict[str, Any], max_failures: int = 10) -> None:
    """Print a human-readable evaluation report."""
    total = metrics["evaluated_count"]

    print("\n" + "=" * 50)
    print("检索评测结果")
    print("=" * 50)
    print(f"评测文件: {metrics['eval_path']}")
    print(f"索引目录: {metrics['index_dir']}")
    print(f"总问题数: {metrics['question_count']}")
    print(f"有效评测数: {metrics['evaluated_count']}")
    print(f"跳过问题数: {metrics['skipped_count']}")
    print(f"Recall@1: {metrics['recall_at_1']:.2%} ({metrics['hits']['hit_at_1']}/{total})")
    print(f"Recall@3: {metrics['recall_at_3']:.2%} ({metrics['hits']['hit_at_3']}/{total})")
    print(f"Recall@5: {metrics['recall_at_5']:.2%} ({metrics['hits']['hit_at_5']}/{total})")
    print(f"MRR: {metrics['mrr']:.4f}")

    failed_cases = metrics["failed_cases"]
    if not failed_cases:
        print("\n全部命中。")
        return

    print(f"\n未命中案例（前 {min(max_failures, len(failed_cases))} 个）：")
    for case in failed_cases[:max_failures]:
        print("-" * 40)
        print(f"ID: {case['id']}")
        print(f"问题: {case['question']}")
        print(f"期望关键词: {case['expected_keywords']}")
        print("Top 5 结果预览:")
        for result in case["top_results"]:
            print(
                f"  {result['rank']}. score={result['score']:.4f} "
                f"title={result['title']} preview={result['preview']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MathRAG retrieval quality.")
    parser.add_argument("--eval-path", default="data/eval/questions.jsonl")
    parser.add_argument("--index-dir", default="data/faiss_index")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-k-embedding", type=int, default=20)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--max-failures", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(project_root)

    try:
        metrics = evaluate_retrieval(
            eval_path=args.eval_path,
            index_dir=args.index_dir,
            top_k=args.top_k,
            top_k_embedding=args.top_k_embedding,
            use_gpu=args.use_gpu,
        )
    except (FileNotFoundError, ValueError, ImportError, RuntimeError, OSError) as exc:
        print(f"评测失败: {exc}")
        return 1

    print_report(metrics, max_failures=args.max_failures)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nJSON 结果已保存: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
