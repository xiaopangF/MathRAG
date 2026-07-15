"""
MathRAG retrieval evaluation.

The script evaluates whether retrieved chunks contain expected keywords for each
question in a JSONL file, and reports Recall@K and MRR.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable


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


def get_expected_page_ranges(item: dict[str, Any]) -> list[tuple[int, int]]:
    """Normalize page annotations to inclusive page ranges."""
    expected = item.get("expected_page_ranges", [])
    if expected is None:
        expected = []
    if isinstance(expected, int) and not isinstance(expected, bool):
        expected = [expected]
    if not isinstance(expected, list):
        raise ValueError("expected_page_ranges 必须是页码或页码范围列表")

    ranges = []
    for value in expected:
        if isinstance(value, int) and not isinstance(value, bool):
            start = end = value
        elif (
            isinstance(value, list)
            and len(value) == 2
            and all(
                isinstance(page, int) and not isinstance(page, bool)
                for page in value
            )
        ):
            start, end = value
        else:
            raise ValueError(f"无效页码范围: {value!r}")
        if start < 1 or end < start:
            raise ValueError(f"无效页码范围: {value!r}")
        ranges.append((start, end))
    return ranges


def get_expected_sections(item: dict[str, Any]) -> list[str]:
    """Return accepted chapter, section, or title labels."""
    expected = item.get("expected_sections") or []
    if isinstance(expected, str):
        expected = [expected]
    if not isinstance(expected, list) or not all(
        isinstance(section, str) for section in expected
    ):
        raise ValueError("expected_sections 必须是字符串或字符串列表")
    return [section.strip() for section in expected if section.strip()]


def normalize_for_match(text: str) -> str:
    """Normalize OCR text for keyword matching."""
    text = text.lower()
    text = re.sub(r"[\s\-‐‑‒–—―_·•'\"“”‘’`´!！?？,，.。:：;；、/\\|()[\]{}<>《》【】（）]", "", text)
    return text


def is_hit(content: str, expected_keywords: list[str]) -> bool:
    """Return True when content contains any expected keyword."""
    normalized_content = normalize_for_match(content)
    return any(
        normalize_for_match(keyword) in normalized_content
        for keyword in expected_keywords
        if normalize_for_match(keyword)
    )


def result_match_text(result: Any) -> str:
    """Combine structural metadata and content for retrieval evaluation."""
    title = str(getattr(result, "title", "") or "").strip()
    content = str(getattr(result, "content", "") or "")
    return f"{title}\n{content}" if title else content


def is_page_hit(result: Any, expected_ranges: list[tuple[int, int]]) -> bool:
    """Return True when a retrieved page range overlaps annotated pages."""
    start = getattr(result, "page_start", None)
    end = getattr(result, "page_end", None)
    if start is None and end is None:
        return False
    start = int(start if start is not None else end)
    end = int(end if end is not None else start)
    return any(
        start <= expected_end and end >= expected_start
        for expected_start, expected_end in expected_ranges
    )


def is_section_hit(result: Any, expected_sections: list[str]) -> bool:
    """Match annotations against returned chapter, section, and title metadata."""
    metadata_text = "\n".join(
        str(getattr(result, field, "") or "")
        for field in ("chapter", "section", "title")
    )
    return is_hit(metadata_text, expected_sections)


def first_matching_rank(
    results: list[Any],
    predicate: Callable[[Any], bool],
) -> int | None:
    for rank, result in enumerate(results, start=1):
        if predicate(result):
            return rank
    return None


def summarize_ranks(ranks: list[int | None]) -> dict[str, Any]:
    evaluated_count = len(ranks)
    hit_at_1 = sum(rank == 1 for rank in ranks)
    hit_at_3 = sum(rank is not None and rank <= 3 for rank in ranks)
    hit_at_5 = sum(rank is not None and rank <= 5 for rank in ranks)
    return {
        "evaluated_count": evaluated_count,
        "recall_at_1": hit_at_1 / evaluated_count if evaluated_count else 0,
        "recall_at_3": hit_at_3 / evaluated_count if evaluated_count else 0,
        "recall_at_5": hit_at_5 / evaluated_count if evaluated_count else 0,
        "mrr": (
            sum(1 / rank if rank is not None else 0 for rank in ranks) / evaluated_count
            if evaluated_count
            else 0
        ),
        "hits": {
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
            "hit_at_5": hit_at_5,
        },
    }


def result_preview(result: Any, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "title": result.title,
        "chapter": result.chapter,
        "section": result.section,
        "page_start": result.page_start,
        "page_end": result.page_end,
        "score": result.rerank_score,
        "retrieval_score": getattr(result, "retrieval_score", result.rerank_score),
        "fusion_score": getattr(result, "fusion_score", 0.0),
        "embedding_rank": getattr(result, "embedding_rank", None),
        "bm25_rank": getattr(result, "bm25_rank", None),
        "preview": result.content[:160] + "...",
    }


def build_failure_case(
    item: dict[str, Any],
    expected_key: str,
    expected: Any,
    results: list[Any],
) -> dict[str, Any]:
    return {
        "id": item.get("id"),
        "question": item["question"],
        expected_key: expected,
        "top_results": [
            result_preview(result, rank)
            for rank, result in enumerate(results[:5], start=1)
        ],
    }


def evaluate_retrieval(
    eval_path: str | Path = "data/eval/questions.jsonl",
    index_dir: str | Path = "data/faiss_index",
    top_k: int = 5,
    top_k_embedding: int = 20,
    top_k_bm25: int = 20,
    rerank_batch_size: int = 32,
    rrf_k: int = 60,
    rrf_weight: float = 1.0,
    use_hybrid_search: bool = True,
    use_reranker: bool = True,
    use_query_rewrite: bool = True,
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
        top_k_bm25=max(top_k_bm25, top_k),
        top_k_rerank=top_k,
        rerank_batch_size=rerank_batch_size,
        rrf_k=rrf_k,
        rrf_weight=rrf_weight,
        use_hybrid_search=use_hybrid_search,
        use_reranker=use_reranker,
        use_query_rewrite=use_query_rewrite,
        use_gpu=use_gpu,
    )
    retriever = MathRAGRetriever(config)

    questions = load_eval_questions(eval_path)
    evaluable_items = []
    for item in questions:
        if (
            get_expected_keywords(item)
            or get_expected_page_ranges(item)
            or get_expected_sections(item)
        ):
            evaluable_items.append(item)
    skipped_count = len(questions) - len(evaluable_items)
    print(f"开始批量检索评测: {len(evaluable_items)} 个有效问题")
    batch_results = retriever.batch_retrieve(
        [item["question"] for item in evaluable_items],
        top_k=top_k,
    )

    keyword_ranks: list[int | None] = []
    page_ranks: list[int | None] = []
    section_ranks: list[int | None] = []
    failed_cases: list[dict[str, Any]] = []
    page_failed_cases: list[dict[str, Any]] = []
    section_failed_cases: list[dict[str, Any]] = []

    for item, results in zip(evaluable_items, batch_results):
        expected_keywords = get_expected_keywords(item)
        expected_pages = get_expected_page_ranges(item)
        expected_sections = get_expected_sections(item)

        if expected_keywords:
            rank = first_matching_rank(
                results,
                lambda result: is_hit(result_match_text(result), expected_keywords),
            )
            keyword_ranks.append(rank)
            if rank is None:
                failed_cases.append(
                    build_failure_case(
                        item,
                        "expected_keywords",
                        expected_keywords,
                        results,
                    )
                )

        if expected_pages:
            rank = first_matching_rank(
                results,
                lambda result: is_page_hit(result, expected_pages),
            )
            page_ranks.append(rank)
            if rank is None:
                page_failed_cases.append(
                    build_failure_case(
                        item,
                        "expected_page_ranges",
                        [list(page_range) for page_range in expected_pages],
                        results,
                    )
                )

        if expected_sections:
            rank = first_matching_rank(
                results,
                lambda result: is_section_hit(result, expected_sections),
            )
            section_ranks.append(rank)
            if rank is None:
                section_failed_cases.append(
                    build_failure_case(
                        item,
                        "expected_sections",
                        expected_sections,
                        results,
                    )
                )

    keyword_metrics = summarize_ranks(keyword_ranks)
    page_metrics = summarize_ranks(page_ranks)
    section_metrics = summarize_ranks(section_ranks)

    metrics = {
        "eval_path": str(eval_path),
        "index_dir": str(index_dir),
        "question_count": len(questions),
        "evaluated_count": keyword_metrics["evaluated_count"],
        "retrieved_question_count": len(evaluable_items),
        "skipped_count": skipped_count,
        "top_k": top_k,
        "top_k_embedding": max(top_k_embedding, top_k),
        "top_k_bm25": max(top_k_bm25, top_k),
        "rerank_batch_size": rerank_batch_size,
        "rrf_k": rrf_k,
        "rrf_weight": rrf_weight,
        "use_hybrid_search": use_hybrid_search,
        "use_reranker": use_reranker,
        "use_query_rewrite": use_query_rewrite,
        "recall_at_1": keyword_metrics["recall_at_1"],
        "recall_at_3": keyword_metrics["recall_at_3"],
        "recall_at_5": keyword_metrics["recall_at_5"],
        "mrr": keyword_metrics["mrr"],
        "hits": keyword_metrics["hits"],
        "keyword_metrics": keyword_metrics,
        "page_metrics": page_metrics,
        "section_metrics": section_metrics,
        "failed_cases": failed_cases,
        "page_failed_cases": page_failed_cases,
        "section_failed_cases": section_failed_cases,
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
    print(f"混合检索: {'开启' if metrics['use_hybrid_search'] else '关闭'}")
    print(f"Reranker: {'开启' if metrics.get('use_reranker', True) else '关闭'}")
    print(f"Query Rewrite: {'开启' if metrics.get('use_query_rewrite', True) else '关闭'}")
    print(f"Embedding Top-K: {metrics['top_k_embedding']}")
    print(f"BM25 Top-K: {metrics['top_k_bm25']}")
    print(f"Rerank Batch Size: {metrics['rerank_batch_size']}")
    print(f"RRF k/weight: {metrics.get('rrf_k', 60)} / {metrics.get('rrf_weight', 1.0)}")
    print(f"Recall@1: {metrics['recall_at_1']:.2%} ({metrics['hits']['hit_at_1']}/{total})")
    print(f"Recall@3: {metrics['recall_at_3']:.2%} ({metrics['hits']['hit_at_3']}/{total})")
    print(f"Recall@5: {metrics['recall_at_5']:.2%} ({metrics['hits']['hit_at_5']}/{total})")
    print(f"MRR: {metrics['mrr']:.4f}")

    for label, key in (("页码", "page_metrics"), ("章节", "section_metrics")):
        grounded = metrics.get(key, {})
        grounded_total = grounded.get("evaluated_count", 0)
        if grounded_total:
            print(
                f"{label} Recall@1/3/5: "
                f"{grounded['recall_at_1']:.2%} / "
                f"{grounded['recall_at_3']:.2%} / "
                f"{grounded['recall_at_5']:.2%} "
                f"(标注 {grounded_total} 题), MRR: {grounded['mrr']:.4f}"
            )

    failed_cases = metrics["failed_cases"]
    page_failed_count = len(metrics.get("page_failed_cases", []))
    section_failed_count = len(metrics.get("section_failed_cases", []))
    if not failed_cases and not page_failed_count and not section_failed_count:
        print("\n全部命中。")
        return

    if page_failed_count or section_failed_count:
        print(
            f"\n结构化标注未命中: 页码 {page_failed_count} 题，"
            f"章节 {section_failed_count} 题"
        )

    if not failed_cases:
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
    parser.add_argument("--top-k-bm25", type=int, default=20)
    parser.add_argument("--rerank-batch-size", type=int, default=32)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rrf-weight", type=float, default=1.0)
    parser.add_argument("--no-hybrid-search", action="store_true")
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--no-query-rewrite", action="store_true")
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
            top_k_bm25=args.top_k_bm25,
            rerank_batch_size=args.rerank_batch_size,
            rrf_k=args.rrf_k,
            rrf_weight=args.rrf_weight,
            use_hybrid_search=not args.no_hybrid_search,
            use_reranker=not args.no_reranker,
            use_query_rewrite=not args.no_query_rewrite,
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
