"""
MathRAG 检索评测脚本（适配 src/retriever 包结构）
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict

# 将项目根目录加入 sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 尝试两种导入方式
try:
    from src.retriever import MathRAGRetriever, RAGConfig
except ImportError:
    try:
        from retriever import MathRAGRetriever, RAGConfig
    except ImportError:
        raise ImportError("无法导入 MathRAGRetriever，请检查 retriever 模块位置")


def load_eval_questions(path: str) -> List[Dict]:
    """加载 jsonl 评测文件"""
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def is_hit(content: str, expected_keywords: List[str]) -> bool:
    """检查 content 是否包含任一 expected_keyword"""
    content_lower = content.lower()
    for kw in expected_keywords:
        if kw.lower() in content_lower:
            return True
    return False


def evaluate_retrieval(
    eval_path: str = "data/eval/questions.jsonl",
    top_k: int = 5,
    use_gpu: bool = False,
):
    # 初始化检索器
    config = RAGConfig(top_k_rerank=top_k, use_gpu=use_gpu)
    retriever = MathRAGRetriever(config)

    questions = load_eval_questions(eval_path)
    total = len(questions)

    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    reciprocal_ranks = []
    failed_cases = []

    for item in questions:
        question = item["question"]
        expected = item.get("expected_chunk_keywords", [])
        if not expected:
            continue

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

        if rank is not None:
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)
            failed_cases.append({
                "id": item.get("id"),
                "question": question,
                "expected": expected,
                "top_results": [
                    result.content[:120] + "..." for result in results[:5]
                ]
            })

    mrr = sum(reciprocal_ranks) / total if total else 0
    recall_1 = hit_at_1 / total if total else 0
    recall_3 = hit_at_3 / total if total else 0
    recall_5 = hit_at_5 / total if total else 0

    print("\n" + "="*50)
    print("📊 检索评测结果")
    print("="*50)
    print(f"总问题数: {total}")
    print(f"Recall@1: {recall_1:.2%} ({hit_at_1}/{total})")
    print(f"Recall@3: {recall_3:.2%} ({hit_at_3}/{total})")
    print(f"Recall@5: {recall_5:.2%} ({hit_at_5}/{total})")
    print(f"MRR: {mrr:.4f}")

    if failed_cases:
        print("\n❌ 未命中案例（前10个）：")
        for case in failed_cases[:10]:
            print("-" * 40)
            print(f"ID: {case['id']}")
            print(f"问题: {case['question']}")
            print(f"期望关键词: {case['expected']}")
            print("Top 5 结果预览:")
            for idx, text in enumerate(case["top_results"], 1):
                print(f"  {idx}. {text}")
    else:
        print("\n🎉 全部命中！")


if __name__ == "__main__":
    # 定位项目根目录
    project_root = Path(__file__).resolve().parent
    while not (project_root / "data").exists():
        if project_root.parent == project_root:
            raise RuntimeError("找不到项目根目录（需包含 data 文件夹）")
        project_root = project_root.parent
    os.chdir(project_root)
    print(f"✅ 当前工作目录: {os.getcwd()}")

    eval_file = Path("data/eval/questions.jsonl")
    if not eval_file.exists():
        print(f"❌ 评测文件不存在: {eval_file}")
        exit(1)

    # 开始评测
    evaluate_retrieval(eval_path=str(eval_file), top_k=5)