"""
MathRAG 完整问答流水线
功能：用户提问 → 双阶段检索 → LLM生成 → 返回答案
"""
import os
from pathlib import Path
from typing import Dict, Any

from src.retriever.retriever import MathRAGRetriever
from src.generation.llm_generator import LLMGenerator

DEFAULT_MIN_RERANK_SCORE = float(os.getenv("MATHRAG_MIN_RERANK_SCORE", "0.2"))
INSUFFICIENT_CONTEXT_ANSWER = "根据当前教材内容，未找到足够可靠的依据。请换一种问法，或先确认教材知识库中包含相关内容。"


class MathRAGPipeline:
    """完整的问答流水线"""

    def __init__(self, min_rerank_score: float | None = None):
        print("🚀 正在初始化 MathRAG 问答系统...")
        self.retriever = MathRAGRetriever()
        self.generator = LLMGenerator()
        self.min_rerank_score = (
            DEFAULT_MIN_RERANK_SCORE
            if min_rerank_score is None
            else min_rerank_score
        )
        print("✅ 系统初始化完成！")

    def ask(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        print(f"\n❓ 问题: {query}")
        print("   🔍 正在检索相关知识点...")
        retrieved_chunks = self.retriever.retrieve(query, top_k=top_k)

        if not retrieved_chunks:
            return {
                "query": query,
                "contexts": [],
                "answer": "❌ 未找到相关知识，请检查教材内容。",
                "confidence": {
                    "is_sufficient": False,
                    "top_rerank_score": None,
                    "min_rerank_score": getattr(self, "min_rerank_score", DEFAULT_MIN_RERANK_SCORE),
                    "reason": "no_context",
                },
            }

        print(f"   ✅ 检索到 {len(retrieved_chunks)} 个相关片段")
        min_rerank_score = getattr(self, "min_rerank_score", DEFAULT_MIN_RERANK_SCORE)
        top_rerank_score = max(chunk.rerank_score for chunk in retrieved_chunks)
        contexts = [
            {
                "content": chunk.content,
                "score": chunk.rerank_score,
                "embedding_score": chunk.embedding_score,
                "bm25_score": chunk.bm25_score,
                "title": chunk.title,
                "chapter": chunk.chapter,
                "section": chunk.section,
                "chunk_type": chunk.chunk_type,
                "file": chunk.file,
                "source_file": chunk.source_file,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "vector_id": chunk.vector_id,
            }
            for chunk in retrieved_chunks
        ]
        confidence = {
            "is_sufficient": top_rerank_score >= min_rerank_score,
            "top_rerank_score": top_rerank_score,
            "min_rerank_score": min_rerank_score,
            "reason": "score_threshold",
        }

        if not confidence["is_sufficient"]:
            return {
                "query": query,
                "contexts": contexts,
                "answer": INSUFFICIENT_CONTEXT_ANSWER,
                "confidence": confidence,
            }

        print("   🤖 正在生成答案...")
        answer = self.generator.generate(query, contexts)

        return {
            "query": query,
            "contexts": contexts,
            "answer": answer,
            "confidence": confidence,
        }


# ---------- 测试 ----------
if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    while not (project_root / "data").exists():
        if project_root.parent == project_root:
            raise RuntimeError("找不到项目根目录")
        project_root = project_root.parent
    os.chdir(project_root)
    print(f"✅ 当前工作目录: {os.getcwd()}")

    pipeline = MathRAGPipeline()
    test_questions = ["什么是导数？", "洛必达法则的适用条件是什么？"]
    for q in test_questions:
        result = pipeline.ask(q)
        print(f"\n📝 答案:\n{result['answer']}")
        print("\n" + "-"*40 + "\n")
