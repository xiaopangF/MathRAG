"""
MathRAG 完整问答流水线
功能：用户提问 → 双阶段检索 → LLM生成 → 返回答案
"""
import os
from pathlib import Path
from typing import Dict, Any

import sys
from pathlib import Path
# 不再需要手动修改 sys.path（app.py 已经做了）
from src.retriever.retriever import MathRAGRetriever
from src.generation.llm_generator import LLMGenerator

class MathRAGPipeline:
    """完整的问答流水线"""

    def __init__(self):
        print("🚀 正在初始化 MathRAG 问答系统...")
        self.retriever = MathRAGRetriever()
        self.generator = LLMGenerator()
        print("✅ 系统初始化完成！")

    def ask(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        处理用户问题
        Returns:
            {
                "query": 原始问题,
                "contexts": 检索到的知识块 [(content, score), ...],
                "answer": 生成的最终答案
            }
        """
        print(f"\n❓ 问题: {query}")

        # 1. 检索
        print("   🔍 正在检索相关知识点...")
        contexts = self.retriever.retrieve(query, top_k=top_k)

        if not contexts:
            return {
                "query": query,
                "contexts": [],
                "answer": "❌ 未找到相关知识，请检查教材内容。"
            }

        print(f"   ✅ 检索到 {len(contexts)} 个相关片段")

        # 2. 生成
        print("   🤖 正在生成答案...")
        answer = self.generator.generate(query, contexts)

        return {
            "query": query,
            "contexts": contexts,
            "answer": answer
        }


# ---------- 测试 ----------
if __name__ == "__main__":
    # 定位项目根目录
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    while not (project_root / "data").exists():
        if project_root.parent == project_root:
            raise RuntimeError("找不到项目根目录")
        project_root = project_root.parent
    os.chdir(project_root)
    print(f"✅ 当前工作目录: {os.getcwd()}")

    # 初始化系统
    pipeline = MathRAGPipeline()

    # 测试问答
    print("\n" + "="*60)
    print("🧪 测试完整问答流程")
    print("="*60)

    test_questions = [
        "什么是导数？",
        "洛必达法则的适用条件是什么？",
        # 可以再加你感兴趣的问题
    ]

    for q in test_questions:
        result = pipeline.ask(q)
        print(f"\n📝 答案:\n{result['answer']}")
        print("\n" + "-"*40 + "\n")