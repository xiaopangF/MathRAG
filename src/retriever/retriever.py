"""
MathRAG 双阶段检索器 + 问答引擎
功能：Embedding粗筛 + Reranker精排 + LLM生成答案
"""
import os
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------- 配置 ----------
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
FAISS_INDEX_PATH = "data/faiss_index"
META_PATH = "data/chunks_meta.pkl"
TOP_K_EMBEDDING = 20  # 第一阶段召回数量
TOP_K_RERANK = 3      # 第二阶段精排后保留数量


class MathRAGRetriever:
    """双阶段检索器"""

    def __init__(self):
        print("🚀 正在初始化检索器...")

        # 1. 加载Embedding模型（用于向量化问题）
        print(f"   - 加载Embedding模型: {EMBEDDING_MODEL}")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)

        # 2. 加载Reranker模型（用于重排序）
        print(f"   - 加载Reranker模型: {RERANKER_MODEL}")
        self.reranker = CrossEncoder(RERANKER_MODEL)

        # 3. 加载FAISS索引
        print(f"   - 加载FAISS索引: {FAISS_INDEX_PATH}")
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        print(f"     ✅ 索引中共有 {self.index.ntotal} 个向量")

        # 4. 加载元数据（知识块内容和标题）
        print(f"   - 加载元数据: {META_PATH}")
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
        self.titles = meta["titles"]
        self.contents = meta["contents"]
        print(f"     ✅ 共加载 {len(self.contents)} 个知识块")

        print("✅ 检索器初始化完成！")

    def retrieve(self, query: str, top_k: int = TOP_K_RERANK) -> List[Tuple[str, float]]:
        """
        双阶段检索：Embedding粗筛 + Reranker精排
        返回: [(content, score), ...] 按相关性从高到低排序
        """
        # ---------- 第一阶段：Embedding检索（粗筛） ----------
        # 将问题向量化
        query_emb = self.embed_model.encode([query])
        faiss.normalize_L2(query_emb)

        # 在FAISS中检索 Top-K_embedding 个最相似的
        scores, indices = self.index.search(query_emb, TOP_K_EMBEDDING)

        # 收集候选知识块
        candidates = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.contents):
                candidates.append({
                    "content": self.contents[idx],
                    "title": self.titles[idx],
                    "embedding_score": float(score)
                })

        # ---------- 第二阶段：Reranker精排 ----------
        # 构造 (query, content) 对，供CrossEncoder打分
        pairs = [(query, c["content"]) for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        # 合并分数并排序
        for i, score in enumerate(rerank_scores):
            candidates[i]["rerank_score"] = float(score)

        # 按rerank分数从高到低排序
        sorted_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        # 返回 top_k 个结果
        results = []
        for c in sorted_candidates[:top_k]:
            results.append((c["content"], c["rerank_score"]))

        return results

    def search(self, query: str, top_k: int = TOP_K_RERANK) -> List[Tuple[str, float]]:
        """简写，等同于retrieve"""
        return self.retrieve(query, top_k)


# ---------- 快速测试 ----------
if __name__ == "__main__":
    # 自动定位项目根目录
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    while not (project_root / "data").exists():
        if project_root.parent == project_root:
            raise RuntimeError("找不到项目根目录")
        project_root = project_root.parent
    os.chdir(project_root)
    print(f"✅ 当前工作目录: {os.getcwd()}")

    # 初始化检索器
    retriever = MathRAGRetriever()

    # 测试检索
    test_queries = [
        "什么是导数？",
        "洛必达法则的适用条件是什么？",
        "如何计算定积分？"
    ]

    print("\n" + "="*60)
    print("🧪 测试双阶段检索效果")
    print("="*60)

    for query in test_queries:
        print(f"\n❓ 问题: {query}")
        print("-"*40)

        results = retriever.retrieve(query, top_k=3)

        for i, (content, score) in enumerate(results):
            print(f"\n  第{i+1}名 (相关性分数: {score:.4f})")
            print(f"  内容预览: {content[:120]}...")