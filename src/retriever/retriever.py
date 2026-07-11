"""
MathRAG 双阶段检索器（工程优化版）
- 保留完整 metadata，按 vector_id 对齐
- 返回结构化检索结果 RetrievedChunk
- 支持 GPU 自动检测
- 缓存机制
- 配置校验
"""
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"

# 设置 HuggingFace 镜像（仅当环境变量未设置时）
os.environ.setdefault("HF_ENDPOINT", os.getenv("HF_ENDPOINT", DEFAULT_HF_ENDPOINT))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.retriever.bm25_retriever import BM25Retriever
from src.loader.math_text import build_math_search_text


def build_ranking_text(content: str, metadata: dict[str, Any]) -> str:
    """Add structural context for ranking without changing displayed content."""
    parts: list[str] = []
    for field in ("chapter", "section", "title"):
        value = str(metadata.get(field) or "").strip()
        if value and value not in parts:
            parts.append(value)
    parts.append(content)
    return "\n".join(parts)


def load_model_cache_first(factory, model_name: str, label: str):
    """Load cached model files first, then allow a remote download if absent."""
    try:
        return factory(model_name, local_files_only=True)
    except (OSError, RuntimeError):
        print(f"   本地未找到完整的 {label} 缓存，尝试联网加载")
        return factory(model_name)


# ============ 配置类 ============
@dataclass
class RAGConfig:
    embedding_model: str = field(
        default_factory=lambda: os.getenv("MATHRAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    )
    reranker_model: str = field(
        default_factory=lambda: os.getenv("MATHRAG_RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
    )
    faiss_index_dir: str = "data/faiss_index"
    index_filename: str = "index.faiss"
    metadata_filename: str = "chunks_meta.jsonl"
    top_k_embedding: int = 20
    top_k_bm25: int = 20
    top_k_rerank: int = 3
    rerank_batch_size: int = 32
    use_hybrid_search: bool = True
    use_gpu: bool = False  # 设为 True 时会尝试使用 GPU
    cache_enabled: bool = True
    cache_ttl: int = 300  # 秒

    def __post_init__(self):
        if self.top_k_embedding < self.top_k_rerank:
            raise ValueError("top_k_embedding 必须大于等于 top_k_rerank")

    @property
    def index_path(self) -> Path:
        return Path(self.faiss_index_dir) / self.index_filename

    @property
    def meta_path(self) -> Path:
        return Path(self.faiss_index_dir) / self.metadata_filename


# ============ 检索结果数据结构 ============
@dataclass
class RetrievedChunk:
    content: str
    rerank_score: float
    embedding_score: float
    vector_id: int
    bm25_score: float = 0.0
    title: str = ""
    parent_id: str = ""
    chapter: str = ""
    section: str = ""
    chunk_type: str = ""
    file: str = ""
    source_file: str = ""
    page_start: int | None = None
    page_end: int | None = None


# ============ 检索器核心类 ============
class MathRAGRetriever:
    """双阶段检索器：Embedding粗筛 + Reranker精排，返回结构化结果"""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self._validate_paths()
        self._load_models()
        self._load_index()
        self._load_metadata()
        self._load_bm25()

        # 缓存：query -> (results, timestamp)
        self._cache: Dict[str, tuple] = {}

    def _validate_paths(self):
        if not self.config.index_path.exists():
            raise FileNotFoundError(f"FAISS 索引不存在: {self.config.index_path}")
        if not self.config.meta_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {self.config.meta_path}")

    def _load_models(self):
        print(f"🚀 加载 Embedding 模型: {self.config.embedding_model}")
        self.embed_model = load_model_cache_first(
            SentenceTransformer,
            self.config.embedding_model,
            "Embedding 模型",
        )

        print(f"🚀 加载 Reranker 模型: {self.config.reranker_model}")
        self.reranker = load_model_cache_first(
            CrossEncoder,
            self.config.reranker_model,
            "Reranker 模型",
        )

        # GPU 支持（自动检测可用性）
        if self.config.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self.embed_model.to('cuda')
                    self.reranker.model.to('cuda')
                    print("   ✅ 已启用 GPU")
                else:
                    print("   ⚠️ GPU 不可用，使用 CPU")
            except ImportError:
                print("   ⚠️ PyTorch 未安装，使用 CPU")

    def _load_index(self):
        print(f"📂 加载 FAISS 索引: {self.config.index_path}")
        self.index = faiss.read_index(str(self.config.index_path))

        # GPU 索引迁移（如果可用且配置启用）
        if self.config.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                print("   ✅ FAISS 索引已迁移到 GPU")
            except Exception as e:
                print(f"   ⚠️ 迁移到 GPU 失败: {e}，使用 CPU")

        print(f"   索引包含 {self.index.ntotal} 个向量")

    def _load_metadata(self):
        """加载 jsonl 元数据，按 vector_id 建立映射"""
        meta_path = self.config.meta_path
        print(f"📂 加载元数据: {meta_path}")

        self.metadata_by_vector_id: Dict[int, Dict[str, Any]] = {}

        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                # 优先使用 vector_id，若没有则用当前索引长度（兼容旧格式）
                vector_id = item.get("vector_id")
                if vector_id is None:
                    vector_id = len(self.metadata_by_vector_id)
                else:
                    vector_id = int(vector_id)
                self.metadata_by_vector_id[vector_id] = item

        if len(self.metadata_by_vector_id) != self.index.ntotal:
            print(
                f"   ⚠️ 警告: 元数据条数 ({len(self.metadata_by_vector_id)}) "
                f"与索引向量数 ({self.index.ntotal}) 不一致"
            )
        else:
            print(f"   ✅ 已加载 {len(self.metadata_by_vector_id)} 条元数据")

    def _load_bm25(self):
        """Build an in-memory BM25 retriever over loaded chunk metadata."""
        if not self.config.use_hybrid_search:
            self.bm25_retriever = None
            return
        print("📚 构建 BM25 词法检索器")
        self.bm25_retriever = BM25Retriever(self.metadata_by_vector_id)
        print(f"   ✅ BM25 文档数: {len(self.bm25_retriever.doc_tokens)}")

    def _encode_query(self, query: str) -> np.ndarray:
        """编码查询并强制转为 float32"""
        search_query = build_math_search_text(query)
        emb = self.embed_model.encode([search_query], normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)

    def _get_cache_key(self, query: str, top_k: int) -> str:
        return f"{query.strip().lower()}_{top_k}"

    def _is_cache_valid(self, key: str) -> bool:
        if not self.config.cache_enabled:
            return False
        if key not in self._cache:
            return False
        _, timestamp = self._cache[key]
        return (time.time() - timestamp) < self.config.cache_ttl

    def _candidate_from_meta(
        self,
        vector_id: int,
        embedding_score: float = 0.0,
        bm25_score: float = 0.0,
    ) -> dict[str, Any] | None:
        meta = self.metadata_by_vector_id.get(vector_id)
        if not meta:
            return None

        content = meta.get("text") or meta.get("content") or ""
        if not content.strip():
            return None

        return {
            "vector_id": vector_id,
            "content": content,
            "embedding_score": float(embedding_score),
            "bm25_score": float(bm25_score),
            "metadata": meta,
        }

    @staticmethod
    def _merge_candidate(
        candidates_by_id: dict[int, dict[str, Any]],
        candidate: dict[str, Any],
    ) -> None:
        vector_id = candidate["vector_id"]
        existing = candidates_by_id.get(vector_id)
        if existing is None:
            candidates_by_id[vector_id] = candidate
            return
        existing["embedding_score"] = max(existing["embedding_score"], candidate["embedding_score"])
        existing["bm25_score"] = max(existing["bm25_score"], candidate["bm25_score"])

    def _build_retrieved_chunk(self, candidate: dict[str, Any]) -> RetrievedChunk:
        meta = candidate["metadata"]
        return RetrievedChunk(
            content=candidate["content"],
            rerank_score=candidate["rerank_score"],
            embedding_score=candidate["embedding_score"],
            bm25_score=candidate.get("bm25_score", 0.0),
            vector_id=candidate["vector_id"],
            title=meta.get("title", ""),
            parent_id=meta.get("parent_id", ""),
            chapter=meta.get("chapter", ""),
            section=meta.get("section", ""),
            chunk_type=meta.get("chunk_type", meta.get("type", "")),
            file=meta.get("file", ""),
            source_file=meta.get("source_file", ""),
            page_start=meta.get("page_start"),
            page_end=meta.get("page_end"),
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        skip_cache: bool = False,
    ) -> List[RetrievedChunk]:
        """
        双阶段检索，返回结构化的 RetrievedChunk 列表。
        """
        if top_k is None:
            top_k = self.config.top_k_rerank

        cache_key = self._get_cache_key(query, top_k)
        if not skip_cache and self._is_cache_valid(cache_key):
            results, _ = self._cache[cache_key]
            return results

        # ---------- 第一阶段：Embedding + BM25 混合召回 ----------
        candidates_by_id: dict[int, dict[str, Any]] = {}

        query_emb = self._encode_query(query)
        scores, indices = self.index.search(query_emb, self.config.top_k_embedding)

        for vector_id, emb_score in zip(indices[0], scores[0]):
            vector_id = int(vector_id)
            if vector_id < 0:
                continue

            candidate = self._candidate_from_meta(vector_id, embedding_score=float(emb_score))
            if candidate:
                self._merge_candidate(candidates_by_id, candidate)

        if self.bm25_retriever is not None:
            bm25_results = self.bm25_retriever.retrieve(query, top_k=self.config.top_k_bm25)
            for item in bm25_results:
                candidate = self._candidate_from_meta(item.vector_id, bm25_score=item.score)
                if candidate:
                    self._merge_candidate(candidates_by_id, candidate)

        candidates = list(candidates_by_id.values())

        if not candidates:
            return []

        # ---------- 第二阶段：Reranker 精排 ----------
        pairs = [
            (query, build_ranking_text(c["content"], c["metadata"]))
            for c in candidates
        ]
        rerank_scores = self.reranker.predict(
            pairs,
            batch_size=self.config.rerank_batch_size,
        )

        for candidate, score in zip(candidates, rerank_scores):
            candidate["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # ---------- 构建返回结果 ----------
        results = [self._build_retrieved_chunk(c) for c in candidates[:top_k]]

        # 缓存
        if self.config.cache_enabled:
            self._cache[cache_key] = (results, time.time())

        return results

    def search(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """简写，等同于 retrieve"""
        return self.retrieve(query, top_k)

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: Optional[int] = None
    ) -> List[List[RetrievedChunk]]:
        """
        批量检索，批量编码查询并集中进行 rerank。
        """
        if top_k is None:
            top_k = self.config.top_k_rerank

        search_queries = [build_math_search_text(query) for query in queries]
        query_embs = self.embed_model.encode(
            search_queries,
            normalize_embeddings=True,
        )
        query_embs = np.asarray(query_embs, dtype=np.float32)

        all_scores, all_indices = self.index.search(query_embs, self.config.top_k_embedding)

        batch_candidates: list[list[dict[str, Any]]] = []
        all_pairs: list[tuple[str, str]] = []
        pair_ranges: list[tuple[int, int]] = []

        for q, scores, indices in zip(queries, all_scores, all_indices):
            candidates_by_id: dict[int, dict[str, Any]] = {}
            for vector_id, emb_score in zip(indices, scores):
                vector_id = int(vector_id)
                if vector_id < 0:
                    continue

                candidate = self._candidate_from_meta(vector_id, embedding_score=float(emb_score))
                if candidate:
                    self._merge_candidate(candidates_by_id, candidate)

            if self.bm25_retriever is not None:
                bm25_results = self.bm25_retriever.retrieve(q, top_k=self.config.top_k_bm25)
                for item in bm25_results:
                    candidate = self._candidate_from_meta(item.vector_id, bm25_score=item.score)
                    if candidate:
                        self._merge_candidate(candidates_by_id, candidate)

            candidates = list(candidates_by_id.values())
            start = len(all_pairs)
            all_pairs.extend(
                (q, build_ranking_text(c["content"], c["metadata"]))
                for c in candidates
            )
            end = len(all_pairs)
            pair_ranges.append((start, end))
            batch_candidates.append(candidates)

        if all_pairs:
            all_rerank_scores = self.reranker.predict(
                all_pairs,
                batch_size=self.config.rerank_batch_size,
            )
        else:
            all_rerank_scores = []

        batch_results = []
        for candidates, (start, end) in zip(batch_candidates, pair_ranges):
            for candidate, score in zip(candidates, all_rerank_scores[start:end]):
                candidate["rerank_score"] = float(score)
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            batch_results.append([
                self._build_retrieved_chunk(c)
                for c in candidates[:top_k]
            ])

        return batch_results


# ============ 快速测试 ============
if __name__ == "__main__":
    # 定位项目根目录
    project_root = Path(__file__).resolve().parent.parent
    while not (project_root / "data").exists():
        if project_root.parent == project_root:
            raise RuntimeError("找不到项目根目录（需包含 data 文件夹）")
        project_root = project_root.parent
    os.chdir(project_root)
    print(f"✅ 当前工作目录: {os.getcwd()}")

    config = RAGConfig()
    if not config.index_path.exists() or not config.meta_path.exists():
        print("⚠️ 索引或元数据文件不存在，请先运行索引构建脚本。")
        print(f"   索引: {config.index_path}")
        print(f"   元数据: {config.meta_path}")
        exit(1)

    retriever = MathRAGRetriever(config)

    test_queries = [
        "什么是导数？",
        "洛必达法则的适用条件是什么？",
        "如何计算定积分？"
    ]

    print("\n" + "="*60)
    print("🧪 测试双阶段检索")
    print("="*60)

    for query in test_queries:
        print(f"\n❓ 问题: {query}")
        print("-"*40)
        results = retriever.retrieve(query, top_k=3)
        if not results:
            print("  没有找到相关内容。")
            continue
        for i, item in enumerate(results, 1):
            print(f"\n  第{i}名")
            print(f"  Rerank 分数: {item.rerank_score:.4f}")
            print(f"  Embedding 分数: {item.embedding_score:.4f}")
            print(f"  标题: {item.title}")
            print(f"  章节: {item.chapter}")
            print(f"  小节: {item.section}")
            print(f"  类型: {item.chunk_type}")
            print(f"  内容预览: {item.content[:120]}...")
