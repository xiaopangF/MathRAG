from types import SimpleNamespace

import numpy as np

from src.retriever.retriever import (
    MathRAGRetriever,
    build_ranking_text,
    load_model_cache_first,
)
from src.retriever.bm25_retriever import BM25Result


class RecordingEmbedder:
    def __init__(self):
        self.sentences = []

    def encode(self, sentences, **kwargs):
        self.sentences = sentences
        return np.zeros((len(sentences), 2), dtype=np.float32)


class EmptyIndex:
    def search(self, query_embeddings, top_k):
        count = len(query_embeddings)
        return (
            np.zeros((count, top_k), dtype=np.float32),
            np.full((count, top_k), -1, dtype=np.int64),
        )


class StaticIndex:
    def search(self, query_embeddings, top_k):
        count = len(query_embeddings)
        scores = np.array([[0.9, 0.8, 0.7][:top_k]] * count, dtype=np.float32)
        indices = np.array([[1, 2, 4][:top_k]] * count, dtype=np.int64)
        return scores, indices


class StaticBM25:
    def __init__(self, metadata):
        self.metadata = metadata
        self.calls = []

    def retrieve(self, query, top_k=20, **kwargs):
        self.calls.append((query, top_k, kwargs))
        return [
            BM25Result(vector_id=2, score=10.0, metadata=self.metadata[2]),
            BM25Result(vector_id=3, score=8.0, metadata=self.metadata[3]),
        ][:top_k]


class RecordingEmptyBM25:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, top_k=20, **kwargs):
        self.calls.append((query, top_k, kwargs))
        return []


def test_build_ranking_text_includes_title_without_mutating_content():
    content = "两者可以互相转化。"

    ranking_text = build_ranking_text(
        content,
        {"title": "数列极限与函数极限的关系"},
    )

    assert ranking_text == f"数列极限与函数极限的关系\n{content}"


def test_build_ranking_text_includes_inherited_structure_once():
    content = "函数在闭区间连续，在开区间可导。"

    ranking_text = build_ranking_text(
        content,
        {
            "chapter": "第二章 一元函数微分学",
            "section": "二、微分中值定理",
            "title": "拉格朗日定理",
        },
    )

    assert ranking_text.splitlines() == [
        "第二章 一元函数微分学",
        "二、微分中值定理",
        "拉格朗日定理",
        content,
    ]


def test_batch_retrieve_normalizes_math_queries_before_embedding():
    retriever = MathRAGRetriever.__new__(MathRAGRetriever)
    retriever.config = SimpleNamespace(
        top_k_rerank=1,
        top_k_embedding=1,
        top_k_bm25=1,
        rerank_batch_size=1,
    )
    retriever.embed_model = RecordingEmbedder()
    retriever.index = EmptyIndex()
    retriever.metadata_by_vector_id = {}
    retriever.bm25_retriever = None
    retriever.reranker = SimpleNamespace(predict=lambda *args, **kwargs: [])

    results = retriever.batch_retrieve(["∫₀¹ x² dx"], top_k=1)

    assert results == [[]]
    assert "integral" in retriever.embed_model.sentences[0]
    assert "^(2)" in retriever.embed_model.sentences[0]


def test_batch_retrieve_can_disable_query_rewrite():
    retriever = MathRAGRetriever.__new__(MathRAGRetriever)
    retriever.config = SimpleNamespace(
        top_k_rerank=1,
        top_k_embedding=1,
        top_k_bm25=1,
        rerank_batch_size=1,
        use_query_rewrite=False,
    )
    retriever.embed_model = RecordingEmbedder()
    retriever.index = EmptyIndex()
    retriever.metadata_by_vector_id = {}
    bm25 = RecordingEmptyBM25()
    retriever.bm25_retriever = bm25
    retriever.reranker = SimpleNamespace(predict=lambda *args, **kwargs: [])

    results = retriever.batch_retrieve(["洛必达法则什么时候能用？"], top_k=1)

    assert results == [[]]
    assert "洛必达法则什么时候能用？" in retriever.embed_model.sentences[0]
    assert "未定式" not in retriever.embed_model.sentences[0]
    assert "适用条件" not in retriever.embed_model.sentences[0]
    assert bm25.calls == [
        ("洛必达法则什么时候能用？", 1, {"rewrite_query": False})
    ]


def test_batch_retrieve_keeps_embedding_query_focused_when_rewrite_is_enabled():
    retriever = MathRAGRetriever.__new__(MathRAGRetriever)
    retriever.config = SimpleNamespace(
        top_k_rerank=1,
        top_k_embedding=1,
        top_k_bm25=1,
        rerank_batch_size=1,
    )
    retriever.embed_model = RecordingEmbedder()
    retriever.index = EmptyIndex()
    retriever.metadata_by_vector_id = {}
    bm25 = RecordingEmptyBM25()
    retriever.bm25_retriever = bm25
    retriever.reranker = SimpleNamespace(predict=lambda *args, **kwargs: [])

    results = retriever.batch_retrieve(["洛必达法则什么时候能用？"], top_k=1)

    assert results == [[]]
    assert "洛必达法则什么时候能用？" in retriever.embed_model.sentences[0]
    assert "未定式" not in retriever.embed_model.sentences[0]
    assert "适用条件" not in retriever.embed_model.sentences[0]
    assert bm25.calls == [
        ("洛必达法则什么时候能用？", 1, {"rewrite_query": True})
    ]


def test_retrieve_uses_rrf_fusion_as_rerank_prior():
    metadata = {
        1: {"text": "只被向量召回的内容", "title": "向量候选"},
        2: {"text": "同时被向量和BM25召回的内容", "title": "融合候选"},
        3: {"text": "只被BM25靠前召回的内容", "title": "词法候选"},
        4: {"text": "向量靠后召回的内容", "title": "向量候选二"},
    }
    retriever = MathRAGRetriever.__new__(MathRAGRetriever)
    retriever.config = SimpleNamespace(
        top_k_rerank=3,
        top_k_embedding=3,
        top_k_bm25=2,
        rerank_batch_size=8,
        rrf_k=60,
        rrf_weight=1.0,
        cache_enabled=False,
        cache_ttl=300,
    )
    retriever.embed_model = RecordingEmbedder()
    retriever.index = StaticIndex()
    retriever.metadata_by_vector_id = metadata
    bm25 = StaticBM25(metadata)
    retriever.bm25_retriever = bm25
    retriever.reranker = SimpleNamespace(
        predict=lambda pairs, **kwargs: [0.0] * len(pairs)
    )
    retriever._cache = {}

    results = retriever.retrieve("融合查询", top_k=3)

    assert [item.vector_id for item in results] == [2, 1, 3]
    assert results[0].embedding_rank == 2
    assert results[0].bm25_rank == 1
    assert results[0].fusion_score > results[1].fusion_score
    assert results[0].retrieval_score == results[0].fusion_score
    assert bm25.calls == [("融合查询", 2, {"rewrite_query": True})]


def test_retrieve_can_disable_reranker_and_rank_by_fusion():
    metadata = {
        1: {"text": "只被向量召回的内容", "title": "向量候选"},
        2: {"text": "同时被向量和 BM25 召回的内容", "title": "融合候选"},
        3: {"text": "只被 BM25 靠前召回的内容", "title": "词法候选"},
        4: {"text": "向量靠后召回的内容", "title": "向量候选二"},
    }
    retriever = MathRAGRetriever.__new__(MathRAGRetriever)
    retriever.config = SimpleNamespace(
        top_k_rerank=3,
        top_k_embedding=3,
        top_k_bm25=2,
        rerank_batch_size=8,
        rrf_k=60,
        rrf_weight=1.0,
        use_reranker=False,
        cache_enabled=False,
        cache_ttl=300,
    )
    retriever.embed_model = RecordingEmbedder()
    retriever.index = StaticIndex()
    retriever.metadata_by_vector_id = metadata
    retriever.bm25_retriever = StaticBM25(metadata)
    retriever.reranker = SimpleNamespace(
        predict=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("reranker should not be called")
        )
    )
    retriever._cache = {}

    results = retriever.retrieve("融合查询", top_k=3)

    assert [item.vector_id for item in results] == [2, 1, 3]
    assert all(item.rerank_score == 0.0 for item in results)
    assert results[0].retrieval_score == results[0].fusion_score


def test_model_loading_prefers_local_cache():
    calls = []

    def factory(model_name, **kwargs):
        calls.append((model_name, kwargs))
        return object()

    loaded = load_model_cache_first(factory, "cached-model", "test model")

    assert loaded is not None
    assert calls == [("cached-model", {"local_files_only": True})]


def test_model_loading_falls_back_to_remote_when_cache_is_missing():
    calls = []

    def factory(model_name, **kwargs):
        calls.append((model_name, kwargs))
        if kwargs.get("local_files_only"):
            raise OSError("cache miss")
        return object()

    loaded = load_model_cache_first(factory, "remote-model", "test model")

    assert loaded is not None
    assert calls == [
        ("remote-model", {"local_files_only": True}),
        ("remote-model", {}),
    ]
