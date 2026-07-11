from types import SimpleNamespace

import numpy as np

from src.retriever.retriever import (
    MathRAGRetriever,
    build_ranking_text,
    load_model_cache_first,
)


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


def test_build_ranking_text_includes_title_without_mutating_content():
    content = "两者可以互相转化。"

    ranking_text = build_ranking_text(
        content,
        {"title": "数列极限与函数极限的关系"},
    )

    assert ranking_text == f"数列极限与函数极限的关系\n{content}"


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
