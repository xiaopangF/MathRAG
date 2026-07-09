from dataclasses import dataclass

from src.pipeline.qa_pipeline import MathRAGPipeline


@dataclass
class FakeChunk:
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


class FakeRetriever:
    def __init__(self, rerank_score=0.91):
        self.rerank_score = rerank_score

    def retrieve(self, query, top_k=3):
        return [
            FakeChunk(
                content="导数表示函数在某一点的变化率。",
                rerank_score=self.rerank_score,
                embedding_score=0.82,
                bm25_score=3.5,
                vector_id=7,
                title="导数的定义",
                chapter="第二章 一元函数微分学",
                section="导数",
                chunk_type="definition",
                file="data/chunks/children/child_0007.txt",
                source_file="高等数学.pdf",
                page_start=47,
                page_end=48,
            )
        ]


class FakeGenerator:
    def __init__(self):
        self.received_contexts = None
        self.call_count = 0

    def generate(self, query, contexts):
        self.call_count += 1
        self.received_contexts = contexts
        return "导数可以理解为函数的变化率。"


def test_pipeline_preserves_retrieval_source_metadata():
    pipeline = object.__new__(MathRAGPipeline)
    pipeline.retriever = FakeRetriever()
    pipeline.generator = FakeGenerator()
    pipeline.min_rerank_score = 0.2

    result = pipeline.ask("什么是导数？")

    assert result["answer"] == "导数可以理解为函数的变化率。"
    assert result["contexts"] == [
        {
            "content": "导数表示函数在某一点的变化率。",
            "score": 0.91,
            "embedding_score": 0.82,
            "bm25_score": 3.5,
            "title": "导数的定义",
            "chapter": "第二章 一元函数微分学",
            "section": "导数",
            "chunk_type": "definition",
            "file": "data/chunks/children/child_0007.txt",
            "source_file": "高等数学.pdf",
            "page_start": 47,
            "page_end": 48,
            "vector_id": 7,
        }
    ]
    assert pipeline.generator.received_contexts == result["contexts"]
    assert pipeline.generator.call_count == 1
    assert result["confidence"]["is_sufficient"] is True


def test_pipeline_refuses_when_retrieval_score_is_too_low():
    pipeline = object.__new__(MathRAGPipeline)
    pipeline.retriever = FakeRetriever(rerank_score=-0.4)
    pipeline.generator = FakeGenerator()
    pipeline.min_rerank_score = 0.2

    result = pipeline.ask("教材里没有的问题")

    assert "未找到足够可靠的依据" in result["answer"]
    assert result["contexts"]
    assert result["confidence"] == {
        "is_sufficient": False,
        "top_rerank_score": -0.4,
        "min_rerank_score": 0.2,
        "reason": "score_threshold",
    }
    assert pipeline.generator.call_count == 0
