from dataclasses import dataclass

from src.pipeline.qa_pipeline import MathRAGPipeline


@dataclass
class FakeChunk:
    content: str
    rerank_score: float
    embedding_score: float
    vector_id: int
    title: str = ""
    parent_id: str = ""
    chapter: str = ""
    section: str = ""
    chunk_type: str = ""
    file: str = ""


class FakeRetriever:
    def retrieve(self, query, top_k=3):
        return [
            FakeChunk(
                content="导数表示函数在某一点的变化率。",
                rerank_score=0.91,
                embedding_score=0.82,
                vector_id=7,
                title="导数的定义",
                chapter="第二章 一元函数微分学",
                section="导数",
                chunk_type="definition",
                file="data/chunks/children/child_0007.txt",
            )
        ]


class FakeGenerator:
    def __init__(self):
        self.received_contexts = None

    def generate(self, query, contexts):
        self.received_contexts = contexts
        return "导数可以理解为函数的变化率。"


def test_pipeline_preserves_retrieval_source_metadata():
    pipeline = object.__new__(MathRAGPipeline)
    pipeline.retriever = FakeRetriever()
    pipeline.generator = FakeGenerator()

    result = pipeline.ask("什么是导数？")

    assert result["answer"] == "导数可以理解为函数的变化率。"
    assert result["contexts"] == [
        {
            "content": "导数表示函数在某一点的变化率。",
            "score": 0.91,
            "embedding_score": 0.82,
            "title": "导数的定义",
            "chapter": "第二章 一元函数微分学",
            "section": "导数",
            "chunk_type": "definition",
            "file": "data/chunks/children/child_0007.txt",
            "vector_id": 7,
        }
    ]
    assert pipeline.generator.received_contexts == [
        ("导数表示函数在某一点的变化率。", 0.91)
    ]
