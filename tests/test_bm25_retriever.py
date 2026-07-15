from src.retriever.bm25_retriever import BM25Retriever, tokenize
from src.loader.math_text import build_math_search_text


def test_tokenize_keeps_chinese_chars_and_alnum_terms():
    assert tokenize("牛顿-莱布尼茨 formula 123") == [
        "牛",
        "顿",
        "莱",
        "布",
        "尼",
        "茨",
        "formula",
        "123",
    ]


def test_bm25_retrieves_keyword_matching_chunk():
    metadata = {
        0: {
            "title": "导数的定义",
            "text": "导数表示函数在某一点的变化率。",
        },
        1: {
            "title": "牛顿-莱布尼茨公式",
            "text": "牛顿莱布尼茨公式用于计算定积分。",
        },
    }
    retriever = BM25Retriever(metadata)

    results = retriever.retrieve("牛顿莱布尼茨公式是什么", top_k=1)

    assert results[0].vector_id == 1
    assert results[0].score > 0


def test_bm25_uses_math_aliases_without_changing_display_text():
    display_text = "计算 ∫₀¹ x² dx。"
    metadata = {
        0: {
            "title": "定积分公式",
            "text": display_text,
            "search_text": build_math_search_text(display_text),
        }
    }
    retriever = BM25Retriever(metadata)

    results = retriever.retrieve("积分的平方项怎么计算", top_k=1)

    assert results[0].vector_id == 0
    assert results[0].metadata["text"] == display_text


def test_bm25_query_rewrite_expands_student_aliases():
    metadata = {
        0: {
            "title": "洛必达法则",
            "text": "洛必达法则用于处理 0/0 或 ∞/∞ 型未定式极限。",
        }
    }
    retriever = BM25Retriever(metadata)

    assert retriever.retrieve("lhospital conditions", top_k=1)[0].vector_id == 0
    assert retriever.retrieve("lhospital conditions", top_k=1, rewrite_query=False) == []
