from src.retriever.bm25_retriever import BM25Retriever, tokenize


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
