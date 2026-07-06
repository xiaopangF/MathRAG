from src.generation.llm_generator import LLMGenerator


def test_normalize_context_supports_structured_context():
    context = {
        "content": "导数表示函数在某一点的变化率。",
        "score": 0.91,
        "title": "导数的定义",
        "chapter": "第二章 一元函数微分学",
        "section": "导数",
        "chunk_type": "definition",
    }

    result = LLMGenerator._normalize_context(context, 1)

    assert result == {
        "index": 1,
        "content": "导数表示函数在某一点的变化率。",
        "score": 0.91,
        "title": "导数的定义",
        "chapter": "第二章 一元函数微分学",
        "section": "导数",
        "chunk_type": "definition",
    }


def test_normalize_context_supports_legacy_tuple_context():
    result = LLMGenerator._normalize_context(("导数是变化率。", 0.8), 2)

    assert result == {
        "index": 2,
        "content": "导数是变化率。",
        "score": 0.8,
        "title": "",
        "chapter": "",
        "section": "",
        "chunk_type": "",
    }
