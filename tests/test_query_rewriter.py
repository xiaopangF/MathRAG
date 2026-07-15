from src.retriever.bm25_retriever import tokenize
from src.retriever.query_rewriter import rewrite_query, rewrite_query_for_retrieval


def test_query_rewriter_expands_math_term_aliases_and_intent_words():
    result = rewrite_query("洛必达法则什么时候能用？")

    assert result.original == "洛必达法则什么时候能用？"
    assert "洛必达法则" in result.search_text
    assert "未定式" in result.search_text
    assert "0/0" in result.search_text
    assert "∞/∞" in result.search_text
    assert "infinity" in tokenize(result.search_text)
    assert "无" in tokenize(result.search_text)
    assert "穷" in tokenize(result.search_text)
    assert "适用条件" in result.search_text


def test_query_rewriter_expands_taylor_and_calculation_intent():
    search_text = rewrite_query_for_retrieval("泰勒公式怎么求近似？")

    assert "泰勒公式" in search_text
    assert "泰勒展开" in search_text
    assert "余项" in search_text
    assert "计算" in search_text
    assert "步骤" in search_text


def test_query_rewriter_keeps_math_symbol_normalization():
    search_text = rewrite_query_for_retrieval("∫₀¹ x² dx 怎么算")

    assert "∫₀¹ x² dx" in search_text
    assert "integral" in search_text
    assert "积分" in search_text
    assert "^(2)" in search_text


def test_query_rewriter_maps_english_intent_to_chinese_textbook_terms():
    search_text = rewrite_query_for_retrieval("L'Hopital conditions")

    assert "洛必达法则" in search_text
    assert "适用条件" in search_text
    assert "前提" in search_text


def test_query_rewriter_does_not_confuse_gaussian_with_gauss_theorem():
    result = rewrite_query("Gaussian integral definition")

    assert "Gauss 公式" not in result.expansions
    assert "散度定理" not in result.expansions
