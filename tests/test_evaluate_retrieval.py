from evaluate_retrieval import get_expected_keywords, is_hit, normalize_for_match


def test_get_expected_keywords_prefers_chunk_keywords():
    item = {
        "expected_keywords": ["答案关键词"],
        "expected_chunk_keywords": ["检索关键词"],
    }

    assert get_expected_keywords(item) == ["检索关键词"]


def test_get_expected_keywords_falls_back_to_answer_keywords():
    item = {"expected_keywords": ["导数", "变化率"]}

    assert get_expected_keywords(item) == ["导数", "变化率"]


def test_normalize_for_match_removes_common_ocr_punctuation():
    assert normalize_for_match('牛顿" 莱布尼茨') == "牛顿莱布尼茨"
    assert normalize_for_match("牛顿-莱布尼茨") == "牛顿莱布尼茨"


def test_is_hit_matches_keyword_across_ocr_punctuation():
    content = "上述公式称为牛顿\" 莱布尼茨公式，它是微积分学中的基本公式。"

    assert is_hit(content, ["牛顿-莱布尼茨"])


def test_is_hit_returns_false_when_no_keyword_matches():
    content = "函数在该区间内连续。"

    assert not is_hit(content, ["洛必达法则"])
