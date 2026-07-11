from types import SimpleNamespace

import pytest

from evaluate_retrieval import (
    get_expected_page_ranges,
    get_expected_keywords,
    get_expected_sections,
    is_page_hit,
    is_section_hit,
    is_hit,
    normalize_for_match,
    result_match_text,
    summarize_ranks,
)


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


def test_result_match_text_includes_title_without_changing_content():
    result = SimpleNamespace(title="数列极限与函数极限的关系", content="两者可以互相转化。")

    assert is_hit(result_match_text(result), ["数列极限与函数极限的关系"])
    assert result.content == "两者可以互相转化。"


def test_page_annotations_accept_single_pages_and_ranges():
    item = {"expected_page_ranges": [109, [164, 165]]}

    assert get_expected_page_ranges(item) == [(109, 109), (164, 165)]


def test_page_annotations_reject_invalid_ranges():
    with pytest.raises(ValueError, match="无效页码范围"):
        get_expected_page_ranges({"expected_page_ranges": [[165, 164]]})
    with pytest.raises(ValueError, match="无效页码范围"):
        get_expected_page_ranges({"expected_page_ranges": 0})


def test_page_hit_uses_overlapping_result_ranges():
    result = SimpleNamespace(page_start=108, page_end=110)

    assert is_page_hit(result, [(109, 109)]) is True
    assert is_page_hit(result, [(111, 112)]) is False


def test_section_hit_checks_chapter_section_and_title():
    result = SimpleNamespace(
        chapter="第二章 一元函数微分学",
        section="微分中值定理",
        title="拉格朗日定理",
    )

    assert get_expected_sections({"expected_sections": "拉格朗日定理"}) == [
        "拉格朗日定理"
    ]
    assert is_section_hit(result, ["微分中值定理"]) is True
    assert is_section_hit(result, ["定积分的性质"]) is False
    with pytest.raises(ValueError, match="expected_sections"):
        get_expected_sections({"expected_sections": [158]})


def test_summarize_ranks_reports_recall_and_mrr():
    summary = summarize_ranks([1, 3, None])

    assert summary["evaluated_count"] == 3
    assert summary["recall_at_1"] == pytest.approx(1 / 3)
    assert summary["recall_at_3"] == pytest.approx(2 / 3)
    assert summary["recall_at_5"] == pytest.approx(2 / 3)
    assert summary["mrr"] == pytest.approx((1 + 1 / 3) / 3)
