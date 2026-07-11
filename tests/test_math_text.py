from src.loader.math_text import (
    build_math_search_text,
    is_formula_candidate,
    normalize_math_text,
)


def test_normalize_math_text_converts_scripts_and_equivalent_operators():
    normalized = normalize_math_text("f(x)=x²+x₁，且 x≤3−y")

    assert "x^(2)" in normalized
    assert "x_(1)" in normalized
    assert "x<=3-y" in normalized


def test_build_math_search_text_preserves_original_and_adds_aliases():
    search_text = build_math_search_text("∫₀¹ x² dx")

    assert "∫₀¹ x² dx" in search_text
    assert "integral" in search_text
    assert "积分" in search_text
    assert "平方" in search_text
    assert "_(0)" in search_text
    assert "^(2)" in search_text


def test_formula_candidate_distinguishes_math_from_normal_prose():
    assert is_formula_candidate("f(x) = x² + 2x + 1") is True
    assert is_formula_candidate("lim x→0 sin(x)/x = 1") is True
    assert is_formula_candidate("这是一个普通的高等数学段落。") is False
